import os
import sys
import time
import logging
import numpy as np
import traci
from gym.spaces import Box


class SumoMergeEnv:
    def __init__(self, cfg_path="target_model/input_sources/config.sumocfg", gui=False):
        self.sumo_cmd = (
            ["sumo-gui", "-c", cfg_path] if gui else ["sumo", "-c", cfg_path]
        )
        self.sumo_conn = None
        logging.info("初始化SUMO环境...")
        self.reset()

        self.episode_length = 1000
        self.current_step = 0

        self.observation_space = Box(low=0, high=100, shape=(6,))
        self.action_space = Box(low=-3, high=3, shape=(1,))

        self.cav_ids = []
        for v_id in traci.vehicle.getIDList():
            try:
                v_type = traci.vehicle.getTypeID(v_id)
                if v_type == "CAV":
                    self.cav_ids.append(v_id)
                    logging.info(f"检测到CAV车辆: {v_id}")
            except Exception as e:
                logging.warning(f"获取车辆 {v_id} 类型失败: {e}")

        logging.info(f"初始化完成，检测到CAV车辆数量: {len(self.cav_ids)}")

    def _get_observations(self):
        if not self.cav_ids:
            logging.warning("未检测到CAV车辆！")
            logging.info(f"当前所有车辆ID: {traci.vehicle.getIDList()}")
            return np.zeros(6)

        try:
            ego_id = self.cav_ids[0]
            logging.debug(f"当前控制车辆ID: {ego_id}")
        except IndexError:
            logging.error("CAV车辆列表异常！")
            logging.error(f"当前CAV列表: {self.cav_ids}")
            return np.zeros(6)

        try:
            ego_speed = traci.vehicle.getSpeed(ego_id)
            ego_pos = traci.vehicle.getPosition(ego_id)
            logging.debug(f"车辆 {ego_id} 速度: {ego_speed:.2f}, 位置: {ego_pos}")
        except Exception as e:
            logging.error(f"获取车辆信息失败: {e}")
            return np.zeros(6)

        try:
            leader_id = traci.vehicle.getLeader(ego_id, 100)
            follower_id = traci.vehicle.getFollower(ego_id, 100)

            if leader_id and leader_id[0]:
                leader_speed = traci.vehicle.getSpeed(leader_id[0])
                leader_dist = leader_id[1]
                logging.debug(
                    f"前车ID: {leader_id[0]}, 距离: {leader_dist:.2f}, 速度: {leader_speed:.2f}"
                )
            else:
                leader_speed = 0
                leader_dist = 100
                logging.debug("未检测到前车")

            if follower_id and follower_id[0]:
                follower_speed = traci.vehicle.getSpeed(follower_id[0])
                follower_dist = follower_id[1]
                logging.debug(
                    f"后车ID: {follower_id[0]}, 距离: {follower_dist:.2f}, 速度: {follower_speed:.2f}"
                )
            else:
                follower_speed = 0
                follower_dist = 100
                logging.debug("未检测到后车")
        except Exception as e:
            logging.error(f"获取周围车辆信息失败: {e}")
            return np.zeros(6)

        try:
            current_lane = traci.vehicle.getLaneID(ego_id)
            target_lane = "mc_1" if current_lane == "mc_0" else "mc_0"
            logging.debug(f"当前车道: {current_lane}, 目标车道: {target_lane}")
        except Exception as e:
            logging.error(f"获取车道信息失败: {e}")
            return np.zeros(6)

        return np.array(
            [
                ego_speed,
                leader_dist,
                leader_speed,
                follower_dist,
                follower_speed,
                1 if current_lane == target_lane else 0,
            ]
        )

    def _apply_action(self, action):
        if not self.cav_ids:
            return

        ego_id = self.cav_ids[0]
        current_speed = traci.vehicle.getSpeed(ego_id)
        new_speed = current_speed + action[0]

        new_speed = np.clip(new_speed, 0, 10)
        traci.vehicle.setSpeed(ego_id, new_speed)

    def _calculate_reward(self):
        if not self.cav_ids:
            return 0

        ego_id = self.cav_ids[0]
        current_speed = traci.vehicle.getSpeed(ego_id)
        leader = traci.vehicle.getLeader(ego_id, 100)

        speed_reward = -abs(current_speed - 8)

        if leader:
            dist = leader[1]
            if dist < 5:
                safety_reward = -10
            elif dist < 10:
                safety_reward = -5
            else:
                safety_reward = 0
        else:
            safety_reward = 0

        current_lane = traci.vehicle.getLaneID(ego_id)
        target_lane = "mc_1" if current_lane == "mc_0" else "mc_0"
        lane_reward = 5 if current_lane == target_lane else -5

        return speed_reward + safety_reward + lane_reward

    def _check_done(self):
        if not self.cav_ids:
            return True

        ego_id = self.cav_ids[0]
        return traci.vehicle.getDistance(ego_id) > 300

    def step(self, action):
        self._apply_action(action)
        traci.simulationStep()
        self.current_step += 1

        obs = self._get_observations()
        reward = self._calculate_reward()
        done = self._check_done()

        return obs, reward, done, {}

    def reset(self):
        logging.info("开始重置环境...")
        if traci.isLoaded():
            logging.info("关闭现有SUMO连接...")
            traci.close()
            sys.stdout.flush()
            time.sleep(0.5)

        try:
            logging.info("启动新的SUMO进程...")
            self.sumo_conn = traci.start(self.sumo_cmd)

            logging.info("执行初始化仿真步骤...")
            for i in range(3):
                traci.simulationStep()
                logging.debug(f"完成第 {i+1} 步初始化")

            all_vehicles = traci.vehicle.getIDList()
            logging.info(f"当前所有车辆ID: {all_vehicles}")

            self.cav_ids = []
            for v_id in all_vehicles:
                try:
                    v_type = traci.vehicle.getTypeID(v_id)
                    if v_type == "CAV":
                        self.cav_ids.append(v_id)
                        logging.info(f"检测到CAV车辆: {v_id}")
                except Exception as e:
                    logging.warning(f"获取车辆 {v_id} 类型失败: {e}")

            logging.info(f"检测到CAV车辆数量: {len(self.cav_ids)}")

            if not self.cav_ids:
                logging.error("初始化后未发现CAV车辆！")
                logging.error("可能原因：")
                logging.error("1. routes.rou.xml中车辆类型未设置'CAV'")
                logging.error("2. 车辆生成时间设置过晚")
                logging.error(f"当前所有车辆ID: {all_vehicles}")
                sys.exit(1)
            else:
                logging.info(f"CAV车辆ID列表: {self.cav_ids}")

        except Exception as e:
            logging.error(f"环境重置失败: {e}")
            sys.exit(1)

        logging.info("环境重置完成")
        return self._get_observations()


def create_sumo_env(gui=False):
    return SumoMergeEnv(cfg_path="target_model/input_sources/config.sumocfg", gui=gui)
