import os
import sys
import time
import logging
import numpy as np
import traci
from gym.spaces import Box
import random  # 添加random模块用于随机数生成

# 配置日志，增加文件处理器
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("sumo_env.log")],
)


class SumoMergeEnv:
    def __init__(self, cfg_path="target_model/input_sources/config.sumocfg", gui=False):
        self.sumo_cmd = (
            ["sumo-gui", "-c", cfg_path] if gui else ["sumo", "-c", cfg_path]
        )
        self.sumo_conn = None
        self.cfg_path = cfg_path
        self.gui = gui
        logging.info("初始化SUMO环境...")

        # 检查配置文件路径是否存在
        if not os.path.exists(cfg_path):
            logging.error(f"配置文件不存在: {cfg_path}")
            raise FileNotFoundError(f"配置文件不存在: {cfg_path}")

        self.episode_length = 1000
        self.current_step = 0

        self.observation_space = Box(low=0, high=100, shape=(6,))
        self.action_space = Box(low=-3, high=3, shape=(1,))

        # 初始化环境但延迟reset到首次需要时
        self.cav_ids = []
        self._is_initialized = False

        # 新增：车流参数设置
        self.vehicle_params = {
            "probs": {"main": 0.5, "ramp": 0.4, "CAV": 0.4},  # 增加生成概率
            "speed": 10.0,  # 最大速度
            "num_vehicles": 50,  # 增加车辆总数
            "accel": 3.0,  # 加速度
            "decel": 5.0,  # 减速度
        }

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

    def _check_done(self, safe_check=True):
        if not self.cav_ids:
            return True

        try:
            ego_id = self.cav_ids[0]
            distance = traci.vehicle.getDistance(ego_id)
            return distance > 300  # 行驶超过300米结束
        except Exception as e:
            if safe_check:
                logging.error(f"检查完成状态时出错: {e}")
                return True
            else:
                raise

    def step(self, action):
        # 确保环境已初始化
        if not self._is_initialized:
            self.reset()

        try:
            self._apply_action(action)
            traci.simulationStep()
            self.current_step += 1

            obs = self._get_observations()
            reward = self._calculate_reward()
            done = self._check_done()

            return obs, reward, done, {}

        except Exception as e:
            logging.error(f"执行步骤时出错: {e}")
            # 尝试重新连接
            self.reset()
            return np.zeros(6), 0, True, {"error": str(e)}

    def generate_routefile(self):
        """生成随机车流的路由文件"""
        logging.info("生成随机车流路由文件...")
        probs = self.vehicle_params["probs"]
        speed = self.vehicle_params["speed"]
        N = self.vehicle_params["num_vehicles"]
        accel = self.vehicle_params["accel"]
        decel = self.vehicle_params["decel"]

        # 获取当前工作目录，确保路径正确
        current_dir = os.path.dirname(os.path.abspath(__file__))
        route_file_path = os.path.join(current_dir, "input_sources", "routes.rou.xml")

        # 确保目录存在
        os.makedirs(os.path.dirname(route_file_path), exist_ok=True)

        vehNr = 0  # 初始化车辆编号

        with open(route_file_path, "w") as routes:
            print(
                f"""<routes>
    <vType id="CAV" accel="{accel}" decel="{decel}" sigma="0.0" length="5" minGap="0.8" tau="0.5" maxSpeed="{speed}" carFollowingModel="IDM" lcStrategic="1" lcCooperative="1" lcAssertive="0.5" lcImpatience="0.0" lcKeepRight="0"/>
    <vType id="HDV" accel="{accel}" decel="{decel}" sigma="0.5" length="5" minGap="2.5" tau="1.2" maxSpeed="{speed}" desiredMaxSpeed="{speed}" speedFactor="normc(1,0.2,0.2,2)" carFollowingModel="Krauss" lcStrategic="0.3" lcAssertive="0.5" lcCooperative="0.2" lcImpatience="0.5"/>
    <route id="main" edges="wm mc ce" />
    <route id="ramp" edges="rm mc ce" />
            """,
                file=routes,
            )

            lam1 = probs["main"]
            lam2 = probs["ramp"]
            lam3 = probs["CAV"]

            # 添加初始车辆以确保仿真开始时有车辆
            print(
                f'    <vehicle id="main_0" type="CAV" route="main" depart="0" color="1,0,0" />',
                file=routes,
            )
            print(
                f'    <vehicle id="ramp_0" type="CAV" route="ramp" depart="0" color="1,0,0" />',
                file=routes,
            )
            vehNr = 2  # 已有两辆车

            for i in range(N):
                if random.uniform(0, 1) < lam1:  # 主路车辆泊松分布
                    vehNr += 1
                    if random.uniform(0, 1) < lam3:  # CAV 车辆泊松分布
                        print(
                            f'    <vehicle id="main_{vehNr}" type="CAV" route="main" depart="{i * 0.5}" color="1,0,0" />',
                            file=routes,
                        )
                    else:
                        hdv_depart_speed = random.uniform(0.8 * speed, speed)
                        print(
                            f'    <vehicle id="main_{vehNr}" type="HDV" route="main" depart="{i * 0.5}" departSpeed="{hdv_depart_speed}" />',
                            file=routes,
                        )
                if random.uniform(0, 1) < lam2:  # 支路车辆泊松分布
                    vehNr += 1
                    if random.uniform(0, 1) < lam3:
                        print(
                            f'    <vehicle id="ramp_{vehNr}" type="CAV" route="ramp" depart="{i * 0.5}" color="1,0,0" />',
                            file=routes,
                        )
                    else:
                        hdv_depart_speed = random.uniform(0.8 * speed, speed)
                        print(
                            f'    <vehicle id="ramp_{vehNr}" type="HDV" route="ramp" depart="{i * 0.5}" departSpeed="{hdv_depart_speed}" />',
                            file=routes,
                        )
            print("</routes>", file=routes)

        logging.info(f"路由文件已生成: {route_file_path}")
        return route_file_path

    def reset(self):
        """重置环境"""
        logging.info("开始重置环境...")
        # 先关闭现有连接
        self.close()

        try:
            # 生成新的随机车流
            route_path = self.generate_routefile()

            # 获取SUMO配置文件的绝对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            cfg_path = os.path.join(current_dir, "input_sources", "config.sumocfg")

            # 确认配置文件存在
            if os.path.exists(cfg_path):
                # 更新配置文件引用正确的路由文件
                self._update_config_file(cfg_path)
            else:
                logging.error(f"找不到配置文件: {cfg_path}")
                raise FileNotFoundError(f"找不到配置文件: {cfg_path}")

            logging.info(f"启动SUMO进程，使用配置文件: {cfg_path}")
            self.sumo_cmd = (
                ["sumo-gui", "-c", cfg_path] if self.gui else ["sumo", "-c", cfg_path]
            )

            # 确保没有旧进程占用端口
            time.sleep(1.0)

            # 启动SUMO进程
            try:
                self.sumo_conn = traci.start(self.sumo_cmd)
                logging.info("SUMO进程启动成功")
            except Exception as e:
                logging.error(f"启动SUMO进程失败: {e}")
                raise

            # 等待连接稳定
            time.sleep(1.0)  # 增加等待时间

            # 增加初始化步骤（增加到10步以确保有车辆生成）
            logging.info("执行初始化仿真步骤...")
            for i in range(10):  # 从5步增加到10步
                traci.simulationStep()
                if i % 2 == 0:  # 每隔两步记录一次
                    vehicles = traci.vehicle.getIDList()
                    logging.info(
                        f"步骤 {i}: 车辆数量 {len(vehicles)}, 车辆ID: {vehicles}"
                    )

            # 获取所有车辆ID
            all_vehicles = traci.vehicle.getIDList()
            logging.info(f"当前所有车辆ID: {all_vehicles}")

            # 获取CAV车辆列表（修改识别逻辑）
            self.cav_ids = []
            hdv_ids = []
            for v_id in all_vehicles:
                try:
                    v_type = traci.vehicle.getTypeID(v_id)
                    if v_type == "CAV":
                        self.cav_ids.append(v_id)
                        logging.info(f"检测到CAV车辆: {v_id}")
                    elif v_type == "HDV":
                        hdv_ids.append(v_id)
                        logging.debug(f"检测到HDV车辆: {v_id}")
                except Exception as e:
                    logging.warning(f"获取车辆 {v_id} 类型失败: {e}")

            logging.info(
                f"检测到CAV车辆数量: {len(self.cav_ids)}, HDV车辆数量: {len(hdv_ids)}"
            )

            # 添加车辆存在性检查
            if not self.cav_ids:
                logging.error("初始化后未发现CAV车辆！")
                logging.error("可能原因：")
                logging.error("1. routes.rou.xml中车辆类型未设置'CAV'")
                logging.error("2. 车辆生成时间设置过晚")
                logging.error(f"当前所有车辆ID: {all_vehicles}")
                # 继续运行而非退出
                # sys.exit(1)
            else:
                logging.info(f"CAV车辆ID列表: {self.cav_ids}")

            self._is_initialized = True

        except Exception as e:
            logging.error(f"环境重置失败: {e}")
            # 记录错误但不中断程序
            # sys.exit(1)

        logging.info("环境重置完成")
        return self._get_observations()

    def _update_config_file(self, cfg_path):
        """更新配置文件，确保路由文件路径正确"""
        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(cfg_path)
            root = tree.getroot()

            # 获取路由文件的绝对路径
            route_dir = os.path.dirname(cfg_path)
            route_path = os.path.join(route_dir, "routes.rou.xml")

            # 使用相对于配置文件的路径
            route_rel_path = "routes.rou.xml"

            # 查找路由文件路径设置
            for input_tag in root.findall(".//input"):
                route_files = input_tag.find("route-files")
                if route_files is not None:
                    route_files.set("value", route_rel_path)
                    logging.info(f"已更新配置文件中的路由文件路径为: {route_rel_path}")
                    tree.write(cfg_path)
                    return

            logging.error("未在配置文件中找到route-files元素")
        except Exception as e:
            logging.error(f"更新配置文件失败: {e}")

    def close(self):
        """关闭环境连接"""
        try:
            if traci.isLoaded():
                logging.info("关闭现有SUMO连接...")
                traci.close()
                self.sumo_conn = None
                sys.stdout.flush()
                time.sleep(1.0)  # 增加等待时间确保连接完全关闭
                self._is_initialized = False
        except Exception as e:
            logging.error(f"关闭连接时出错: {e}")


def create_sumo_env(gui=False):
    """创建并返回带gui可视化选项的SUMO环境"""
    # 确保使用绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(current_dir, "input_sources", "config.sumocfg")

    if not os.path.exists(cfg_path):
        logging.error(f"配置文件不存在: {cfg_path}")
        # 检查parent_dir是否存在
        parent_dir = os.path.dirname(current_dir)
        alt_cfg_path = os.path.join(
            parent_dir, "target_model", "input_sources", "config.sumocfg"
        )

        if os.path.exists(alt_cfg_path):
            logging.info(f"使用替代配置文件路径: {alt_cfg_path}")
            cfg_path = alt_cfg_path
        else:
            raise FileNotFoundError(
                f"找不到配置文件，尝试的路径: {cfg_path}, {alt_cfg_path}"
            )

    # 检查config.sumocfg文件是否引用了正确的routes.rou.xml路径
    if os.path.exists(cfg_path):
        # 更新配置文件以确保它引用了正确的路由文件
        update_config_file(cfg_path)

    logging.info(f"使用配置文件: {cfg_path}")
    return SumoMergeEnv(cfg_path=cfg_path, gui=gui)


def update_config_file(cfg_path):
    """更新配置文件，确保路由文件路径正确"""
    try:
        import xml.etree.ElementTree as ET

        tree = ET.parse(cfg_path)
        root = tree.getroot()

        # 查找路由文件路径设置
        for input_tag in root.findall(".//input"):
            route_files = input_tag.find("route-files")
            if route_files is not None:
                route_files.set("value", "routes.rou.xml")
                logging.info(f"已更新配置文件中的路由文件路径为: routes.rou.xml")
                tree.write(cfg_path)
                return
    except Exception as e:
        logging.error(f"更新配置文件失败: {e}")


def test_sumo_env():
    """测试SUMO环境"""
    env = None
    try:
        logging.info("开始测试SUMO环境...")
        env = create_sumo_env(gui=False)

        logging.info("调用reset()...")
        obs = env.reset()
        logging.info(f"获得初始观测值: {obs}")

        logging.info("测试执行动作...")
        for i in range(5):
            action = np.array([0.0])
            obs, reward, done, _ = env.step(action)
            logging.info(f"Step {i}: obs={obs}, reward={reward}, done={done}")
            if done:
                break

        logging.info("环境测试完成")

    except Exception as e:
        logging.error(f"测试过程中出错: {e}")
    finally:
        # 确保连接被关闭
        if env:
            logging.info("关闭环境连接...")
            env.close()


if __name__ == "__main__":
    test_sumo_env()
