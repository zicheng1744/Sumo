import os
import sys
import time
import logging
import numpy as np
import traci
from gym.spaces import Box

# 配置日志，增加文件处理器
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("sumo_env.log")],
)


class SumoMergeEnv:
    # 初始化环境
    def __init__(self, cfg_path="target_model/input_sources/config.sumocfg", gui=False):
        # 输入参数：cfg_path是SUMO配置文件的路径，gui是是否使用GUI
        # sumo_cmd是启动SUMO的命令，sumo_conn是SUMO连接，cfg_path是配置文件路径，gui是是否使用GUI
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

        # 初始化环境参数
        # episode_length是每个episode的最大步数，current_step是当前步数
        self.episode_length = 1000000
        self.current_step = 0

        # 观测空间和动作空间
        # 观测空间是一个6维的Box空间，动作空间是一个1维的Box空间
        self.observation_space = Box(low=0, high=100, shape=(6,))
        self.action_space = Box(low=-3, high=3, shape=(1,))

        # cav_ids是CAV车辆的ID列表，_is_initialized是环境是否初始化的标志
        self.cav_ids = []
        self.hdv_ids = []
        self._is_initialized = False

        # 设置车辆参数（其实在这里没用）
        """self.vehicle_params = {
            "probs": {"main": 0.5, "ramp": 0.4, "CAV": 0.4},  # 增加生成概率
            "speed": 10.0,  # 最大速度
            "num_vehicles": 50,  # 增加车辆总数
            "accel": 3.0,  # 加速度
            "decel": 5.0,  # 减速度
        }"""

    # 获取观测值
    def _get_observations(self):
        """获取所有CAV车辆的观测状态"""
        logging.debug("获取所有CAV车辆的观测状态...")
        if not self.cav_ids:
            logging.warning("未检测到CAV车辆！")
            logging.info(f"当前所有车辆ID: {traci.vehicle.getIDList()}")
            return np.zeros((1, 6))  # 返回一个默认状态（批次大小为1）

        all_observations = []
        for ego_id in self.cav_ids:
            try:
                logging.debug(f"获取车辆 {ego_id} 的观测状态...")
                ego_speed = traci.vehicle.getSpeed(ego_id)
                logging.debug(f"车辆 {ego_id} 的速度: {ego_speed}")

                # 获取前车信息
                leader_id = traci.vehicle.getLeader(ego_id, 100)
                # 判断leader_id是否存在并且包含了前车ID（leader_id[0]）
                # 在SUMO中，getLeader()会返回最近前车的ID和与该车的距离
                if leader_id and leader_id[0]:
                    leader_speed = traci.vehicle.getSpeed(leader_id[0])
                    leader_dist = leader_id[1]
                else:
                    leader_speed = 0
                    leader_dist = 100

                # 获取后车信息
                follower_id = traci.vehicle.getFollower(ego_id, 100)
                if follower_id and follower_id[0]:
                    follower_speed = traci.vehicle.getSpeed(follower_id[0])
                    follower_dist = follower_id[1]
                else:
                    follower_speed = 0
                    follower_dist = 100

                # 获取车道信息
                current_lane = traci.vehicle.getLaneID(ego_id)
                target_lane = "mc_1" if current_lane == "mc_0" else "mc_0"

                # 构建单车观测
                vehicle_obs = [
                    ego_speed,
                    leader_dist,
                    leader_speed,
                    follower_dist,
                    follower_speed,
                    1 if current_lane == target_lane else 0,
                ]
                all_observations.append(vehicle_obs)

            except Exception as e:
                logging.error(f"获取车辆 {ego_id} 的观测失败: {e}")
                # 使用默认观测
                all_observations.append([0, 100, 0, 100, 0, 0])

        # 如果没有CAV，返回单个全零观测
        if not all_observations:
            return np.zeros((1, 6))

        return np.array(all_observations)

    def _apply_action(self, actions):
        """将动作应用到所有CAV车辆"""
        if not self.cav_ids:
            return

        # 确保动作数量与车辆数量匹配
        actions = np.atleast_2d(actions)
        n_actions = min(len(actions), len(self.cav_ids))

        for i in range(n_actions):
            ego_id = self.cav_ids[i]
            # print(f"ego_id: {ego_id}")
            try:
                current_speed = traci.vehicle.getSpeed(ego_id)
                new_speed = current_speed + actions[i][0]  # 动作是速度调整
                new_speed = np.clip(new_speed, 0, 10)
                traci.vehicle.setSpeed(ego_id, new_speed)
            except Exception as e:
                logging.error(f"应用动作到车辆 {ego_id} 失败: {e}")

    def _calculate_reward(self):
        """计算全局奖励 - 考虑所有CAV车辆的整体表现"""
        if not self.cav_ids:
            return 0

        # 全局效率奖励 - 平均速度
        total_speed = 0
        target_speed = 100  # 期望的平均速度

        # 安全奖励 - 与前车保持安全距离
        safety_reward = 0

        # 协作奖励 - 车辆间速度协调
        speed_variance = 0
        speeds = []

        for ego_id in self.cav_ids:
            try:
                # 计算速度奖励
                current_speed = traci.vehicle.getSpeed(ego_id)
                speeds.append(current_speed)
                total_speed += current_speed

                # 计算安全奖励
                leader = traci.vehicle.getLeader(ego_id, 100)
                if leader:
                    dist = leader[1]
                    if dist < 5:  # 危险距离
                        safety_reward -= 10
                    elif dist < 10:  # 警告距离
                        safety_reward -= 5
            except Exception as e:
                logging.error(f"计算车辆 {ego_id} 的奖励失败: {e}")

        # 计算平均速度
        if len(self.cav_ids) > 0:
            avg_speed = total_speed / len(self.cav_ids)
            speed_reward = -abs(avg_speed - target_speed)  # 鼓励接近目标速度

            # 计算速度方差 (如果有多辆车)
            if len(speeds) > 1:
                speed_variance = np.var(speeds)
                cooperation_reward = -speed_variance  # 鼓励速度协调
            else:
                cooperation_reward = 0
        else:
            speed_reward = 0
            cooperation_reward = 0

        # 综合奖励 = 速度奖励 + 安全奖励 + 协作奖励
        total_reward = speed_reward + safety_reward + cooperation_reward
        return total_reward

    def _check_done(self, safe_check=True):
        try:
            return len(traci.vehicle.getIDList()) == 0
        except Exception as e:
            if safe_check:
                logging.error(f"检查完成状态时出错: {e}")
                return True
            else:
                raise

    def step(self, action):
        if not self._is_initialized:
            self.reset()

        try:
            self._apply_action(action)
            traci.simulationStep()
            # 使用新逻辑更新车辆类型并划分CAV/HDV
            all_vehicles = traci.vehicle.getIDList()
            vehicles = [
                {
                    "id": v_id,
                    "type": traci.vehicle.getTypeID(v_id),
                }
                for v_id in all_vehicles
            ]
            self.cav_ids = [v["id"] for v in vehicles if "CAV" in v["type"]]
            self.hdv_ids = [v["id"] for v in vehicles if "HDV" in v["type"]]

            self.current_step += 1

            obs = self._get_observations()
            reward = self._calculate_reward()
            done = self._check_done()

            return obs, reward, done, {}

        except Exception as e:
            logging.error(f"执行步骤时出错: {e}")
            self.reset()
            return np.zeros((1, 6)), 0, True, {"error": str(e)}

    def reset(self):
        """重置环境"""
        logging.info("开始重置环境...")
        # 先关闭现有连接
        self.close()

        try:

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

            # 初始化步骤
            logging.info("执行初始化仿真步骤...")
            i = 0
            while True:
                traci.simulationStep()
                vehicles = [
                    {
                        "id": v_id,
                        "type": traci.vehicle.getTypeID(v_id),
                    }
                    for v_id in traci.vehicle.getIDList()
                ]
                cav_vehicles = [v for v in vehicles if "CAV" in v["type"]]
                if cav_vehicles:  # 检测到CAV车辆后退出
                    logging.info(
                        f"步骤 {i}: CAV车辆数量 {len(cav_vehicles)}, 车辆ID: {[v['id'] for v in cav_vehicles]}"
                    )
                    break  # 退出循环
                if i % 5 == 0:  # 每隔五步记录一次
                    logging.info(
                        f"步骤 {i}: 车辆数量 {len(vehicles)}, 车辆ID: {vehicles}"
                    )
                i += 1
                if i >= 100:  # 防止无限循环，最多执行20步
                    logging.warning("在50步内未检测到车辆")
                    break

            # 获取所有车辆ID
            all_vehicles = [
                {
                    "id": v_id,
                    "type": traci.vehicle.getTypeID(v_id),
                }
                for v_id in traci.vehicle.getIDList()
            ]
            logging.info(f"当前所有车辆ID: {all_vehicles}")

            # 获取CAV车辆列表
            self.cav_ids = [v["id"] for v in all_vehicles if "CAV" in v["type"]]
            self.hdv_ids = [v["id"] for v in all_vehicles if "HDV" in v["type"]]

            logging.info(
                f"检测到CAV车辆数量: {len(self.cav_ids)}, HDV车辆数量: {len(self.hdv_ids)}"
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
            # 使用相对路径
            import xml.etree.ElementTree as ET

            # 读取配置文件
            # tree是ElementTree对象，root是根元素
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
        logging.info(f"检测CAV车辆数量: {len(env.cav_ids)}")
        logging.info(f"检测HDV车辆数量: {len(env.hdv_ids)}")

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
