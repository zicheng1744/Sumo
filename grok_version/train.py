import os
import sys
import time
import argparse
import logging
import datetime
import numpy as np
import matplotlib.pyplot as plt
import traci
import gymnasium as gym
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList, EvalCallback
# 导入TensorboardCallback，或使用参数启用tensorboard
from stable_baselines3.common.logger import configure
import math
import json
import traceback

# 添加matplotlib字体设置
import matplotlib
# 设置matplotlib使用Agg后端，避免GUI相关问题
matplotlib.use('Agg')
# 设置全局字体为不需要特殊中文支持的字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 确保SUMO环境变量设置正确
if 'SUMO_HOME' not in os.environ:
    sumo_home = 'D:\\sumo'  # 修改为您的SUMO安装路径
    os.environ['SUMO_HOME'] = sumo_home
    sys.path.append(os.path.join(sumo_home, 'tools'))

# 设置日志格式
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"training_{current_time}.log"
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
root_logger = logging.getLogger()
root_logger.setLevel(logging.ERROR)

# 文件处理程序
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)

# 控制台处理程序
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)

class SumoMergeEnv(gym.Env):
    def __init__(self, cfg_path=None, gui=False, need_reset=True, max_episode_length=10000, action_scale=10.0, max_speed=30.0):
        super(SumoMergeEnv, self).__init__()
        # 获取配置文件路径
        if cfg_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            cfg_path = os.path.join(base_dir, "input_sources", "config.sumocfg")

        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"SUMO 配置文件未找到: {cfg_path}")
        
        logging.info(f"初始化SUMO环境: 配置文件={cfg_path}, GUI={gui}")
        self.sumo_cmd = ["sumo-gui" if gui else "sumo", "-c", cfg_path]
        self.cfg_path = cfg_path
        self.gui = gui
        self.episode_length = max_episode_length
        self.current_step = 0
        self.cav_ids = []
        self.hdv_ids = []
        self._is_initialized = False
        self.max_cavs = 50  # 最大支持的 CAV 数量
        self.need_reset = need_reset
        self.action_scale = action_scale  # 动作缩放因子，影响速度变化幅度
        self.max_speed = max_speed  # 最大速度限制
        
        # 新增：拥堵检测和异常检测计数器
        self.congestion_count = 0  # 拥堵检测计数器
        self.low_reward_count = 0  # 低奖励检测计数器
        
        # 初始状态记录
        self.initial_vehicle_count = 0
        self.initial_avg_speed = 0
        
        # 统计数据
        self.stats = {
            'rewards': [],
            'avg_speeds': [],
            'vehicle_counts': [],
            'lead_cav_speeds': []  # 领头CAV速度记录
        }
        
        # 动作历史记录
        self.actions_history = []

        # 固定观测空间和动作空间
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.max_cavs * 7,), dtype=np.float32
        )
        self.action_space = Box(
            low=-1, high=1, shape=(self.max_cavs,), dtype=np.float32
        )
        
        logging.info(f"环境参数: 动作缩放={self.action_scale}, 最大速度={self.max_speed}, 最大回合长度={self.episode_length}")
        
        # 如果需要在初始化时就重置环境
        if need_reset:
            self.reset()

    def _get_observations(self):
        """获取所有 CAV 的观测并展平"""
        if not self.cav_ids:
            return np.zeros(self.max_cavs * 7, dtype=np.float32)

        obs = []
        for cav_id in self.cav_ids:
            try:
                speed = traci.vehicle.getSpeed(cav_id)
                leader = traci.vehicle.getLeader(cav_id, 100)
                
                # 修改：添加无领车标志，修改默认值
                leader_exists = 1 if leader and leader[0] else 0
                leader_dist = leader[1] if leader and leader[0] else 100
                leader_speed = (
                    traci.vehicle.getSpeed(leader[0]) if leader and leader[0] else 0  # 修改默认值为0
                )
                
                follower = traci.vehicle.getFollower(cav_id, 100)
                follower_dist = follower[1] if follower and follower[0] else 100
                follower_speed = (
                    traci.vehicle.getSpeed(follower[0])
                    if follower and follower[0]
                    else 0
                )
                lane = traci.vehicle.getLaneID(cav_id)
                target_lane = "mc_1" if lane == "mc_0" else "mc_0"
                lane_flag = 1 if lane == target_lane else 0
                
                # 修改：添加无领车标志到观测中
                obs.extend(
                    [
                        speed,
                        leader_dist,
                        leader_speed,
                        follower_dist,
                        follower_speed,
                        lane_flag,
                        leader_exists,  # 新增：无领车标志
                    ]
                )
            except Exception as e:
                logging.warning(f"获取 {cav_id} 观测失败: {e}")
                obs.extend([0, 100, 0, 100, 0, 0, 0])  # 添加对应的默认值

        # 填充到固定长度
        flat_obs = np.zeros(self.max_cavs * 7, dtype=np.float32)
        flat_obs[: len(obs)] = obs
        return flat_obs

    def _apply_action(self, actions):
        """应用动作到所有 CAV"""
        actions = actions * self.action_scale  # 使用增大的动作范围
        
        # 确保actions的长度与cav_ids匹配
        if len(self.cav_ids) > 0:
            # 截断actions到当前CAV数量
            actions_to_apply = actions[:len(self.cav_ids)]
            
            lead_cavs = []  # 记录领头CAV
            
            # 识别所有领头CAV（前方无车或距离很远的CAV）
            for i, cav_id in enumerate(self.cav_ids):
                try:
                    leader = traci.vehicle.getLeader(cav_id, 100)
                    if not leader or leader[1] > 50:  # 前方无车或距离很远
                        lead_cavs.append((i, cav_id))
                except Exception:
                    pass
            
            for i, cav_id in enumerate(self.cav_ids):
                try:
                    current_speed = traci.vehicle.getSpeed(cav_id)
                    
                    # 对领头CAV特殊处理，鼓励加速
                    if (i, cav_id) in lead_cavs:
                        # 如果是领头CAV且速度较低，施加额外加速
                        if current_speed < self.max_speed * 0.5:
                            new_speed = np.clip(current_speed + max(actions_to_apply[i], 0.5), 0, self.max_speed)
                            # 记录首车速度变化日志
                            logging.info(f"Step {self.current_step}, 领头CAV {cav_id}: 当前速度={current_speed:.2f}, 动作值={actions_to_apply[i]:.2f}, 新速度={new_speed:.2f}")
                        else:
                            new_speed = np.clip(current_speed + actions_to_apply[i], 0, self.max_speed)
                    else:
                        new_speed = np.clip(current_speed + actions_to_apply[i], 0, self.max_speed)
                    
                    traci.vehicle.setSpeed(cav_id, new_speed)
                except Exception as e:
                    logging.warning(f"应用动作到 {cav_id} 失败: {e}")

    def _identify_lead_cavs(self):
        """识别所有领头CAV（前方无车或距离很远的CAV）"""
        lead_cavs = []
        lead_speeds = []
        
        for cav_id in self.cav_ids:
            try:
                leader = traci.vehicle.getLeader(cav_id, 100)
                if not leader or leader[1] > 50:  # 前方无车或距离很远
                    lead_cavs.append(cav_id)
                    lead_speeds.append(traci.vehicle.getSpeed(cav_id))
            except Exception:
                pass
                
        return lead_cavs, lead_speeds

    def _calculate_reward(self):
        """计算当前状态的奖励值，标准化到0-1范围"""
        # 安全检查：如果没有车辆，返回中等奖励
        if not self.cav_ids:
            return 0.5  # 中等奖励，既不奖励也不惩罚
        
        # --- 速度奖励部分 ---
        # 计算当前所有CAV的平均速度
        speeds = []
        for cav_id in self.cav_ids:
            try:
                speed = traci.vehicle.getSpeed(cav_id)
                speeds.append(speed)
            except traci.exceptions.TraCIException:
                continue  # 忽略可能消失的车辆
        
        if not speeds:  # 如果无法获取任何速度
            return 0.5  # 中等奖励
        
        avg_speed = sum(speeds) / len(speeds)
        
        # 将平均速度标准化为0-1分数，假设理想速度为最大速度的80%
        ideal_speed = self.max_speed * 0.8
        speed_score = min(avg_speed / ideal_speed, 1.0)  # 限制在1.0以内
        
        # --- 领头车辆速度奖励 ---
        # 获取领头CAV及其速度
        lead_cavs, lead_speeds = self._identify_lead_cavs()
        
        # 计算领头CAV平均速度的奖励
        lead_speed_score = 0.0
        if lead_speeds:
            avg_lead_speed = sum(lead_speeds) / len(lead_speeds)
            # 领头车速标准化
            lead_speed_score = min(avg_lead_speed / ideal_speed, 1.0)
        
        # --- 总体奖励计算 ---
        # 权重分配
        w_speed = 0.7      # 速度奖励权重
        w_lead = 0.3        # 领头车辆速度奖励权重

        
        # 计算总奖励，标准化到0-1范围
        reward = (
            w_speed * speed_score +
            w_lead * lead_speed_score
        )
        
        # 记录日志
        if self.current_step % 100 == 0:  # 每100步记录一次，减少日志量
            logging.info(f"奖励计算: 速度={speed_score:.2f}, 领头速度={lead_speed_score:.2f}, 总奖励={reward:.2f}")
        
        return reward  # 最终奖励在0-1范围内

    def step(self, action):
        """执行一个步骤，应用动作并获取观测、奖励和完成状态"""
        if not self._is_initialized:
            self.reset()
        
        # 保存动作历史
        self.actions_history.append(action)
        
        # 应用动作
        self._apply_action(action)
        
        # 获取下一个状态
        traci.simulationStep()
        self.current_step += 1
        
        # 更新车辆ID列表
        self._update_vehicle_ids()
        
        # 获取观测
        obs = self._get_observations()
        
        # 计算奖励
        reward = self._calculate_reward()
        
        # 判断是否完成
        done = self._is_done()
        
        # 计算并保存统计信息
        speeds = [traci.vehicle.getSpeed(veh_id) for veh_id in self.cav_ids if veh_id in traci.vehicle.getIDList()]
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        self.stats['avg_speeds'].append(avg_speed)
        self.stats['vehicle_counts'].append(len(self.cav_ids))
        self.stats['rewards'].append(reward)
        
        # 记录领头CAV速度
        lead_cavs, lead_speeds = self._identify_lead_cavs()
        avg_lead_speed = sum(lead_speeds) / len(lead_speeds) if lead_speeds else 0
        self.stats['lead_cav_speeds'].append(avg_lead_speed)
        
        # 早期终止检查: 检查拥堵情况
        if avg_speed < (self.max_speed * 0.1) and len(self.cav_ids) > 5:  # 如果平均速度低于最大速度的10%且有足够车辆
            self.congestion_count += 1
        else:
            self.congestion_count = 0
            
        # 早期终止检查: 检查长时间低奖励
        # 由于奖励已经标准化到0-1范围，使用更合适的阈值
        if reward < 0.3:  # 修改：使用0.3作为阈值，之前的50远远超过了标准化奖励范围
            self.low_reward_count += 1
        else:
            self.low_reward_count = 0
            
        # 终止条件: 如果拥堵持续过长或连续低奖励，提前结束
        # 增加计数器阈值，避免过早终止
        early_termination = (self.congestion_count > 100) or (self.low_reward_count > 200)
        # 终止条件: 如果没有车辆了，提前结束
        no_vehicles = len(self.cav_ids) == 0 and len(self.hdv_ids) == 0
        
        # 判断终止(terminated)和截断(truncated)
        # terminated: 环境自然终止（如无车辆、拥堵或持续低奖励）
        terminated = no_vehicles or early_termination
        # truncated: 由于外部原因提前结束（如达到最大步数）
        truncated = self.current_step >= self.episode_length
        
        # 准备信息字典
        info = {
            "current_step": self.current_step,
            "max_steps": self.episode_length,
            "cav_count": len(self.cav_ids),
            "hdv_count": len(self.hdv_ids),
            "avg_speed": avg_speed,
            "avg_lead_speed": avg_lead_speed,
            "congestion_count": self.congestion_count,
            "low_reward_count": self.low_reward_count,
            "early_termination": early_termination,
            "no_vehicles": no_vehicles,
            "is_success": truncated,  # 如果达到最大步数则视为成功
        }
        
        # 如果回合结束，打印详细信息
        if terminated or truncated:
            reason = "到达最大步数"
            if early_termination:
                if self.congestion_count > 100:
                    reason = "交通拥堵"
                else:
                    reason = "持续低奖励"
            elif no_vehicles:
                reason = "无车辆"
                
            logging.info(f"回合结束 (原因: {reason}) - 步数: {self.current_step}, 平均速度: {avg_speed:.2f}, "
                        f"车辆数: {len(self.cav_ids)}, 奖励: {reward:.2f}")
        
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """重置环境，可以包含预热阶段"""
        # 关闭现有的SUMO连接
        self.close()
        
        # 等待一小段时间确保资源释放
        time.sleep(0.5)
        
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
        
        # 重置计数器
        self.congestion_count = 0  # 拥堵检测计数器
        self.low_reward_count = 0  # 低奖励检测计数器
        
        # 清空统计信息
        self.stats = {
            'rewards': [],
            'avg_speeds': [],
            'vehicle_counts': [],
            'lead_cav_speeds': []  # 重置领头CAV速度统计
        }
        
        # 重置动作历史
        self.actions_history = []
        
        # 强制确保没有活动的TRACI连接
        try:
            # 检查是否有活动连接
            connection_exists = False
            try:
                connection_exists = traci.isLoaded()
            except:
                connection_exists = False
                
            if connection_exists:
                try:
                    traci.close()
                    logging.info("关闭已有的TRACI连接")
                except Exception as e:
                    logging.warning(f"关闭已有TRACI连接时出现警告: {e}")
                time.sleep(1)  # 等待连接完全关闭
        except Exception as e:
            logging.warning(f"检查TRACI连接状态时出现异常: {e}")
            # 继续尝试，因为可能是连接已经关闭导致的错误
        
        # 尝试启动SUMO，有重试机制
        max_tries = 3
        for attempt in range(max_tries):
            try:
                # 使用唯一标签启动新连接，避免默认连接冲突
                unique_label = f"sim_{int(time.time())}_{attempt}"
                traci.start(self.sumo_cmd, label=unique_label)
                self._is_initialized = True
                logging.info(f"成功启动SUMO (尝试 {attempt+1}/{max_tries}), 连接标签: {unique_label}")
                break
            except Exception as e:
                logging.error(f"SUMO 启动失败 (尝试 {attempt+1}/{max_tries}): {e}")
                # 尝试清理可能存在的连接
                try:
                    # 安全地尝试切换和关闭
                    try:
                        traci.switch(unique_label)
                        traci.close()
                    except:
                        pass
                except:
                    pass
                time.sleep(2)  # 等待一段时间再重试
                if attempt == max_tries - 1:  # 最后一次尝试失败
                    raise RuntimeError(f"无法启动SUMO: {e}")

        # 预热阶段：运行10秒（约100步），让车流稳定下来
        prewarming_steps = 100  # 10秒（每步0.1秒）
        logging.info(f"开始环境预热: {prewarming_steps}步")
        
        # 初始化车辆列表
        all_vehicles = []
        self.cav_ids = []
        self.hdv_ids = []
        
        # 执行预热阶段的模拟
        for step in range(prewarming_steps):
            try:
                traci.simulationStep()
                all_vehicles = traci.vehicle.getIDList()
                self.cav_ids = [v for v in all_vehicles if "CAV" in traci.vehicle.getTypeID(v)]
                self.hdv_ids = [v for v in all_vehicles if "HDV" in traci.vehicle.getTypeID(v)]
                
                # 记录预热阶段的信息
                if step % 20 == 0 or (step == prewarming_steps - 1):
                    logging.info(f"预热步骤 {step}: 总车辆={len(all_vehicles)}, CAV={len(self.cav_ids)}, HDV={len(self.hdv_ids)}")
            except Exception as e:
                logging.error(f"预热阶段出错: {e}")
                self.close()
                time.sleep(1)
                # 重新启动SUMO，使用新标签
                unique_label = f"sim_retry_{int(time.time())}"
                traci.start(self.sumo_cmd, label=unique_label)
        
        # 如果预热后仍无CAV车辆，继续等待直到出现
        if not self.cav_ids:
            logging.warning("预热后无CAV车辆，继续等待...")
            for step in range(50):  # 额外等待5秒
                traci.simulationStep()
                all_vehicles = traci.vehicle.getIDList()
                self.cav_ids = [v for v in all_vehicles if "CAV" in traci.vehicle.getTypeID(v)]
                if self.cav_ids:
                    logging.info(f"在额外等待的第{step}步检测到CAV车辆: {len(self.cav_ids)}辆")
                    break
        
        # 如果仍无CAV车辆，这是一个问题
        if not self.cav_ids:
            logging.error("预热后仍无CAV车辆，请检查车流生成设置!")
        
        # 重置步数计数
        self.current_step = 0
        
        # 获取初始观测
        obs = self._get_observations()
        
        # 记录初始状态
        self._record_initial_state()
        
        # 创建info字典
        info = {
            "cav_count": len(self.cav_ids),
            "hdv_count": len(self.hdv_ids),
            "total_count": len(all_vehicles),
            "is_success": False,
            "prewarming_steps": prewarming_steps
        }
        
        return obs, info
    
    def _record_initial_state(self):
        """记录环境初始状态，用于后续评估"""
        # 记录初始车辆数量
        self.initial_vehicle_count = len(self.cav_ids) + len(self.hdv_ids)
        
        # 记录初始平均速度
        total_speed = sum([traci.vehicle.getSpeed(veh_id) for veh_id in (self.cav_ids + self.hdv_ids)])
        self.initial_avg_speed = total_speed / self.initial_vehicle_count if self.initial_vehicle_count > 0 else 0
        
        logging.info(f"初始状态: 车辆数={self.initial_vehicle_count}, 平均速度={self.initial_avg_speed:.2f}")

    def close(self):
        """关闭环境，释放TRACI连接资源"""
        try:
            # 先检查连接是否仍然存在
            connection_exists = False
            try:
                connection_exists = traci.isLoaded()
            except:
                connection_exists = False
                
            if connection_exists:
                # 记录当前连接标签以便调试
                try:
                    connection_label = traci.getConnection()
                    logging.info(f"正在关闭TRACI连接: {connection_label}")
                except:
                    logging.info("正在关闭TRACI连接")
                
                # 关闭连接
                try:
                    traci.close()
                    self._is_initialized = False
                except Exception as e:
                    logging.warning(f"关闭TRACI连接时出现警告: {e}")
                
                # 等待确保资源释放
                time.sleep(0.5)
            else:
                logging.info("没有活动的TRACI连接需要关闭")
        except Exception as e:
            # 降级为警告级别，避免错误日志
            logging.warning(f"关闭TRACI连接过程中出现异常: {e}")
            
        # 无论如何都要确保环境标记为未初始化
        self._is_initialized = False
    
    def _update_vehicle_ids(self):
        """更新车辆ID列表"""
        try:
            all_vehicles = traci.vehicle.getIDList()
            self.cav_ids = [v for v in all_vehicles if "CAV" in traci.vehicle.getTypeID(v)]
            self.hdv_ids = [v for v in all_vehicles if "HDV" in traci.vehicle.getTypeID(v)]
        except traci.exceptions.TraCIException as e:
            logging.warning(f"更新车辆ID时发生TraCI异常: {e}")
            self.cav_ids = []
            self.hdv_ids = []

    def _is_done(self):
        """判断当前回合是否应该自然终止（不包括达到最大步数的情况）
        
        Returns:
            bool: 如果环境应该自然终止，则返回True，否则返回False
        """
        # 注意：到达最大步数的情况将在step方法中被作为truncated处理
        # 这个方法只检查其他终止条件，用于决定terminated的值
        
        # 检查是否所有车辆已离开
        if len(self.cav_ids) == 0 and len(self.hdv_ids) == 0:
            logging.info("回合结束: 所有车辆已离开")
            return True
        
        # 其他可能的终止条件可以在这里添加
        
        return False


# 自定义回调，用于记录训练信息
class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0, save_dir="./results", save_freq=1000):
        super(TrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.average_speeds = []  # 新增：记录每个episode的平均速度
        self.vehicle_counts = []  # 新增：记录每个episode的平均车辆数
        self.success_rates = []   # 新增：记录每个episode的成功率
        self.episode_count = 0
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.save_dir = save_dir
        self.save_freq = save_freq
        
        # 用于记录当前episode的统计数据
        self.current_speeds = []
        self.current_vehicle_counts = []
        
        # 确保结果目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 创建CSV文件记录每回合的结果
        # 修改：使用当前时间戳创建唯一的CSV文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # 获取data目录（在上一级目录中）
        data_dir = os.path.dirname(os.path.dirname(self.save_dir)) + "/data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.csv_file = os.path.join(data_dir, f"training_episodes_{timestamp}.csv")
        with open(self.csv_file, 'w', encoding='utf-8') as f:
            f.write("episode,reward,length,avg_speed,avg_vehicle_count,success\n")

    def _on_training_start(self) -> None:
        # 不再记录INFO级别的日志
        pass

    def _on_step(self) -> bool:
        # 累积奖励和步数
        self.current_episode_reward += self.locals["rewards"][0]
        self.current_episode_length += 1
        
        # 收集当前步骤的统计信息
        info = self.locals["infos"][0]
        if "avg_speed" in info:
            self.current_speeds.append(info["avg_speed"])
        if "cav_count" in info:
            self.current_vehicle_counts.append(info["cav_count"])
        
        # 检查是否完成一个回合 (终止或截断)
        # 在新的Gymnasium API中，episode_done = terminated or truncated
        terminated = self.locals.get("terminated", [False])[0]
        truncated = self.locals.get("truncated", [False])[0]
        episode_done = terminated or truncated
        
        if episode_done:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # 计算并保存本回合的平均统计数据
            avg_speed = sum(self.current_speeds) / len(self.current_speeds) if self.current_speeds else 0
            avg_vehicle_count = sum(self.current_vehicle_counts) / len(self.current_vehicle_counts) if self.current_vehicle_counts else 0
            self.average_speeds.append(avg_speed)
            self.vehicle_counts.append(avg_vehicle_count)
            
            # 判断回合是否成功（例如，如果达到最大步数则视为成功）
            is_success = info.get("is_success", False)
            self.success_rates.append(1 if is_success else 0)
            
            # 修改：使用ERROR级别记录回合完成信息
            if self.episode_count % 5 == 0:  # 每5回合输出一次，减少输出频率
                logging.error(f"回合 {self.episode_count} 完成: 奖励={self.current_episode_reward:.2f}, 平均速度={avg_speed:.2f}")
            
            # 将结果写入CSV文件
            with open(self.csv_file, 'a', encoding='utf-8') as f:
                f.write(f"{self.episode_count},{self.current_episode_reward:.6f},{self.current_episode_length},"
                       f"{avg_speed:.6f},{avg_vehicle_count:.6f},{1 if is_success else 0}\n")
            
            # 重置当前回合的统计信息
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.current_speeds = []
            self.current_vehicle_counts = []
            
            # 每5个回合更新图表（但不保存，只在训练结束时保存）
            if self.episode_count % 5 == 0:
                self.update_plots()
        
        # 如果达到保存频率且不是训练结束，只保存临时文件，不保存模型
        # 模型只在训练结束时保存
        return True

    def _on_rollout_end(self) -> None:
        # 轻量级输出，不再详细记录rollout信息
        pass

    def _on_training_end(self) -> None:
        # 训练结束时保存所有图表
        self.plot_training_progress()
    
    def update_plots(self):
        """更新图表但不保存，只在内存中更新以便于训练结束时保存"""
        if not self.episode_rewards:
            return
        
        # 此方法只更新图表数据，但不保存，减少文件I/O
        pass
    
    def plot_training_progress(self):
        """绘制训练进度的图表，专注于episode-based的统计"""
        if not self.episode_rewards:
            return
            
        # 绘制回合奖励曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.episode_rewards) + 1), self.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, "episode_rewards.png"))
        plt.close()
        
        # 绘制回合长度曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.episode_lengths) + 1), self.episode_lengths)
        plt.title('Episode Length')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, "episode_lengths.png"))
        plt.close()
        
        # 绘制平均速度曲线
        if self.average_speeds:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self.average_speeds) + 1), self.average_speeds)
            plt.title('Average Speed per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Speed (m/s)')
            plt.grid(True)
            plt.savefig(os.path.join(self.save_dir, "episode_avg_speeds.png"))
            plt.close()
        
        # 绘制平均车辆数曲线
        if self.vehicle_counts:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self.vehicle_counts) + 1), self.vehicle_counts)
            plt.title('Average Vehicle Count per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Vehicle Count')
            plt.grid(True)
            plt.savefig(os.path.join(self.save_dir, "episode_vehicle_counts.png"))
            plt.close()
        
        # 绘制成功率曲线
        if self.success_rates:
            # 计算滑动平均成功率
            window_size = min(10, len(self.success_rates))
            if window_size > 0:
                success_avg = np.convolve(self.success_rates, np.ones(window_size)/window_size, mode='valid')
                plt.figure(figsize=(10, 6))
                plt.plot(range(window_size, len(self.success_rates) + 1), success_avg)
                plt.title(f'{window_size}-Episode Moving Average Success Rate')
                plt.xlabel('Episode')
                plt.ylabel('Success Rate')
                plt.ylim(0, 1)
                plt.grid(True)
                plt.savefig(os.path.join(self.save_dir, "episode_success_rate.png"))
                plt.close()
        
        # 计算并绘制移动平均奖励
        window_size = min(10, len(self.episode_rewards))
        if window_size > 0:
            moving_avg = np.convolve(self.episode_rewards, np.ones(window_size)/window_size, mode='valid')
            plt.figure(figsize=(10, 6))
            plt.plot(range(window_size, len(self.episode_rewards) + 1), moving_avg)
            plt.title(f'{window_size}-Episode Moving Average Reward')
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
            plt.grid(True)
            plt.savefig(os.path.join(self.save_dir, "moving_avg_rewards.png"))
            plt.close()
        
        print(f"训练图表已保存到 {self.save_dir}")


# 在TrainingCallback类之后，添加新的PPOMetricsCallback类
class PPOMetricsCallback(BaseCallback):
    """记录PPO训练指标的回调"""
    def __init__(self, verbose=0, save_dir="./results"):
        super(PPOMetricsCallback, self).__init__(verbose)
        self.save_dir = save_dir
        self.metrics_history = {
            'approx_kl': [],
            'clip_fraction': [],
            'entropy_loss': [],
            'explained_variance': [],
            'learning_rate': [],
            'loss': [],
            'n_updates': [],
            'policy_gradient_loss': [],
            'std': [],
            'value_loss': [],
            'fps': [],
            'time_elapsed': [],
            'total_timesteps': [],
            'iterations': []
        }
        self.reward_decay_analysis = {
            'episode': [],
            'reward': [],
            'episode_length': [],
            'avg_speed': [],
            'vehicle_count': [],
            'decay_rate': []  # 奖励衰减率
        }
        
        # 确保结果目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 创建CSV文件
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = os.path.join(save_dir, f"ppo_metrics.csv")
        self.reward_analysis_file = os.path.join(save_dir, f"reward_decay.csv")
        
        with open(self.csv_file, 'w', encoding='utf-8') as f:
            header = ','.join(self.metrics_history.keys())
            f.write(f"{header}\n")
            
        with open(self.reward_analysis_file, 'w', encoding='utf-8') as f:
            header = ','.join(self.reward_decay_analysis.keys())
            f.write(f"{header}\n")

    def _on_step(self) -> bool:
        # 不需要每步执行操作
        return True

    def _on_rollout_end(self) -> None:
        # 结束一个rollout后，收集指标数据
        for key in self.metrics_history.keys():
            if key in self.locals:
                self.metrics_history[key].append(self.locals[key])
                
        # 分析奖励衰减
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            last_episode = self.model.ep_info_buffer[-1]
            if 'r' in last_episode and 'l' in last_episode:
                episode = len(self.reward_decay_analysis['episode']) + 1
                reward = last_episode['r']
                episode_length = last_episode['l']
                
                # 获取平均速度和车辆数量
                avg_speed = 0
                vehicle_count = 0
                if 'avg_speed' in last_episode:
                    avg_speed = last_episode['avg_speed']
                if 'cav_count' in last_episode:
                    vehicle_count = last_episode['cav_count']
                
                # 计算衰减率
                decay_rate = 0
                if len(self.reward_decay_analysis['reward']) > 0:
                    prev_reward = self.reward_decay_analysis['reward'][-1]
                    if prev_reward > 0:
                        decay_rate = (prev_reward - reward) / prev_reward
                    
                # 保存数据
                self.reward_decay_analysis['episode'].append(episode)
                self.reward_decay_analysis['reward'].append(reward)
                self.reward_decay_analysis['episode_length'].append(episode_length)
                self.reward_decay_analysis['avg_speed'].append(avg_speed)
                self.reward_decay_analysis['vehicle_count'].append(vehicle_count)
                self.reward_decay_analysis['decay_rate'].append(decay_rate)
                
                # 如果衰减率超过阈值，记录警告
                if decay_rate > 0.2:  # 衰减率超过20%
                    logging.error(f"奖励明显下降! 回合{episode}: 前值={prev_reward:.2f}, 当前值={reward:.2f}, 衰减率={decay_rate:.2f}")
                
                # 将数据写入CSV
                with open(self.reward_analysis_file, 'a', encoding='utf-8') as f:
                    f.write(f"{episode},{reward},{episode_length},{avg_speed},{vehicle_count},{decay_rate}\n")
        
        # 将当前指标写入CSV
        metrics_values = []
        for key in self.metrics_history.keys():
            value = self.metrics_history[key][-1] if self.metrics_history[key] else 0
            metrics_values.append(str(value))
            
        with open(self.csv_file, 'a', encoding='utf-8') as f:
            f.write(','.join(metrics_values) + '\n')
        
        # 打印格式化的训练指标表格
        self._print_formatted_metrics()

    def _print_formatted_metrics(self):
        """按照指定格式打印训练指标"""
        # 确保有指标数据
        if not any(self.metrics_history.values()):
            return
            
        # 获取最新指标
        latest_metrics = {}
        for key, values in self.metrics_history.items():
            if values:
                latest_metrics[key] = values[-1]
        
        # 获取回合信息
        ep_len_mean = 0
        ep_rew_mean = 0
        success_rate = 0
        
        if hasattr(self.model, 'ep_info_buffer') and self.model.ep_info_buffer:
            # 计算平均回合长度和奖励
            ep_lengths = [ep_info['l'] for ep_info in self.model.ep_info_buffer if 'l' in ep_info]
            ep_rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer if 'r' in ep_info]
            successes = [1 if ep_info.get('is_success', False) else 0 for ep_info in self.model.ep_info_buffer]
            
            if ep_lengths:
                ep_len_mean = sum(ep_lengths) / len(ep_lengths)
            if ep_rewards:
                ep_rew_mean = sum(ep_rewards) / len(ep_rewards)
            if successes:
                success_rate = sum(successes) / len(successes)
        
        # 打印格式化的训练统计表格
        print("\n" + "-" * 40)
        print("| rollout/                |            |")
        print(f"|    ep_len_mean          | {ep_len_mean:.2e}   |")
        print(f"|    ep_rew_mean          | {ep_rew_mean:.0f}        |")
        print(f"|    success_rate         | {success_rate:.1f}          |")
        print("| time/                   |            |")
        print(f"|    fps                  | {latest_metrics.get('fps', 0):.0f}         |")
        print(f"|    iterations           | {latest_metrics.get('iterations', 0):.0f}         |")
        print(f"|    time_elapsed         | {latest_metrics.get('time_elapsed', 0):.0f}       |")
        print(f"|    total_timesteps      | {latest_metrics.get('total_timesteps', 0):.0f}      |")
        print("| train/                  |            |")
        print(f"|    approx_kl            | {latest_metrics.get('approx_kl', 0):.8f} |")
        print(f"|    clip_fraction        | {latest_metrics.get('clip_fraction', 0):.3f}      |")
        print(f"|    clip_range           | {latest_metrics.get('clip_range', 0.2):.1f}        |")
        print(f"|    entropy_loss         | {latest_metrics.get('entropy_loss', 0):.0f}        |")
        print(f"|    explained_variance   | {latest_metrics.get('explained_variance', 0):.3f}      |")
        print(f"|    learning_rate        | {latest_metrics.get('learning_rate', 0):.6f}   |")
        print(f"|    loss                 | {latest_metrics.get('loss', 0):.3f}     |")
        print(f"|    n_updates            | {latest_metrics.get('n_updates', 0):.0f}        |")
        print(f"|    policy_gradient_loss | {latest_metrics.get('policy_gradient_loss', 0):.4f}    |")
        print(f"|    std                  | {latest_metrics.get('std', 0):.2f}       |")
        print(f"|    value_loss           | {latest_metrics.get('value_loss', 0):.2f}       |")
        print("-" * 40)

    def _on_training_end(self) -> None:
        # 训练结束时，生成图表
        self.plot_metrics()
        self.analyze_reward_decay()
        
    def plot_metrics(self):
        """绘制PPO指标图表"""
        # 绘制所有指标
        for key, values in self.metrics_history.items():
            if not values:
                continue
                
            plt.figure(figsize=(10, 6))
            plt.plot(values)
            plt.title(f'PPO {key} over Training')
            plt.xlabel('Iteration')
            plt.ylabel(key)
            plt.grid(True)
            plt.savefig(os.path.join(self.save_dir, f"ppo_{key}.png"))
            plt.close()
            
    def analyze_reward_decay(self):
        """分析奖励衰减并生成图表"""
        if not self.reward_decay_analysis['episode']:
            return
            
        # 绘制奖励随回合变化图
        plt.figure(figsize=(10, 6))
        plt.plot(self.reward_decay_analysis['episode'], self.reward_decay_analysis['reward'])
        plt.title('Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, "reward_analysis.png"))
        plt.close()
        
        # 绘制奖励衰减率图
        plt.figure(figsize=(10, 6))
        plt.plot(self.reward_decay_analysis['episode'][1:], self.reward_decay_analysis['decay_rate'][1:])  # 跳过第一个元素，因为初始衰减率无意义
        plt.title('Reward Decay Rate')
        plt.xlabel('Episode')
        plt.ylabel('Decay Rate')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, "reward_decay_rate.png"))
        plt.close()
        
        # 绘制奖励与车辆数量的关系
        plt.figure(figsize=(10, 6))
        plt.scatter(self.reward_decay_analysis['vehicle_count'], self.reward_decay_analysis['reward'])
        plt.title('Reward vs Vehicle Count')
        plt.xlabel('Vehicle Count')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, "reward_vs_vehicles.png"))
        plt.close()
        
        # 绘制奖励与平均速度的关系
        plt.figure(figsize=(10, 6))
        plt.scatter(self.reward_decay_analysis['avg_speed'], self.reward_decay_analysis['reward'])
        plt.title('Reward vs Average Speed')
        plt.xlabel('Average Speed')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, "reward_vs_speed.png"))
        plt.close()


def setup_logging(log_file=None):
    """设置日志配置
    
    Args:
        log_file: 可选的日志文件路径，如果提供则同时记录到文件
    """
    # 创建根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.ERROR)  # 修改：只记录ERROR级别的日志
    
    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)  # 修改：控制台只显示ERROR级别
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # 添加处理器到根日志器
    root_logger.addHandler(console_handler)
    
    # 如果提供了日志文件路径，添加文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.ERROR)  # 修改：文件也只记录ERROR级别
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logging.error(f"日志文件路径: {log_file}")  # 使用error级别记录这条信息
        
    return root_logger


def train(
    save_dir="./results",
    gui=False,
    total_timesteps=300000,
    learning_rate=1e-3,
    min_learning_rate=1e-5,
    lr_schedule="linear",
    episode_length=15000,
    action_scale=10.0,
    max_speed=30.0,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    max_grad_norm=0.5,
    policy="MlpPolicy"
):
    """配置和开始训练过程
    
    Args:
        save_dir: 结果保存目录
        gui: 是否启用GUI模式
        total_timesteps: 训练总步数
        learning_rate: 初始学习率
        min_learning_rate: 最低学习率（用于学习率衰减）
        lr_schedule: 学习率衰减策略 ("linear", "exp"或"constant")
        episode_length: 单个回合最大步数
        action_scale: 动作缩放因子
        max_speed: 最大车速
        n_steps: 更新前收集的步数
        batch_size: batch大小
        n_epochs: 每次更新的epoch数
        gamma: 折扣因子
        gae_lambda: GAE参数
        clip_range: PPO裁剪范围
        ent_coef: 熵系数
        max_grad_norm: 梯度裁剪阈值
        policy: 策略类型
    """
    # 创建基本的结果目录
    base_dir = save_dir
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # 创建当前训练会话专用目录（使用时间戳）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(base_dir, f"training_session_{timestamp}")
    
    # 创建所需子目录
    models_dir = os.path.join(session_dir, "models")
    plots_dir = os.path.join(session_dir, "plots")
    logs_dir = os.path.join(session_dir, "logs")
    data_dir = os.path.join(session_dir, "data")
    
    # 确保所有目录都存在
    for dir_path in [session_dir, models_dir, plots_dir, logs_dir, data_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(logs_dir, f"training_{timestamp}.log")
    setup_logging(log_file)
    
    # 记录训练参数
    logging.error(f"训练会话目录: {session_dir}")
    
    # 将训练参数存储到元数据文件
    metadata = {
        "timestamp": timestamp,
        "total_timesteps": total_timesteps,
        "learning_rate": learning_rate,
        "min_learning_rate": min_learning_rate,
        "lr_schedule": lr_schedule,
        "episode_length": episode_length,
        "action_scale": action_scale,
        "max_speed": max_speed,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "max_grad_norm": max_grad_norm,
        "policy": policy,
        "gui": gui
    }
    
    # 保存元数据
    with open(os.path.join(session_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    
    # 配置学习率调度函数
    def get_lr_schedule(curr_progress_remaining):
        """学习率衰减调度函数"""
        if lr_schedule == "constant":
            return learning_rate
        elif lr_schedule == "linear":
            return min_learning_rate + (learning_rate - min_learning_rate) * curr_progress_remaining
        elif lr_schedule == "exp":
            return min_learning_rate + (learning_rate - min_learning_rate) * math.exp(5 * (curr_progress_remaining - 1))
        else:
            return learning_rate  # 默认使用常数学习率
    
    try:
        # 创建环境
        env = SumoMergeEnv(
            gui=gui,
            max_episode_length=episode_length,
            action_scale=action_scale,
            max_speed=max_speed
        )
        
        # 检查环境是否符合gym标准
        check_env(env)
        logging.error("环境检查通过")
        
        # 创建回调
        # 删除中间检查点，只保存最终模型
        final_model_path = os.path.join(models_dir, "final_model")
        
        # 修改: 使用TrainingCallback记录训练进度
        training_callback = TrainingCallback(
            save_dir=plots_dir,
            save_freq=total_timesteps  # 设置为总步数，确保只在最后保存
        )
        
        # 创建PPO指标回调，保存到data目录
        metrics_callback = PPOMetricsCallback(
            save_dir=data_dir
        )
        
        # 组合所有回调
        all_callbacks = [training_callback, metrics_callback]
        
        # 创建模型
        model = PPO(
            policy=policy,
            env=env,
            learning_rate=get_lr_schedule,  # 使用学习率衰减函数
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            verbose=1,
            tensorboard_log=None  # 移除tensorboard日志
        )
        
        # 开始训练
        print(f"开始训练 PPO 模型，总步数: {total_timesteps}")
        model.learn(
            total_timesteps=total_timesteps,
            callback=all_callbacks
        )
        
        # 保存最终模型
        model.save(final_model_path)
        print(f"最终模型已保存到 {final_model_path}")
        
        # 创建训练总结报告
        with open(os.path.join(session_dir, "training_summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"训练会话: {timestamp}\n")
            f.write("="*50 + "\n")
            f.write("参数:\n")
            for key, value in metadata.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            f.write("训练结果:\n")
            f.write(f"  总回合数: {model.num_timesteps // episode_length}\n")
            f.write(f"  总步数: {model.num_timesteps}\n")
            if hasattr(model, "ep_info_buffer") and len(model.ep_info_buffer) > 0:
                rewards = [ep["r"] for ep in model.ep_info_buffer]
                f.write(f"  平均回合奖励: {sum(rewards)/len(rewards):.2f}\n")
                f.write(f"  最高回合奖励: {max(rewards):.2f}\n")
                f.write(f"  最低回合奖励: {min(rewards):.2f}\n")
            f.write("\n")
            f.write("文件位置:\n")
            f.write(f"  日志: {log_file}\n")
            f.write(f"  模型: {final_model_path}\n")
            f.write(f"  图表: {plots_dir}\n")
            f.write(f"  数据: {data_dir}\n")
        
        print(f"训练完成，结果保存在: {session_dir}")
        
        # 在会话结束前关闭环境
        env.close()
        return model, session_dir
        
    except Exception as e:
        logging.error(f"训练过程中出错: {e}")
        traceback.print_exc()
        # 确保环境关闭
        try:
            env.close()
        except:
            pass
        raise


def test(
    model_path, 
    gui=True, 
    episode_length=10000, 
    action_scale=10.0, 
    max_speed=30.0, 
    test_episodes=5
):
    """测试保存的模型
    
    Args:
        model_path: 模型文件路径
        gui: 是否启用GUI模式
        episode_length: 每个回合的最大步数
        action_scale: 动作缩放因子
        max_speed: 最大速度限制
        test_episodes: 测试回合数
    """
    logging.info(f"开始测试模型: {model_path}")
    logging.info(f"测试参数: GUI={gui}, 回合={test_episodes}, 回合长度={episode_length}")
    
    # 创建环境
    env = SumoMergeEnv(
        gui=gui,
        max_episode_length=episode_length,
        action_scale=action_scale,
        max_speed=max_speed
    )
    
    try:
        # 加载模型
        model = PPO.load(model_path)
        logging.info(f"模型已加载: {model_path}")
        
        # 运行测试回合
        total_reward = 0
        rewards = []
        avg_speeds = []
        
        for ep in range(test_episodes):
            logging.info(f"开始测试回合 {ep+1}/{test_episodes}")
            obs, _ = env.reset()
            ep_reward = 0
            ep_steps = 0
            terminated = False
            truncated = False
            
            # 收集回合统计数据
            episode_speeds = []
            
            while not (terminated or truncated):
                # 获取模型预测的动作
                action, _ = model.predict(obs, deterministic=True)
                
                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                
                # 记录统计信息
                ep_reward += reward
                ep_steps += 1
                
                # 记录当前步的速度
                if "avg_speed" in info:
                    episode_speeds.append(info["avg_speed"])
                
                # 记录详细的状态信息（每100步一次）
                if ep_steps % 100 == 0 or terminated or truncated:
                    cav_count = info.get("cav_count", 0)
                    avg_speed = info.get("avg_speed", 0)
                    logging.info(f"  步骤: {ep_steps}, CAVs: {cav_count}, 平均速度: {avg_speed:.2f}, 奖励: {reward:.2f}")
            
            # 计算回合平均速度
            avg_speed = sum(episode_speeds) / len(episode_speeds) if episode_speeds else 0
            
            # 记录回合结果
            logging.info(f"回合 {ep+1} 完成: 步数={ep_steps}, 奖励={ep_reward:.2f}, 平均速度={avg_speed:.2f}")
            rewards.append(ep_reward)
            avg_speeds.append(avg_speed)
            total_reward += ep_reward
        
        # 计算并显示测试结果
        avg_reward = total_reward / test_episodes
        avg_speed_overall = sum(avg_speeds) / len(avg_speeds) if avg_speeds else 0
        
        logging.info(f"测试完成: {test_episodes}个回合")
        logging.info(f"平均奖励: {avg_reward:.2f}")
        logging.info(f"平均速度: {avg_speed_overall:.2f}")
        
        # 绘制测试结果图表
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.bar(range(1, test_episodes+1), rewards)
        plt.title('Test Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.bar(range(1, test_episodes+1), avg_speeds)
        plt.title('Test Episode Average Speeds')
        plt.xlabel('Episode')
        plt.ylabel('Speed (m/s)')
        plt.grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        test_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"test_results_{test_time}.png")
        plt.close()
        
        return avg_reward, avg_speed_overall
        
    except Exception as e:
        logging.error(f"测试过程中出错: {e}")
        traceback.print_exc()
    finally:
        # 确保环境关闭
        env.close()


def main():
    """主函数，处理命令行参数并启动训练或测试"""
    parser = argparse.ArgumentParser(description='训练或测试SUMO-RL连接模型')
    
    # 模式选择
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                      help='运行模式: 训练(train)或测试(test)')
    
    # 通用参数
    parser.add_argument('--gui', action='store_true', help='使用SUMO GUI模式')
    parser.add_argument('--save_dir', type=str, default='./results', help='结果保存目录')
    parser.add_argument('--episode_length', type=int, default=10000, help='每回合最大步数')
    parser.add_argument('--action_scale', type=float, default=10.0, help='动作缩放因子')
    parser.add_argument('--max_speed', type=float, default=30.0, help='最大车速')
    
    # 训练特定参数
    parser.add_argument('--total_timesteps', type=int, default=300000, help='训练总步数')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='初始学习率')
    parser.add_argument('--min_learning_rate', type=float, default=1e-5, help='最低学习率')
    parser.add_argument('--lr_schedule', type=str, default='linear', 
                      choices=['linear', 'exp', 'constant'], help='学习率调度策略')
    parser.add_argument('--n_steps', type=int, default=2048, help='每次更新前收集的步数')
    parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
    parser.add_argument('--n_epochs', type=int, default=10, help='每次更新的epoch数')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE参数')
    parser.add_argument('--clip_range', type=float, default=0.2, help='PPO裁剪范围')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='熵系数')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='梯度裁剪阈值')
    
    # 测试特定参数
    parser.add_argument('--model_path', type=str, help='用于测试的模型路径')
    parser.add_argument('--test_episodes', type=int, default=5, help='测试回合数')
    
    args = parser.parse_args()
    
    # 记录运行模式
    mode = args.mode.lower()
    logging.info(f"运行模式: {mode}")
    
    if mode == 'train':
        # 训练模式
        train(
            save_dir=args.save_dir,
            gui=args.gui,
            total_timesteps=args.total_timesteps,
            learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            lr_schedule=args.lr_schedule,
            episode_length=args.episode_length,
            action_scale=args.action_scale,
            max_speed=args.max_speed,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            max_grad_norm=args.max_grad_norm
        )
    elif mode == 'test':
        if not args.model_path:
            logging.error("测试模式需要指定--model_path")
            return
        test(
            model_path=args.model_path,
            gui=args.gui,
            episode_length=args.episode_length,
            action_scale=args.action_scale,
            max_speed=args.max_speed,
            test_episodes=args.test_episodes
        )
    else:
        logging.error(f"未知模式: {mode}")
        
if __name__ == "__main__":
    # 设置基本日志配置
    setup_logging()
    
    # 输出基本信息
    logging.info(f"SUMO-RL训练脚本已启动")
    logging.info(f"Python版本: {sys.version}")
    logging.info(f"当前工作目录: {os.getcwd()}")
    
    # 检查SUMO是否可用
    try:
        import traci
        logging.info("TRACI 可用")
    except ImportError:
        logging.error("TRACI模块不可用。请确保SUMO已正确安装并且sumo/tools在您的PYTHONPATH中。")
        sys.exit(1)
    
    try:
        main()
    except Exception as e:
        logging.error(f"程序执行出错: {e}")
        traceback.print_exc()
        # 确保在发生错误时清理资源
        try:
            if traci.isLoaded():
                traci.close(False)
        except:
            pass 