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
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure

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
root_logger.setLevel(logging.INFO)

# 文件处理程序
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)

# 控制台处理程序
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)

class SumoMergeEnv(gym.Env):
    def __init__(self, cfg_path=None, gui=False, need_reset=True, max_episode_length=10000, action_scale=3.0, max_speed=100.0):
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
        self.action_scale = action_scale
        self.max_speed = max_speed
        self.stats = {
            'rewards': [],
            'avg_speeds': [],
            'vehicle_counts': []
        }

        # 固定观测空间和动作空间
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.max_cavs * 6,), dtype=np.float32
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
            return np.zeros(self.max_cavs * 6, dtype=np.float32)

        obs = []
        for cav_id in self.cav_ids:
            try:
                speed = traci.vehicle.getSpeed(cav_id)
                leader = traci.vehicle.getLeader(cav_id, 100)
                leader_dist = leader[1] if leader and leader[0] else 100
                leader_speed = (
                    traci.vehicle.getSpeed(leader[0]) if leader and leader[0] else 0
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
                obs.extend(
                    [
                        speed,
                        leader_dist,
                        leader_speed,
                        follower_dist,
                        follower_speed,
                        lane_flag,
                    ]
                )
            except Exception as e:
                logging.warning(f"获取 {cav_id} 观测失败: {e}")
                obs.extend([0, 100, 0, 100, 0, 0])

        # 填充到固定长度
        flat_obs = np.zeros(self.max_cavs * 6, dtype=np.float32)
        flat_obs[: len(obs)] = obs
        return flat_obs

    def _apply_action(self, actions):
        """应用动作到所有 CAV"""
        actions = actions * self.action_scale  # 缩放到 [-action_scale, action_scale]
        
        # 确保actions的长度与cav_ids匹配
        if len(self.cav_ids) > 0:
            # 截断actions到当前CAV数量
            actions_to_apply = actions[:len(self.cav_ids)]
            
            for i, cav_id in enumerate(self.cav_ids):
                try:
                    current_speed = traci.vehicle.getSpeed(cav_id)
                    new_speed = np.clip(current_speed + actions_to_apply[i], 0, self.max_speed)
                    traci.vehicle.setSpeed(cav_id, new_speed)
                except Exception as e:
                    logging.warning(f"应用动作到 {cav_id} 失败: {e}")

    def _calculate_reward(self):
        """计算全局奖励"""
        if not self.cav_ids:
            return 0
        
        total_speed = 0
        speeds = []
        
        for cav_id in self.cav_ids:
            try:
                speed = traci.vehicle.getSpeed(cav_id)
                speeds.append(speed)
                total_speed += speed
                
                
            except Exception as e:
                logging.warning(f"计算 {cav_id} 奖励失败: {e}")
        
        # 平均速度奖励
        avg_speed = total_speed / len(self.cav_ids) if self.cav_ids else 0
        
        
        
        # 总奖励
        speed_reward = avg_speed  # 基本奖励为平均速度
        total_reward = speed_reward
        
        # 记录统计信息
        self.stats['rewards'].append(total_reward)
        self.stats['avg_speeds'].append(avg_speed)
        self.stats['vehicle_counts'].append(len(self.cav_ids))
        
        # 调试信息
        if self.current_step % 100 == 0:
            logging.info(f"Step {self.current_step}, CAVs: {len(self.cav_ids)}, Reward: {total_reward:.2f} = "
                       f"Speed: {speed_reward:.2f}")
        
        return total_reward

    def step(self, action):
        if not self._is_initialized:
            obs, _ = self.reset()
            return obs, 0, False, False, {}
        
        self._apply_action(action)
        
        try:
            traci.simulationStep()
            self.current_step += 1
            
            # 更新车辆列表
            all_vehicles = traci.vehicle.getIDList()
            self.cav_ids = [v for v in all_vehicles if "CAV" in traci.vehicle.getTypeID(v)]
            self.hdv_ids = [v for v in all_vehicles if "HDV" in traci.vehicle.getTypeID(v)]
            
            # 获取观测和奖励
            obs = self._get_observations()
            reward = self._calculate_reward()
            
            # 判断是否结束
            terminated = len(all_vehicles) == 0
            truncated = self.current_step >= self.episode_length
            
            info = {"cav_count": len(self.cav_ids), "step": self.current_step, "vehicle_count": len(all_vehicles)}
            
            return obs, reward, terminated, truncated, info
            
        except traci.exceptions.FatalTraCIError as e:
            logging.error(f"TRACI 错误: {e}")
            # 尝试重新连接
            self.reset()
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0, True, False, {"error": str(e)}

    def reset(self, seed=None, options=None):
        """重置环境"""
        # 关闭现有的SUMO连接
        self.close()
        
        # 等待一小段时间确保资源释放
        time.sleep(0.5)
        
        # 清空统计信息
        self.stats = {
            'rewards': [],
            'avg_speeds': [],
            'vehicle_counts': []
        }
        
        # 尝试启动SUMO，有重试机制
        max_tries = 3
        for attempt in range(max_tries):
            try:
                traci.start(self.sumo_cmd)
                self._is_initialized = True
                break
            except Exception as e:
                logging.error(f"SUMO 启动失败 (尝试 {attempt+1}/{max_tries}): {e}")
                time.sleep(2)  # 等待一段时间再重试
                if attempt == max_tries - 1:  # 最后一次尝试失败
                    raise RuntimeError(f"无法启动SUMO: {e}")

        # 运行一段时间，等待车辆生成
        logging.info("等待车辆生成...")
        for step in range(100):
            try:
                traci.simulationStep()
                all_vehicles = traci.vehicle.getIDList()
                self.cav_ids = [v for v in all_vehicles if "CAV" in traci.vehicle.getTypeID(v)]
                self.hdv_ids = [v for v in all_vehicles if "HDV" in traci.vehicle.getTypeID(v)]
                
                # 如果已有CAV车辆，就可以开始了
                if self.cav_ids:
                    logging.info(f"在第 {step} 步生成了 {len(self.cav_ids)} 辆CAV")
                    break
                    
                # 每20步输出一次状态
                if step % 20 == 0:
                    logging.debug(f"步骤 {step}: {len(all_vehicles)} 辆车, {len(self.cav_ids)} 辆CAV")
            except Exception as e:
                logging.error(f"初始化时出错: {e}")
                self.close()
                time.sleep(1)
                traci.start(self.sumo_cmd)
        
        self.current_step = 0
        obs = self._get_observations()
        return obs, {}

    def close(self):
        """关闭环境"""
        try:
            if traci.isLoaded():
                traci.close()
                self._is_initialized = False
                time.sleep(0.5)  # 等待一下确保资源释放
        except Exception as e:
            logging.error(f"关闭TRACI时出错: {e}")
    
    def plot_stats(self, save_dir="./results"):
        """绘制并保存统计信息的图表"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 绘制奖励曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.stats['rewards'])
        plt.title('奖励随时间变化')
        plt.xlabel('时间步')
        plt.ylabel('奖励')
        plt.grid(True)
        plt.savefig(f"{save_dir}/rewards_{current_time}.png")
        
        # 绘制平均速度曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.stats['avg_speeds'])
        plt.title('平均速度随时间变化')
        plt.xlabel('时间步')
        plt.ylabel('平均速度 (m/s)')
        plt.grid(True)
        plt.savefig(f"{save_dir}/avg_speeds_{current_time}.png")
        
        # 绘制车辆数量曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.stats['vehicle_counts'])
        plt.title('CAV车辆数量随时间变化')
        plt.xlabel('时间步')
        plt.ylabel('车辆数量')
        plt.grid(True)
        plt.savefig(f"{save_dir}/vehicle_counts_{current_time}.png")
        
        logging.info(f"统计图表已保存到 {save_dir} 目录")


# 自定义回调，用于记录训练信息
class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0, save_dir="./results", save_freq=1000):
        super(TrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.save_dir = save_dir
        self.save_freq = save_freq
        
        # 确保结果目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 创建CSV文件记录每回合的结果
        self.csv_file = os.path.join(save_dir, f"training_episodes_{current_time}.csv")
        with open(self.csv_file, 'w', encoding='utf-8') as f:
            f.write("episode,reward,length\n")

    def _on_training_start(self) -> None:
        logging.info("开始训练...")

    def _on_step(self) -> bool:
        # 累积奖励和步数
        self.current_episode_reward += self.locals["rewards"][0]
        self.current_episode_length += 1
        
        # 检查是否完成一个回合
        if self.locals["dones"][0]:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            logging.info(f"回合 {self.episode_count} 完成: 奖励={self.current_episode_reward:.2f}, 步数={self.current_episode_length}")
            
            # 将结果写入CSV文件
            with open(self.csv_file, 'a', encoding='utf-8') as f:
                f.write(f"{self.episode_count},{self.current_episode_reward:.6f},{self.current_episode_length}\n")
            
            # 重置当前回合的统计信息
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
            # 每几个回合绘制并保存图表
            if self.episode_count % 5 == 0:
                self.plot_training_progress()
        
        # 如果达到保存频率，保存当前模型
        if self.num_timesteps % self.save_freq == 0:
            model_path = os.path.join(self.save_dir, f"model_{self.num_timesteps}_steps")
            self.model.save(model_path)
            logging.info(f"已保存模型到 {model_path}")
            
        return True

    def _on_rollout_end(self) -> None:
        # 在完成一个rollout后输出统计信息
        avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
        logging.info(f"完成一个采样周期，当前总步数: {self.num_timesteps}, 最近10回合平均奖励: {avg_reward:.2f}")

    def _on_training_end(self) -> None:
        logging.info(f"训练结束，总步数: {self.num_timesteps}, 总回合数: {self.episode_count}")
        self.plot_training_progress()
    
    def plot_training_progress(self):
        """绘制训练进度的图表"""
        if not self.episode_rewards:
            return
            
        # 绘制回合奖励曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.episode_rewards)
        plt.title('回合奖励随训练进度变化')
        plt.xlabel('回合')
        plt.ylabel('奖励')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, f"episode_rewards_{current_time}.png"))
        
        # 绘制回合长度曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.episode_lengths)
        plt.title('回合长度随训练进度变化')
        plt.xlabel('回合')
        plt.ylabel('步数')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, f"episode_lengths_{current_time}.png"))
        
        # 计算并绘制移动平均线
        window_size = min(10, len(self.episode_rewards))
        if window_size > 0:
            moving_avg = np.convolve(self.episode_rewards, np.ones(window_size)/window_size, mode='valid')
            plt.figure(figsize=(10, 6))
            plt.plot(moving_avg)
            plt.title(f'回合奖励的{window_size}回合移动平均')
            plt.xlabel('回合')
            plt.ylabel('平均奖励')
            plt.grid(True)
            plt.savefig(os.path.join(self.save_dir, f"moving_avg_rewards_{current_time}.png"))
        
        logging.info(f"训练进度图表已更新")


def train(args):
    """训练模型"""
    # 创建结果目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # 配置TensorBoard日志
    tensorboard_log = os.path.join(args.save_dir, "tensorboard")
    if not os.path.exists(tensorboard_log):
        os.makedirs(tensorboard_log)
    
    logging.info(f"训练参数: GUI={args.gui}, 学习率={args.learning_rate}, 总步数={args.total_timesteps}, "
               f"批量大小={args.batch_size}, n_steps={args.n_steps}, epochs={args.n_epochs}")
    
    # 创建环境，禁用自动重置以便check_env能正常工作
    env = SumoMergeEnv(
        gui=args.gui, 
        need_reset=False,
        max_episode_length=args.episode_length,
        action_scale=args.action_scale,
        max_speed=args.max_speed
    )
    
    # 检查环境是否符合gym规范
    try:
        logging.info("检查环境...")
        check_env(env)
        logging.info("环境检查通过!")
    except Exception as e:
        logging.warning(f"环境检查失败: {e}")
        # 即使检查失败，我们也继续训练
    
    # 现在手动重置环境，为训练做准备
    env.reset()
    
    # 使用向量化环境包装器
    vec_env = DummyVecEnv([lambda: SumoMergeEnv(
        gui=args.gui,
        max_episode_length=args.episode_length,
        action_scale=args.action_scale,
        max_speed=args.max_speed
    )])

    # 初始化 PPO 模型
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )

    # 配置回调
    training_callback = TrainingCallback(save_dir=args.save_dir, save_freq=args.save_freq)
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=os.path.join(args.save_dir, "checkpoints"),
        name_prefix="ppo_sumo",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    callbacks = CallbackList([training_callback, checkpoint_callback])
    
    try:
        logging.info("开始训练...")
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
        logging.info("训练完成!")
        
        # 保存最终模型
        final_model_path = os.path.join(args.save_dir, f"ppo_sumo_model_final_{current_time}")
        model.save(final_model_path)
        logging.info(f"最终模型已保存为 {final_model_path}")
        
        # 绘制环境统计图表
        if isinstance(vec_env.envs[0], SumoMergeEnv):
            vec_env.envs[0].plot_stats(save_dir=args.save_dir)
    except Exception as e:
        logging.error(f"训练过程中出错: {e}")
    finally:
        # 确保环境关闭
        vec_env.close()


def test_saved_model(args):
    """测试已保存的模型"""
    model_path = args.load_model
    if not os.path.exists(model_path):
        logging.error(f"模型文件不存在: {model_path}")
        return
        
    logging.info(f"加载模型: {model_path}")
    
    # 创建结果目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # 创建环境
    env = SumoMergeEnv(
        gui=args.gui, 
        max_episode_length=args.episode_length,
        action_scale=args.action_scale,
        max_speed=args.max_speed
    )
    
    # 加载模型
    model = PPO.load(model_path)
    
    try:
        # 重置环境
        obs, _ = env.reset()
        
        # 运行模型
        done = False
        total_reward = 0
        step = 0
        
        while not done and step < args.test_steps:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated
            
            if step % 100 == 0:
                logging.info(f"测试步骤 {step}, 累积奖励: {total_reward:.2f}")
                
        logging.info(f"测试完成: 总步数={step}, 总奖励={total_reward:.2f}")
        
        # 绘制并保存统计信息
        env.plot_stats(save_dir=args.save_dir)
    finally:
        # 关闭环境
        env.close()


def main():
    parser = argparse.ArgumentParser(description="SUMO强化学习训练与测试")
    
    # 训练或测试模式选择
    subparsers = parser.add_subparsers(dest="mode", help="运行模式: train或test")
    
    # 训练模式参数
    train_parser = subparsers.add_parser("train", help="训练模式")
    train_parser.add_argument("--gui", action="store_true", help="使用SUMO的GUI模式")
    train_parser.add_argument("--save_dir", type=str, default="./results", help="结果保存目录")
    train_parser.add_argument("--episode_length", type=int, default=10000, help="单回合最大步数")
    train_parser.add_argument("--action_scale", type=float, default=3.0, help="动作缩放因子")
    train_parser.add_argument("--max_speed", type=float, default=100.0, help="最大速度限制")
    train_parser.add_argument("--learning_rate", type=float, default=3e-4, help="学习率")
    train_parser.add_argument("--total_timesteps", type=int, default=200000, help="总训练步数")
    train_parser.add_argument("--n_steps", type=int, default=8192, help="每次更新收集的步数")
    train_parser.add_argument("--batch_size", type=int, default=64, help="批量大小")
    train_parser.add_argument("--n_epochs", type=int, default=5, help="每批数据训练轮数")
    train_parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    train_parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda参数")
    train_parser.add_argument("--clip_range", type=float, default=0.2, help="PPO裁剪范围")
    train_parser.add_argument("--save_freq", type=int, default=10000, help="模型保存频率(步数)")
    
    # 测试模式参数
    test_parser = subparsers.add_parser("test", help="测试模式")
    test_parser.add_argument("--gui", action="store_true", help="使用SUMO的GUI模式")
    test_parser.add_argument("--save_dir", type=str, default="./results", help="结果保存目录")
    test_parser.add_argument("--episode_length", type=int, default=10000, help="单回合最大步数")
    test_parser.add_argument("--action_scale", type=float, default=3.0, help="动作缩放因子")
    test_parser.add_argument("--max_speed", type=float, default=100.0, help="最大速度限制")
    test_parser.add_argument("--load_model", type=str, required=True, help="要加载的模型路径")
    test_parser.add_argument("--test_steps", type=int, default=1000, help="测试步数")
    
    args = parser.parse_args()
    
    # 默认为训练模式
    if args.mode is None:
        args.mode = "train"
        
    logging.info(f"运行模式: {args.mode}")
    
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test_saved_model(args)
    else:
        logging.error(f"未知的运行模式: {args.mode}")


if __name__ == "__main__":
    main() 