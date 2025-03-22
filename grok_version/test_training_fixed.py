import os
import sys
import time
import numpy as np
import traci
import gymnasium as gym
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# 确保SUMO环境变量设置正确
if 'SUMO_HOME' not in os.environ:
    sumo_home = 'D:\\sumo'  # 修改为您的SUMO安装路径
    os.environ['SUMO_HOME'] = sumo_home
    sys.path.append(os.path.join(sumo_home, 'tools'))

class SumoMergeEnv(gym.Env):
    def __init__(self, cfg_path=None, gui=False, need_reset=True):
        super(SumoMergeEnv, self).__init__()
        # 获取配置文件路径
        if cfg_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            cfg_path = os.path.join(base_dir, "input_sources", "config.sumocfg")

        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"SUMO 配置文件未找到: {cfg_path}")

        self.sumo_cmd = ["sumo-gui" if gui else "sumo", "-c", cfg_path]
        self.cfg_path = cfg_path
        self.gui = gui
        self.episode_length = 1000
        self.current_step = 0
        self.cav_ids = []
        self.hdv_ids = []
        self._is_initialized = False
        self.max_cavs = 50  # 最大支持的 CAV 数量
        self.need_reset = need_reset

        # 固定观测空间和动作空间
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.max_cavs * 6,), dtype=np.float32
        )
        self.action_space = Box(
            low=-1, high=1, shape=(self.max_cavs,), dtype=np.float32
        )
        self.action_scale = 3  # 动作缩放因子，从 [-1, 1] 到 [-3, 3]
        
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
                print(f"获取 {cav_id} 观测失败: {e}")
                obs.extend([0, 100, 0, 100, 0, 0])

        # 填充到固定长度
        flat_obs = np.zeros(self.max_cavs * 6, dtype=np.float32)
        flat_obs[: len(obs)] = obs
        return flat_obs

    def _apply_action(self, actions):
        """应用动作到所有 CAV"""
        actions = actions * self.action_scale  # 缩放到 [-3, 3]
        
        # 确保actions的长度与cav_ids匹配
        if len(self.cav_ids) > 0:
            # 截断actions到当前CAV数量
            actions_to_apply = actions[:len(self.cav_ids)]
            
            for i, cav_id in enumerate(self.cav_ids):
                try:
                    current_speed = traci.vehicle.getSpeed(cav_id)
                    new_speed = np.clip(current_speed + actions_to_apply[i], 0, 15)
                    traci.vehicle.setSpeed(cav_id, new_speed)
                except Exception as e:
                    print(f"应用动作到 {cav_id} 失败: {e}")

    def _calculate_reward(self):
        """计算全局奖励"""
        if not self.cav_ids:
            return 0
        
        total_speed = 0
        speeds = []
        safety_reward = 0
        target_speed = 8  # 目标速度
        
        for cav_id in self.cav_ids:
            try:
                speed = traci.vehicle.getSpeed(cav_id)
                speeds.append(speed)
                total_speed += speed
                
                '''# 安全奖励计算
                leader = traci.vehicle.getLeader(cav_id, 100)
                if leader and leader[1] < 5:
                    safety_reward -= 10
                elif leader and leader[1] < 10:
                    safety_reward -= 5'''
            except Exception as e:
                print(f"计算 {cav_id} 奖励失败: {e}")
        
        # 平均速度奖励
        avg_speed = total_speed / len(self.cav_ids) if self.cav_ids else 0
        speed_reward = -abs(avg_speed - target_speed)
        
        '''# 协作奖励（速度方差）
        coop_reward = -np.var(speeds) if len(speeds) > 1 else 0'''
        
        total_reward = speed_reward
        
        # 调试信息
        if self.current_step % 100 == 0:
            print(f"Step {self.current_step}, CAVs: {len(self.cav_ids)}, Reward: {total_reward:.2f} = Speed: {speed_reward:.2f}")
        
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
            
            info = {"cav_count": len(self.cav_ids), "step": self.current_step}
            
            return obs, reward, terminated, truncated, info
            
        except traci.exceptions.FatalTraCIError as e:
            print(f"TRACI 错误: {e}")
            # 尝试重新连接
            self.reset()
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0, True, False, {"error": str(e)}

    def reset(self, seed=None, options=None):
        """重置环境"""
        # 关闭现有的SUMO连接
        self.close()
        
        # 等待一小段时间确保资源释放
        time.sleep(0.5)
        
        # 尝试启动SUMO，有重试机制
        max_tries = 3
        for attempt in range(max_tries):
            try:
                traci.start(self.sumo_cmd)
                self._is_initialized = True
                break
            except Exception as e:
                print(f"SUMO 启动失败 (尝试 {attempt+1}/{max_tries}): {e}")
                time.sleep(2)  # 等待一段时间再重试
                if attempt == max_tries - 1:  # 最后一次尝试失败
                    raise RuntimeError(f"无法启动SUMO: {e}")

        # 运行一段时间，等待车辆生成
        print("等待车辆生成...")
        for step in range(100):
            try:
                traci.simulationStep()
                all_vehicles = traci.vehicle.getIDList()
                self.cav_ids = [v for v in all_vehicles if "CAV" in traci.vehicle.getTypeID(v)]
                self.hdv_ids = [v for v in all_vehicles if "HDV" in traci.vehicle.getTypeID(v)]
                
                # 如果已有CAV车辆，就可以开始了
                if self.cav_ids:
                    print(f"在第 {step} 步生成了 {len(self.cav_ids)} 辆CAV")
                    break
                    
                # 每20步输出一次状态
                if step % 20 == 0:
                    print(f"步骤 {step}: {len(all_vehicles)} 辆车, {len(self.cav_ids)} 辆CAV")
            except Exception as e:
                print(f"初始化时出错: {e}")
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
            print(f"关闭TRACI时出错: {e}")


# 自定义回调，用于记录训练信息
class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_training_start(self) -> None:
        print("开始训练...")

    def _on_step(self) -> bool:
        # 累积奖励和步数
        self.current_episode_reward += self.locals["rewards"][0]
        self.current_episode_length += 1
        
        # 检查是否完成一个回合
        if self.locals["dones"][0]:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            print(f"回合 {self.episode_count} 完成: 奖励={self.current_episode_reward:.2f}, 步数={self.current_episode_length}")
            
            # 重置当前回合的统计信息
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        return True

    def _on_rollout_end(self) -> None:
        # 在完成一个rollout后输出统计信息
        avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
        print(f"完成一个采样周期，当前总步数: {self.num_timesteps}, 最近10回合平均奖励: {avg_reward:.2f}")

    def _on_training_end(self) -> None:
        print(f"训练结束，总步数: {self.num_timesteps}, 总回合数: {self.episode_count}")


def test_training_episode():
    """测试一个训练回合的运行情况"""
    # 创建环境，禁用自动重置以便check_env能正常工作
    env = SumoMergeEnv(gui=True, need_reset=False)
    
    # 检查环境是否符合gym规范
    try:
        print("检查环境...")
        check_env(env)
        print("环境检查通过!")
    except Exception as e:
        print(f"环境检查失败: {e}")
        # 即使检查失败，我们也继续训练
    
    # 现在手动重置环境，为训练做准备
    env.reset()
    
    # 使用向量化环境包装器
    vec_env = DummyVecEnv([lambda: SumoMergeEnv(gui=True)])

    # 初始化 PPO 模型
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=5,  # 减少每批数据的训练轮数，避免过拟合
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./ppo_tensorboard_test/",
    )

    # 使用 learn() 运行训练，并添加回调
    callback = TrainingCallback()
    
    try:
        print("开始训练...")
        model.learn(total_timesteps=20000, callback=callback)
        print("训练完成!")
        
        # 保存模型
        model.save("ppo_sumo_model")
        print("模型已保存为 ppo_sumo_model")
    except Exception as e:
        print(f"训练过程中出错: {e}")
    finally:
        # 确保环境关闭
        vec_env.close()


def test_saved_model():
    """测试已保存的模型"""
    if not os.path.exists("ppo_sumo_model.zip"):
        print("模型文件不存在，请先训练模型")
        return
        
    # 创建环境
    env = SumoMergeEnv(gui=True)
    
    # 加载模型
    model = PPO.load("ppo_sumo_model")
    
    try:
        # 重置环境
        obs, _ = env.reset()
        
        # 运行模型
        done = False
        total_reward = 0
        step = 0
        
        while not done and step < 1000:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated
            
            if step % 100 == 0:
                print(f"步骤 {step}, 累积奖励: {total_reward:.2f}")
                
        print(f"测试完成: 总步数={step}, 总奖励={total_reward:.2f}")
    finally:
        # 关闭环境
        env.close()


if __name__ == "__main__":
    print("开始SUMO-RL训练测试")
    test_training_episode()
    # test_saved_model() 