import os
import numpy as np
import traci
import gymnasium as gym
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback


class SumoMergeEnv(gym.Env):
    def __init__(self, cfg_path=None, gui=True):
        super(SumoMergeEnv, self).__init__()
        if cfg_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            cfg_path = os.path.join(base_dir, "input_sources", "config.sumocfg")

        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"SUMO 配置文件未找到: {cfg_path}")

        self.sumo_cmd = ["sumo-gui" if gui else "sumo", "-c", cfg_path]
        self.gui = gui
        self.episode_length = 1000
        self.current_step = 0
        self.cav_ids = []
        self.hdv_ids = []
        self._is_initialized = False
        self.max_cavs = 50

        self.observation_space = Box(
            low=0, high=100, shape=(self.max_cavs * 6,), dtype=np.float32
        )
        self.action_space = Box(
            low=-1, high=1, shape=(self.max_cavs,), dtype=np.float32
        )
        self.action_scale = 3

    def _get_observations(self):
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
                print(f"获取 {cav_id} 观测失败: {e}")  # 使用 print 替代 logging
                obs.extend([0, 100, 0, 100, 0, 0])

        flat_obs = np.zeros(self.max_cavs * 6, dtype=np.float32)
        flat_obs[: len(obs)] = obs
        return flat_obs

    def _apply_action(self, actions):
        actions = actions * self.action_scale
        for i, cav_id in enumerate(self.cav_ids):
            try:
                current_speed = traci.vehicle.getSpeed(cav_id)
                new_speed = np.clip(current_speed + actions[i], 0, 10)
                traci.vehicle.setSpeed(cav_id, new_speed)
            except Exception as e:
                print(f"应用动作到 {cav_id} 失败: {e}")  # 使用 print 替代 logging

    def _calculate_reward(self):
        if not self.cav_ids:
            return 0
        total_speed = 0
        speeds = []
        safety_reward = 0
        target_speed = 10
        for cav_id in self.cav_ids:
            try:
                speed = traci.vehicle.getSpeed(cav_id)
                speeds.append(speed)
                total_speed += speed
                leader = traci.vehicle.getLeader(cav_id, 100)
                if leader and leader[1] < 5:
                    safety_reward -= 10
                elif leader and leader[1] < 10:
                    safety_reward -= 5
            except Exception as e:
                print(f"计算 {cav_id} 奖励失败: {e}")  # 使用 print 替代 logging
        avg_speed = total_speed / len(self.cav_ids) if self.cav_ids else 0
        speed_reward = -abs(avg_speed - target_speed)
        coop_reward = -np.var(speeds) if len(speeds) > 1 else 0
        return speed_reward + safety_reward + coop_reward

    def step(self, action):
        if not self._is_initialized:
            self.reset()
        self._apply_action(action)
        traci.simulationStep()
        self.current_step += 1
        all_vehicles = traci.vehicle.getIDList()
        self.cav_ids = [v for v in all_vehicles if "CAV" in traci.vehicle.getTypeID(v)]
        self.hdv_ids = [v for v in all_vehicles if "HDV" in traci.vehicle.getTypeID(v)]
        obs = self._get_observations()
        reward = self._calculate_reward()
        terminated = len(all_vehicles) == 0
        truncated = self.current_step >= self.episode_length
        info = {"cav_count": len(self.cav_ids)}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.close()
        try:
            traci.start(self.sumo_cmd)
            self._is_initialized = True
        except Exception as e:
            print(f"SUMO 启动失败: {e}")  # 使用 print 替代 logging
            raise

        for _ in range(100):
            traci.simulationStep()
            all_vehicles = traci.vehicle.getIDList()
            self.cav_ids = [
                v for v in all_vehicles if "CAV" in traci.vehicle.getTypeID(v)
            ]
            self.hdv_ids = [
                v for v in all_vehicles if "HDV" in traci.vehicle.getTypeID(v)
            ]
            if self.cav_ids:
                break
        self.current_step = 0
        return self._get_observations(), {}

    def close(self):
        if traci.isLoaded():
            traci.close()
            self._is_initialized = False


# 自定义回调，用于记录训练信息
class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)

    def _on_training_start(self) -> None:
        print("开始训练测试...")

    def _on_step(self) -> bool:
        # 每一步记录信息（可选）
        return True

    def _on_rollout_end(self) -> None:
        print(f"完成一个回合，当前总步数: {self.num_timesteps}")

    def _on_training_end(self) -> None:
        print(f"训练结束，总步数: {self.num_timesteps}")


def test_training_episode():
    """测试一个训练回合的运行情况"""
    # 创建环境并检查
    env = SumoMergeEnv(gui=True)
    check_env(env)  # 使用 SB3 自带的 check_env 验证环境
    env = DummyVecEnv([lambda: env])

    # 初始化 PPO 模型
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./ppo_tensorboard_test/",
    )

    # 获取初始观测
    obs = env.reset()
    print(f"初始观测: {obs[0][:12]}")  # 使用 print 替代 logging

    # 使用 learn() 运行训练，并添加回调
    callback = TrainingCallback()
    model.learn(total_timesteps=1024, callback=callback)

    # 环境关闭由 DummyVecEnv 自动处理，无需手动调用 env.close()


if __name__ == "__main__":
    test_training_episode()
