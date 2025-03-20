import os
import time
import logging
import numpy as np
import traci
import gymnasium as gym
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("sumo_ppo_training.log")],
)


class SumoMergeEnv(gym.Env):
    def __init__(self, cfg_path=None, gui=False):
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
        self.episode_length = 10000
        self.current_step = 0
        self.cav_ids = []
        self.hdv_ids = []
        self._is_initialized = False
        self.max_cavs = 50  # 最大支持的 CAV 数量

        # 固定观测空间和动作空间
        self.observation_space = Box(
            low=0, high=100, shape=(self.max_cavs * 6,), dtype=np.float32
        )
        self.action_space = Box(
            low=-1, high=1, shape=(self.max_cavs,), dtype=np.float32
        )
        self.action_scale = 3  # 动作缩放因子，从 [-1, 1] 到 [-3, 3]

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
                logging.error(f"获取 {cav_id} 观测失败: {e}")
                obs.extend([0, 100, 0, 100, 0, 0])

        # 填充到固定长度
        flat_obs = np.zeros(self.max_cavs * 6, dtype=np.float32)
        flat_obs[: len(obs)] = obs
        return flat_obs

    def _apply_action(self, actions):
        """应用动作到所有 CAV"""
        actions = actions * self.action_scale  # 缩放到 [-3, 3]
        for i, cav_id in enumerate(self.cav_ids):
            try:
                current_speed = traci.vehicle.getSpeed(cav_id)
                new_speed = np.clip(current_speed + actions[i], 0, 10)
                traci.vehicle.setSpeed(cav_id, new_speed)
            except Exception as e:
                logging.error(f"应用动作到 {cav_id} 失败: {e}")

    def _calculate_reward(self):
        """计算全局奖励"""
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
                logging.error(f"计算 {cav_id} 奖励失败: {e}")
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
            logging.error(f"SUMO 启动失败: {e}")
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
            time.sleep(1)


def train_ppo():
    env = SumoMergeEnv(gui=False)
    check_env(env)
    env = DummyVecEnv([lambda: SumoMergeEnv(gui=False)])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/",
    )

    model.learn(total_timesteps=100000)
    model.save("ppo_sumo_merge")
    logging.info("训练完成，模型已保存")
    env.close()


def test_ppo():
    env = SumoMergeEnv(gui=True)
    model = PPO.load("ppo_sumo_merge")
    obs, _ = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()


if __name__ == "__main__":
    train_ppo()
    # test_ppo()
