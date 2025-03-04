import os
import sys
import traci
from sumolib import checkBinary
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


# 创建自定义SUMO环境
class SumoEnv(gym.Env):
    def __init__(self):
        super(SumoEnv, self).__init__()

        # 定义动作空间（示例：3个离散动作）
        self.action_space = spaces.Discrete(3)  # 加速/保持/减速

        # 定义观察空间（示例：速度、位置）
        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([30, 1000], dtype=np.float32),
            dtype=np.float32,
        )

        # SUMO配置
        self.sumo_cmd = [
            checkBinary("sumo"),  # 或sumo-gui
            "-c",
            "cross.sumocfg",
            "--collision.action",
            "warn",
        ]

        self.simulation_steps = 0

    def reset(self):
        if traci.isLoaded():
            traci.close()

        # 启动SUMO
        traci.start(self.sumo_cmd)
        self.simulation_steps = 0
        return self._get_observation()

    def step(self, action):
        # 执行动作（示例动作）
        vehicle_id = "veh0"  # 假设存在这个车辆
        if action == 0:
            traci.vehicle.slowDown(vehicle_id, 5, 1)  # 减速
        elif action == 1:
            current_speed = traci.vehicle.getSpeed(vehicle_id)
            traci.vehicle.setSpeed(vehicle_id, current_speed)  # 保持
        else:
            traci.vehicle.slowDown(vehicle_id, 15, 1)  # 加速

        # 推进仿真
        traci.simulationStep()
        self.simulation_steps += 1

        # 获取观测值
        obs = self._get_observation()

        # 计算奖励（示例）
        reward = traci.vehicle.getSpeed(vehicle_id)  # 使用速度作为奖励

        # 终止条件
        done = self.simulation_steps >= 1000  # 限制仿真步数
        if traci.vehicle.getIDCount() == 0:
            done = True

        return obs, reward, done, {}

    def _get_observation(self):
        # 获取观测值（示例）
        if traci.vehicle.getIDCount() > 0:
            vehicle_id = traci.vehicle.getIDList()[0]
            speed = traci.vehicle.getSpeed(vehicle_id)
            pos = traci.vehicle.getPosition(vehicle_id)[0]
            return np.array([speed, pos], dtype=np.float32)
        else:
            return np.zeros(2, dtype=np.float32)

    def close(self):
        traci.close()


# 验证环境
env = SumoEnv()
check_env(env)  # 检查环境是否符合gym规范

# 初始化PPO模型
model = PPO("MlpPolicy", env, verbose=1)

# 训练（示例）
model.learn(total_timesteps=10000)

# 测试训练后的模型
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        obs = env.reset()

env.close()
