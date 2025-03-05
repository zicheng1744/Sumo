import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import traci
from gym import spaces
from gym.spaces import Box
import time


class SumoMergeEnv:
    def __init__(self, cfg_path="config.sumocfg"):
        # 初始化SUMO连接
        self.sumo_cmd = ["sumo", "-c", cfg_path]
        self.sumo_conn = None
        self.reset()

        # 环境参数设置
        self.episode_length = 1000
        self.current_step = 0

        # 定义观测空间和动作空间
        self.observation_space = Box(
            low=0, high=100, shape=(6,)
        )  # 速度，位置，前后车距等
        self.action_space = Box(low=-3, high=3, shape=(1,))  # 加速度控制范围

        # 获取CAV车辆ID列表
        self.cav_ids = [v for v in traci.vehicle.getIDList() if "CAV" in v]

    def _get_observations(self):
        """获取CAV车辆的状态信息"""
        if not self.cav_ids:
            return np.zeros(6)

        # 获取主车信息
        ego_id = self.cav_ids[0]
        ego_speed = traci.vehicle.getSpeed(ego_id)
        ego_pos = traci.vehicle.getPosition(ego_id)

        # 获取周围车辆信息
        leader_id = traci.vehicle.getLeader(ego_id, 100)
        follower_id = traci.vehicle.getFollower(ego_id, 100)

        if leader_id:
            leader_speed = traci.vehicle.getSpeed(leader_id[0])
            leader_dist = leader_id[1]
        else:
            leader_speed = 0
            leader_dist = 100

        if follower_id:
            follower_speed = traci.vehicle.getSpeed(follower_id[0])
            follower_dist = follower_id[1]
        else:
            follower_speed = 0
            follower_dist = 100

        # 获取车道信息
        current_lane = traci.vehicle.getLaneID(ego_id)
        target_lane = "mc_1" if current_lane == "mc_0" else "mc_0"

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
        """应用控制动作"""
        if not self.cav_ids:
            return

        ego_id = self.cav_ids[0]
        current_speed = traci.vehicle.getSpeed(ego_id)
        new_speed = current_speed + action[0]

        # 限制速度在合理范围内
        new_speed = np.clip(new_speed, 0, 10)
        traci.vehicle.setSpeed(ego_id, new_speed)

    def _calculate_reward(self):
        """计算奖励"""
        if not self.cav_ids:
            return 0

        ego_id = self.cav_ids[0]
        current_speed = traci.vehicle.getSpeed(ego_id)
        leader = traci.vehicle.getLeader(ego_id, 100)

        # 速度奖励
        speed_reward = -abs(current_speed - 8)  # 鼓励保持8m/s的速度

        # 安全距离奖励
        if leader:
            dist = leader[1]
            if dist < 5:  # 距离过近
                safety_reward = -10
            elif dist < 10:  # 距离较近
                safety_reward = -5
            else:
                safety_reward = 0
        else:
            safety_reward = 0

        # 车道奖励
        current_lane = traci.vehicle.getLaneID(ego_id)
        target_lane = "mc_1" if current_lane == "mc_0" else "mc_0"
        lane_reward = 5 if current_lane == target_lane else -5

        return speed_reward + safety_reward + lane_reward

    def _check_done(self):
        """检查是否结束"""
        if not self.cav_ids:
            return True

        ego_id = self.cav_ids[0]
        return traci.vehicle.getDistance(ego_id) > 300  # 行驶超过300米结束

    def step(self, action):
        """执行一步仿真"""
        self._apply_action(action)
        traci.simulationStep()  # 使用traci API而不是self.sumo_conn
        self.current_step += 1

        obs = self._get_observations()
        reward = self._calculate_reward()
        done = self._check_done()

        return obs, reward, done, {}

    def reset(self):
        """重置环境"""
        if traci.isLoaded():
            traci.close()
            sys.stdout.flush()
            time.sleep(0.5)  # 增加关闭等待时间

        # 启动新连接
        try:
            self.sumo_conn = traci.start(self.sumo_cmd)
            traci.simulationStep()  # 推进初始步
            self.cav_ids = [v for v in traci.vehicle.getIDList() if "CAV" in v]
        except Exception as e:
            print(f"连接失败: {e}")
            sys.exit(1)

        return self._get_observations()


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # 共享网络层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh()
        )
        # Actor网络
        self.actor = nn.Sequential(nn.Linear(64, action_dim), nn.Tanh())
        # Critic网络
        self.critic = nn.Sequential(nn.Linear(64, 1))

    def forward(self, state):
        shared_features = self.shared(state)
        action_mean = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_mean, value


class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.clip_epsilon = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95

    def select_action(self, state):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0)
        action_mean, _ = self.policy(state)
        dist = Normal(action_mean, 0.1)  # 固定标准差
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy()[0], log_prob.detach()

    def update(self, states, actions, old_log_probs, rewards, next_states, dones):
        """更新策略"""
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)

        # 计算优势函数
        with torch.no_grad():
            _, values = self.policy(states)
            _, next_values = self.policy(torch.FloatTensor(next_states))
            advantages = self._compute_gae(rewards, values, next_values, dones)

        # 计算新的动作概率
        action_means, new_values = self.policy(states)
        dist = Normal(action_means, 0.1)
        new_log_probs = dist.log_prob(actions)

        # 计算比率
        ratio = torch.exp(new_log_probs - old_log_probs)

        # PPO裁剪目标
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            * advantages
        )
        policy_loss = -torch.min(surr1, surr2).mean()

        # 价值函数损失
        value_loss = nn.MSELoss()(new_values.squeeze(), rewards)

        # 总损失
        loss = policy_loss + 0.5 * value_loss

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _compute_gae(self, rewards, values, next_values, dones):
        """计算广义优势估计"""
        advantages = torch.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards) - 1)):
            if dones[t]:
                gae = 0
            delta = (
                rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        return advantages


def train():
    """训练主循环"""
    env = SumoMergeEnv()
    agent = PPO(state_dim=6, action_dim=1)

    for episode in range(1000):
        state = env.reset()
        episode_reward = 0
        states, actions, log_probs, rewards, next_states, dones = [], [], [], [], [], []

        for step in range(env.episode_length):
            # 选择动作
            action, log_prob = agent.select_action(state)

            # 执行动作
            next_state, reward, done, _ = env.step(action)

            # 存储经验
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            # 更新状态
            state = next_state
            episode_reward += reward

            # 如果结束则更新策略
            if done or step == env.episode_length - 1:
                loss = agent.update(
                    np.array(states),
                    np.array(actions),
                    torch.cat(log_probs),
                    np.array(rewards),
                    np.array(next_states),
                    np.array(dones),
                )
                print(
                    f"Episode {episode} | Step {step} | Reward: {episode_reward:.1f} | Loss: {loss:.4f}"
                )
                break


if __name__ == "__main__":
    train()
