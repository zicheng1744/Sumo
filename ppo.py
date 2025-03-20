import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


class MultiCAVActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, max_vehicles=50):
        super().__init__()
        self.max_vehicles = max_vehicles
        # 状态编码器 - 处理每个车辆的状态
        self.vehicle_encoder = nn.Sequential(
            nn.Linear(state_dim, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU()
        )

        # 全局特征提取器 - 处理所有车辆的编码状态
        self.global_net = nn.Sequential(
            nn.Linear(32 * max_vehicles, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )

        # 动作生成器 - 为每个车辆生成动作
        self.actor = nn.Linear(128, action_dim * max_vehicles)

        # 价值评估 - 评估状态的价值
        self.critic = nn.Linear(128, 1)

    def forward(self, states):
        batch_size = states.shape[0]
        n_vehicles = min(states.shape[1], self.max_vehicles)

        # 填充或裁剪车辆数量
        if states.shape[1] < self.max_vehicles:
            # 填充缺失的车辆状态为0
            padding = torch.zeros(
                batch_size, self.max_vehicles - states.shape[1], states.shape[2]
            ).to(states.device)
            states = torch.cat([states, padding], dim=1)
        elif states.shape[1] > self.max_vehicles:
            # 裁剪多余的车辆
            states = states[:, : self.max_vehicles, :]

        # 重塑状态以便于处理
        reshaped_states = states.reshape(batch_size * self.max_vehicles, -1)

        # 对每个车辆的状态进行编码
        vehicle_features = self.vehicle_encoder(reshaped_states)
        vehicle_features = vehicle_features.reshape(batch_size, self.max_vehicles * 32)

        # 提取全局特征
        global_features = self.global_net(vehicle_features)

        # 生成每个车辆的动作
        action_means = self.actor(global_features)
        action_means = action_means.reshape(batch_size, self.max_vehicles, -1)

        # 限制只返回实际车辆数量的动作
        action_means = action_means[:, :n_vehicles, :]

        # 评估状态价值
        value = self.critic(global_features)

        return action_means, value


class PPO:
    def __init__(self, state_dim, action_dim, max_vehicles=10):
        self.policy = MultiCAVActorCritic(state_dim, action_dim, max_vehicles)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.clip_epsilon = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95

    def select_action(self, state):
        """为多辆车辆选择动作"""
        # 将状态转换为Tensor，确保形状为 [batch_size, n_vehicles, state_dim]
        state = torch.FloatTensor(state).unsqueeze(0)  # 添加batch维度

        # 通过网络获取动作均值
        action_means, _ = self.policy(state)

        # 创建对角正态分布
        std = torch.ones_like(action_means) * 0.1
        dist = Normal(action_means, std)

        # 从分布中采样动作
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        return actions.detach().numpy()[0], log_probs.detach()

    def update(self, states, actions, old_log_probs, rewards, next_states, dones):
        """更新策略网络"""
        # 将输入转换为Tensor
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # 计算优势函数
        with torch.no_grad():
            _, values = self.policy(states)
            _, next_values = self.policy(next_states)
            advantages = self._compute_gae(rewards, values, next_values, dones)

        # 计算新的动作分布
        action_means, new_values = self.policy(states)
        std = torch.ones_like(action_means) * 0.1
        dist = Normal(action_means, std)
        new_log_probs = dist.log_prob(actions)

        # 计算策略比率
        ratio = torch.exp(new_log_probs.sum(-1) - old_log_probs.sum(-1))

        # 计算裁剪的策略目标
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            * advantages
        )
        policy_loss = -torch.min(surr1, surr2).mean()

        # 计算价值损失
        value_loss = nn.MSELoss()(new_values.squeeze(), rewards)

        # 总损失
        loss = policy_loss + 0.5 * value_loss

        # 执行优化步骤
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _compute_gae(self, rewards, values, next_values, dones):
        """计算广义优势估计"""
        # 创建一个全零张量，用于存储GAE
        advantages = torch.zeros_like(rewards)
        gae = 0

        # 从后向前计算GAE
        for t in reversed(range(len(rewards))):
            # dones[t]用于标记是否到达终止状态
            if dones[t]:
                gae = 0
            # 计算delta，delta是TD误差，是状态值的估计值和真实值之间的差
            delta = (
                rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        return advantages
