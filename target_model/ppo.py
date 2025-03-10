import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()  # 继承父类的所有属性
        # sequential是一个容器，可以将模块按顺序排列
        # nn.Linear是一个全连接层，输入特征数为state_dim，输出特征数为64
        # nn.Tanh是一个激活函数，用于增加网络的非线性
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh()
        )
        # actor和critic是两个独立的网络，用于输出动作和状态值
        # actor是如何选择动作的？ actor输出的是动作的均值
        # critic是如何评价状态的？ critic输出的是
        self.actor = nn.Sequential(nn.Linear(64, action_dim), nn.Tanh())
        self.critic = nn.Sequential(nn.Linear(64, 1))

    # 输入状态，输出动作和状态值
    def forward(self, state):
        # shared_features是共享网络的输出
        shared_features = self.shared(state)
        action_mean = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_mean, value


class PPO:
    # self是指向类的实例本身的指针
    # state_dim是状态的维度，action_dim是动作的维度
    def __init__(self, state_dim, action_dim):
        # 创建一个ActorCritic网络
        self.policy = ActorCritic(state_dim, action_dim)
        # 创建一个Adam优化器
        # Adam是一种自适应学习率的优化算法
        # lr是学习率，3e-4表示3*10^-4
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        # clip_epsilon是PPO算法中的一个超参数
        # gamma是折扣因子，用于计算折扣奖励
        # gae_lambda是GAE算法中的一个超参数
        self.clip_epsilon = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95

    # 输入状态，输出动作和动作的对数概率
    def select_action(self, state):
        # 将状态转换为张量，并增加一个维度
        state = torch.FloatTensor(state).unsqueeze(0)
        # 通过ActorCritic网络输出动作均值和状态值
        action_mean, _ = self.policy(state)
        # 创建一个正态分布，均值为action_mean，标准差为0.1
        dist = Normal(action_mean, 0.1)
        # 从正态分布中采样一个动作
        action = dist.sample()
        # 计算动作的对数概率
        log_prob = dist.log_prob(action)
        return action.detach().numpy()[0], log_prob.detach()

    # 输入状态、动作、动作的对数概率、奖励、下一个状态和done标志
    # 输出损失
    def update(self, states, actions, old_log_probs, rewards, next_states, dones):
        # 将输入数据转换为张量
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        # dones用于标记是否到达终止状态
        dones = torch.FloatTensor(dones)

        # 计算GAE
        # GAE是一种用于估计优势函数的算法
        # 优势函数是指一个状态相对于平均状态的价值
        # 优势函数可以用于计算动作的优势
        # 优势越大，动作越好
        # no_grad是一个上下文管理器，用于关闭梯度计算
        with torch.no_grad():
            # 通过ActorCritic网络输出状态值
            _, values = self.policy(states)
            _, next_values = self.policy(next_states)
            # 计算GAE
            # _compute_gae是一个私有方法，用于计算GAE
            advantages = self._compute_gae(rewards, values, next_values, dones)

        # 计算新的动作均值和状态值
        action_means, new_values = self.policy(states)
        # 创建一个正态分布，均值为action_means，标准差为0.1
        dist = Normal(action_means, 0.1)
        # 计算新的动作的对数概率
        new_log_probs = dist.log_prob(actions)

        # 计算比率，用于计算策略损失
        ratio = torch.exp(new_log_probs - old_log_probs)

        # 计算策略损失
        # surr1是PPO算法中的第一个损失项
        # surr1代表了新的动作和旧的动作之间的比率
        surr1 = ratio * advantages
        # surr2是PPO算法中的第二个损失项
        # surr2代表了新的动作和旧的动作之间的比率，但是被截断了
        # clip_epsilon是PPO算法中的一个超参数
        surr2 = (
            torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            * advantages
        )
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = nn.MSELoss()(new_values.squeeze(), rewards)

        loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # 计算GAE
    def _compute_gae(self, rewards, values, next_values, dones):
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
