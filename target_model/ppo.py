import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh()
        )
        self.actor = nn.Sequential(nn.Linear(64, action_dim), nn.Tanh())
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
        state = torch.FloatTensor(state).unsqueeze(0)
        action_mean, _ = self.policy(state)
        dist = Normal(action_mean, 0.1)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy()[0], log_prob.detach()

    def update(self, states, actions, old_log_probs, rewards, next_states, dones):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        with torch.no_grad():
            _, values = self.policy(states)
            _, next_values = self.policy(next_states)
            advantages = self._compute_gae(rewards, values, next_values, dones)

        action_means, new_values = self.policy(states)
        dist = Normal(action_means, 0.1)
        new_log_probs = dist.log_prob(actions)

        ratio = torch.exp(new_log_probs - old_log_probs)

        surr1 = ratio * advantages
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

    def _compute_gae(self, rewards, values, next_values, dones):
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
