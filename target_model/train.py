import numpy as np
import torch
import sys
import os
import logging

# 添加父目录到搜索路径
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from target_model.envir import create_sumo_env
from target_model.ppo import PPO


def train_ppo(env):
    """训练主循环"""
    # 创建PPO代理
    agent = PPO(state_dim=6, action_dim=1)

    try:
        for episode in range(100):
            state = env.reset()
            episode_reward = 0
            states, actions, log_probs, rewards, next_states, dones = (
                [],
                [],
                [],
                [],
                [],
                [],
            )

            for step in range(env.episode_length):
                action, log_prob = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)

                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

                state = next_state
                episode_reward += reward

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

    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        # 确保环境正确关闭
        print("Closing environment...")
        env.close()


if __name__ == "__main__":
    try:
        env = create_sumo_env(gui=True)
        train_ppo(env)
    except Exception as e:
        print(f"Failed to create or train environment: {e}")
        # 如果在这里出现异常，可能没有env变量，所以不调用env.close()
