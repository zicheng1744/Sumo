import numpy as np
import torch
import sys
import os
import logging

# 添加父目录到搜索路径
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from target_model.envir import create_sumo_env
from target_model.ppo import PPO


def train_ppo(env, max_vehicles=50):
    """训练主循环"""
    # 创建PPO代理，支持多车辆控制
    agent = PPO(state_dim=6, action_dim=1, max_vehicles=max_vehicles)

    try:
        # 训练100个episode
        for episode in range(100):
            # 重置环境
            state = env.reset()
            # 记录每个episode的总奖励
            episode_reward = 0

            # 初始化数据列表
            states, actions, log_probs, rewards, next_states, dones = (
                [],
                [],
                [],
                [],
                [],
                [],
            )

            for step in range(env.episode_length):
                # 选择动作 - 现在可以处理多辆车
                action, log_prob = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)

                # 存储数据
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

                # 更新状态和累积奖励
                state = next_state
                episode_reward += reward

                # 如果episode结束或达到最大步数，则更新策略
                if done or step == env.episode_length - 1:
                    loss = agent.update(
                        np.array(states),
                        np.array(actions),
                        torch.cat([lp.flatten() for lp in log_probs]).reshape(
                            len(log_probs), -1
                        ),
                        np.array(rewards),
                        np.array(next_states),
                        np.array(dones),
                    )
                    n_vehicles = (
                        state.shape[0]
                        if hasattr(state, "shape") and len(state.shape) > 1
                        else 1
                    )
                    print(
                        f"Episode {episode} | Step {step} | Vehicles {n_vehicles} | "
                        f"Reward: {episode_reward:.1f} | Loss: {loss:.4f}"
                    )
                    break

    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        print("Closing environment...")
        env.close()


if __name__ == "__main__":
    try:
        env = create_sumo_env(gui=True)
        train_ppo(env, max_vehicles=50)  # 支持最多100辆CAV
    except Exception as e:
        print(f"Failed to create or train environment: {e}")
        # 如果在这里出现异常，可能没有env变量，所以不调用env.close()
