import numpy as np
import torch
import sys
import os
import logging
import time
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

# 添加父目录到搜索路径
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from target_model.envir import create_sumo_env
from target_model.ppo import PPO


# 设置日志
def setup_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    # 创建logger
    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)

    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 格式化器
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file


# 训练函数
def train_ppo_nogui(
    env,
    max_vehicles=50,
    n_episodes=100,
    log_dir="logs",
    save_dir="models",
    save_interval=10,
):
    """无GUI训练主循环，记录详细日志"""
    # 设置日志和保存目录
    logger, log_file = setup_logger(log_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 创建PPO代理，支持多车辆控制
    agent = PPO(state_dim=6, action_dim=1, max_vehicles=max_vehicles)

    # 记录训练指标
    all_episode_rewards = []
    all_episode_steps = []
    all_episode_losses = []
    all_vehicle_counts = []

    # 记录开始时间
    start_time = time.time()
    logger.info(f"开始训练，最大车辆数: {max_vehicles}，训练回合数: {n_episodes}")

    try:
        # 训练n_episodes个episode
        for episode in range(n_episodes):
            # 重置环境
            state = env.reset()
            episode_reward = 0
            episode_losses = []

            # 获取车辆数量
            n_vehicles = (
                state.shape[0]
                if hasattr(state, "shape") and len(state.shape) > 1
                else 1
            )
            logger.info(f"Episode {episode} 开始，当前车辆数: {n_vehicles}")

            # 初始化数据列表
            states, actions, log_probs, rewards, next_states, dones = (
                [],
                [],
                [],
                [],
                [],
                [],
            )

            # 当前episode的起始时间
            episode_start = time.time()

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
                    padded_states = []
                    for s in states:
                        padded = np.zeros((max_vehicles, 6))  # 6是观测向量维度
                        padded[: s.shape[0]] = s
                        padded_states.append(padded)
                    padded_actions = []
                    for a in actions:
                        padded = np.zeros((max_vehicles, 1))
                        padded[: len(a)] = a
                        padded_actions.append(padded)
                    padded_next_states = []
                    for ns in next_states:
                        padded = np.zeros((max_vehicles, 6))
                        padded[: ns.shape[0]] = ns
                        padded_next_states.append(padded)

                    np_states = np.array(padded_states)
                    np_actions = np.array(padded_actions)
                    np_next_states = np.array(padded_next_states)

                    loss = agent.update(
                        np_states,
                        np_actions,
                        torch.cat([lp.flatten() for lp in log_probs]).reshape(
                            len(log_probs), -1
                        ),
                        np.array(rewards),
                        np_next_states,
                        np.array(dones),
                    )

                    episode_losses.append(loss)
                    episode_duration = time.time() - episode_start
                    n_vehicles = (
                        state.shape[0]
                        if hasattr(state, "shape") and len(state.shape) > 1
                        else 1
                    )

                    # 记录本次episode的结果
                    logger.info(
                        f"Episode {episode}/{n_episodes-1} | Steps: {step} | "
                        f"Vehicles: {n_vehicles} | Reward: {episode_reward:.1f} | "
                        f"Loss: {loss:.4f} | Duration: {episode_duration:.1f}s"
                    )

                    # 保存训练指标
                    all_episode_rewards.append(episode_reward)
                    all_episode_steps.append(step)
                    all_episode_losses.append(
                        np.mean(episode_losses) if episode_losses else 0
                    )
                    all_vehicle_counts.append(n_vehicles)

                    break

            # 定期保存模型
            if (episode + 1) % save_interval == 0 or episode == n_episodes - 1:
                model_path = os.path.join(save_dir, f"ppo_model_ep{episode+1}.pt")
                torch.save(agent.policy.state_dict(), model_path)
                logger.info(f"模型已保存至: {model_path}")

                # 生成并保存训练曲线
                plot_training_curves(
                    all_episode_rewards,
                    all_episode_losses,
                    all_episode_steps,
                    all_vehicle_counts,
                    log_dir,
                    episode,
                )

    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中出错: {e}", exc_info=True)
    finally:
        # 确保环境正确关闭
        logger.info("关闭环境...")
        env.close()

        # 训练总结
        total_duration = time.time() - start_time
        logger.info(f"训练结束，总用时: {total_duration:.1f}秒")
        logger.info(f"最终平均奖励: {np.mean(all_episode_rewards[-10:]):.2f}")

        # 保存最终训练曲线
        plot_training_curves(
            all_episode_rewards,
            all_episode_losses,
            all_episode_steps,
            all_vehicle_counts,
            log_dir,
            "final",
        )

        # 返回训练日志路径和最终模型路径
        return log_file, os.path.join(save_dir, f"ppo_model_ep{n_episodes}.pt")


def plot_training_curves(rewards, losses, steps, vehicle_counts, log_dir, episode):
    """绘制训练曲线并保存"""
    plt.figure(figsize=(12, 10))

    # 绘制奖励曲线
    plt.subplot(2, 2, 1)
    plt.plot(rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    # 绘制损失曲线
    plt.subplot(2, 2, 2)
    plt.plot(losses)
    plt.title("Episode Losses")
    plt.xlabel("Episode")
    plt.ylabel("Loss")

    # 绘制步数曲线
    plt.subplot(2, 2, 3)
    plt.plot(steps)
    plt.title("Episode Steps")
    plt.xlabel("Episode")
    plt.ylabel("Steps")

    # 绘制车辆数量曲线
    plt.subplot(2, 2, 4)
    plt.plot(vehicle_counts)
    plt.title("Vehicle Count")
    plt.xlabel("Episode")
    plt.ylabel("Number of Vehicles")

    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"training_curves_ep{episode}.png"))
    plt.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description="无GUI模式的PPO训练脚本")
    parser.add_argument("--episodes", type=int, default=100, help="训练回合数")
    parser.add_argument("--max-vehicles", type=int, default=50, help="最大车辆数")
    parser.add_argument("--log-dir", type=str, default="logs", help="日志保存目录")
    parser.add_argument("--save-dir", type=str, default="models", help="模型保存目录")
    parser.add_argument(
        "--save-interval", type=int, default=10, help="模型保存间隔(回合)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()

    try:
        # 创建无GUI环境
        env = create_sumo_env(gui=False)  # 关闭GUI以提高训练速度

        # 开始训练
        log_file, model_path = train_ppo_nogui(
            env,
            max_vehicles=args.max_vehicles,
            n_episodes=args.episodes,
            log_dir=args.log_dir,
            save_dir=args.save_dir,
            save_interval=args.save_interval,
        )

        print(f"\n训练完成!")
        print(f"日志文件: {log_file}")
        print(f"最终模型: {model_path}")
        print(f"训练曲线: {os.path.join(args.log_dir, 'training_curves_final.png')}")

    except Exception as e:
        print(f"训练失败: {e}")
