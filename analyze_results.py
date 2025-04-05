import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from glob import glob
import datetime

def load_ppo_metrics(session_dir):
    """加载PPO指标数据"""
    try:
        # 尝试在data目录和session_dir目录中查找
        data_dir = os.path.join(session_dir, "data")
        if not os.path.exists(data_dir):
            data_dir = session_dir
            
        # 检查典型的指标文件名
        for filename in ["ppo_metrics.csv", "metrics.csv", "ppo_data.csv", "training_metrics.csv"]:
            metrics_path = os.path.join(data_dir, filename)
            if os.path.exists(metrics_path):
                df = pd.read_csv(metrics_path)
                # 检查是否包含关键指标列
                required_columns = ["explained_variance", "approx_kl", "loss"]
                # 如果至少有一个关键列，认为这是有效的指标数据
                if any(col in df.columns for col in required_columns):
                    print(f"从 {metrics_path} 加载PPO指标数据")
                    return df
                    
        # 尝试搜索所有CSV文件
        for root, _, files in os.walk(session_dir):
            for file in files:
                if file.endswith(".csv") and "metric" in file.lower():
                    metrics_path = os.path.join(root, file)
                    df = pd.read_csv(metrics_path)
                    if any(col in df.columns for col in ["loss", "kl", "explained_variance"]):
                        print(f"从 {metrics_path} 加载PPO指标数据")
                        return df
        
        print("未找到PPO指标数据或格式不正确")
        return None
    except Exception as e:
        print(f"加载PPO指标数据时出错: {str(e)}")
        return None

def load_training_data(session_dir):
    """
    加载训练数据，包括回合统计和PPO指标
    
    Args:
        session_dir: 训练会话目录路径
    
    Returns:
        episodes_df: 回合统计数据
        metrics_df: PPO训练指标数据
        metadata: 训练元数据
    """
    # 查找CSV文件
    data_dir = os.path.join(session_dir, "data")
    
    # 如果data目录不存在，可能数据直接保存在session_dir中
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}，尝试在会话根目录中查找...")
        data_dir = session_dir
    
    # 添加更多文件搜索位置
    possible_data_dirs = [
        data_dir,  # 标准位置
        session_dir,  # 会话根目录
        os.path.join(session_dir, "logs"),  # 日志目录
        os.path.join(session_dir, "results")  # 结果目录
    ]
    
    # 搜索更多可能的文件名模式
    episode_csv_patterns = [
        "training_episodes_*.csv",  # 带时间戳
        "training_episodes.csv",    # 无时间戳
        "episodes_*.csv",           # 简化名称
        "episode_info_*.csv",       # 另一种可能的命名
        "episodes.csv",             # 更简化的名称
        "*.csv"                     # 最后尝试任何CSV
    ]
    
    # 寻找匹配的文件
    episode_csv = []
    
    # 在所有可能的目录中搜索
    for dir_path in possible_data_dirs:
        if not os.path.exists(dir_path):
            continue
            
        # 搜索回合数据
        for pattern in episode_csv_patterns:
            pattern_path = os.path.join(dir_path, pattern)
            found_files = glob(pattern_path)
            if found_files:
                # 如果找到了多个文件，优先选择含有'episode'关键字的
                for file in found_files:
                    if 'episode' in file.lower() and file not in episode_csv:
                        episode_csv.append(file)
                # 如果没有找到含有'episode'的文件，添加所有找到的文件
                if not episode_csv and not any('episode' in f.lower() for f in found_files):
                    episode_csv.extend([f for f in found_files if f not in episode_csv])
    
    episodes_df = None
    metadata = None
    
    # 加载回合数据
    if episode_csv:
        # 尝试从找到的文件中加载数据，直到成功
        for csv_file in episode_csv:
            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                    # 检查是否包含必要的列
                    if 'episode' in df.columns or 'reward' in df.columns or 'length' in df.columns:
                        episodes_df = df
                        print(f"成功加载回合数据: {csv_file}")
                        break
                    else:
                        print(f"文件 {csv_file} 不包含回合数据所需的列")
                except Exception as e:
                    print(f"加载 {csv_file} 出错: {e}")
        
        if episodes_df is None:
            print("未能从找到的文件中加载回合数据")
    else:
        print("未找到任何可能的回合数据文件")
    
    # 使用专门的函数加载PPO指标数据
    metrics_df = load_ppo_metrics(session_dir)
    
    # 尝试从其他文件中提取回合数据（如果仍然没有）
    if episodes_df is None and metrics_df is not None:
        # 尝试从metrics数据中提取回合信息
        episode_related_cols = ['ep_len_mean', 'ep_rew_mean', 'episode_reward', 'episode_length', 'success_rate']
        if any(col in metrics_df.columns for col in episode_related_cols):
            print("尝试从指标数据中提取回合信息...")
            # 创建一个新的DataFrame作为回合数据
            try:
                episodes_df = pd.DataFrame()
                # 添加episode列
                episodes_df['episode'] = range(1, len(metrics_df) + 1)
                
                # 映射metrics_df中的列到episodes_df
                column_mapping = {
                    'ep_rew_mean': 'reward',
                    'episode_reward': 'reward', 
                    'ep_len_mean': 'length',
                    'episode_length': 'length',
                    'success_rate': 'success'
                }
                
                for metrics_col, episode_col in column_mapping.items():
                    if metrics_col in metrics_df.columns:
                        episodes_df[episode_col] = metrics_df[metrics_col]
                
                print("成功从指标数据中提取回合信息")
            except Exception as e:
                print(f"从指标数据中提取回合信息失败: {e}")
                episodes_df = None
    
    # 加载元数据
    metadata_file = os.path.join(session_dir, "metadata.json")
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"成功加载训练元数据")
        except Exception as e:
            print(f"加载元数据出错: {e}")
    else:
        print("未找到元数据文件")
    
    return episodes_df, metrics_df, metadata

def plot_learning_curve(episodes_df, save_dir):
    """绘制学习曲线：ep_rew_mean和ep_len_mean vs episode"""
    if episodes_df is None or episodes_df.empty:
        print("无法绘制学习曲线：缺少回合数据或数据为空")
        return
    
    # 检查必要的列是否存在
    required_cols = ['episode', 'reward', 'length']
    missing_cols = [col for col in required_cols if col not in episodes_df.columns]
    if missing_cols:
        print(f"无法绘制学习曲线：缺少必要的列 {missing_cols}")
        return
    
    try:
        plt.figure(figsize=(12, 6))
        
        # 创建双Y轴
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # 绘制奖励曲线
        line1, = ax1.plot(episodes_df['episode'], episodes_df['reward'], 'b-', label='Episode Reward')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # 绘制回合长度曲线
        line2, = ax2.plot(episodes_df['episode'], episodes_df['length'], 'r-', label='Episode Length')
        ax2.set_ylabel('Length', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # 设置图例
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        plt.legend(lines, labels, loc='upper left')
        
        plt.title('Learning Curve: Reward and Episode Length')
        plt.grid(True)
        
        # 添加10回合移动平均线
        if len(episodes_df) >= 10:
            window_size = 10
            # 计算移动平均
            reward_ma = episodes_df['reward'].rolling(window=window_size).mean()
            # 绘制移动平均线
            ax1.plot(episodes_df['episode'][window_size-1:], reward_ma[window_size-1:], 'g--', 
                    label=f'{window_size}-Episode Moving Average')
        
        # 保存图表
        save_path = os.path.join(save_dir, "learning_curve.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"学习曲线已保存至: {save_path}")
    except Exception as e:
        print(f"绘制学习曲线时出错: {e}")

def plot_loss_curves(metrics_df, save_dir):
    """绘制损失曲线：value_loss、policy_loss和approx_kl"""
    if metrics_df is None or metrics_df.empty:
        print("无法绘制损失曲线：缺少PPO指标数据或数据为空")
        return
    
    # 检查必要的列是否存在
    required_cols = ['value_loss', 'policy_gradient_loss', 'approx_kl']
    missing_cols = [col for col in required_cols if col not in metrics_df.columns]
    if missing_cols:
        print(f"无法绘制损失曲线：缺少必要的列 {missing_cols}")
        return
    
    try:
        plt.figure(figsize=(12, 6))
        
        # 创建主Y轴和一个次Y轴（用于KL散度，它可能有不同的量级）
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # 索引或迭代次数
        iterations = range(1, len(metrics_df) + 1)
        
        # 绘制value_loss和policy_loss在主Y轴
        line1, = ax1.plot(iterations, metrics_df['value_loss'], 'b-', label='Value Loss')
        line2, = ax1.plot(iterations, metrics_df['policy_gradient_loss'], 'g-', label='Policy Gradient Loss')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss Value')
        
        # 绘制approx_kl在次Y轴
        line3, = ax2.plot(iterations, metrics_df['approx_kl'], 'r-', label='Approx KL Divergence')
        ax2.set_ylabel('KL Divergence')
        
        # 设置图例
        lines = [line1, line2, line3]
        labels = [l.get_label() for l in lines]
        plt.legend(lines, labels, loc='upper right')
        
        plt.title('PPO Training Losses and KL Divergence')
        plt.grid(True)
        
        # 保存图表
        save_path = os.path.join(save_dir, "loss_curves.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"损失曲线已保存至: {save_path}")
    except Exception as e:
        print(f"绘制损失曲线时出错: {e}")

def plot_exploration_analysis(metrics_df, save_dir):
    """绘制探索与利用分析：entropy_loss和std"""
    if metrics_df is None or metrics_df.empty:
        print("无法绘制探索分析：缺少PPO指标数据或数据为空")
        return
    
    # 检查必要的列是否存在
    required_cols = ['entropy_loss', 'std']
    missing_cols = [col for col in required_cols if col not in metrics_df.columns]
    if missing_cols:
        print(f"无法绘制探索分析：缺少必要的列 {missing_cols}")
        return
    
    try:
        plt.figure(figsize=(12, 6))
        
        # 创建双Y轴
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # 索引或迭代次数
        iterations = range(1, len(metrics_df) + 1)
        
        # 绘制entropy_loss在主Y轴
        line1, = ax1.plot(iterations, metrics_df['entropy_loss'], 'b-', label='Entropy Loss')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Entropy Loss')
        
        # 绘制std在次Y轴
        line2, = ax2.plot(iterations, metrics_df['std'], 'r-', label='Action Standard Deviation')
        ax2.set_ylabel('Standard Deviation')
        
        # 设置图例
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        plt.legend(lines, labels, loc='upper right')
        
        plt.title('Exploration Analysis: Entropy Loss and Action STD')
        plt.grid(True)
        
        # 保存图表
        save_path = os.path.join(save_dir, "exploration_analysis.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"探索分析图已保存至: {save_path}")
    except Exception as e:
        print(f"绘制探索分析图时出错: {e}")

def plot_task_metrics(episodes_df, save_dir):
    """绘制任务指标：success_rate和ep_len_mean"""
    if episodes_df is None or episodes_df.empty:
        print("无法绘制任务指标：缺少回合数据或数据为空")
        return
    
    # 检查必要的列是否存在
    required_cols = ['episode', 'length']
    if 'success' not in episodes_df.columns:
        print("警告：缺少success列，将只绘制回合长度")
        # 可以继续，因为我们可以只绘制回合长度
    
    missing_cols = [col for col in required_cols if col not in episodes_df.columns]
    if missing_cols:
        print(f"无法绘制任务指标：缺少必要的列 {missing_cols}")
        return
    
    try:
        plt.figure(figsize=(12, 6))
        
        # 创建双Y轴
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        lines = []
        labels = []
        
        # 如果有success列，绘制成功率
        if 'success' in episodes_df.columns:
            # 计算滑动平均成功率
            window_size = min(10, len(episodes_df))
            if window_size > 0:
                # 计算移动平均成功率
                success_ma = episodes_df['success'].rolling(window=window_size).mean()
                episodes = episodes_df['episode']
                
                # 绘制移动平均成功率
                line1, = ax1.plot(episodes[window_size-1:], success_ma[window_size-1:], 'g-', 
                                label=f'{window_size}-Episode Moving Average Success Rate')
                ax1.set_ylabel('Success Rate')
                ax1.set_ylim([0, 1])
                
                lines.append(line1)
                labels.append(line1.get_label())
        
        # 绘制回合长度
        line2, = ax2.plot(episodes_df['episode'], episodes_df['length'], 'b-', label='Episode Length')
        ax2.set_ylabel('Episode Length')
        
        lines.append(line2)
        labels.append(line2.get_label())
        
        # 设置图例
        plt.legend(lines, labels, loc='upper left')
        
        plt.xlabel('Episode')
        plt.title('Task Metrics: Success Rate and Episode Length')
        plt.grid(True)
        
        # 保存图表
        save_path = os.path.join(save_dir, "task_metrics.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"任务指标图已保存至: {save_path}")
    except Exception as e:
        print(f"绘制任务指标图时出错: {e}")

def print_hyperparameters(metadata, metrics_df):
    """打印使用的超参数配置"""
    if metadata is None:
        print("未找到训练元数据，无法显示超参数配置")
        return
    
    print("\n--- 超参数配置 ---")
    # 打印基本超参数
    params_to_print = [
        'learning_rate', 'min_learning_rate', 'lr_schedule',
        'n_steps', 'batch_size', 'n_epochs', 'gamma',
        'gae_lambda', 'clip_range', 'ent_coef', 'max_grad_norm'
    ]
    
    for param in params_to_print:
        if param in metadata:
            print(f"{param}: {metadata[param]}")
    
    # 从metrics中检索最终值（如果可用）
    if metrics_df is not None and not metrics_df.empty:
        last_metrics = metrics_df.iloc[-1]
        final_metrics = {
            'final_learning_rate': last_metrics.get('learning_rate', 'N/A'),
            'final_clip_range': last_metrics.get('clip_range', 'N/A'),
            'final_value_loss': last_metrics.get('value_loss', 'N/A'),
            'final_policy_loss': last_metrics.get('policy_gradient_loss', 'N/A'),
            'final_entropy_loss': last_metrics.get('entropy_loss', 'N/A'),
            'final_approx_kl': last_metrics.get('approx_kl', 'N/A')
        }
        
        print("\n--- 训练结束时的指标 ---")
        for metric, value in final_metrics.items():
            print(f"{metric}: {value}")

def create_summary_report(session_dir, episodes_df, metrics_df, metadata):
    """创建训练结果摘要报告，保存为markdown格式"""
    if not os.path.exists(session_dir):
        print(f"错误：会话目录 {session_dir} 不存在")
        return
        
    try:
        # 创建报告文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(session_dir, f"training_summary_{timestamp}.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # 写入标题
            f.write(f"# PPO训练结果摘要报告\n\n")
            f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 写入会话信息
            session_name = os.path.basename(session_dir)
            f.write(f"## 训练会话信息\n\n")
            f.write(f"- 会话目录: `{session_dir}`\n")
            f.write(f"- 会话名称: `{session_name}`\n\n")
            
            # 写入超参数信息
            f.write(f"## 训练超参数\n\n")
            if metadata:
                f.write("| 参数 | 值 |\n")
                f.write("|------|------|\n")
                
                params_to_print = [
                    'learning_rate', 'min_learning_rate', 'lr_schedule',
                    'n_steps', 'batch_size', 'n_epochs', 'gamma',
                    'gae_lambda', 'clip_range', 'ent_coef', 'max_grad_norm',
                    'total_timesteps', 'episode_length'
                ]
                
                for param in params_to_print:
                    if param in metadata:
                        f.write(f"| {param} | {metadata[param]} |\n")
            else:
                f.write("*元数据不可用*\n\n")
            
            # 写入训练结果统计
            f.write(f"\n## 训练结果统计\n\n")
            
            # 回合统计
            f.write(f"### 回合统计\n\n")
            if episodes_df is not None and not episodes_df.empty:
                episode_count = len(episodes_df)
                avg_reward = episodes_df['reward'].mean() if 'reward' in episodes_df.columns else 'N/A'
                max_reward = episodes_df['reward'].max() if 'reward' in episodes_df.columns else 'N/A'
                avg_length = episodes_df['length'].mean() if 'length' in episodes_df.columns else 'N/A'
                success_rate = episodes_df['success'].mean() if 'success' in episodes_df.columns else 'N/A'
                
                f.write("| 指标 | 值 |\n")
                f.write("|------|------|\n")
                f.write(f"| 回合数 | {episode_count} |\n")
                f.write(f"| 平均奖励 | {avg_reward:.2f} |\n") if avg_reward != 'N/A' else f.write(f"| 平均奖励 | {avg_reward} |\n")
                f.write(f"| 最高奖励 | {max_reward:.2f} |\n") if max_reward != 'N/A' else f.write(f"| 最高奖励 | {max_reward} |\n")
                f.write(f"| 平均回合长度 | {avg_length:.2f} |\n") if avg_length != 'N/A' else f.write(f"| 平均回合长度 | {avg_length} |\n")
                f.write(f"| 成功率 | {success_rate:.2%} |\n") if success_rate != 'N/A' else f.write(f"| 成功率 | {success_rate} |\n")
            else:
                f.write("*回合数据不可用*\n\n")
            
            # PPO指标统计
            f.write(f"\n### PPO训练指标\n\n")
            if metrics_df is not None and not metrics_df.empty:
                last_metrics = metrics_df.iloc[-1].to_dict()
                
                f.write("| 指标 | 最终值 |\n")
                f.write("|------|------|\n")
                
                metrics_to_print = [
                    'approx_kl', 'clip_fraction', 'entropy_loss', 
                    'explained_variance', 'learning_rate', 'loss',
                    'policy_gradient_loss', 'value_loss', 'std'
                ]
                
                for metric in metrics_to_print:
                    if metric in last_metrics:
                        value = last_metrics[metric]
                        f.write(f"| {metric} | {value:.6f} |\n")
            else:
                f.write("*PPO指标数据不可用*\n\n")
            
            # 写入性能指标
            f.write(f"\n### 性能指标\n\n")
            if metrics_df is not None and not metrics_df.empty and 'fps' in metrics_df.columns:
                avg_fps = metrics_df['fps'].mean()
                total_time = metrics_df['time_elapsed'].max() if 'time_elapsed' in metrics_df.columns else 'N/A'
                
                f.write("| 指标 | 值 |\n")
                f.write("|------|------|\n")
                f.write(f"| 平均FPS | {avg_fps:.2f} |\n")
                f.write(f"| 总训练时间 | {total_time:.2f}秒 |\n") if total_time != 'N/A' else f.write(f"| 总训练时间 | {total_time} |\n")
            else:
                f.write("*性能指标数据不可用*\n\n")
            
            # 写入图表引用
            f.write(f"\n## 结果图表\n\n")
            
            plots_dir = os.path.join(session_dir, "plots")
            if os.path.exists(plots_dir):
                plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
                
                if plot_files:
                    for plot_file in plot_files:
                        plot_path = os.path.join("plots", plot_file)
                        plot_name = os.path.splitext(plot_file)[0].replace('_', ' ').title()
                        f.write(f"### {plot_name}\n\n")
                        f.write(f"![{plot_name}]({plot_path})\n\n")
                else:
                    f.write("*没有可用的图表*\n\n")
            else:
                f.write("*图表目录不存在*\n\n")
            
            # 写入结论
            f.write(f"\n## 结论\n\n")
            f.write("根据训练结果分析，得出以下结论：\n\n")
            
            # 简单的自动结论生成
            conclusions = []
            
            # 基于回合奖励的结论
            if episodes_df is not None and not episodes_df.empty and 'reward' in episodes_df.columns:
                rewards = episodes_df['reward']
                if len(rewards) > 1:
                    first_half = rewards[:len(rewards)//2].mean()
                    second_half = rewards[len(rewards)//2:].mean()
                    
                    if second_half > first_half * 1.2:
                        conclusions.append("- 模型训练有明显进步，后半段奖励明显高于前半段。")
                    elif second_half > first_half:
                        conclusions.append("- 模型训练有一定进步，后半段奖励略高于前半段。")
                    elif second_half < first_half * 0.8:
                        conclusions.append("- 模型训练出现退化，后半段奖励明显低于前半段，可能出现过拟合或奖励函数设计问题。")
                    else:
                        conclusions.append("- 模型训练稳定，奖励波动不大。")
            
            # 基于KL散度的结论
            if metrics_df is not None and not metrics_df.empty and 'approx_kl' in metrics_df.columns:
                kl_values = metrics_df['approx_kl']
                if any(kl > 0.2 for kl in kl_values):
                    conclusions.append("- 训练中出现较大的策略更新(KL散度>0.2)，可能需要减小学习率或增大clip_range。")
            
            # 基于熵损失的结论
            if metrics_df is not None and not metrics_df.empty and 'entropy_loss' in metrics_df.columns:
                entropy_values = metrics_df['entropy_loss']
                if len(entropy_values) > 1:
                    first = entropy_values.iloc[0]
                    last = entropy_values.iloc[-1]
                    if abs(last) < abs(first) * 0.5:
                        conclusions.append("- 策略熵明显降低，表明探索减少，可能需要增加熵系数以鼓励更多探索。")
            
            # 如果没有结论，添加一个默认的
            if not conclusions:
                conclusions.append("- 由于数据有限，无法得出具体结论。建议进行更长时间的训练。")
                
            for conclusion in conclusions:
                f.write(f"{conclusion}\n")
        
        print(f"摘要报告已保存至: {report_path}")
        return report_path
        
    except Exception as e:
        print(f"创建摘要报告时出错: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='分析PPO训练结果并生成图表')
    parser.add_argument('--session_dir', type=str, 
                        help='训练会话目录路径，包含data、plots等子目录。如果不提供，将自动查找最新的训练会话')
    parser.add_argument('--generate_report', action='store_true', 
                        help='是否生成摘要报告')
    args = parser.parse_args()
    
    # 如果未提供session_dir，自动寻找最新的训练会话目录
    session_dir = args.session_dir
    if not session_dir:
        results_dir = "./results"
        if not os.path.exists(results_dir):
            print(f"错误：默认结果目录 {results_dir} 不存在")
            return
            
        # 获取所有训练会话目录
        session_dirs = [d for d in os.listdir(results_dir) 
                        if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("training_session_")]
        
        if not session_dirs:
            print(f"错误：在 {results_dir} 中未找到任何训练会话目录")
            return
            
        # 按照目录名（包含时间戳）排序，获取最新的
        session_dirs.sort(reverse=True)
        session_dir = os.path.join(results_dir, session_dirs[0])
        print(f"自动选择最新的训练会话: {session_dir}")
    
    if not os.path.exists(session_dir):
        print(f"错误：指定的会话目录 {session_dir} 不存在")
        return
    
    # 确保plots目录存在
    plots_dir = os.path.join(session_dir, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # 加载训练数据
    episodes_df, metrics_df, metadata = load_training_data(session_dir)
    
    # 绘制学习曲线
    plot_learning_curve(episodes_df, plots_dir)
    
    # 绘制损失曲线
    plot_loss_curves(metrics_df, plots_dir)
    
    # 绘制探索分析
    plot_exploration_analysis(metrics_df, plots_dir)
    
    # 绘制任务指标
    plot_task_metrics(episodes_df, plots_dir)
    
    # 打印超参数配置
    print_hyperparameters(metadata, metrics_df)
    
    # 如果指定了生成报告，则创建摘要报告
    if args.generate_report:
        create_summary_report(session_dir, episodes_df, metrics_df, metadata)
    
    print(f"\n分析完成！所有图表已保存到 {plots_dir} 目录")

if __name__ == "__main__":
    # 当作为脚本直接运行时调用main函数
    main()

# 添加一个run_analysis函数作为入口点
def run_analysis(session_dir=None, generate_report=False):
    """
    提供一个简单的函数入口点用于执行分析
    
    Args:
        session_dir: 可选，训练会话目录路径
        generate_report: 是否生成摘要报告
    """
    import sys
    
    # 构建命令行参数
    args = []
    if session_dir:
        args.extend(["--session_dir", session_dir])
    if generate_report:
        args.append("--generate_report")
    
    # 保存原始命令行参数
    old_args = sys.argv
    
    try:
        # 设置新的命令行参数
        sys.argv = [sys.argv[0]] + args
        # 执行主函数
        main()
    finally:
        # 恢复原始命令行参数
        sys.argv = old_args

# 当从其他脚本导入时可以使用这个简化的入口点
if __name__ == "__main__":
    # 当作为脚本直接运行时调用main函数
    main() 