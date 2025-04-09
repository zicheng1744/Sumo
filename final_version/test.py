import os
import sys
import time
import argparse
import logging
import datetime
import numpy as np
import matplotlib.pyplot as plt
import traceback
import json

# 确保SUMO环境变量设置正确
if 'SUMO_HOME' not in os.environ:
    sumo_home = 'D:\\sumo'  # 修改为您的SUMO安装路径
    os.environ['SUMO_HOME'] = sumo_home
    sys.path.append(os.path.join(sumo_home, 'tools'))

try:
    import traci
except ImportError:
    raise ImportError("无法导入traci模块，请确保SUMO已正确安装")

# 导入PPO算法
try:
    from stable_baselines3 import PPO
except ImportError:
    raise ImportError("无法导入stable_baselines3，请安装: pip install stable-baselines3")

# 从train.py导入环境类
try:
    from train import SumoMergeEnv
except ImportError:
    raise ImportError("无法从train.py导入SumoMergeEnv类，请确保train.py在当前目录")

# 设置matplotlib
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，避免GUI相关问题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def setup_test_logging(log_dir=None):
    """设置测试专用日志配置
    
    Args:
        log_dir: 日志目录，如果不提供则使用当前目录
    """
    if log_dir is None:
        log_dir = os.getcwd()
    
    # 确保日志目录存在
    logs_dir = os.path.join(log_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # 设置日志文件路径
    log_file = os.path.join(logs_dir, "test_log.txt")
    
    # 创建根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建格式化器
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    logging.info(f"测试日志文件: {log_file}")
    return log_file

def validate_model_path(model_path):
    """验证模型路径并返回正确的模型文件路径
    
    如果路径是目录，尝试查找.zip文件或检查是否是标准模型目录结构
    
    Args:
        model_path: 输入的模型路径
        
    Returns:
        有效的模型文件路径或None（如果无法确定）
    """
    if not os.path.exists(model_path):
        logging.error(f"模型路径不存在: {model_path}")
        return None
        
    if os.path.isfile(model_path):
        # 如果是文件且以.zip结尾，直接返回
        if model_path.endswith('.zip'):
            return model_path
        else:
            logging.warning(f"模型文件不是.zip格式: {model_path}")
            return model_path  # 仍然返回，让Stable-Baselines3尝试加载
    
    # 如果是目录，检查几种可能的情况
    if os.path.isdir(model_path):
        # 情况1: 目录中存在final_model.zip
        zip_path = os.path.join(model_path, "final_model.zip")
        if os.path.exists(zip_path):
            logging.info(f"在目录中找到模型文件: {zip_path}")
            return zip_path
            
        # 情况2: 目录本身是一个模型目录（如final_model）
        # 检查是否包含policy.pth等文件
        if all(os.path.exists(os.path.join(model_path, f)) for f in ['policy.pth', 'pytorch_variables.pth']):
            # 这是一个模型目录，但我们需要.zip文件
            parent_dir = os.path.dirname(model_path)
            model_name = os.path.basename(model_path)
            zip_path = os.path.join(parent_dir, f"{model_name}.zip")
            
            if os.path.exists(zip_path):
                logging.info(f"找到对应的zip文件: {zip_path}")
                return zip_path
            else:
                logging.error(f"找到模型目录但没有对应的zip文件: {model_path}")
                logging.error(f"请使用训练脚本保存的.zip文件，通常位于: {zip_path}")
                return None
    
    # 如果都不匹配，返回原始路径，让Stable-Baselines3尝试处理
    logging.warning(f"无法确定有效的模型文件路径: {model_path}")
    return model_path

def find_latest_model(base_dir="./results"):
    """查找最新的训练模型
    
    Args:
        base_dir: 结果目录的基路径
        
    Returns:
        最新模型的路径，如果没有找到则返回None
    """
    # 查找所有训练会话目录
    session_dirs = []
    for folder in os.listdir(base_dir):
        if folder.startswith("training_session_"):
            full_path = os.path.join(base_dir, folder)
            if os.path.isdir(full_path):
                session_dirs.append(full_path)
    
    if not session_dirs:
        return None
    
    # 按修改时间排序，获取最新的会话目录
    latest_session = max(session_dirs, key=os.path.getmtime)
    logging.info(f"找到最新的训练会话: {latest_session}")
    
    # 检查模型文件
    models_dir = os.path.join(latest_session, "models")
    if os.path.exists(models_dir):
        # 优先查找.zip文件
        for model_file in os.listdir(models_dir):
            if model_file.endswith(".zip"):
                return os.path.join(models_dir, model_file)
        
        # 如果没找到.zip文件，检查是否有final_model目录对应的zip文件
        fm_dir = os.path.join(models_dir, "final_model")
        if os.path.exists(fm_dir) and os.path.isdir(fm_dir):
            # 查找上层目录中是否有final_model.zip
            zip_path = os.path.join(models_dir, "final_model.zip")
            if os.path.exists(zip_path):
                return zip_path
            else:
                logging.warning(f"找到模型目录但没有对应的zip文件: {fm_dir}")
                logging.warning(f"请检查训练输出，确保生成了.zip格式的模型文件")
    
    logging.error(f"在最新会话 {latest_session} 中未找到有效的模型文件")
    return None

def test(
    model_path, 
    gui=True, 
    episode_length=10000, 
    action_scale=10.0, 
    max_speed=15.0, 
    test_episodes=1,
    result_dir="./test_results",
    pretest_session_dir=None  # 新增：预测试会话目录，用于结果比较
):
    """测试保存的模型
    
    Args:
        model_path: 模型文件路径
        gui: 是否启用GUI模式
        episode_length: 每个回合的最大步数
        action_scale: 动作缩放因子
        max_speed: 最大速度限制
        test_episodes: 测试回合数
        result_dir: 结果保存目录
        pretest_session_dir: 预测试会话目录，用于结果比较
    """
    # 创建结果目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(result_dir, f"test_session_{timestamp}")
    
    # 创建子目录
    plots_dir = os.path.join(session_dir, "plots")
    data_dir = os.path.join(session_dir, "data")
    comparison_dir = os.path.join(session_dir, "comparison")  # 新增：比较结果目录
    
    for dir_path in [session_dir, plots_dir, data_dir, comparison_dir]:  # 添加comparison_dir
        os.makedirs(dir_path, exist_ok=True)
    
    # 设置日志
    log_file = setup_test_logging(session_dir)
    
    logging.info(f"开始测试模型: {model_path}")
    logging.info(f"测试参数: GUI={gui}, 回合={test_episodes}, 回合长度={episode_length}")
    
    # 保存测试配置
    test_config = {
        "timestamp": timestamp,
        "model_path": model_path,
        "gui": gui,
        "episode_length": episode_length,
        "action_scale": action_scale,
        "max_speed": max_speed,
        "test_episodes": test_episodes,
        "pretest_session_dir": pretest_session_dir  # 添加预测试会话目录
    }
    
    with open(os.path.join(session_dir, "test_config.json"), "w", encoding="utf-8") as f:
        json.dump(test_config, f, indent=4)
    
    # 创建环境
    env = SumoMergeEnv(
        gui=gui,
        max_episode_length=episode_length,
        action_scale=action_scale,
        max_speed=max_speed
    )
    
    try:
        # 加载模型
        try:
            model = PPO.load(model_path)
            logging.info(f"模型已加载: {model_path}")
        except Exception as e:
            logging.error(f"加载模型失败: {e}")
            # 如果加载失败，尝试验证路径并重新加载
            valid_path = validate_model_path(model_path)
            if valid_path and valid_path != model_path:
                logging.info(f"尝试使用修正后的路径: {valid_path}")
                model = PPO.load(valid_path)
                logging.info(f"使用修正后的路径成功加载模型: {valid_path}")
                model_path = valid_path  # 更新模型路径以便测试报告中显示
            else:
                raise  # 如果没有修正路径或修正后仍失败，抛出原异常
        
        # 准备记录数据的CSV文件
        episode_stats_file = os.path.join(data_dir, "episode_stats.csv")
        with open(episode_stats_file, "w", encoding="utf-8") as f:
            f.write("episode,reward,steps,avg_speed,max_speed,min_speed,cav_count\n")
        
        # 准备绘图数据
        episode_numbers = []
        rewards = []
        avg_speeds = []
        step_rewards = []  # 记录每步的奖励
        step_speeds = []   # 记录每步的速度
        step_numbers = []  # 记录步骤编号
        cav_counts = []    # 记录CAV数量
        hdv_counts = []    # 记录HDV数量
        
        # 运行测试回合
        total_reward = 0
        
        for ep in range(test_episodes):
            logging.info(f"开始测试回合 {ep+1}/{test_episodes}")
            obs, _ = env.reset()
            ep_reward = 0
            ep_steps = 0
            terminated = False
            truncated = False
            
            # 收集回合统计数据
            episode_speeds = []
            episode_rewards = []
            episode_cav_counts = []
            episode_hdv_counts = []
            
            # 记录详细的每步数据
            step_data_file = os.path.join(data_dir, f"episode_{ep+1}_steps.csv")
            with open(step_data_file, "w", encoding="utf-8") as f:
                f.write("step,reward,avg_speed,cav_count,hdv_count\n")
            
            while ep_steps < episode_length:
                # 获取模型预测的动作
                action, _ = model.predict(obs, deterministic=True)
                
                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                
                # 记录统计信息
                ep_reward += reward
                ep_steps += 1
                
                # 记录步骤编号
                step_numbers.append(ep_steps)
                
                # 记录当前步的速度和奖励
                if "avg_speed" in info:
                    avg_speed = info["avg_speed"]
                    episode_speeds.append(avg_speed)
                    step_speeds.append(avg_speed)
                else:
                    avg_speed = 0
                
                # 记录CAV和HDV数量
                cav_count = info.get("cav_count", 0)
                hdv_count = info.get("hdv_count", 0)
                episode_cav_counts.append(cav_count)
                episode_hdv_counts.append(hdv_count)
                cav_counts.append(cav_count)
                hdv_counts.append(hdv_count)
                
                episode_rewards.append(reward)
                step_rewards.append(reward)
                
                # 记录详细的状态信息（每100步一次）
                if ep_steps % 100 == 0 or terminated or truncated:
                    logging.info(f"  步骤: {ep_steps}, CAVs: {cav_count}, HDVs: {hdv_count}, 平均速度: {avg_speed:.2f}, 奖励: {reward:.2f}")
                
                # 将每步数据写入CSV
                with open(step_data_file, "a", encoding="utf-8") as f:
                    f.write(f"{ep_steps},{reward:.6f},{avg_speed:.6f},{cav_count},{hdv_count}\n")
                
                # 记录终止状态，但不提前结束模拟
                if terminated or truncated:
                    reason = "模拟继续中"
                    if info.get("no_vehicles", False):
                        reason = "无车辆 (继续模拟)"
                    elif info.get("early_termination", False):
                        if info.get("congestion_count", 0) > 100:
                            reason = "交通拥堵 (继续模拟)"
                        else:
                            reason = "持续低奖励 (继续模拟)"
                    
                    logging.info(f"注意事件: {reason}, 当前步数: {ep_steps}, 但模拟将继续到指定步数")
            
            logging.info(f"测试回合 {ep+1} 完成，达到指定步数: {ep_steps}")
            
            # 计算回合统计数据
            avg_speed = sum(episode_speeds) / len(episode_speeds) if episode_speeds else 0
            max_speed_val = max(episode_speeds) if episode_speeds else 0
            min_speed_val = min(episode_speeds) if episode_speeds else 0
            avg_cav_count = sum(episode_cav_counts) / len(episode_cav_counts) if episode_cav_counts else 0
            avg_hdv_count = sum(episode_hdv_counts) / len(episode_hdv_counts) if episode_hdv_counts else 0
            
            # 记录回合结果
            logging.info(f"回合 {ep+1} 完成: 步数={ep_steps}, 奖励={ep_reward:.2f}, 平均速度={avg_speed:.2f}")
            logging.info(f"平均车辆数: CAV={avg_cav_count:.2f}, HDV={avg_hdv_count:.2f}")
            episode_numbers.append(ep + 1)
            rewards.append(ep_reward)
            avg_speeds.append(avg_speed)
            total_reward += ep_reward
            
            # 将回合统计数据写入CSV
            with open(episode_stats_file, "a", encoding="utf-8") as f:
                f.write(f"{ep+1},{ep_reward:.6f},{ep_steps},{avg_speed:.6f},{max_speed_val:.6f},{min_speed_val:.6f},{avg_cav_count:.2f}\n")
        
        # 计算并显示测试结果
        avg_reward = total_reward / test_episodes
        avg_speed_overall = sum(avg_speeds) / len(avg_speeds) if avg_speeds else 0
        avg_cav_count_overall = sum(cav_counts) / len(cav_counts) if cav_counts else 0
        avg_hdv_count_overall = sum(hdv_counts) / len(hdv_counts) if hdv_counts else 0
        
        logging.info(f"测试完成: {test_episodes}个回合")
        logging.info(f"平均奖励: {avg_reward:.2f}")
        logging.info(f"平均速度: {avg_speed_overall:.2f}")
        logging.info(f"平均CAV数量: {avg_cav_count_overall:.2f}")
        logging.info(f"平均HDV数量: {avg_hdv_count_overall:.2f}")
        
        # 生成更多可视化
        # 1. 回合奖励和速度
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.bar(episode_numbers, rewards, color='blue')
        plt.title('测试回合奖励')
        plt.xlabel('回合')
        plt.ylabel('奖励')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.bar(episode_numbers, avg_speeds, color='green')
        plt.title('测试回合平均速度')
        plt.xlabel('回合')
        plt.ylabel('速度 (m/s)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "episode_stats.png"))
        plt.close()
        
        # 2. 每步奖励和速度趋势图
        if step_rewards and step_speeds:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            plt.plot(step_rewards, color='blue')
            plt.title('每步奖励趋势')
            plt.xlabel('步骤')
            plt.ylabel('奖励')
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            plt.plot(step_speeds, color='green')
            plt.title('每步平均速度趋势')
            plt.xlabel('步骤')
            plt.ylabel('速度 (m/s)')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "step_trends.png"))
            plt.close()
            
        # 3. 车辆数量趋势图
        plt.figure(figsize=(12, 6))
        plt.plot(cav_counts, color='red', label='CAV')
        plt.plot(hdv_counts, color='blue', label='HDV')
        plt.title('车辆数量趋势图 (有模型控制)')
        plt.xlabel('步骤')
        plt.ylabel('车辆数量')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, "vehicle_count_trend.png"))
        plt.close()
            
        # 4. 测试摘要图
        plt.figure(figsize=(10, 6))
        
        # 创建数据表格
        cell_text = [
            [f"{avg_reward:.2f}"],
            [f"{avg_speed_overall:.2f} m/s"],
            [f"{avg_cav_count_overall:.2f}"],
            [f"{avg_hdv_count_overall:.2f}"],
            [f"{sum([len(r) for r in [episode_rewards]])}"]
        ]
        
        row_labels = ['平均奖励', '平均速度', '平均CAV数量', '平均HDV数量', '总步数']
        col_labels = ['值']
        
        plt.axis('off')
        table = plt.table(cellText=cell_text,
                          rowLabels=row_labels,
                          colLabels=col_labels,
                          loc='center',
                          cellLoc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        plt.title('测试结果摘要 (有模型控制)', fontsize=16, pad=20)
        
        plt.savefig(os.path.join(plots_dir, "test_summary.png"))
        plt.close()
        
        # 如果有预测试目录，创建比较图表
        if pretest_session_dir and os.path.exists(pretest_session_dir):
            try:
                logging.info(f"创建与预测试结果的比较: {pretest_session_dir}")
                
                # 加载预测试数据
                pretest_data_file = os.path.join(pretest_session_dir, "data", "step_data.csv")
                if os.path.exists(pretest_data_file):
                    pretest_data = {}
                    with open(pretest_data_file, 'r', encoding='utf-8') as f:
                        import csv
                        reader = csv.DictReader(f)
                        pretest_data['steps'] = []
                        pretest_data['rewards'] = []
                        pretest_data['avg_speeds'] = []
                        pretest_data['cav_counts'] = []
                        pretest_data['hdv_counts'] = []
                        
                        for row in reader:
                            pretest_data['steps'].append(int(row['step']))
                            pretest_data['rewards'].append(float(row['reward']))
                            pretest_data['avg_speeds'].append(float(row['avg_speed']))
                            pretest_data['cav_counts'].append(int(row['cav_count']))
                            pretest_data['hdv_counts'].append(int(row['hdv_count']))
                    
                    # 截取与预测试相同长度的数据进行比较
                    max_len = min(len(pretest_data['steps']), len(step_speeds))
                    
                    # 比较速度趋势
                    plt.figure(figsize=(12, 6))
                    plt.plot(pretest_data['steps'][:max_len], pretest_data['avg_speeds'][:max_len], 
                            color='blue', linestyle='--', label='无控制')
                    plt.plot(step_numbers[:max_len], step_speeds[:max_len], 
                            color='red', linestyle='-', label='模型控制')
                    plt.title('速度比较: 模型控制 vs 无控制')
                    plt.xlabel('步骤')
                    plt.ylabel('平均速度 (m/s)')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(os.path.join(comparison_dir, "speed_comparison.png"))
                    plt.close()
                    
                    # 比较奖励趋势
                    plt.figure(figsize=(12, 6))
                    plt.plot(pretest_data['steps'][:max_len], pretest_data['rewards'][:max_len], 
                            color='blue', linestyle='--', label='无控制')
                    plt.plot(step_numbers[:max_len], step_rewards[:max_len], 
                            color='red', linestyle='-', label='模型控制')
                    plt.title('奖励比较: 模型控制 vs 无控制')
                    plt.xlabel('步骤')
                    plt.ylabel('奖励')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(os.path.join(comparison_dir, "reward_comparison.png"))
                    plt.close()
                    
                    # 比较车辆数量趋势
                    plt.figure(figsize=(12, 6))
                    plt.plot(pretest_data['steps'][:max_len], pretest_data['cav_counts'][:max_len], 
                            color='blue', linestyle='--', label='CAV (无控制)')
                    plt.plot(pretest_data['steps'][:max_len], pretest_data['hdv_counts'][:max_len], 
                            color='green', linestyle='--', label='HDV (无控制)')
                    plt.plot(step_numbers[:max_len], cav_counts[:max_len], 
                            color='red', linestyle='-', label='CAV (模型控制)')
                    plt.plot(step_numbers[:max_len], hdv_counts[:max_len], 
                            color='orange', linestyle='-', label='HDV (模型控制)')
                    plt.title('车辆数量比较: 模型控制 vs 无控制')
                    plt.xlabel('步骤')
                    plt.ylabel('车辆数量')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(os.path.join(comparison_dir, "vehicle_count_comparison.png"))
                    plt.close()
                    
                    # 创建摘要比较图
                    plt.figure(figsize=(10, 8))
                    
                    # 计算预测试平均值
                    pretest_avg_reward = sum(pretest_data['rewards']) / len(pretest_data['rewards'])
                    pretest_avg_speed = sum(pretest_data['avg_speeds']) / len(pretest_data['avg_speeds'])
                    pretest_avg_cav = sum(pretest_data['cav_counts']) / len(pretest_data['cav_counts'])
                    pretest_avg_hdv = sum(pretest_data['hdv_counts']) / len(pretest_data['hdv_counts'])
                    
                    # 创建比较表格
                    cell_text = [
                        [f"{pretest_avg_reward:.2f}", f"{avg_reward:.2f}", f"{(avg_reward-pretest_avg_reward):.2f}"],
                        [f"{pretest_avg_speed:.2f} m/s", f"{avg_speed_overall:.2f} m/s", f"{(avg_speed_overall-pretest_avg_speed):.2f} m/s"],
                        [f"{pretest_avg_cav:.2f}", f"{avg_cav_count_overall:.2f}", f"{(avg_cav_count_overall-pretest_avg_cav):.2f}"],
                        [f"{pretest_avg_hdv:.2f}", f"{avg_hdv_count_overall:.2f}", f"{(avg_hdv_count_overall-pretest_avg_hdv):.2f}"]
                    ]
                    
                    row_labels = ['平均奖励', '平均速度', '平均CAV数量', '平均HDV数量']
                    col_labels = ['无控制', '模型控制', '差异']
                    
                    plt.axis('off')
                    table = plt.table(cellText=cell_text,
                                    rowLabels=row_labels,
                                    colLabels=col_labels,
                                    loc='center',
                                    cellLoc='center')
                    
                    table.auto_set_font_size(False)
                    table.set_fontsize(12)
                    table.scale(1, 2)
                    
                    plt.title('测试与预测试比较摘要', fontsize=16, pad=20)
                    
                    plt.savefig(os.path.join(comparison_dir, "comparison_summary.png"))
                    plt.close()
                    
                    # 创建比较摘要文本文件
                    with open(os.path.join(comparison_dir, "comparison_summary.txt"), "w", encoding="utf-8") as f:
                        f.write(f"测试与预测试比较\n")
                        f.write("="*50 + "\n\n")
                        f.write("指标比较:\n")
                        f.write(f"{'指标':<15}{'无控制':<15}{'模型控制':<15}{'差异':<15}{'改进百分比':<15}\n")
                        f.write("-"*75 + "\n")
                        
                        # 奖励比较
                        improvement = (avg_reward-pretest_avg_reward)/abs(pretest_avg_reward)*100 if pretest_avg_reward != 0 else float('inf')
                        f.write(f"{'平均奖励':<15}{pretest_avg_reward:<15.2f}{avg_reward:<15.2f}{(avg_reward-pretest_avg_reward):<15.2f}{improvement:<15.1f}%\n")
                        
                        # 速度比较
                        improvement = (avg_speed_overall-pretest_avg_speed)/pretest_avg_speed*100 if pretest_avg_speed != 0 else float('inf')
                        f.write(f"{'平均速度':<15}{pretest_avg_speed:<15.2f}{avg_speed_overall:<15.2f}{(avg_speed_overall-pretest_avg_speed):<15.2f}{improvement:<15.1f}%\n")
                        
                        # CAV数量比较
                        improvement = (avg_cav_count_overall-pretest_avg_cav)/pretest_avg_cav*100 if pretest_avg_cav != 0 else float('inf')
                        f.write(f"{'平均CAV数量':<15}{pretest_avg_cav:<15.2f}{avg_cav_count_overall:<15.2f}{(avg_cav_count_overall-pretest_avg_cav):<15.2f}{improvement:<15.1f}%\n")
                        
                        # HDV数量比较
                        improvement = (avg_hdv_count_overall-pretest_avg_hdv)/pretest_avg_hdv*100 if pretest_avg_hdv != 0 else float('inf')
                        f.write(f"{'平均HDV数量':<15}{pretest_avg_hdv:<15.2f}{avg_hdv_count_overall:<15.2f}{(avg_hdv_count_overall-pretest_avg_hdv):<15.2f}{improvement:<15.1f}%\n")
                        
                        f.write("\n")
                        f.write("结论:\n")
                        speed_change = avg_speed_overall - pretest_avg_speed
                        if speed_change > 0:
                            f.write(f"模型控制提高了交通流的平均速度，增加了 {speed_change:.2f} m/s ({(speed_change/pretest_avg_speed*100):.1f}%)。\n")
                        else:
                            f.write(f"模型控制降低了交通流的平均速度，减少了 {abs(speed_change):.2f} m/s ({(abs(speed_change)/pretest_avg_speed*100):.1f}%)。\n")
                            
                        f.write("\n")
                        f.write("文件位置:\n")
                        f.write(f"预测试结果: {pretest_session_dir}\n")
                        f.write(f"测试结果: {session_dir}\n")
                        f.write(f"比较图表: {comparison_dir}\n")
                
                else:
                    logging.warning(f"未找到预测试数据文件: {pretest_data_file}")
            except Exception as e:
                logging.error(f"创建比较图表时出错: {e}")
                traceback.print_exc()
        
        # 创建测试摘要文本文件
        with open(os.path.join(session_dir, "test_summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"测试会话: {timestamp}\n")
            f.write("="*50 + "\n")
            f.write("测试参数:\n")
            for key, value in test_config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            f.write("测试结果:\n")
            f.write(f"  回合数: {test_episodes}\n")
            f.write(f"  平均奖励: {avg_reward:.2f}\n")
            f.write(f"  平均速度: {avg_speed_overall:.2f} m/s\n")
            f.write(f"  平均CAV数量: {avg_cav_count_overall:.2f}\n")
            f.write(f"  平均HDV数量: {avg_hdv_count_overall:.2f}\n")
            f.write(f"  总步数: {sum([len(r) for r in [episode_rewards]])}\n")
            f.write("\n")
            f.write("回合详情:\n")
            for ep in range(test_episodes):
                f.write(f"  回合 {ep+1}: 奖励={rewards[ep]:.2f}, 平均速度={avg_speeds[ep]:.2f} m/s\n")
            f.write("\n")
            f.write("文件位置:\n")
            f.write(f"  日志: {log_file}\n")
            f.write(f"  图表: {plots_dir}\n")
            f.write(f"  数据: {data_dir}\n")
            if pretest_session_dir:
                f.write(f"  比较: {comparison_dir}\n")
        
        logging.info(f"测试结果已保存到: {session_dir}")
        return avg_reward, avg_speed_overall, session_dir
        
    except Exception as e:
        logging.error(f"测试过程中出错: {e}")
        traceback.print_exc()
        return None, None, None
    finally:
        # 确保环境关闭
        env.close()

def main():
    """主函数，处理命令行参数并启动测试"""
    parser = argparse.ArgumentParser(description='测试SUMO-RL连接模型')
    
    # 测试参数
    parser.add_argument('--model_path', type=str, help='模型文件路径，如果不指定则使用最新训练的模型')
    parser.add_argument('--gui', action='store_true', help='使用SUMO GUI模式')
    parser.add_argument('--test_episodes', type=int, default=1, help='测试回合数')
    parser.add_argument('--episode_length', type=int, default=3000, help='每回合最大步数')
    parser.add_argument('--action_scale', type=float, default=10.0, help='动作缩放因子')
    parser.add_argument('--max_speed', type=float, default=15.0, help='最大车速')
    parser.add_argument('--result_dir', type=str, default='./test_results', help='结果保存目录')
    parser.add_argument('--pretest_dir', type=str, help='预测试会话目录，用于结果比较')
    
    args = parser.parse_args()
    
    # 确保结果目录存在
    os.makedirs(args.result_dir, exist_ok=True)
    
    # 如果未指定模型路径，则查找最新训练的模型
    model_path = args.model_path
    if not model_path:
        model_path = find_latest_model()
        if not model_path:
            logging.error("未找到训练模型，请使用--model_path指定模型路径")
            return
    else:
        # 验证指定的模型路径
        valid_path = validate_model_path(model_path)
        if valid_path:
            model_path = valid_path
        else:
            logging.error("无效的模型路径，请确保提供正确的模型文件路径(.zip文件)")
            return
    
    # 输出测试信息
    print(f"开始测试模型: {model_path}")
    print(f"测试参数: GUI={args.gui}, 回合数={args.test_episodes}")
    
    # 执行测试
    test(
        model_path=model_path,
        gui=args.gui,
        episode_length=args.episode_length,
        action_scale=args.action_scale,
        max_speed=args.max_speed,
        test_episodes=args.test_episodes,
        result_dir=args.result_dir,
        pretest_session_dir=args.pretest_dir
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序执行出错: {e}")
        traceback.print_exc()
        # 确保在发生错误时清理资源
        try:
            if traci.isLoaded():
                traci.close(False)
        except:
            pass 