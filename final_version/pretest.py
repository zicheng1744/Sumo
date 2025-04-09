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

def setup_pretest_logging(log_dir=None):
    """设置预测试专用日志配置
    
    Args:
        log_dir: 日志目录，如果不提供则使用当前目录
    """
    if log_dir is None:
        log_dir = os.getcwd()
    
    # 确保日志目录存在
    logs_dir = os.path.join(log_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # 设置日志文件路径
    log_file = os.path.join(logs_dir, "pretest_log.txt")
    
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
    
    logging.info(f"预测试日志文件: {log_file}")
    return log_file

def pretest(
    gui=True, 
    episode_length=10000, 
    max_speed=15.0, 
    result_dir="./pretest_results"
):
    """执行预测试，不使用任何模型控制，观察基本交通行为
    
    Args:
        gui: 是否启用GUI模式
        episode_length: 测试步数
        max_speed: 最大速度限制
        result_dir: 结果保存目录
    """
    # 创建结果目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(result_dir, f"pretest_session_{timestamp}")
    
    # 创建子目录
    plots_dir = os.path.join(session_dir, "plots")
    data_dir = os.path.join(session_dir, "data")
    
    for dir_path in [session_dir, plots_dir, data_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 设置日志
    log_file = setup_pretest_logging(session_dir)
    
    logging.info(f"开始预测试（无模型控制）")
    logging.info(f"测试参数: GUI={gui}, 步数={episode_length}")
    
    # 保存测试配置
    test_config = {
        "timestamp": timestamp,
        "gui": gui,
        "episode_length": episode_length,
        "max_speed": max_speed
    }
    
    with open(os.path.join(session_dir, "pretest_config.json"), "w", encoding="utf-8") as f:
        json.dump(test_config, f, indent=4)
    
    # 创建环境
    env = SumoMergeEnv(
        gui=gui,
        max_episode_length=episode_length,
        action_scale=0.0,  # 动作缩放设为0，表示不会影响车速
        max_speed=max_speed
    )
    
    try:
        logging.info("开始无干预测试，观察原始交通流行为")
        
        # 准备记录数据的CSV文件
        steps_data_file = os.path.join(data_dir, "step_data.csv")
        with open(steps_data_file, "w", encoding="utf-8") as f:
            f.write("step,reward,avg_speed,cav_count,hdv_count\n")
        
        # 准备绘图数据
        step_numbers = []
        rewards = []
        avg_speeds = []
        cav_counts = []
        hdv_counts = []
        
        # 重置环境
        obs, _ = env.reset()
        cur_step = 0
        terminated = False
        truncated = False
        
        # 执行指定步数的模拟
        while cur_step < episode_length:
            # 空动作（不控制CAV）
            action = np.zeros(env.action_space.shape)
            
            # 执行一步模拟
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 记录数据
            step_numbers.append(cur_step)
            rewards.append(reward)
            
            # 从info中获取指标
            avg_speed = info.get("avg_speed", 0)
            cav_count = info.get("cav_count", 0)
            hdv_count = info.get("hdv_count", 0)
            
            avg_speeds.append(avg_speed)
            cav_counts.append(cav_count)
            hdv_counts.append(hdv_count)
            
            # 将步骤数据写入CSV
            with open(steps_data_file, "a", encoding="utf-8") as f:
                f.write(f"{cur_step},{reward:.6f},{avg_speed:.6f},{cav_count},{hdv_count}\n")
            
            # 每100步输出一次状态
            if cur_step % 100 == 0:
                logging.info(f"步骤: {cur_step}, 车辆: CAV={cav_count}, HDV={hdv_count}, 速度: {avg_speed:.2f}, 奖励: {reward:.2f}")
            
            cur_step += 1
            
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
                
                logging.info(f"注意事件: {reason}, 当前步数: {cur_step}, 但模拟将继续到指定步数")
        
        logging.info(f"预测试完成，达到指定步数: {cur_step}")
        
        # 计算统计数据
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        avg_speed_overall = sum(avg_speeds) / len(avg_speeds) if avg_speeds else 0
        avg_cav_count = sum(cav_counts) / len(cav_counts) if cav_counts else 0
        avg_hdv_count = sum(hdv_counts) / len(hdv_counts) if hdv_counts else 0
        
        logging.info(f"预测试完成: 总步数={cur_step}")
        logging.info(f"平均奖励: {avg_reward:.2f}")
        logging.info(f"平均速度: {avg_speed_overall:.2f}")
        logging.info(f"平均CAV数量: {avg_cav_count:.2f}")
        logging.info(f"平均HDV数量: {avg_hdv_count:.2f}")
        
        # 生成可视化图表
        # 1. 速度趋势图
        plt.figure(figsize=(12, 6))
        plt.plot(step_numbers, avg_speeds, color='blue')
        plt.title('速度趋势图 (无干预)')
        plt.xlabel('步骤')
        plt.ylabel('平均速度 (m/s)')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, "speed_trend.png"))
        plt.close()
        
        # 2. 车辆数量趋势图
        plt.figure(figsize=(12, 6))
        plt.plot(step_numbers, cav_counts, color='red', label='CAV')
        plt.plot(step_numbers, hdv_counts, color='blue', label='HDV')
        plt.title('车辆数量趋势图 (无干预)')
        plt.xlabel('步骤')
        plt.ylabel('车辆数量')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, "vehicle_count_trend.png"))
        plt.close()
        
        # 3. 奖励趋势图
        plt.figure(figsize=(12, 6))
        plt.plot(step_numbers, rewards, color='green')
        plt.title('奖励趋势图 (无干预)')
        plt.xlabel('步骤')
        plt.ylabel('奖励')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, "reward_trend.png"))
        plt.close()
        
        # 4. 统计摘要图
        plt.figure(figsize=(10, 6))
        
        # 创建数据表格
        cell_text = [
            [f"{avg_reward:.2f}"],
            [f"{avg_speed_overall:.2f} m/s"],
            [f"{avg_cav_count:.2f}"],
            [f"{avg_hdv_count:.2f}"],
            [f"{cur_step}"]
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
        
        plt.title('预测试结果摘要 (无干预)', fontsize=16, pad=20)
        
        plt.savefig(os.path.join(plots_dir, "pretest_summary.png"))
        plt.close()
        
        # 创建预测试摘要文本文件
        with open(os.path.join(session_dir, "pretest_summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"预测试会话: {timestamp}\n")
            f.write("="*50 + "\n")
            f.write("预测试参数:\n")
            for key, value in test_config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            f.write("预测试结果:\n")
            f.write(f"  总步数: {cur_step}\n")
            f.write(f"  平均奖励: {avg_reward:.2f}\n")
            f.write(f"  平均速度: {avg_speed_overall:.2f} m/s\n")
            f.write(f"  平均CAV数量: {avg_cav_count:.2f}\n")
            f.write(f"  平均HDV数量: {avg_hdv_count:.2f}\n")
            f.write("\n")
            f.write("文件位置:\n")
            f.write(f"  日志: {log_file}\n")
            f.write(f"  图表: {plots_dir}\n")
            f.write(f"  数据: {data_dir}\n")
        
        logging.info(f"预测试结果已保存到: {session_dir}")
        return avg_reward, avg_speed_overall, session_dir
        
    except Exception as e:
        logging.error(f"预测试过程中出错: {e}")
        traceback.print_exc()
        return None, None, None
    finally:
        # 确保环境关闭
        env.close()

def main():
    """主函数，处理命令行参数并启动预测试"""
    parser = argparse.ArgumentParser(description='SUMO交通流预测试（无模型干预）')
    
    # 测试参数
    parser.add_argument('--gui', action='store_true', help='使用SUMO GUI模式')
    parser.add_argument('--episode_length', type=int, default=3000, help='测试步数')
    parser.add_argument('--max_speed', type=float, default=15.0, help='最大车速')
    parser.add_argument('--result_dir', type=str, default='./pretest_results', help='结果保存目录')
    
    args = parser.parse_args()
    
    # 确保结果目录存在
    os.makedirs(args.result_dir, exist_ok=True)
    
    # 输出测试信息
    print(f"开始预测试（无模型干预）")
    print(f"测试参数: GUI={args.gui}, 步数={args.episode_length}")
    
    # 执行预测试
    pretest(
        gui=args.gui,
        episode_length=args.episode_length,
        max_speed=args.max_speed,
        result_dir=args.result_dir
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