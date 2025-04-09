#!/usr/bin/env python
"""
分析SUMO训练中的平均车速数据
这个脚本用于分析和可视化训练过程中每个step的平均车速
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

def find_latest_speed_data():
    """查找最新的速度数据文件"""
    results_dir = "./results"
    if not os.path.exists(results_dir):
        print(f"结果目录不存在: {results_dir}")
        return None

    # 查找最新的训练会话文件夹
    training_sessions = glob.glob(os.path.join(results_dir, "training_session_*"))
    if not training_sessions:
        print("未找到训练会话文件夹")
        return None

    # 按修改时间排序，获取最新的
    training_sessions.sort(key=os.path.getmtime, reverse=True)
    latest_session = training_sessions[0]

    # 在最新的训练会话文件夹中查找速度数据文件
    speed_dir = os.path.join(latest_session, "data")
    if not os.path.exists(speed_dir):
        print(f"速度数据目录不存在: {speed_dir}")
        return None

    files = glob.glob(os.path.join(speed_dir, "speed_data_*.csv"))
    if not files:
        print("未找到速度数据文件")
        return None

    # 按修改时间排序，获取最新的
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0], latest_session

def load_speed_data(file_path):
    """加载速度数据"""
    try:
        # 指定列名
        df = pd.read_csv(file_path, names=['step', 'avg_speed'], header=None)
        print(f"已加载速度数据: {file_path}")
        print(f"数据点数量: {len(df)}")
        return df
    except Exception as e:
        print(f"加载速度数据时出错: {e}")
        return None

def plot_speed_vs_step(df, save_dir=None):
    """绘制平均车速与步数的关系图"""
    if df is None or df.empty:
        print("无数据可绘制")
        return
        
    plt.figure(figsize=(12, 6))
    plt.plot(df['step'], df['avg_speed'])
    plt.title('Average Speed vs Training Steps')
    plt.xlabel('Step')
    plt.ylabel('Average Speed (m/s)')
    plt.grid(True)
    
    # 添加移动平均线
    window_size = min(100, len(df) // 10) if len(df) > 100 else 10
    if window_size > 1:
        df['moving_avg'] = df['avg_speed'].rolling(window=window_size).mean()
        plt.plot(df['step'][window_size-1:], df['moving_avg'][window_size-1:], 'r--', 
                label=f'{window_size}-Step Moving Average')
        plt.legend()
    
    # 保存图表到最新训练会话的plots文件夹
    if save_dir:
        plots_dir = os.path.join(save_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        save_path = os.path.join(plots_dir, f"speed_vs_step_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(save_path)
        print(f"图表已保存至: {save_path}")
    
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='分析SUMO训练中的平均车速数据')
    parser.add_argument('--file', type=str, help='指定速度数据文件路径，不提供则使用最新的')
    parser.add_argument('--show', action='store_true', help='显示图表(默认只保存不显示)')
    args = parser.parse_args()
    
    # 获取速度数据文件
    file_path = args.file
    if not file_path:
        file_path, save_dir = find_latest_speed_data()
        if not file_path:
            print("无法找到速度数据文件")
            return
    
    # 加载数据
    df = load_speed_data(file_path)
    if df is None:
        return
    
    # 保存图表而不显示（如果没有--show参数）
    if not args.show:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
    
    # 分析和绘图，只绘制平均速度与步数关系图
    plot_speed_vs_step(df, save_dir)
    
    print(f"\n分析完成。图表已保存到 {save_dir} 目录")

if __name__ == "__main__":
    main() 