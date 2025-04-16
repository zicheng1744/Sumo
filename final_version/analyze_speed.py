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
from scipy.signal import savgol_filter  # 添加savgol_filter导入

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

def plot_speed_vs_step(df, save_dir=None, use_savgol=False, use_ewma=True):
    """绘制平均车速与步数的关系图"""
    if df is None or df.empty:
        print("无数据可绘制")
        return
        
    plt.figure(figsize=(12, 6))
    
    # 绘制原始数据
    plt.plot(df['step'], df['avg_speed'], color='#1f77b4', alpha=0.3, label='原始数据')
    
    # 计算多种平滑结果
    window_size = max(200, len(df) // 5)
    df['moving_avg'] = df['avg_speed'].rolling(window=window_size).mean()
    
    if use_ewma:
        df['ewma'] = df['avg_speed'].ewm(alpha=0.05).mean()
    
    if use_savgol and len(df) > 501:  # 确保数据量足够应用Savitzky-Golay滤波
        window_length = min(501, len(df) // 2)
        # 确保window_length是奇数
        if window_length % 2 == 0:
            window_length -= 1
        df['savgol'] = savgol_filter(df['avg_speed'], window_length=window_length, polyorder=3)
    
    # 绘制平滑曲线
    plt.plot(df['step'], df['moving_avg'], color='#ff7f0e', linewidth=2, label=f'{window_size}步移动平均')
    
    if use_ewma:
        plt.plot(df['step'], df['ewma'], color='#2ca02c', linewidth=2, linestyle='--', label='指数加权移动平均')
    
    if use_savgol and 'savgol' in df.columns:
        plt.plot(df['step'], df['savgol'], color='#d62728', linewidth=2, linestyle='-.', label='Savitzky-Golay滤波')
    
    # 添加置信区间
    std = df['avg_speed'].rolling(window=window_size).std()
    plt.fill_between(df['step'][window_size-1:], 
                     df['moving_avg'][window_size-1:] - std[window_size-1:],
                     df['moving_avg'][window_size-1:] + std[window_size-1:],
                     color='#ff7f0e', alpha=0.2)
    
    # 图表美化
    plt.title('训练步数与平均车速关系', fontsize=16)
    plt.xlabel('训练步数', fontsize=14)
    plt.ylabel('平均车速 (m/s)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tick_params(labelsize=12)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))  # 科学计数法显示x轴
    plt.ylim(bottom=0)  # 设置y轴下限为0
    plt.tight_layout()
    
    # 创建plots目录（如果不存在）
    if save_dir:
        plots_dir = os.path.join(save_dir, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        save_path = os.path.join(plots_dir, f"speed_vs_step_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(save_path, dpi=300)
        print(f"图表已保存到: {save_path}")
    else:
        plt.show()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='分析SUMO训练中的平均车速数据')
    parser.add_argument('--file', type=str, help='指定速度数据文件路径，不提供则使用最新的')
    parser.add_argument('--show', action='store_true', help='显示图表(默认只保存不显示)')
    parser.add_argument('--savgol', action='store_true', help='使用Savitzky-Golay滤波')
    parser.add_argument('--no-ewma', action='store_true', help='不使用指数加权移动平均')
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
    
    # 分析和绘图，传递滤波器选项
    plot_speed_vs_step(df, save_dir, use_savgol=args.savgol, use_ewma=not args.no_ewma)
    
    print(f"\n分析完成。图表已保存到 {save_dir} 目录")

if __name__ == "__main__":
    main() 