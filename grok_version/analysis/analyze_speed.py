#!/usr/bin/env python
"""
Analyze average speed data during SUMO training
This script is used to analyze and visualize the average speed at each step during training
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



def load_speed_data(file_path):
    """Load speed data"""
    try:
        # Specify column names
        df = pd.read_csv(file_path, names=['step', 'avg_speed'], header=None)
        print(f"Loaded speed data: {file_path}")
        print(f"Number of data points: {len(df)}")
        return df
    except Exception as e:
        print(f"Error loading speed data: {e}")
        return None

def plot_speed_vs_step(df, save_dir=None):
    """Plot the relationship between average speed and training steps"""
    if df is None or df.empty:
        print("No data to plot")
        return
        
    plt.figure(figsize=(12, 6))
    
    # Plot raw data
    plt.plot(df['step'], df['avg_speed'], color='#1f77b4', alpha=0.3, label='Raw Data')
    
    # Calculate various smoothing results
    window_size = 1000
    df['moving_avg'] = df['avg_speed'].rolling(window=window_size).mean()
    
    # Plot smoothed curve
    plt.plot(df['step'], df['moving_avg'], color='#ff7f0e', linewidth=2, label=f'{window_size}-Step Moving Average')
    
    # Add confidence interval
    std = df['avg_speed'].rolling(window=window_size).std()
    plt.fill_between(df['step'][window_size-1:], 
                     df['moving_avg'][window_size-1:] - std[window_size-1:],
                     df['moving_avg'][window_size-1:] + std[window_size-1:],
                     color='#ff7f0e', alpha=0.2)
    
    # Beautify the chart
    plt.title('Relationship between Training Steps and Average Speed', fontsize=16)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Average Speed (m/s)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tick_params(labelsize=12)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))  # Scientific notation for x-axis
    plt.ylim(bottom=0)  # Set y-axis lower limit to 0
    plt.tight_layout()
    
    # Create plots directory (if it doesn't exist)
    
    plt.show()
    
    
    
    plt.savefig(os.path.join(save_dir), dpi=300, bbox_inches='tight')

def main():
    # Get speed data file
    file_path = "C:/Users/18576/Sumo/grok_version/analysis/speed_data_20250406_151319.csv"
    save_dir = "C:/Users/18576/Sumo/grok_version/analysis"
    
    # Load data
    df = load_speed_data(file_path)

    # Analyze and plot
    plot_speed_vs_step(df, save_dir)

if __name__ == "__main__":
    main() 