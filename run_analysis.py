#!/usr/bin/env python
"""
运行PPO训练结果分析的简单脚本
"""

import os
import sys
from analyze_results import run_analysis

def main():
    """
    主函数，处理命令行参数并调用分析函数
    """
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='分析PPO训练结果')
    parser.add_argument('--session_dir', type=str, 
                      help='训练会话目录路径，不提供则自动查找最新的')
    parser.add_argument('--generate_report', action='store_true',
                      help='是否生成摘要报告')
    args = parser.parse_args()
    
    # 调用分析函数
    run_analysis(
        session_dir=args.session_dir,
        generate_report=args.generate_report
    )

if __name__ == "__main__":
    main() 