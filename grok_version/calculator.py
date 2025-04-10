import pandas as pd
import os

def calculate_average_speed(csv_path):
    """
    读取指定CSV文件的最后10000个数据点并计算均值
    
    参数:
        csv_path: CSV文件的路径
        
    返回:
        平均速度值
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(csv_path):
            print(f"错误: 文件 {csv_path} 不存在")
            return None
            
        # 读取CSV文件
        # 假设CSV格式为两列：第一列是时间步，第二列是速度
        df = pd.read_csv(csv_path, header=None, names=['step', 'speed'])
        
        # 获取最后10000行
        if len(df) > 10000:
            df = df.tail(10000)
            
        # 计算速度均值
        average_speed = df['speed'].mean()
        
        print(f"从文件 {os.path.basename(csv_path)} 中读取了 {len(df)} 个数据点")
        print(f"平均速度: {average_speed:.6f}")
        
        return average_speed
        
    except Exception as e:
        print(f"计算平均速度时出错: {e}")
        return None

if __name__ == "__main__":
    
    csv_path = "grok_version/results/training_session_20250409_010111/data/speed_data_20250409_010111.csv"
        
    calculate_average_speed(csv_path)
