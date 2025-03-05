import os
import sys
import traci
from sumolib import checkBinary
import time
import subprocess
import socket
import xml.etree.ElementTree as ET


def find_free_port():
    """查找可用端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def check_config_file(config_file):
    """检查配置文件的有效性"""
    try:
        # 检查文件是否存在
        if not os.path.exists(config_file):
            print(f"错误: 配置文件不存在: {config_file}")
            return False

        # 解析XML文件
        tree = ET.parse(config_file)
        root = tree.getroot()

        # 检查必要的文件引用
        net_file = root.find(".//net-file")
        route_files = root.find(".//route-files")

        if net_file is None or route_files is None:
            print("错误: 配置文件中缺少必要的文件引用")
            return False

        # 检查引用的文件是否存在
        net_file_path = os.path.join(
            os.path.dirname(config_file), net_file.get("value")
        )
        route_file_path = os.path.join(
            os.path.dirname(config_file), route_files.get("value")
        )

        if not os.path.exists(net_file_path):
            print(f"错误: 网络文件不存在: {net_file_path}")
            return False
        if not os.path.exists(route_file_path):
            print(f"错误: 路由文件不存在: {route_file_path}")
            return False

        print("配置文件检查通过")
        return True

    except ET.ParseError as e:
        print(f"错误: 配置文件格式无效: {str(e)}")
        return False
    except Exception as e:
        print(f"错误: 检查配置文件时出错: {str(e)}")
        return False


def test_traci_connection():
    try:
        # 确保SUMO_HOME环境变量已设置
        if "SUMO_HOME" not in os.environ:
            print("错误: 未设置SUMO_HOME环境变量")
            return False

        print(f"SUMO_HOME: {os.environ['SUMO_HOME']}")

        # 获取当前目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_dir, "cross.sumocfg")

        # 检查配置文件
        if not check_config_file(config_file):
            return False

        # 获取可用端口
        port = find_free_port()
        print(f"使用端口: {port}")

        # 构建SUMO命令
        sumo_cmd = [
            checkBinary("sumo-gui"),
            "-c",
            config_file,
            "--no-step-log",
            "true",
            "--no-warnings",
            "true",
            "--remote-port",
            str(port),
            "--step-length",
            "0.1",
            "--time-to-teleport",
            "-1",
            "--verbose",  # 添加详细输出
            "true",
        ]

        print("\n=== 测试步骤 1: 启动SUMO ===")
        print(f"SUMO命令: {' '.join(sumo_cmd)}")

        # 启动SUMO进程
        sumo_process = subprocess.Popen(
            sumo_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NEW_CONSOLE,  # 在新控制台窗口中运行
        )

        # 等待SUMO启动
        print("等待SUMO启动...")
        time.sleep(3)  # 增加等待时间

        # 检查进程是否还在运行
        if sumo_process.poll() is not None:
            stdout, stderr = sumo_process.communicate()
            print(f"错误: SUMO进程已退出\n标准输出: {stdout}\n错误输出: {stderr}")
            return False

        print("\n=== 测试步骤 2: 尝试TraCI连接 ===")
        try:
            print("正在连接TraCI...")
            traci.start(sumo_cmd)
            print("TraCI连接成功！")

            # 测试基本TraCI功能
            print("\n=== 测试步骤 3: 测试TraCI功能 ===")
            print(f"当前仿真时间: {traci.simulation.getTime()}")
            print(f"当前车辆数量: {traci.vehicle.getIDCount()}")

            # 执行一个仿真步骤
            print("执行仿真步骤...")
            traci.simulationStep()
            print("仿真步骤执行成功")

            return True

        except Exception as e:
            print(f"TraCI连接失败: {str(e)}")
            stdout, stderr = sumo_process.communicate()
            print(f"SUMO输出:\n标准输出: {stdout}\n错误输出: {stderr}")
            return False

        finally:
            # 清理
            if traci.isLoaded():
                traci.close()
            if sumo_process.poll() is None:
                sumo_process.terminate()
                sumo_process.wait()

    except Exception as e:
        print(f"测试过程中出错: {str(e)}")
        return False


if __name__ == "__main__":
    print("开始TraCI连接测试...")
    success = test_traci_connection()
    print(f"\n测试结果: {'成功' if success else '失败'}")
