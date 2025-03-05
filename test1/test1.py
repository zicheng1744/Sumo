import os
import sys
import traci
from sumolib import checkBinary
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import time
import socket
import subprocess
import signal

# 确保SUMO_HOME环境变量已设置
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("请设置SUMO_HOME环境变量")


def find_free_port():
    """查找可用端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def kill_process_on_port(port):
    """终止指定端口上的进程"""
    try:
        if os.name == "nt":  # Windows
            cmd = f"netstat -ano | findstr :{port}"
            result = subprocess.check_output(cmd, shell=True).decode()
            if result:
                pid = result.split()[-1]
                subprocess.run(f"taskkill /F /PID {pid}", shell=True)
        else:  # Linux/Mac
            cmd = f"lsof -ti:{port}"
            pid = subprocess.check_output(cmd, shell=True).decode().strip()
            if pid:
                os.kill(int(pid), signal.SIGTERM)
    except Exception as e:
        print(f"终止端口 {port} 上的进程时出错: {str(e)}")


class SumoEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # 定义动作空间（5个离散动作）
        self.action_space = spaces.Discrete(
            5
        )  # 0: 减速, 1: 保持, 2: 加速, 3: 左转, 4: 右转

        # 定义观察空间（扩展状态空间）
        self.observation_space = spaces.Box(
            low=np.array(
                [0, 0, 0, 0, 0, 0, 0], dtype=np.float32
            ),  # 速度, x位置, y位置, 到下一个红绿灯距离, 到下一个红绿灯时间, 前方车辆距离, 前方车辆速度
            high=np.array([30, 1000, 1000, 1000, 100, 100, 30], dtype=np.float32),
            dtype=np.float32,
        )

        # 获取可用端口
        self.port = find_free_port()
        print(f"使用端口: {self.port}")

        # 确保配置文件存在
        self.config_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "cross.sumocfg")
        )
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"找不到配置文件: {self.config_file}")

        # 检查网络文件是否存在
        self.net_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "cross.net.xml")
        )
        if not os.path.exists(self.net_file):
            raise FileNotFoundError(f"找不到网络文件: {self.net_file}")

        # SUMO配置
        self.sumo_cmd = [
            checkBinary("sumo-gui"),  # 使用sumo-gui而不是sumo
            "-c",
            self.config_file,
            "--no-step-log",
            "true",
            "--no-warnings",
            "true",
            "--remote-port",
            str(self.port),
            "--waiting-time-memory",
            "1000",
            "--device.rerouting.probability",
            "0",
            "--collision.action",
            "warn",
            "--collision.mingap-factor",
            "0.5",
            "--collision.stoptime",
            "1",
            "--ignore-route-errors",
            "true",
            "--step-length",
            "0.1",
            "--time-to-teleport",
            "-1",
            "--mesosim",
            "false",  # 使用正确的布尔值语法
        ]

        self.simulation_steps = 0
        self.max_steps = 1000
        self.target_vehicle = "veh0"  # 目标控制车辆
        self.sumo_process = None

    def reset(self, seed=None, options=None):
        try:
            # 确保之前的连接已关闭
            if traci.isLoaded():
                traci.close()
                time.sleep(1)

            # 终止可能占用端口的进程
            kill_process_on_port(self.port)
            time.sleep(1)

            print("正在启动SUMO...")
            print(f"配置文件路径: {self.config_file}")
            print(f"SUMO命令: {' '.join(self.sumo_cmd)}")

            # 启动SUMO进程
            try:
                self.sumo_process = subprocess.Popen(
                    self.sumo_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                    text=True,  # 使用文本模式
                )
            except Exception as e:
                raise RuntimeError(f"启动SUMO进程失败: {str(e)}")

            # 等待SUMO启动并检查输出
            time.sleep(2)

            # 检查进程是否还在运行
            if self.sumo_process.poll() is not None:
                stdout, stderr = self.sumo_process.communicate()
                error_msg = f"SUMO进程已退出\n标准输出: {stdout}\n错误输出: {stderr}"
                raise RuntimeError(error_msg)

            # 尝试连接TraCI
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    print(f"尝试连接TraCI (尝试 {attempt + 1}/{max_retries})...")
                    traci.start(self.sumo_cmd)
                    print("SUMO启动成功")
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        stdout, stderr = self.sumo_process.communicate()
                        error_msg = (
                            f"TraCI连接失败\n"
                            f"错误信息: {str(e)}\n"
                            f"SUMO标准输出: {stdout}\n"
                            f"SUMO错误输出: {stderr}"
                        )
                        raise RuntimeError(error_msg)
                    print(f"TraCI连接尝试 {attempt + 1} 失败: {str(e)}")
                    time.sleep(2)

            # 等待SUMO完全启动
            time.sleep(2)

            self.simulation_steps = 0
            return self._get_observation(), {}
        except Exception as e:
            print(f"重置环境时出错: {str(e)}")
            if self.sumo_process:
                self.sumo_process.terminate()
                self.sumo_process.wait()
            raise

    def step(self, action):
        # 执行动作
        if traci.vehicle.getIDCount() > 0:
            if action == 0:  # 减速
                traci.vehicle.slowDown(self.target_vehicle, 5, 1)
            elif action == 1:  # 保持
                current_speed = traci.vehicle.getSpeed(self.target_vehicle)
                traci.vehicle.setSpeed(self.target_vehicle, current_speed)
            elif action == 2:  # 加速
                traci.vehicle.slowDown(self.target_vehicle, 15, 1)
            elif action == 3:  # 左转
                traci.vehicle.changeLane(self.target_vehicle, 0, duration=1)
            elif action == 4:  # 右转
                traci.vehicle.changeLane(self.target_vehicle, 1, duration=1)

        traci.simulationStep()
        self.simulation_steps += 1

        obs = self._get_observation()
        reward = self._calculate_reward()
        terminated = self._is_terminated()
        truncated = self.simulation_steps >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def _get_observation(self):
        if traci.vehicle.getIDCount() > 0:
            # 获取目标车辆信息
            speed = traci.vehicle.getSpeed(self.target_vehicle)
            pos = traci.vehicle.getPosition(self.target_vehicle)

            # 获取下一个红绿灯信息
            next_tls = traci.vehicle.getNextTLS(self.target_vehicle)
            if next_tls:
                tls_distance = next_tls[0][2]
                tls_state = next_tls[0][3]
            else:
                tls_distance = 1000
                tls_state = "G"

            # 获取前方车辆信息
            leader = traci.vehicle.getLeader(self.target_vehicle)
            if leader:
                leader_distance = leader[1]
                leader_speed = traci.vehicle.getSpeed(leader[0])
            else:
                leader_distance = 100
                leader_speed = 0

            return np.array(
                [
                    speed,
                    pos[0],
                    pos[1],
                    tls_distance,
                    1 if tls_state == "G" else 0,
                    leader_distance,
                    leader_speed,
                ],
                dtype=np.float32,
            )
        else:
            return np.zeros(7, dtype=np.float32)

    def _calculate_reward(self):
        if traci.vehicle.getIDCount() == 0:
            return -100  # 车辆消失的惩罚

        # 基础奖励：速度奖励
        speed = traci.vehicle.getSpeed(self.target_vehicle)
        reward = speed / 10  # 速度奖励

        # 碰撞惩罚
        if traci.simulation.getCollidingVehiclesNumber() > 0:
            reward -= 50

        # 等待时间惩罚
        waiting_time = traci.vehicle.getWaitingTime(self.target_vehicle)
        reward -= waiting_time / 10

        # 到达目标奖励
        if self._is_terminated() and traci.vehicle.getIDCount() > 0:
            reward += 100

        return reward

    def _is_terminated(self):
        # 检查是否到达终点或发生碰撞
        if traci.vehicle.getIDCount() == 0:
            return True

        # 检查是否到达终点（这里需要根据您的具体场景定义）
        # 示例：检查是否到达特定位置
        if traci.vehicle.getIDCount() > 0:
            pos = traci.vehicle.getPosition(self.target_vehicle)
            # 修改终点判断条件，根据十字路口场景调整
            if pos[0] > 400 or pos[1] > 400:  # 假设十字路口中心在(400,400)附近
                return True

        return False

    def close(self):
        """关闭环境"""
        try:
            if traci.isLoaded():
                traci.close()
            if self.sumo_process:
                self.sumo_process.terminate()
                self.sumo_process.wait()
        except Exception as e:
            print(f"关闭环境时出错: {str(e)}")


# 主程序
if __name__ == "__main__":
    try:
        print("正在创建环境...")
        env = SumoEnv()

        print("正在检查环境...")
        check_env(env)

        print("正在创建PPO模型...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1,
        )

        print("开始训练...")
        model.learn(total_timesteps=1000)

        print("保存模型...")
        model.save("ppo_sumo_model")

        print("开始测试...")
        obs, _ = env.reset()
        for _ in range(100):
            action, _states = model.predict(obs)
            obs, rewards, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()

        env.close()
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        if "env" in locals():
            env.close()
