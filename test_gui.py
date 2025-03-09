import os
import sys
import time
import traci
import logging
import numpy as np

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from target_model.envir import create_sumo_env

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("test_gui.log")],
)


def test_gui():
    """测试SUMO-GUI能否正常运行"""
    logging.info("开始测试SUMO-GUI...")
    env = None

    try:
        # 创建带GUI的环境
        logging.info("正在创建带GUI的环境...")
        env = create_sumo_env(gui=True)
        logging.info("成功创建带GUI的环境")

        # 重置环境前增加等待时间，确保之前的进程已完全终止
        time.sleep(2)

        # 重置环境
        logging.info("重置环境...")
        try:
            obs = env.reset()
            logging.info(f"初始观测: {obs}")
        except Exception as e:
            logging.error(f"重置环境失败: {e}")
            import traceback

            logging.error(traceback.format_exc())
            # 尝试重新创建环境
            env.close()
            time.sleep(3)  # 等待更长时间
            env = create_sumo_env(gui=True)
            obs = env.reset()
            logging.info("第二次尝试重置环境成功")

        # 等待GUI充分显示
        time.sleep(3)  # 增加等待时间

        # 检查TraCI连接是否正常
        if not traci.isLoaded():
            logging.error("TraCI连接未建立，尝试重新连接...")
            raise RuntimeError("TraCI连接未建立")

        # 获取所有车辆
        try:
            all_vehicles = traci.vehicle.getIDList()
            logging.info(f"初始车辆列表: {all_vehicles}")
        except traci.exceptions.FatalTraCIError as e:
            logging.error(f"获取车辆列表失败: {e}")
            return

        cav_ids = env.cav_ids
        logging.info(f"CAV车辆ID列表: {cav_ids}")

        # 执行额外的步骤以等待车辆生成
        if not all_vehicles:
            logging.warning("未检测到任何车辆，尝试推进仿真...")
            # 尝试推进仿真更多步骤，等待车辆产生
            for step in range(20):
                traci.simulationStep()
                all_vehicles = traci.vehicle.getIDList()
                if all_vehicles:
                    logging.info(f"步骤 {step}: 检测到车辆: {all_vehicles}")
                    break
                time.sleep(0.1)

        # 获取最新的车辆列表
        all_vehicles = traci.vehicle.getIDList()
        if not all_vehicles:
            logging.error("仍然没有检测到车辆，请检查路由文件和配置")
            # 打印路由文件内容以便调试
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                route_file = os.path.join(
                    current_dir, "target_model", "input_sources", "routes.rou.xml"
                )
                if os.path.exists(route_file):
                    with open(route_file, "r") as f:
                        logging.info(f"路由文件内容: \n{f.read()}")
                else:
                    logging.error(f"路由文件不存在: {route_file}")
            except Exception as e:
                logging.error(f"读取路由文件失败: {e}")
            return

        # 对所有车辆添加颜色以便观察
        for veh_id in all_vehicles:
            try:
                vtype = traci.vehicle.getTypeID(veh_id)
                if vtype == "CAV":
                    traci.vehicle.setColor(veh_id, (255, 0, 0, 255))  # 红色CAV
                else:
                    traci.vehicle.setColor(veh_id, (0, 0, 255, 255))  # 蓝色HDV
            except:
                pass

        # 执行一些仿真步骤以观察GUI响应
        logging.info("执行仿真步骤...")
        for i in range(100):  # 执行100步，给足够的时间观察GUI
            # 对所有车辆应用控制
            for veh_id in traci.vehicle.getIDList():
                if "CAV" in veh_id or traci.vehicle.getTypeID(veh_id) == "CAV":
                    # 对CAV应用智能控制
                    action = 0.1  # 轻微加速
                    current_speed = traci.vehicle.getSpeed(veh_id)
                    traci.vehicle.setSpeed(veh_id, current_speed + action)

            # 推进仿真
            traci.simulationStep()

            # 每10步打印一次状态信息
            if i % 10 == 0:
                vehicles = traci.vehicle.getIDList()
                logging.info(f"Step {i}: 当前车辆数量: {len(vehicles)}")
                if vehicles:
                    for veh_id in vehicles[:3]:  # 只打印前3辆车的信息
                        try:
                            speed = traci.vehicle.getSpeed(veh_id)
                            pos = traci.vehicle.getPosition(veh_id)
                            lane = traci.vehicle.getLaneID(veh_id)
                            vtype = traci.vehicle.getTypeID(veh_id)
                            logging.info(
                                f"  车辆{veh_id} ({vtype}): 速度={speed:.2f}, 位置={pos}, 车道={lane}"
                            )
                        except:
                            pass

            # 等待一小段时间使动画平滑
            time.sleep(0.05)

            # 检查是否还有车辆
            if not traci.vehicle.getIDList():
                logging.info("没有车辆在仿真中，结束测试")
                break

        logging.info("GUI测试完成")

    except Exception as e:
        logging.error(f"GUI测试出错: {e}")
        import traceback

        logging.error(traceback.format_exc())
    finally:
        # 确保环境正确关闭
        if env:
            logging.info("关闭环境...")
            try:
                env.close()
            except Exception as e:
                logging.error(f"关闭环境时出错: {e}")


if __name__ == "__main__":
    test_gui()
