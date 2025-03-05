import os
import sys
import traci


def test_traci_connection():
    """测试TraCI连接"""
    try:
        # 启动SUMO
        sumo_cmd = ["sumo", "-c", "config.sumocfg"]
        traci.start(sumo_cmd)

        # 执行一个仿真步骤
        traci.simulationStep()

        # 获取所有车辆ID
        vehicle_ids = traci.vehicle.getIDList()
        print(f"当前场景中的车辆数量: {len(vehicle_ids)}")
        print(f"车辆ID列表: {vehicle_ids}")

        # 关闭连接
        traci.close()
        print("TraCI连接测试成功！")

    except Exception as e:
        print(f"连接测试失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    test_traci_connection()
