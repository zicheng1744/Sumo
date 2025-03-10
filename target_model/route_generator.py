import os
import random
import logging


def generate_routefile(probs, speed, N, accel, decel):
    lam1 = probs["main"]
    lam2 = probs["ramp"]
    lam3 = probs["CAV"]
    route_file_path = os.path.join(
        os.path.dirname(__file__), "input_sources", "routes.rou.xml"
    )
    try:
        with open(route_file_path, "w") as routes:
            print(
                f"""<routes>
        <vType id="CAV" accel="{accel}" decel="{decel}" sigma="0.0" length="5" minGap="0.8" tau="0.5" maxSpeed="{speed}" carFollowingModel="IDM" lcStrategic="1" lcCooperative="1" lcAssertive="0.5" lcImpatience="0.0" lcKeepRight="0"/>
        <vType id="HDV" accel="{accel}" decel="{decel}" sigma="0.5" length="5" minGap="2.5" tau="1.2" maxSpeed="{speed}" desiredMaxSpeed="{speed}" speedFactor="normc(1,0.2,0.2,2)" carFollowingModel="Krauss" lcStrategic="0.3" lcAssertive="0.5" lcCooperative="0.2" lcImpatience="0.5"/>
        <route id="main" edges="wm mc ce" />
        <route id="ramp" edges="rm mc ce" />
        """,
                file=routes,
            )
            for i in range(N):
                if random.uniform(0, 1) < lam1:  # 主路车辆泊松分布
                    if random.uniform(0, 1) < lam3:  # CAV 车辆泊松分布
                        print(
                            f'    <vehicle id="main_{i}" type="CAV" route="main" depart="{i}" color="1,0,0" />',
                            file=routes,
                        )
                    else:
                        hdv_depart_speed = random.uniform(0.8 * speed, speed)
                        print(
                            f'    <vehicle id="main_{i}" type="HDV" route="main" depart="{i}"  departSpeed="{hdv_depart_speed}" />',
                            file=routes,
                        )
                if random.uniform(0, 1) < lam2:  # 支路车辆泊松分布
                    if random.uniform(0, 1) < lam3:
                        print(
                            f'    <vehicle id="ramp_{i}" type="CAV" route="ramp" depart="{i}" color="1,0,0" />',
                            file=routes,
                        )
                    else:
                        hdv_depart_speed = random.uniform(0.8 * speed, speed)
                        print(
                            f'    <vehicle id="ramp_{i}" type="HDV" route="ramp" depart="{i}"  departSpeed="{hdv_depart_speed}" />',
                            file=routes,
                        )
            print("</routes>", file=routes)
        logging.info(f"路由文件已生成: {route_file_path}")
    except OSError as e:
        logging.error(f"写入路由文件时出错: {e}")
        raise

    return route_file_path


if __name__ == "__main__":
    probs = {"main": 0.5, "ramp": 0.4, "CAV": 0.4}
    speed = 10.0
    N = 50
    accel = 3.0
    decel = 5.0
    print("开始生成路由文件...")
    route_file_path = generate_routefile(probs, speed, N, accel, decel)
    print(f"已更新路由文件: {route_file_path}")
