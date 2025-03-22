import os
import random
import argparse
import time

def generate_routefile(probs, speed, N, accel, decel):
    """生成SUMO所需的路由文件，包含随机生成的车流
    
    Args:
        probs: 包含各种概率的字典 {'main': 主路概率, 'ramp': 匝道概率, 'CAV': 智能车概率}
        speed: 车辆最大速度
        N: 模拟时长/车辆生成间隔
        accel: 加速度
        decel: 减速度
    
    Returns:
        路由文件的路径
    """
    lam1 = probs["main"]  # 主路车辆生成概率
    lam2 = probs["ramp"]  # 匝道车辆生成概率
    lam3 = probs["CAV"]   # 智能车辆比例
    
    # 获取路由文件的路径
    route_file_path = os.path.join(
        os.path.dirname(__file__), "input_sources", "routes.rou.xml"
    )
    
    veh_nr_main = 0  # 主路车辆计数
    veh_nr_ramp = 0  # 匝道车辆计数
    
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
            
            # 生成N个时间步长的车辆
            for i in range(N):
                # 主路车辆
                if random.uniform(0, 1) < lam1:
                    if random.uniform(0, 1) < lam3:  # CAV车辆
                        print(
                            f'    <vehicle id="main_{veh_nr_main}" type="CAV" route="main" depart="{i}" color="1,0,0" />',
                            file=routes,
                        )
                    else:  # HDV车辆
                        hdv_depart_speed = random.uniform(0.8 * speed, speed)
                        print(
                            f'    <vehicle id="main_{veh_nr_main}" type="HDV" route="main" depart="{i}" departSpeed="{hdv_depart_speed}" />',
                            file=routes,
                        )
                    veh_nr_main += 1
                
                # 匝道车辆
                if random.uniform(0, 1) < lam2:
                    if random.uniform(0, 1) < lam3:  # CAV车辆
                        print(
                            f'    <vehicle id="ramp_{veh_nr_ramp}" type="CAV" route="ramp" depart="{i}" color="1,0,0" />',
                            file=routes,
                        )
                    else:  # HDV车辆
                        hdv_depart_speed = random.uniform(0.8 * speed, speed)
                        print(
                            f'    <vehicle id="ramp_{veh_nr_ramp}" type="HDV" route="ramp" depart="{i}" departSpeed="{hdv_depart_speed}" />',
                            file=routes,
                        )
                    veh_nr_ramp += 1
                    
            print("</routes>", file=routes)
        print(f"路由文件已生成: {route_file_path}")
    except OSError as e:
        print(f"写入路由文件时出错: {e}")
        raise

    return route_file_path

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="生成SUMO随机车流")
    parser.add_argument("--main_prob", type=float, default=0.5, help="主路车辆生成概率")
    parser.add_argument("--ramp_prob", type=float, default=0.3, help="匝道车辆生成概率")
    parser.add_argument("--cav_prob", type=float, default=0.4, help="智能车辆比例")
    parser.add_argument("--speed", type=float, default=8.0, help="车辆最大速度")
    parser.add_argument("--duration", type=int, default=100, help="模拟时长")
    parser.add_argument("--accel", type=float, default=3.0, help="加速度")
    parser.add_argument("--decel", type=float, default=5.0, help="减速度")
    
    args = parser.parse_args()
    
    # 设置车流生成参数
    probs = {
        "main": args.main_prob,
        "ramp": args.ramp_prob,
        "CAV": args.cav_prob
    }
    
    print(f"生成参数：主路概率={args.main_prob}, 匝道概率={args.ramp_prob}, CAV比例={args.cav_prob}")
    print(f"车辆速度={args.speed}, 模拟时长={args.duration}, 加速度={args.accel}, 减速度={args.decel}")
    
    # 生成路由文件
    start_time = time.time()
    route_file_path = generate_routefile(probs, args.speed, args.duration, args.accel, args.decel)
    end_time = time.time()
    
    print(f"成功生成路由文件，用时 {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main() 