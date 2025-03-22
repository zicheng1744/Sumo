import os
import random
import argparse
import time

def generate_routefile(probs, speed, N, accel, decel, max_vehicles=64):
    """生成SUMO所需的路由文件，包含随机生成的车流
    
    Args:
        probs: 包含各种概率的字典 {'main': 主路概率, 'ramp': 匝道概率, 'CAV': 智能车概率}
        speed: 车辆最大速度
        N: 模拟时长/车辆生成间隔
        accel: 加速度
        decel: 减速度
        max_vehicles: 最大生成车辆数量
    
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
    total_vehicles = 0  # 总车辆计数
    
    # 匝道车辆的最大速度需要降低，避免速度过高导致冲突
    ramp_speed = min(speed, 15.0)  # 匝道车辆最大速度限制为15
    
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
            
            # 生成车辆，但限制最大数量为max_vehicles
            for i in range(N):
                if total_vehicles >= max_vehicles:
                    break
                    
                # 主路车辆
                if random.uniform(0, 1) < lam1 and total_vehicles < max_vehicles:
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
                    total_vehicles += 1
                
                # 匝道车辆
                if random.uniform(0, 1) < lam2 and total_vehicles < max_vehicles:
                    if random.uniform(0, 1) < lam3:  # CAV车辆
                        # 匝道车辆以较低速度进入
                        print(
                            f'    <vehicle id="ramp_{veh_nr_ramp}" type="CAV" route="ramp" depart="{i}" departSpeed="10" color="1,0,0" />',
                            file=routes,
                        )
                    else:  # HDV车辆
                        # 匝道车辆以较低速度进入
                        hdv_depart_speed = random.uniform(5.0, ramp_speed)
                        print(
                            f'    <vehicle id="ramp_{veh_nr_ramp}" type="HDV" route="ramp" depart="{i}" departSpeed="{hdv_depart_speed}" />',
                            file=routes,
                        )
                    veh_nr_ramp += 1
                    total_vehicles += 1
                    
            print("</routes>", file=routes)
        print(f"路由文件已生成: {route_file_path}")
        print(f"总共生成了 {total_vehicles} 辆车 (主路: {veh_nr_main}, 匝道: {veh_nr_ramp})")
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
    parser.add_argument("--speed", type=float, default=30.0, help="车辆最大速度")  # 降低默认速度至30
    parser.add_argument("--duration", type=int, default=1000, help="模拟时长")
    parser.add_argument("--accel", type=float, default=3.0, help="加速度")
    parser.add_argument("--decel", type=float, default=5.0, help="减速度")
    parser.add_argument("--max_vehicles", type=int, default=64, help="最大生成车辆数量")
    
    args = parser.parse_args()
    
    # 设置车流生成参数
    probs = {
        "main": args.main_prob,
        "ramp": args.ramp_prob,
        "CAV": args.cav_prob
    }
    
    print(f"生成参数：主路概率={args.main_prob}, 匝道概率={args.ramp_prob}, CAV比例={args.cav_prob}")
    print(f"车辆速度={args.speed}, 模拟时长={args.duration}, 加速度={args.accel}, 减速度={args.decel}")
    print(f"最大车辆数量={args.max_vehicles}")
    
    # 生成路由文件
    start_time = time.time()
    route_file_path = generate_routefile(probs, args.speed, args.duration, args.accel, args.decel, args.max_vehicles)
    end_time = time.time()
    
    print(f"成功生成路由文件，用时 {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main() 