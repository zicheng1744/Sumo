import os
import random
import argparse
import time

def generate_routefile(probs, speed, duration, accel, decel, max_vehicles=64):
    """
    Generate route file for SUMO using flow tags to create continuous traffic
    
    Args:
        probs: Dictionary containing probabilities {'main': main road prob, 'ramp': ramp prob, 'CAV': CAV ratio}
        speed: Maximum vehicle speed
        duration: Simulation duration
        accel: Acceleration
        decel: Deceleration
        max_vehicles: If specified, limits the maximum number of vehicles
    
    Returns:
        Path to the route file
    """
    main_prob = probs["main"]  # Main road vehicle generation probability
    ramp_prob = probs["ramp"]  # Ramp vehicle generation probability
    cav_prob = probs["CAV"]    # CAV proportion
    
    # Get the path to the route file
    route_file_path = os.path.join(
        os.path.dirname(__file__), "input_sources", "routes.rou.xml"
    )
    
    # Ramp vehicles should have lower max speed to avoid conflicts
    ramp_speed = min(speed, 15.0)  # Limit ramp vehicle max speed to 15
    
    try:
        with open(route_file_path, "w", encoding="utf-8") as routes:
            # Start routes tag
            routes.write("<routes>\n")
            
            # Write vehicle type distribution
            routes.write("    <!-- Vehicle Type Distribution for Flow -->\n")
            routes.write("    <vTypeDistribution id=\"mixed\">\n")
            routes.write(f"        <vType id=\"CAV\" accel=\"{accel}\" decel=\"{decel}\" sigma=\"0.0\" length=\"5\" minGap=\"0.8\" tau=\"0.5\" maxSpeed=\"{speed}\" \n")
            routes.write(f"               carFollowingModel=\"IDM\" lcStrategic=\"1\" lcCooperative=\"1\" lcAssertive=\"0.5\" lcImpatience=\"0.0\" lcKeepRight=\"0\" color=\"1,0,0\" probability=\"{cav_prob}\"/>\n")
            routes.write(f"        <vType id=\"HDV\" accel=\"{accel}\" decel=\"{decel}\" sigma=\"0.5\" length=\"5\" minGap=\"2.5\" tau=\"1.2\" maxSpeed=\"{speed}\" \n")
            routes.write(f"               desiredMaxSpeed=\"{speed}\" speedFactor=\"normc(1,0.2,0.2,2)\" carFollowingModel=\"Krauss\" \n")
            routes.write(f"               lcStrategic=\"0.3\" lcAssertive=\"0.5\" lcCooperative=\"0.2\" lcImpatience=\"0.5\" probability=\"{1-cav_prob}\"/>\n")
            routes.write("    </vTypeDistribution>\n\n")
            
            # Write route definitions
            routes.write("    <!-- Route Definitions -->\n")
            routes.write("    <route id=\"main\" edges=\"wm mc ce\" />\n")
            routes.write("    <route id=\"ramp\" edges=\"rm mc ce\" />\n\n")
            
            # Write flow tags
            routes.write("    <!-- Use Flow Tags for Continuous Traffic -->\n")
            routes.write(f"    <flow id=\"main_flow\" route=\"main\" begin=\"0\" end=\"{duration}\" probability=\"{main_prob}\" type=\"mixed\" departSpeed=\"max\"/>\n")
            routes.write(f"    <flow id=\"ramp_flow\" route=\"ramp\" begin=\"0\" end=\"{duration}\" probability=\"{ramp_prob}\" type=\"mixed\" departSpeed=\"10\"/>\n")
            
            # Close routes tag
            routes.write("</routes>\n")
            
        print(f"Route file generated: {route_file_path}")
        print(f"Configuration: Main prob={main_prob}, Ramp prob={ramp_prob}, CAV ratio={cav_prob}, Duration={duration}")
    except OSError as e:
        print(f"Error writing route file: {e}")
        raise

    return route_file_path

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate continuous traffic for SUMO")
    parser.add_argument("--main_prob", type=float, default=0.7, help="Main road vehicle generation probability")
    parser.add_argument("--ramp_prob", type=float, default=0.3, help="Ramp vehicle generation probability")
    parser.add_argument("--cav_prob", type=float, default=0.5, help="CAV proportion")
    parser.add_argument("--speed", type=float, default=15.0, help="Maximum vehicle speed")
    parser.add_argument("--duration", type=int, default=3600, help="Simulation duration (seconds)")
    parser.add_argument("--accel", type=float, default=3.0, help="Acceleration")
    parser.add_argument("--decel", type=float, default=5.0, help="Deceleration")
    parser.add_argument("--max_vehicles", type=int, default=0, help="Maximum number of vehicles, 0 means unlimited")
    
    args = parser.parse_args()
    
    # Set traffic generation parameters
    probs = {
        "main": args.main_prob,
        "ramp": args.ramp_prob,
        "CAV": args.cav_prob
    }
    
    print(f"Generation parameters: Main prob={args.main_prob}, Ramp prob={args.ramp_prob}, CAV ratio={args.cav_prob}")
    print(f"Vehicle speed={args.speed}, Duration={args.duration}s, Accel={args.accel}, Decel={args.decel}")
    if args.max_vehicles > 0:
        print(f"Maximum vehicles={args.max_vehicles}")
    else:
        print("No limit on maximum vehicles")
    
    # Generate route file
    start_time = time.time()
    route_file_path = generate_routefile(probs, args.speed, args.duration, args.accel, args.decel, args.max_vehicles)
    end_time = time.time()
    
    print(f"Successfully generated route file in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 