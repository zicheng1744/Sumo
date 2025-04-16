import os
import sys
import csv
import pandas as pd

# Add the parent directory to sys.path to import calculate_throughput
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calculate_throughput import calculate_average_throughput

def main():
    """
    Calculate average throughput for different CAV penetration rates (0.1-1.0)
    and generate final_throughput.csv file
    """
    # Base path for analysis folders (01-10 for different CAV probabilities)
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # List to store throughput values
    throughput_values = []
    
    # Process each folder (01 to 10 representing 0.1 to 1.0 CAV probability)
    for i in range(1, 11):
        folder_name = f"{i:02d}"  # Format: 01, 02, ..., 10
        throughput_file = os.path.join(base_path, folder_name, "data", "throughput.csv")
        
        if os.path.exists(throughput_file):
            # Calculate average throughput
            avg_throughput = calculate_average_throughput(throughput_file)
            if avg_throughput is not None:
                throughput_values.append(avg_throughput)
                print(f"CAV Probability {i/10:.1f}: Average Throughput = {avg_throughput:.6f}")
            else:
                print(f"Error: Could not calculate average throughput for CAV Probability {i/10:.1f}")
                throughput_values.append(0)  # Add 0 as placeholder for missing data
        else:
            print(f"Warning: File not found - {throughput_file}")
            throughput_values.append(0)  # Add 0 as placeholder for missing data
    
    # Save throughput values to final_throughput.csv
    output_file = os.path.join(base_path, "final_throughput.csv")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(throughput_values)
    
    print(f"\nFinal throughput values saved to {output_file}")
    print(f"Values: {', '.join([f'{val:.6f}' for val in throughput_values])}")

if __name__ == "__main__":
    main() 