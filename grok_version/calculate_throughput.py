import pandas as pd
import os

def calculate_average_throughput(csv_path):
    """
    Calculate the average throughput from the last 10000 data points in a CSV file
    
    Parameters:
        csv_path: Path to the CSV file
        
    Returns:
        Average throughput value
    """
    try:
        # Check if file exists
        if not os.path.exists(csv_path):
            print(f"Error: File {csv_path} does not exist")
            return None
            
        # Read CSV file
        # Assume CSV format has columns: window, step, throughput, cumulative_throughput
        df = pd.read_csv(csv_path)
        
        # Get the last 10000 rows
        if len(df) > 10000:
            df = df.tail(10000)
            
        # Calculate throughput average
        average_throughput = df['throughput'].mean()
        
        print(f"Read {len(df)} data points from file {os.path.basename(csv_path)}")
        print(f"Average throughput: {average_throughput:.6f}")
        
        return average_throughput
        
    except Exception as e:
        print(f"Error calculating average throughput: {e}")
        return None

if __name__ == "__main__":
    # You can replace this path with your actual throughput.csv path
    csv_path = "grok_version/analysis/01/data/throughput.csv"
        
    calculate_average_throughput(csv_path) 