import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
import os
import csv

# Set font to support English
plt.rcParams['font.sans-serif'] = ['Arial']  # Use Arial for English labels
plt.rcParams['axes.unicode_minus'] = False  # To display negative signs correctly

# Read CSV data
data_path = os.path.join(os.path.dirname(__file__), "final_speed.csv")
with open(data_path, 'r') as f:
    reader = csv.reader(f)
    speeds = list(reader)[0]
    speeds = [float(speed.strip()) for speed in speeds]

# Define probability values (from 0.1 to 1.0, step 0.1)
cav_probabilities = np.arange(0.1, 1.1, 0.1)

# Set chart style
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'Times New Roman'

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Plot scatter plot
ax.scatter(cav_probabilities, speeds, color='#1f77b4', s=80, alpha=0.7, 
           label='Average Speed Data', edgecolors='black', linewidths=0.5)

# Add trend line (polynomial fitting)
z = np.polyfit(cav_probabilities, speeds, 2)
p = np.poly1d(z)
x_trend = np.linspace(0.1, 1.0, 100)
y_trend = p(x_trend)
ax.plot(x_trend, y_trend, color='#ff7f0e', linestyle='-', linewidth=2, 
        label='Trend Line (Quadratic Fit)')

# Add grid lines
ax.grid(True, linestyle='--', alpha=0.7)

# Set axes
ax.set_xlabel('CAV Proportion', fontsize=14, fontweight='bold')
ax.set_ylabel('Final Average Speed (m/s)', fontsize=14, fontweight='bold')
ax.set_xlim(0.05, 1.05)
ax.set_xticks(np.arange(0.1, 1.1, 0.1))
ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))

# Add title
ax.set_title('Impact of CAV Proportion on Final Average Speed', fontsize=16, fontweight='bold', pad=15)

# Add legend
ax.legend(loc='best', frameon=True, framealpha=0.9, fontsize=12)

# Beautify the chart
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=12)

# Add value labels for data points
for i, speed in enumerate(speeds):
    ax.annotate(f'{speed:.2f}', 
                xy=(cav_probabilities[i], speed),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

# Save chart
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'cav_speed_analysis.png'), dpi=300, bbox_inches='tight')

print("Chart saved as 'cav_speed_analysis.png' and 'cav_speed_analysis.pdf'")
plt.show() 