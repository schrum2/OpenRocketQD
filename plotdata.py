import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t

# Directory containing the JSON files
data_dir = "evolve_rockets_output"

# File patterns for both algorithms
cma_pattern = "cma_me_imp_stabilitynose_altitude_{}_metrics.json"
map_elites_pattern = "map_elites_stabilitynose_altitude_{}_metrics.json"

# Number of files to process for each algorithm
num_files = 30

# Lists to store data for both algorithms
cma_coverage_data = []
map_elites_coverage_data = []
cma_qd_score_data = []
map_elites_qd_score_data = []

# Load data from CMA-ES files
for i in range(num_files):
    cma_file_path = os.path.join(data_dir, cma_pattern.format(i))
    with open(cma_file_path, 'r') as f:
        cma_data = json.load(f)
        cma_coverage_data.append(cma_data["Archive Coverage"]["y"])
        cma_qd_score_data.append(cma_data["QD Score"]["y"])

# Load data from MAP-Elites files
for i in range(num_files):
    map_elites_file_path = os.path.join(data_dir, map_elites_pattern.format(i))
    with open(map_elites_file_path, 'r') as f:
        map_elites_data = json.load(f)
        map_elites_coverage_data.append(map_elites_data["Archive Coverage"]["y"])
        map_elites_qd_score_data.append(map_elites_data["QD Score"]["y"])

# Convert lists to numpy arrays for easier manipulation
cma_coverage_data = np.array(cma_coverage_data)
map_elites_coverage_data = np.array(map_elites_coverage_data)
cma_qd_score_data = np.array(cma_qd_score_data)
map_elites_qd_score_data = np.array(map_elites_qd_score_data)

# Function to calculate means and 95% confidence intervals
def calculate_mean_and_ci(data):
    mean_values = np.mean(data, axis=0)
    ci_values = sem(data, axis=0) * t.ppf((1 + 0.95) / 2, data.shape[0] - 1)
    return mean_values, ci_values

# Calculate means and confidence intervals for both algorithms
cma_coverage_mean, cma_coverage_ci = calculate_mean_and_ci(cma_coverage_data)
map_elites_coverage_mean, map_elites_coverage_ci = calculate_mean_and_ci(map_elites_coverage_data)
cma_qd_score_mean, cma_qd_score_ci = calculate_mean_and_ci(cma_qd_score_data)
map_elites_qd_score_mean, map_elites_qd_score_ci = calculate_mean_and_ci(map_elites_qd_score_data)

# X values (assumed to be the same across all files)
x_values = cma_data["Archive Coverage"]["x"]

# Configure matplotlib to embed fonts in the PDF
plt.rcParams['pdf.fonttype'] = 42
# Increase font sizes
plt.rcParams.update({'font.size': 14})

# Fixed left margin to ensure consistent plot box width
left_margin = 0.15  # Adjust this value if necessary

# Plot Archive Coverage comparison
plt.figure(figsize=(10, 6))
plt.plot(x_values, cma_coverage_mean, label="CMA-ES", color="blue")
plt.fill_between(x_values, cma_coverage_mean - cma_coverage_ci, cma_coverage_mean + cma_coverage_ci, color="blue", alpha=0.2)
plt.plot(x_values, map_elites_coverage_mean, label="MAP-Elites", color="green")
plt.fill_between(x_values, map_elites_coverage_mean - map_elites_coverage_ci, map_elites_coverage_mean + map_elites_coverage_ci, color="green", alpha=0.2)
plt.xlabel("Generations", fontsize=20)
plt.ylabel("Average Archive Coverage", fontsize=20)
plt.title("Average Archive Coverage", fontsize=20)
plt.legend(loc="lower right", fontsize=20)
plt.subplots_adjust(left=left_margin)
plt.savefig("Archive_Coverage_Comparison.pdf", bbox_inches='tight')
plt.close()

# Plot QD Score comparison
plt.figure(figsize=(10, 6))
plt.plot(x_values, cma_qd_score_mean, label="CMA-ES", color="blue")
plt.fill_between(x_values, cma_qd_score_mean - cma_qd_score_ci, cma_qd_score_mean + cma_qd_score_ci, color="blue", alpha=0.2)
plt.plot(x_values, map_elites_qd_score_mean, label="MAP-Elites", color="green")
plt.fill_between(x_values, map_elites_qd_score_mean - map_elites_qd_score_ci, map_elites_qd_score_mean + map_elites_qd_score_ci, color="green", alpha=0.2)
plt.xlabel("Generations", fontsize=20)
plt.ylabel("Average QD Score", fontsize=20)
plt.title("Average QD Score", fontsize=20)
plt.legend(loc="lower right", fontsize=20)
plt.subplots_adjust(left=left_margin)
plt.savefig("QD_Score_Comparison.pdf", bbox_inches='tight')
plt.close()

average_scaling_factor = 10000.0

# Convert CMA-ES and MAP-Elites Archive Coverage to counts (Occupied Cells)
cma_occupied_cells = cma_coverage_mean * average_scaling_factor
map_elites_occupied_cells = map_elites_coverage_mean * average_scaling_factor

# Plot Number of Individuals in Archive (Occupied Cells) comparison
plt.figure(figsize=(10, 6))
plt.plot(x_values, cma_occupied_cells, label="CMA-ES", color="blue")
plt.fill_between(x_values, (cma_occupied_cells - cma_coverage_ci * average_scaling_factor),
                 (cma_occupied_cells + cma_coverage_ci * average_scaling_factor), color="blue", alpha=0.2)
plt.plot(x_values, map_elites_occupied_cells, label="MAP-Elites", color="green")
plt.fill_between(x_values, (map_elites_occupied_cells - map_elites_coverage_ci * average_scaling_factor),
                 (map_elites_occupied_cells + map_elites_coverage_ci * average_scaling_factor), color="green", alpha=0.2)
plt.xlabel("Generations", fontsize=20)
plt.ylabel("Occupied Cells", fontsize=20)
plt.title("Number of Individuals in Archive", fontsize=20)
plt.legend(loc="lower right", fontsize=20)
plt.subplots_adjust(left=left_margin)
plt.savefig("Occupied_Cells_Comparison.pdf", bbox_inches='tight')
plt.close()


