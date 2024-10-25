import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t

# Directory containing the JSON files
data_dir = "evolve_rockets_output"
file_pattern = "cma_me_imp_stabilitynose_altitude_{}_metrics.json"

# Number of files to process
num_files = 30

# Lists to store data for averaging
coverage_data = []
qd_score_data = []

# Load data from each file
for i in range(num_files):
    file_path = os.path.join(data_dir, file_pattern.format(i))
    with open(file_path, 'r') as f:
        data = json.load(f)
        coverage_data.append(data["Archive Coverage"]["y"])
        qd_score_data.append(data["QD Score"]["y"])

# Convert lists to numpy arrays for easier manipulation
coverage_data = np.array(coverage_data)
qd_score_data = np.array(qd_score_data)

# Calculate means and 95% confidence intervals
def calculate_mean_and_ci(data):
    mean_values = np.mean(data, axis=0)
    ci_values = sem(data, axis=0) * t.ppf((1 + 0.95) / 2, data.shape[0] - 1)
    return mean_values, ci_values

coverage_mean, coverage_ci = calculate_mean_and_ci(coverage_data)
qd_score_mean, qd_score_ci = calculate_mean_and_ci(qd_score_data)

# X values (assumed to be the same across all files)
x_values = data["Archive Coverage"]["x"]

# Plot Archive Coverage
plt.figure(figsize=(10, 6))
plt.plot(x_values, coverage_mean, label="Average Archive Coverage", color="blue")
plt.fill_between(x_values, coverage_mean - coverage_ci, coverage_mean + coverage_ci, color="blue", alpha=0.2)
plt.xlabel("Generations")
plt.ylabel("Average Archive Coverage")
plt.title("Average Archive Coverage with 95% Confidence Interval")
plt.legend()
plt.savefig("Average_Archive_Coverage.pdf")
plt.close()

# Plot QD Score
plt.figure(figsize=(10, 6))
plt.plot(x_values, qd_score_mean, label="Average QD Score", color="green")
plt.fill_between(x_values, qd_score_mean - qd_score_ci, qd_score_mean + qd_score_ci, color="green", alpha=0.2)
plt.xlabel("Generations")
plt.ylabel("Average QD Score")
plt.title("Average QD Score with 95% Confidence Interval")
plt.legend()
plt.savefig("Average_QD_Score.pdf")
plt.close()
