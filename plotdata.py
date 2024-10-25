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
