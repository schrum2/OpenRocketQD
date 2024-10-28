import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker
from ribs.archives import GridArchive
from ribs.visualize import cvt_archive_heatmap, grid_archive_heatmap

from rocket_evaluate import MAX_FITNESS

# Constants from original configuration
BUFFER = 0.5
MIN_STABILITY = 1.0
MAX_STABILITY = 3.0
MIN_ALTITUDE = 0.0
MAX_ALTITUDE = 90.0
MIN_NOSE_TYPE_INDEX = 0
MAX_NOSE_TYPE_INDEX = 5

def load_grid_archive_from_csv(filepath, config=None):
    """
    Load a previously saved GridArchive from a CSV file.
    
    Args:
        filepath: Path to the CSV file containing the archive data
        config: Optional configuration dictionary containing archive kwargs
        
    Returns:
        GridArchive: Reconstructed archive with the saved solutions
    """
    global bounds

    # Load the CSV data
    df = pd.read_csv(filepath)
    
    # Extract dimensions of the problem
    solution_cols = [col for col in df.columns if col.startswith('solution_')]
    solution_dim = len(solution_cols)
    
    # Use the original archive parameters
    archive_dims = (100, 100)
    bounds = [
        (0, ((BUFFER + (MAX_STABILITY - MIN_STABILITY)) * (MAX_NOSE_TYPE_INDEX + 1))),
        (MIN_ALTITUDE, MAX_ALTITUDE)
    ]
    
    # Create new archive with original parameters
    archive_kwargs = {} if config is None else config.get("archive", {}).get("kwargs", {})
    
    archive = GridArchive(
        solution_dim=solution_dim,
        dims=archive_dims,
        ranges=bounds,
        **archive_kwargs
    )
    
    # Add each solution to the archive
    for _, row in df.iterrows():
        # Reshape arrays to match expected batch dimensions
        solution = row[solution_cols].values.reshape(1, -1)  # Shape: (1, solution_dim)
        objective = np.array([row['objective']])  # Shape: (1,)
        measures = row[[f'measures_{i}' for i in range(2)]].values.reshape(1, -1)  # Shape: (1, 2)
        
        # Add the solution to the archive
        archive.add(solution, objective, measures)
    
    return archive

# Custom plotting function
def plot_custom_heatmap(archive):
    # Load data from the archive
    data = archive.data(return_type='pandas')
    fitness_values = data["objective"].values
    measures = data[["measures_0", "measures_1"]].values

    # Extract stability and altitude values
    stability_values = measures[:, 0]
    altitude_values = measures[:, 1]
    
    # Set up parameters for the grid and plot
    num_nose_types = 6
    stability_bins = 100
    altitude_bins = 90

    # Create a grid to populate fitness values based on stability and altitude positions
    heatmap_grid = np.full((altitude_bins, stability_bins * num_nose_types), np.nan)

    stability_range = MAX_STABILITY - MIN_STABILITY
    total_segment_width = stability_range + BUFFER
    
    # Map each fitness value to its corresponding grid cell
    for i in range(len(fitness_values)):
        stability = stability_values[i]
        altitude = altitude_values[i]
        nose_index = int(stability // total_segment_width)
        local_stability = (stability % total_segment_width) + MIN_STABILITY
        
        # Calculate grid indices
        x_idx = int((nose_index * stability_bins) + ((local_stability - MIN_STABILITY) / stability_range * stability_bins))
        y_idx = int((altitude / MAX_ALTITUDE) * (altitude_bins - 1))

        # Ensure indices are within bounds
        if 0 <= x_idx < stability_bins * num_nose_types and 0 <= y_idx < altitude_bins:
            # Place the fitness value in the appropriate cell of the heatmap grid
            heatmap_grid[y_idx, x_idx] = fitness_values[i]

    # Prepare the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Display the grid as a heatmap
    cmap = plt.cm.inferno
    norm = mcolors.Normalize(vmin=0, vmax=MAX_FITNESS)
    cbar = ax.pcolormesh(heatmap_grid, cmap=cmap, norm=norm, shading='nearest')

    # Set limits and labels for the axes
    ax.set_ylim(0, altitude_bins)
    ax.set_yticks(np.linspace(0, altitude_bins - 1, 10))
    ax.set_yticklabels(np.linspace(0, MAX_ALTITUDE, 10).astype(int))
    ax.set_ylabel("Altitude")
    ax.set_xlabel("Stability / Nose Type")

    # Custom x-axis labels for nose types
    ax.set_xticks([(i + 0.5) * stability_bins for i in range(num_nose_types)])
    ax.set_xticklabels(["OGIVE", "CONICAL", "ELLIPSOID", "POWER", "PARABOLIC", "HAACK"])
    
    # Secondary x-axis for stability labels
    secax = ax.secondary_xaxis('top')
    stability_ticks = [i * stability_bins + offset for i in range(num_nose_types) for offset in [0, stability_bins / 2, stability_bins]]
    secax.set_xticks(stability_ticks)
    secax.set_xticklabels([f"{val:.1f}" for val in [1.0, 2.0, 3.0] * num_nose_types])
    secax.set_xlabel("Stability")

    # Draw vertical lines for stability boundaries (1.0 and 3.0) for each nose type segment
    for i in range(num_nose_types):
        left_edge = i * stability_bins
        ax.axvline(x=left_edge, color="black", linewidth=1)  # Left boundary of each nose type
        ax.axvline(x=left_edge + stability_bins, color="black", linewidth=1)  # 3.0 stability line within each nose type

    # Add color bar with label
    colorbar = fig.colorbar(cbar, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    colorbar.set_label("Consistency")

    plt.tight_layout()
    plt.savefig("test3.png")
    plt.close(fig)

# Example usage:
if __name__ == "__main__":
    # Set up configuration with threshold_min
    config = {
        "archive": {
            "kwargs": {
                "threshold_min": -np.inf
            }
        }
    }
    
    # Load the archive
    archive = load_grid_archive_from_csv(
        "evolve_rockets_output/cma_me_imp_stabilitynose_altitude_0_archive.csv",
        config=config
    )
    
    # Verify the loaded archive
    print(f"Number of elite solutions: {len(archive)}")
    print(f"Grid dimensions: {archive.dims}")
    print(f"Measure ranges: {list(zip(archive.lower_bounds, archive.upper_bounds))}")
    print(f"First measure range: {bounds[0]}")  # Stability-nose combined range
    print(f"Second measure range: {bounds[1]}")  # Altitude range

    # Plot the custom heatmap
    plot_custom_heatmap(archive)

#    import matplotlib
#    matplotlib.use('Agg')
#    import matplotlib.pyplot as plt
#
#    plt.figure(figsize=(8, 6))
#    grid_archive_heatmap(archive, vmin=0, vmax=MAX_FITNESS)
#    plt.tight_layout()
#    plt.savefig("test.png")
#    plt.close(plt.gcf())