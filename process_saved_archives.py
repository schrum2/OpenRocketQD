import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    data = archive.data(return_type="pandas")
    print(data.columns)

    # Extract measures and fitness
    measures = data[["measures_0", "measures_1"]]
    fitness = data["objective"]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot (square cells) and remove the title
    scatter = ax.scatter(
        measures["measures_0"], measures["measures_1"],
        c=fitness, cmap="magma", norm=Normalize(vmin=0, vmax=MAX_FITNESS),
        s=15, marker="s"  # square cells
    )
    
    # Configure color bar with "Consistency" label
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Consistency", fontsize=12)

    # Set limits and ticks on y-axis (Altitude)
    ax.set_ylim(MIN_ALTITUDE, MAX_ALTITUDE)
    ax.set_ylabel("Altitude", fontsize=12)

    # Set limits on x-axis and add nose type labels
    ax.set_xlim(0, ((BUFFER + (MAX_STABILITY - MIN_STABILITY)) * (MAX_NOSE_TYPE_INDEX + 1)))
    ax.set_xlabel("Stability", fontsize=12)

    # Create custom ticks and labels for stability sections
    ticks = []
    tick_labels = []
    stability_ticks = np.linspace(MIN_STABILITY, MAX_STABILITY, num=5)
    
    for i in range(MIN_NOSE_TYPE_INDEX, MAX_NOSE_TYPE_INDEX + 1):
        section_start = i * (BUFFER + (MAX_STABILITY - MIN_STABILITY))
        section_ticks = section_start + stability_ticks
        
        ticks.extend(section_ticks)
        tick_labels.extend([f"{stability:.1f}" for stability in stability_ticks])

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    
    # Add nose type labels
    nose_labels = ["OGIVE", "CONICAL", "ELLIPSOID", "POWER", "PARABOLIC", "HAACK"]
    nose_positions = [(i + 0.5) * (BUFFER + (MAX_STABILITY - MIN_STABILITY)) for i in range(6)]
    ax_secondary = ax.secondary_xaxis("top")
    ax_secondary.set_xticks(nose_positions)
    ax_secondary.set_xticklabels(nose_labels, fontsize=10)
    ax_secondary.set_xlabel("Nose Type", fontsize=12)

    plt.tight_layout()
    plt.show()

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