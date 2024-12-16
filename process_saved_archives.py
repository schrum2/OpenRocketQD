import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker
from ribs.archives import GridArchive

#from ribs.visualize import cvt_archive_heatmap, grid_archive_heatmap

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
def plot_custom_heatmap(archive, save_path="custom_heatmap.png"):
    """
    Custom implementation of a heatmap plot for a GridArchive without using ribs.visualize.

    Args:
        archive: A GridArchive instance containing the archive data.
        save_path: Path to save the generated heatmap image.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import numpy as np

    # Extract the archive data (fitness and measures) into a 2D grid.
    data = archive.data(return_type='pandas')
    fitness_values = data['objective'].values
    measures = data[['measures_0', 'measures_1']].values

    # Set up grid dimensions based on the archive
    grid_width, grid_height = archive.dims  # Archive resolution
    lower_bounds = archive.lower_bounds
    upper_bounds = archive.upper_bounds

    # Calculate grid cell sizes
    cell_width = (upper_bounds[0] - lower_bounds[0]) / grid_width
    cell_height = (upper_bounds[1] - lower_bounds[1]) / grid_height

    # Initialize a grid to hold fitness values
    heatmap_grid = np.full((grid_height, grid_width), np.nan)

    # Fill the grid with fitness values
    for i in range(len(fitness_values)):
        stability, altitude = measures[i]
        x_idx = int((stability - lower_bounds[0]) / cell_width)
        y_idx = int((altitude - lower_bounds[1]) / cell_height)
        
        if 0 <= x_idx < grid_width and 0 <= y_idx < grid_height:
            heatmap_grid[y_idx, x_idx] = fitness_values[i]

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.inferno  # Color map for consistency
    norm = Normalize(vmin=0, vmax=MAX_FITNESS)  # Normalize color scale
    
    c = ax.imshow(
        heatmap_grid,
        origin="lower",
        extent=[lower_bounds[0], upper_bounds[0], lower_bounds[1], upper_bounds[1]],
        cmap=cmap,
        norm=norm,
        aspect="auto"
    )

    # Add a colorbar
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label("Objective (Consistency)")

    # Labels and ticks
    ax.set_xlabel("Stability / Nose Type")
    ax.set_ylabel("Altitude")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Integer ticks for x-axis

    plt.tight_layout()
    plt.savefig(save_path)
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

    # Plot the custom heatmap: once this call is enabled, it replaces the code below
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