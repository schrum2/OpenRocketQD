import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def plot_custom_heatmap(archive):
    # Retrieve archive data in pandas format
    archive_data = archive.data(return_type='pandas')

    # Extract columns
    stability_nose_values = archive_data['measures_0']
    altitude_values = archive_data['measures_1']
    fitness_values = archive_data['objective']

    # Plotting configuration
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate by nose type and plot each as a column
    for nose_type_index in range(MIN_NOSE_TYPE_INDEX, MAX_NOSE_TYPE_INDEX + 1):
        mask = (stability_nose_values >= nose_type_index * (BUFFER + (MAX_STABILITY - MIN_STABILITY))) & \
               (stability_nose_values < (nose_type_index + 1) * (BUFFER + (MAX_STABILITY - MIN_STABILITY)))
        
        # Select rows that match the current nose type
        nose_data = archive_data[mask]
        stabilities = nose_data['measures_0'] % (BUFFER + (MAX_STABILITY - MIN_STABILITY)) + MIN_STABILITY
        altitudes = nose_data['measures_1']
        fitnesses = nose_data['objective']

        # Create a scatter plot for the current nose type
        scatter = ax.scatter(
            stabilities + nose_type_index * (BUFFER + (MAX_STABILITY - MIN_STABILITY)),
            altitudes,
            c=fitnesses,
            cmap="magma",
            vmin=0,
            vmax=MAX_FITNESS,
            s=5
        )
    
    # Color bar and labels
    fig.colorbar(scatter, ax=ax, label="Fitness")
    ax.set_xlabel("Stability + Nose Type Index")
    ax.set_ylabel("Altitude")
    ax.set_title("Custom Heatmap by Nose Type and Stability")

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