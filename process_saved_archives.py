import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker
from ribs.archives import GridArchive

import argparse
import os

from rocket_evaluate import MAX_FITNESS

# Constants from original configuration
BUFFER = 0.5
MIN_STABILITY = 1.0
MAX_STABILITY = 3.0
MIN_ALTITUDE = 0.0
MAX_ALTITUDE = 90.0
MIN_NOSE_TYPE_INDEX = 0
MAX_NOSE_TYPE_INDEX = 5
NOSE_TYPE_LABELS = ["OGIVE", "CONICAL", "ELLIPSOID", "POWER", "PARABOLIC", "HAACK"]
NUM_NOSE_TYPES = len(NOSE_TYPE_LABELS)

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
   
    return add_data_frame_to_archive(df, archive)

    
def add_data_frame_to_archive(df, archive):
    """
    Takes all the solutions in a pandas dataframe and puts them into a pyribs GridArchive.

    Args:
        df: pandas dataframe
        archive: pyribs GridArchive (would other types work too?)

    Returns:
        GridArchive with the solutions added
    """
    # Redundant calculation?
    solution_cols = [col for col in df.columns if col.startswith('solution_')]

    # Add each solution to the archive
    for _, row in df.iterrows():
        # Reshape arrays to match expected batch dimensions
        solution = row[solution_cols].values.reshape(1, -1)  # Shape: (1, solution_dim)
        objective = np.array([row['objective']])  # Shape: (1,)
        measures = row[[f'measures_{i}' for i in range(2)]].values.reshape(1, -1)  # Shape: (1, 2)
        
        # Add the solution to the archive
        archive.add(solution, objective, measures)
    
    return archive

def load_multiple_archives(prefix=None, start_index=None, end_index=None, config=None):
    """
    Load multiple archives from CSV files based on provided file prefix and index range
    
    Args:
        prefix: Text prefix that all file names start with
        start_index: Starting index for archive files (inclusive)
        end_index: Ending index for archive files (inclusive)
        config: Optional configuration dictionary
        
    Returns:
        GridArchive: Merged archive containing solutions from multiple files
    """
    global bounds

    # Get information from first CSV file in collection
    df = pd.read_csv(f"{prefix}_{start_index}_archive.csv")

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

    archive = add_data_frame_to_archive(df, archive)
 
    # Start at +1 since first was read and added above
    for i in range(start_index+1, end_index + 1):
        # Load the CSV data
        df = pd.read_csv(f"{prefix}_{i}_archive.csv")
        archive = add_data_frame_to_archive(df, archive)

    return archive

# Custom plotting function
def plot_custom_heatmap(archive, save_path="custom_heatmap.pdf"):
    """
    Custom implementation of a heatmap plot for a GridArchive with labeled intervals on the x-axis.

    Args:
        archive: A GridArchive instance containing the archive data.
        save_path: Path to save the generated heatmap image.
    """
    # Extract the archive data (fitness and measures) into a 2D grid
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

            # If fitness value is 0, print coordinates
            #if fitness_values[i] == 0:
            #    print(f"Point with Objective 0 found at coordinates: (Stability: {stability}, Altitude: {altitude})")



    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
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

    # Add vertical lines to divide the x-axis into nose type segments
    total_x_range = upper_bounds[0] - lower_bounds[0]
    stability_range_width = MAX_STABILITY - MIN_STABILITY

    # Update x-axis ticks and labels
    stability_ticks = [
        lower_bounds[0] + i * (stability_range_width + BUFFER) + stability_range_width / 2.0
        for i in range(NUM_NOSE_TYPES)
    ]

    # Add stability labels as secondary ticks
    secax = ax.secondary_xaxis('top')

    # Define ticks for the boundaries of each nose type, including BUFFER
    nose_type_width = stability_range_width + BUFFER
    secax_ticks = [i * nose_type_width for i in range(NUM_NOSE_TYPES + 1)]  # Include all 6 boundaries

    # Set stability range labels for each segment
    stability_labels = []
    for i in range(NUM_NOSE_TYPES):
        stability_labels.extend([
            f"{MIN_STABILITY:.1f}", 
            f"{(MIN_STABILITY + MAX_STABILITY) / 2:.1f}", 
            f"{MAX_STABILITY:.1f}"
        ])

    secax.set_xticks(secax_ticks)  # Boundary ticks
    secax.set_xticklabels([""] * len(secax_ticks))  # No labels on boundaries
    secax.set_xlabel("Stability Score", fontsize=20)

    # Main axis stability labels within each nose type
    stability_positions = [
        i * nose_type_width + offset
        for i in range(NUM_NOSE_TYPES)
        for offset in [0, stability_range_width / 2, stability_range_width]
    ]
    stability_labels = [
        f"{MIN_STABILITY:.1f}", f"{(MIN_STABILITY + MAX_STABILITY) / 2:.1f}", f"{MAX_STABILITY:.1f}"
    ] * NUM_NOSE_TYPES

    # Set the x-axis ticks for stability
    secax.set_xticks(stability_positions)  # The main stability positions
    secax.set_xticklabels(stability_labels, fontsize=10)  # Stability values like 1.0, 2.0, etc.

    # Draw vertical lines to separate each nose type
    for i in range(NUM_NOSE_TYPES+1):
         line_position = i * nose_type_width - (BUFFER /2.0)
         ax.axvline(x=line_position, color="black", linewidth=1, linestyle="-")
    
    # Add primary x-axis nose type labels
    nose_type_positions = [i * nose_type_width + stability_range_width / 2 for i in range(NUM_NOSE_TYPES)]
    nose_types = ["OGIVE", "CONICAL", "ELLIPSOID", "POWER", "PARABOLIC", "HAACK"]

    #print(stability_ticks)
    ax.set_xticks(stability_ticks)
    ax.set_xticklabels(NOSE_TYPE_LABELS, fontsize=15)

    # Set the x-axis ticks for the nose types (minor ticks) and their labels
    #ax.set_xticks(nose_type_positions, minor=True)  # The positions where nose types should be
    #ax.tick_params(axis='x', which='minor', length=0)  # Hide minor ticks for better visibility
    #ax.set_xticklabels(nose_types, fontsize=20, minor=True)  # Nose type labels below stability ticks


    # Labels and formatting
    ax.set_xlabel("Nose Type", fontsize=20)
    ax.set_ylabel("Altitude", fontsize=20)

    # Add a colorbar
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label("Consistency Score", fontsize=20)

    # Configure matplotlib to embed fonts in the PDF
    plt.rcParams['pdf.fonttype'] = 42
    # Increase font sizes
    plt.rcParams.update({'font.size': 14})
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def compare_archives(archive1, archive2):
    """Creates a new GridArchive showing the overlap between two existing archives.
    
    Args:
        archive1 (GridArchive): First archive to compare
        archive2 (GridArchive): Second archive to compare
        
    Returns:
        GridArchive: A new archive where:
        - Cells with solutions in both archives have score 100
        - Cells with solutions only in archive1 have score 75
        - Cells with solutions only in archive2 have score 25
        - Empty cells in both archives remain empty
    
    Raises:
        ValueError: If the archives have different dimensions or ranges
    """
    # Verify archives have same configuration
    if not np.array_equal(archive1.dims, archive2.dims):
        raise ValueError("Archives must have same dimensions")
    if not (np.array_equal(archive1.lower_bounds, archive2.lower_bounds) and 
            np.array_equal(archive1.upper_bounds, archive2.upper_bounds)):
        raise ValueError("Archives must have same ranges")
    
    # Create new archive with same configuration
    result_archive = GridArchive(
        solution_dim=archive1.solution_dim,
        dims=archive1.dims,
        ranges=list(zip(archive1.lower_bounds, archive1.upper_bounds))
    )
    
    # Get data from both archives
    _, data1 = archive1._store.retrieve(np.arange(archive1.cells))
    _, data2 = archive2._store.retrieve(np.arange(archive2.cells))
    
    # Create masks for occupied cells
    mask1 = np.vectorize(lambda x: x > 0)(data1['objective'])
    mask2 = np.vectorize(lambda x: x > 0)(data2['objective'])

    #print(data1['objective'])
    #print(len(data1['objective']))
    #print(mask1)
    #print(len(mask1))
    #print(mask1.sum())
    
    # Create solution indices
    indices = np.arange(archive1.cells)
    
    # Add solutions for cells where both archives have content (score 100)
    both_mask = mask1 & mask2
    if np.any(both_mask):
        indices_both = indices[both_mask]
        for idx in indices_both:
            # Use solution from archive1 arbitrarily since we only care about presence
            result_archive.add_single(
                solution=data1['solution'][idx],
                objective=100.0,
                measures=data1['measures'][idx]
            )
    
    # Add solutions for cells only in archive1 (score 75)
    only1_mask = mask1 & ~mask2
    if np.any(only1_mask):
        indices_only1 = indices[only1_mask]
        for idx in indices_only1:
            result_archive.add_single(
                solution=data1['solution'][idx],
                objective=75.0,
                measures=data1['measures'][idx]
            )
    
    # Add solutions for cells only in archive2 (score 25)
    only2_mask = ~mask1 & mask2
    if np.any(only2_mask):
        indices_only2 = indices[only2_mask]
        for idx in indices_only2:
            result_archive.add_single(
                solution=data2['solution'][idx],
                objective=25.0,
                measures=data2['measures'][idx]
            )
    
    return result_archive
   
def main():
    """
    Set up and validate argument parser for rocket design archive loading.
    
    Requires either:
    1. A specific file with --file, OR
    2. Both --prefix and --range together
    
    Output file is optional in both cases.
    """
    parser = argparse.ArgumentParser(description="Load and visualize rocket design archives")
    
    # Create a group to enforce the file XOR (prefix + range) requirement
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-f', '--file', 
                             help='Specific CSV file to load')
    input_group.add_argument('-p', '--prefix', 
                             help='Path and file name prefix shared by all CSV files in range to be loaded')
    
    # Add range as a required argument when prefix is used
    parser.add_argument('-r', '--range', 
                        nargs=2, 
                        type=int, 
                        metavar=('START', 'END'),
                        help='Range of archive indices to load (inclusive)',
                        required=False)
    
    # Optional secondary archive to compare against
    parser.add_argument('-c', '--compare', 
                        default=None, 
                        help='File or prefix to compare against')

    # Optional output file specification
    parser.add_argument('-o', '--output', 
                        default='custom_heatmap.pdf', 
                        help='Output file path for the heatmap')
    
    # Parse initial arguments
    args = parser.parse_args()
    
    # Custom validation
    if args.prefix is not None and args.range is None:
        parser.error("--prefix requires --range to be specified")
    
    # Set up configuration with threshold_min
    config = {
        "archive": {
            "kwargs": {
                "threshold_min": -np.inf
            }
        }
    }
    
    # Load archives based on input type
    if args.file:
        archive = load_grid_archive_from_csv(args.file, config)
        if args.compare:
            other = load_grid_archive_from_csv(args.compare, config)
            archive = compare_archives(archive, other)
    else:
        archive = load_multiple_archives(prefix=args.prefix,
                                         start_index=args.range[0], 
                                         end_index=args.range[1], 
                                         config=config)

        if args.compare:
            other = load_multiple_archives(prefix=args.compare,
                                         start_index=args.range[0], 
                                         end_index=args.range[1], 
                                         config=config)
            archive = compare_archives(archive, other)
    
    # Verify the loaded archive
    print(f"Number of elite solutions: {len(archive)}")
    print(f"Grid dimensions: {archive.dims}")
    print(f"Measure ranges: {list(zip(archive.lower_bounds, archive.upper_bounds))}")
    
    # Plot the custom heatmap
    plot_custom_heatmap(archive, args.output)


# Load a single specific file
# python process_saved_archives.py -f evolve_rockets_output/cma_me_imp_stabilitynose_altitude_5_archive.csv

# Load archives with indices 0 through 29 and specify output file
# python process_saved_archives.py -r 0 29 -p evolve_rockets_output/cma_me_imp_stabilitynose_altitude -o cma_me_imp_stabilitynose_altitude.pdf

if __name__ == "__main__":
    main()
