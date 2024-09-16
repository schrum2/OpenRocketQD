"""Runs various QD algorithms to evolve model rockets

First run: pip install -r requirements.txt

There are two measures in this example. 
The rocket Stability and the Maximum Altitude.

The supported algorithms are:
- `map_elites`: GridArchive with GaussianEmitter.
- `line_map_elites`: GridArchive with IsoLineEmitter.
- `cvt_map_elites`: CVTArchive with GaussianEmitter.
- `line_cvt_map_elites`: CVTArchive with IsoLineEmitter.
- `me_map_elites`: MAP-Elites with Bandit Scheduler.
- `cma_me_imp`: GridArchive with EvolutionStrategyEmitter using
  TwoStageImprovmentRanker.
- `cma_me_imp_mu`: GridArchive with EvolutionStrategyEmitter using
  TwoStageImprovmentRanker and mu selection rule.
- `cma_me_rd`: GridArchive with EvolutionStrategyEmitter using
  RandomDirectionRanker.
- `cma_me_rd_mu`: GridArchive with EvolutionStrategyEmitter using
  TwoStageRandomDirectionRanker and mu selection rule.
- `cma_me_opt`: GridArchive with EvolutionStrategyEmitter using ObjectiveRanker
  with mu selection rule.
- `cma_me_mixed`: GridArchive with EvolutionStrategyEmitter, where half (7) of
  the emitter are using TwoStageRandomDirectionRanker and half (8) are
  TwoStageImprovementRanker.
- `og_map_elites`: GridArchive with GradientOperatorEmitter, does not use
  measure gradients.
- `omg_mega`: GridArchive with GradientOperatorEmitter, uses measure gradients.
- `cma_mega`: GridArchive with GradientArborescenceEmitter.
- `cma_mega_adam`: GridArchive with GradientArborescenceEmitter using Adam
  Optimizer.
- `cma_mae`: GridArchive (learning_rate = 0.01) with EvolutionStrategyEmitter
  using ImprovementRanker.
- `cma_maega`: GridArchive (learning_rate = 0.01) with
  GradientArborescenceEmitter using ImprovementRanker.

The parameters for each algorithm are stored in CONFIG. 

Outputs are saved in the `evolve_rockets_output/` directory by default. The archive is
saved as a CSV named `{algorithm}_archive.csv`, while snapshots of the
heatmap are saved as `{algorithm}_heatmap_{iteration}.png`. Metrics about
the run are also saved in `{algorithm}_metrics.json`, and plots of the
metrics are saved in PNG's with the name `{algorithm}_metric_name.png`.

To generate a video of the heatmap from the heatmap images, use a tool like
ffmpeg. For example, the following will generate a 6FPS video showing the
heatmap for cma_me_imp with 20 dims.

    ffmpeg -r 6 -i "evolve_rockets_output/cma_me_imp_heatmap_%*.png" \
        evolve_rockets_output/cma_me_imp_heatmap_video.mp4

Usage (see evolve_main function for all args or run `python evolve_rockets.py --help`):
    python evolve_rockets.py ALGORITHM
Example:
    python evolve_rockets.py map_elites

    # To make numpy and sklearn run single-threaded, set env variables for BLAS
    # and OpenMP:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python evolve_rockets.py map_elites 20
Help:
    python evolve_rockets.py --help
"""
import os
import copy
import json
import time
from pathlib import Path

import fire
import numpy as np
import tqdm

from ribs.archives import CVTArchive, GridArchive
from ribs.emitters import (EvolutionStrategyEmitter, GaussianEmitter,
                           GradientArborescenceEmitter, GradientOperatorEmitter,
                           IsoLineEmitter)
from ribs.schedulers import BanditScheduler, Scheduler
from ribs.visualize import cvt_archive_heatmap, grid_archive_heatmap

# Might use to distribute evaluations later. See Pyribs Lunar Lander tutorial
#from dask.distributed import Client

import rocket_evaluate as re
from rocket_evaluate import MAX_FITNESS
import rocket_design as rd
from rocket_design import GENOME_LENGTH

import orhelper
from orhelper import FlightDataType, FlightEvent

MAX_GENOME_VALUE = 1.0
# Do not need bounds if sigmoid confines the values
BOUNDS = None #[(0.0,MAX_GENOME_VALUE)] * GENOME_LENGTH 
STARTING_SOLUTION = [MAX_GENOME_VALUE / 2.0] * GENOME_LENGTH

CONFIG = {
    "map_elites": {
        "use_result_archive": False,
        "is_dqd": False,
        "batch_size": 37,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "threshold_min": -np.inf
            }
        },
        "emitters": [{
            "class": GaussianEmitter,
            "kwargs": {
                "sigma": 0.5,
                "bounds" : BOUNDS
            },
            "num_emitters": 10
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "line_map_elites": {
        "use_result_archive": False,
        "is_dqd": False,
        "batch_size": 37,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "threshold_min": -np.inf
            }
        },
        "emitters": [{
            "class": IsoLineEmitter,
            "kwargs": {
                "iso_sigma": 0.1,
                "line_sigma": 0.2,
                "bounds" : BOUNDS
            },
            "num_emitters": 10
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "cvt_map_elites": {
        "use_result_archive": False,
        "is_dqd": False,
        "batch_size": 37,
        "archive": {
            "class": CVTArchive,
            "kwargs": {
                "cells": 10_000,
                "samples": 100_000,
                "use_kd_tree": True
            }
        },
        "emitters": [{
            "class": GaussianEmitter,
            "kwargs": {
                "sigma": 0.5,
                "bounds" : BOUNDS
            },
            "num_emitters": 10
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "line_cvt_map_elites": {
        "use_result_archive": False,
        "is_dqd": False,
        "batch_size": 37,
        "archive": {
            "class": CVTArchive,
            "kwargs": {
                "cells": 10_000,
                "samples": 100_000,
                "use_kd_tree": True
            }
        },
        "emitters": [{
            "class": IsoLineEmitter,
            "kwargs": {
                "iso_sigma": 0.1,
                "line_sigma": 0.2,
                "bounds" : BOUNDS
            },
            "num_emitters": 10
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "me_map_elites": {
        "use_result_archive": False,
        "is_dqd": False,
        "batch_size": 50,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "threshold_min": -np.inf
            }
        },
        "emitters": [{
            "class": EvolutionStrategyEmitter,
            "kwargs": {
                "sigma0": 0.5,
                "ranker": "obj",
                "bounds" : BOUNDS
            },
            "num_emitters": 7
        }, {
            "class": EvolutionStrategyEmitter,
            "kwargs": {
                "sigma0": 0.5,
                "ranker": "2rd",
                "bounds" : BOUNDS
            },
            "num_emitters": 7
        }, {
            "class": EvolutionStrategyEmitter,
            "kwargs": {
                "sigma0": 0.5,
                "ranker": "2imp",
                "bounds" : BOUNDS
            },
            "num_emitters": 7
        }, {
            "class": IsoLineEmitter,
            "kwargs": {
                "iso_sigma": 0.01,
                "line_sigma": 0.1,
                "bounds" : BOUNDS
            },
            "num_emitters": 7
        }],
        "scheduler": {
            "class": BanditScheduler,
            "kwargs": {
                "num_active": 12,
                "reselect": "terminated"
            }
        }
    },
    "cma_me_mixed": {
        "use_result_archive": False,
        "is_dqd": False,
        "batch_size": 37,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "threshold_min": -np.inf
            }
        },
        "emitters": [{
            "class": EvolutionStrategyEmitter,
            "kwargs": {
                "sigma0": 0.5,
                "ranker": "2rd",
                "bounds" : BOUNDS
            },
            "num_emitters": 5
        }, {
            "class": EvolutionStrategyEmitter,
            "kwargs": {
                "sigma0": 0.5,
                "ranker": "2imp",
                "bounds" : BOUNDS
            },
            "num_emitters": 5
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "cma_me_imp": {
        "use_result_archive": False,
        "is_dqd": False,
        "batch_size": 37,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "threshold_min": -np.inf
            }
        },
        "emitters": [{
            "class": EvolutionStrategyEmitter,
            "kwargs": {
                "sigma0": 0.5,
                "ranker": "2imp",
                "selection_rule": "filter",
                "restart_rule": "no_improvement",
                "bounds" : BOUNDS
            },
            "num_emitters": 10
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "cma_me_imp_mu": {
        "use_result_archive": False,
        "is_dqd": False,
        "batch_size": 37,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "threshold_min": -np.inf
            }
        },
        "emitters": [{
            "class": EvolutionStrategyEmitter,
            "kwargs": {
                "sigma0": 0.5,
                "ranker": "2imp",
                "selection_rule": "mu",
                "restart_rule": "no_improvement",
                "bounds" : BOUNDS
            },
            "num_emitters": 10
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "cma_me_rd": {
        "use_result_archive": False,
        "is_dqd": False,
        "batch_size": 37,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "threshold_min": -np.inf
            }
        },
        "emitters": [{
            "class": EvolutionStrategyEmitter,
            "kwargs": {
                "sigma0": 0.5,
                "ranker": "2rd",
                "selection_rule": "filter",
                "restart_rule": "no_improvement",
                "bounds" : BOUNDS
            },
            "num_emitters": 10
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "cma_me_rd_mu": {
        "use_result_archive": False,
        "is_dqd": False,
        "batch_size": 37,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "threshold_min": -np.inf
            }
        },
        "emitters": [{
            "class": EvolutionStrategyEmitter,
            "kwargs": {
                "sigma0": 0.5,
                "ranker": "2rd",
                "selection_rule": "mu",
                "restart_rule": "no_improvement",
                "bounds" : BOUNDS
            },
            "num_emitters": 10
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "cma_me_opt": {
        "use_result_archive": False,
        "is_dqd": False,
        "batch_size": 37,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "threshold_min": -np.inf
            }
        },
        "emitters": [{
            "class": EvolutionStrategyEmitter,
            "kwargs": {
                "sigma0": 0.5,
                "ranker": "obj",
                "selection_rule": "mu",
                "restart_rule": "basic",
                "bounds" : BOUNDS
            },
            "num_emitters": 10
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "og_map_elites": {
        "use_result_archive": False,
        "is_dqd": True,
        # Divide by 2 since half of the 36 solutions are used in ask_dqd(), and
        # the other half are used in ask().
        "batch_size": 36 // 2,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "threshold_min": -np.inf
            }
        },
        "emitters": [{
            "class": GradientOperatorEmitter,
            "kwargs": {
                "sigma": 0.5,
                "sigma_g": 0.5,
                "measure_gradients": False,
                "normalize_grad": False,
                "bounds" : BOUNDS
            },
            "num_emitters": 1
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "omg_mega": {
        "use_result_archive": False,
        "is_dqd": True,
        # Divide by 2 since half of the 36 solutions are used in ask_dqd(), and
        # the other half are used in ask().
        "batch_size": 36 // 2,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "threshold_min": -np.inf
            }
        },
        "emitters": [{
            "class": GradientOperatorEmitter,
            "kwargs": {
                "sigma": 0.0,
                "sigma_g": 10.0,
                "measure_gradients": True,
                "normalize_grad": True,
                "bounds" : BOUNDS
            },
            "num_emitters": 1
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "cma_mega": {
        "use_result_archive": False,
        "is_dqd": True,
        "batch_size": 35,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "threshold_min": -np.inf
            }
        },
        "emitters": [{
            "class": GradientArborescenceEmitter,
            "kwargs": {
                "sigma0": 0.5, #10.0,
                "lr": 1.0,
                "grad_opt": "gradient_ascent",
                "selection_rule": "mu",
                "bounds" : BOUNDS
            },
            "num_emitters": 1
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "cma_mega_adam": {
        "use_result_archive": False,
        "is_dqd": True,
        "batch_size": 35,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "threshold_min": -np.inf
            }
        },
        "emitters": [{
            "class": GradientArborescenceEmitter,
            "kwargs": {
                "sigma0": 0.5, #10.0,
                "lr": 0.002,
                "grad_opt": "adam",
                "selection_rule": "mu",
                "bounds" : BOUNDS
            },
            "num_emitters": 1
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "cma_mae": {
        "use_result_archive": True,
        "is_dqd": False,
        "batch_size": 36,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "threshold_min": 0,
                "learning_rate": 0.01
            }
        },
        "emitters": [{
            "class": EvolutionStrategyEmitter,
            "kwargs": {
                "sigma0": 0.5,
                "ranker": "imp",
                "selection_rule": "mu",
                "restart_rule": "basic",
                "bounds" : BOUNDS
            },
            "num_emitters": 10
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "cma_maega": {
        "use_result_archive": True,
        "is_dqd": True,
        "batch_size": 35,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "threshold_min": 0,
                "learning_rate": 0.01
            }
        },
        "emitters": [{
            "class": GradientArborescenceEmitter,
            "kwargs": {
                "sigma0": 0.5, #10.0,
                "lr": 1.0,
                "ranker": "imp",
                "grad_opt": "gradient_ascent",
                "restart_rule": "basic",
                "bounds" : BOUNDS
            },
            "num_emitters": 10
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    }
}

def sigmoid(arr):
    return 1/(1 + np.exp(-arr))

def evaluate_rocket_genome(genome):
    """
        Evaluate a rocket, then return fitness and BC/measure information

        genome -- numbers that are mapped to rocket design parameters

        Return: (fitness, stability, max altitude, nose type)
    """
    global sim
    global opts
    global orh
    global doc
    rocket = opts.getRocket()
    # Confine to [0,1] by scaling
    #squeezed_genome = list(map(lambda x : x / MAX_GENOME_VALUE, genome))
    # Confine to (0,1) with sigmoid function
    squeezed_genome = sigmoid(genome)
    rd.apply_genome_to_rocket(orh, rocket, squeezed_genome)
    return re.simulate_rocket(orh, sim, opts, doc)

def evolve_rockets(solution_batch):
    """Sphere function evaluation and measures for a batch of solutions.

    Args:
        solution_batch (np.ndarray): (batch_size, dim) batch of solutions.
    Returns:
        objective_batch (np.ndarray): (batch_size,) batch of objectives.
        objective_grad_batch (np.ndarray): (batch_size, solution_dim) batch of
            objective gradients.
        measures_batch (np.ndarray): (batch_size, 2) batch of measures.
        measures_grad_batch (np.ndarray): (batch_size, 2, solution_dim) batch of
            measure gradients.
    """
    dim = solution_batch.shape[1]

    # These are made up objectives and BCs. Replace with real later
    results = map(evaluate_rocket_genome, solution_batch)

    # Was for debugging
    #results = list()
    #for genome in solution_batch:
    #    print("Genome")
    #    print(genome)
    #    results.append(evaluate_rocket_genome(genome))
    #    print(results)
    #    input("iteration")

    # Collect the objectives and measures in a manner similar to the Lunar Lander example
    objective_batch = [] 
    measures_batch = []

    global global_bin_model
    global MAX_STABILITY
    global MIN_STABILITY
    global BUFFER

    for obj, stability, altitude, nose_type in results:

        # Punish low stability: Cody says stability should be at least 1.0
        if stability < MIN_STABILITY:
            obj = 0

        objective_batch.append(obj)
        if global_bin_model == "stability_altitude":
            measures_batch.append([stability, altitude])
        elif global_bin_model == "stabilitynose_altitude":
            nose_index = rd.nose_type_index(nose_type)
            if stability < MIN_STABILITY:
                stability = MIN_STABILITY # Not strictly true, but makes the mapping correct
            elif stability > MAX_STABILITY:
                stability = MAX_STABILITY # Also inaccurate, but less likely to cause issues
            stabilitynose = (nose_index * (BUFFER + (MAX_STABILITY - MIN_STABILITY))) + (stability - MIN_STABILITY)
            measures_batch.append([stabilitynose, altitude])
        #elif global_bin_model == "stability_nose_altitude":
        #    nose_index = rd.nose_type_index(nose_type)
        #    measures_batch.append([stability, nose_index, altitude])
        else:
            print("global_bin_model not valid:", global_bin_model)
            quit()

    # I have no idea how to compute gradients for a problem like this, if it is even possible
    objective_grad_batch = None 
    measures_grad_batch = None 

    return (
        objective_batch,
        objective_grad_batch,
        measures_batch,
        measures_grad_batch,
    )


def create_scheduler(config, algorithm, seed=None):
    """Creates a scheduler based on the algorithm.

    Args:
        config (dict): Configuration dictionary with parameters for the various
            components.
        algorithm (string): Name of the algorithm.
        seed (int): Main seed for the various components.
    Returns:
        ribs.schedulers.Scheduler: A ribs scheduler for running the algorithm.
    """
    solution_dim = GENOME_LENGTH
    learning_rate = 1.0 if "learning_rate" not in config["archive"]["kwargs"] else config["archive"]["kwargs"]["learning_rate"]
    use_result_archive = config["use_result_archive"]
    
    # These are guesses that go beyond what I expect is reasonable
    global MAX_STABILITY
    global MIN_STABILITY
    global MAX_NOSE_TYPE_INDEX
    global BUFFER

    BUFFER = 0.5 # Create space between nose cone types in 3D archived mapped to 2D

    MIN_STABILITY = 1.0
    MAX_STABILITY = 3.0
    MIN_ALTITUDE = 0.0
    MAX_ALTITUDE = 110.0 
    MIN_NOSE_TYPE_INDEX = 0
    MAX_NOSE_TYPE_INDEX = 5

    global global_bin_model
    global_bin_model = config["bin_model"]

    if config["bin_model"] == "stability_altitude":
        archive_dims = (100,100) # Hard-coding archive size 
        bounds = [(MIN_STABILITY, MAX_STABILITY), (MIN_ALTITUDE, MAX_ALTITUDE)]
    elif config["bin_model"] == "stabilitynose_altitude":
        archive_dims = (100,100) # Hard-coding archive size 
        bounds = [(0, ( (BUFFER + (MAX_STABILITY - MIN_STABILITY)) * (MAX_NOSE_TYPE_INDEX+1)) ), (MIN_ALTITUDE, MAX_ALTITUDE)]
    #elif config["bin_model"] == "stability_nose_altitude":
    #    archive_dims = (100,6,100) # Hard-coding archive size 
    #    bounds = [(MIN_STABILITY, MAX_STABILITY), (MIN_NOSE_TYPE_INDEX, MAX_NOSE_TYPE_INDEX), (MIN_ALTITUDE, MAX_ALTITUDE)]
    else:
        print("Invalid bin_model:", config["bin_model"])
        quit()

    initial_sol = np.array(STARTING_SOLUTION)
    mode = "batch"

    # Create archive.
    archive_class = config["archive"]["class"]
    if archive_class == GridArchive:
        archive = archive_class(solution_dim=solution_dim,
                                ranges=bounds,
                                dims=archive_dims,
                                seed=seed,
                                **config["archive"]["kwargs"])
    else:
        archive = archive_class(solution_dim=solution_dim,
                                ranges=bounds,
                                seed=seed,
                                **config["archive"]["kwargs"])

    # Create result archive.
    result_archive = None
    if use_result_archive:
        result_archive = GridArchive(solution_dim=solution_dim,
                                     dims=archive_dims,
                                     ranges=bounds,
                                     seed=seed)

    # Create emitters. Each emitter needs a different seed so that they do not
    # all do the same thing, hence we create an rng here to generate seeds. The
    # rng may be seeded with None or with a user-provided seed.
    seed_sequence = np.random.SeedSequence(seed)
    emitters = []
    for e in config["emitters"]:
        emitter_class = e["class"]
        emitters += [
            emitter_class(
                archive,
                x0=initial_sol,
                **e["kwargs"],
                batch_size=config["batch_size"],
                seed=s,
            ) for s in seed_sequence.spawn(e["num_emitters"])
        ]

    # Create Scheduler
    scheduler_class = config["scheduler"]["class"]
    scheduler = scheduler_class(archive,
                                emitters,
                                result_archive=result_archive,
                                add_mode=mode,
                                **config["scheduler"]["kwargs"])
    scheduler_name = scheduler.__class__.__name__

    print(f"Create {scheduler_name} for {algorithm} with learning rate "
          f"{learning_rate} and add mode {mode}, using solution dim "
          f"{solution_dim}, archive dims {archive_dims}, and "
          f"{len(emitters)} emitters.")
    return scheduler

def save_heatmap(plt, archive, heatmap_path):
    """Saves a heatmap of the archive to the given path.

    Args:
        archive (GridArchive or CVTArchive): The archive to save.
        heatmap_path: Image path for the heatmap.
    """
    if isinstance(archive, GridArchive):
        plt.figure(figsize=(8, 6))
        grid_archive_heatmap(archive, vmin=0, vmax=MAX_FITNESS)
        plt.tight_layout()
        plt.savefig(heatmap_path)
    elif isinstance(archive, CVTArchive):
        plt.figure(figsize=(16, 12))
        cvt_archive_heatmap(archive, vmin=0, vmax=MAX_FITNESS)
        plt.tight_layout()
        plt.savefig(heatmap_path)
    plt.close(plt.gcf())


def evolve_rockets_main(algorithm,
                run_num=0,
                bin_model="stability_altitude",
                itrs=300,
                learning_rate=None,
                es=None,
                outdir="evolve_rockets_output",
                log_freq=20,
                seed=None):
    """Evolve model rockets with Open Rocket.

    Args:
        algorithm (str): Name of the algorithm.
        itrs (int): Iterations to run.
        learning_rate (float): The archive learning rate.
        es (str): If passed, this will set the ES for all
            EvolutionStrategyEmitter instances.
        outdir (str): Directory to save output.
        log_freq (int): Number of iterations to wait before recording metrics
            and saving heatmap.
        seed (int): Seed for the algorithm. By default, there is no seed.
    """

    # Adding matplotlib imports here to avoid problems when combined with JPype
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    config = copy.deepcopy(CONFIG[algorithm])

    # Use default itrs for each algorithm.
    if itrs is not None:
        config["iters"] = itrs

    # Use default learning_rate for each algorithm.
    if learning_rate is not None:
        config["archive"]["kwargs"]["learning_rate"] = learning_rate

    # Set ES for all EvolutionStrategyEmitter.
    if es is not None:
        for e in config["emitters"]:
            if e["class"] == EvolutionStrategyEmitter:
                e["kwargs"]["es"] = es

    name = f"{algorithm}_{bin_model}_{run_num}"
    if es is not None:
        name += f"_{es}"
    outdir = Path(outdir)
    if not outdir.is_dir():
        outdir.mkdir()

    # Allow different bin models
    config["bin_model"] = bin_model
    # Default is "stability_altitude" with fitness of altitude consistency
    # Other options:
    # "stabilitynose_altitude" with fitness of altitude consistency

    scheduler = create_scheduler(config, algorithm, seed=seed)
    result_archive = scheduler.result_archive
    is_dqd = config["is_dqd"]
    itrs = config["iters"]
    metrics = {
        "QD Score": {
            "x": [0],
            "y": [0.0],
        },
        "Archive Coverage": {
            "x": [0],
            "y": [0.0],
        },
    }

    #print("TEST")
    #test_batch = scheduler.ask()
    #print(evaluate_rocket_genome(test_batch[0]))
    #input("done numpy 1")
    #print(evaluate_rocket_genome(test_batch[1]))
    #input("done numpy 2")

    non_logging_time = 0.0
    save_heatmap(plt, result_archive, str(outdir / f"{name}_heatmap_{0:05d}.png"))

    for itr in tqdm.trange(1, itrs + 1):
        itr_start = time.time()

        if is_dqd:
            solution_batch = scheduler.ask_dqd()
            (objective_batch, objective_grad_batch, measures_batch,
             measures_grad_batch) = evolve_rockets(solution_batch)
            objective_grad_batch = np.expand_dims(objective_grad_batch, axis=1)
            jacobian_batch = np.concatenate(
                (objective_grad_batch, measures_grad_batch), axis=1)
            scheduler.tell_dqd(objective_batch, measures_batch, jacobian_batch)

        solution_batch = scheduler.ask()
        objective_batch, _, measure_batch, _ = evolve_rockets(solution_batch)
        scheduler.tell(objective_batch, measure_batch)
        non_logging_time += time.time() - itr_start

        # Logging and output.
        final_itr = itr == itrs
        if itr % log_freq == 0 or final_itr:
            if final_itr:
                result_archive.data(return_type="pandas").to_csv(
                    outdir / f"{name}_archive.csv")

            # Record and display metrics.
            metrics["QD Score"]["x"].append(itr)
            metrics["QD Score"]["y"].append(result_archive.stats.qd_score)
            metrics["Archive Coverage"]["x"].append(itr)
            metrics["Archive Coverage"]["y"].append(
                result_archive.stats.coverage)
            tqdm.tqdm.write(
                f"Iteration {itr} | Archive Coverage: "
                f"{metrics['Archive Coverage']['y'][-1] * 100:.3f}% "
                f"QD Score: {metrics['QD Score']['y'][-1]:.3f}")

            save_heatmap(plt, result_archive,
                         str(outdir / f"{name}_heatmap_{itr:05d}.png"))

    # Plot metrics.
    print(f"Algorithm Time (Excludes Logging and Setup): {non_logging_time}s")
    for metric, values in metrics.items():
        plt.plot(values["x"], values["y"])
        plt.title(metric)
        plt.xlabel("Iteration")
        plt.savefig(
            str(outdir / f"{name}_{metric.lower().replace(' ', '_')}.png"))
        plt.clf()
    with (outdir / f"{name}_metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)


if __name__ == '__main__':
    with orhelper.OpenRocketInstance() as instance:
        global orh
        global sim
        global opts
        global rocket
        global doc
    
        #from net.sf.openrocket.preset import ComponentPreset
        #body_tube_presets = instance.preset_loader.getDatabase().listForType(ComponentPreset.Type.BODY_TUBE)

        orh = orhelper.Helper(instance)
        doc = orh.load_doc(os.path.join('examples', 'base_15.03.ork')) # File was modified to replace Trapezoidal fin set with Freeform fin set
        sim = doc.getSimulation(0)
        opts = sim.getOptions()
        rocket = opts.getRocket()

        re.prepare_for_rocket_simulation(sim) # Sets some global variables for rocket evaluation
        nose = orh.get_component_named(rocket, 'Nose cone')
        rd.define_nose_types(nose)
        #rd.define_body_tube_presets(body_tube_presets)

        #import random
        #genome = list()
        #for _ in range(GENOME_LENGTH):
        #    genome.append(random.random())
        #print(evaluate_rocket_genome(genome))
        #input("done 1")
        #genome = list()
        #for _ in range(GENOME_LENGTH):
        #    genome.append(random.random())
        #print(evaluate_rocket_genome(genome))
        #input("done 2")

        fire.Fire(evolve_rockets_main)
