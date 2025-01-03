
from orhelper import FlightDataType, FlightEvent
import numpy as np
import statistics

import sys
import csv
import os
import math

# To test the robustness of the rocket design, it is evaluated at varying wind speeds
WIND_SPEED_MIN = 2.0   # m/s 
WIND_SPEED_MAX = 6.7   # m/s (6.7 m/s = 14.988 mph)
# NOTE: The MAX values shown here (above and below) are misleading.
#       I meant for these to be the final values tested at, but because
#       the increments are defined by dividing by NUM_ROCKET_EVALS,
#       the max values tested at are actually one increment short of
#       the maxes I intended. These settings are left as-is to remain
#       consistent with experiments done for publication.
WIND_DEVIATION_MIN = 0.2  # m/s
WIND_DEVIATION_MAX = 4.0  # m/s

NUM_ROCKET_EVALS = 3

MAX_FITNESS = 40 # Higher?

DEBUG = False

def prepare_for_rocket_simulation(sim):
    # Once the instance starts, Java classes can be imported using JPype
    from net.sf.openrocket.masscalc import BasicMassCalculator
    from net.sf.openrocket.masscalc import MassCalculator
    from net.sf.openrocket.aerodynamics import WarningSet
    from net.sf.openrocket.aerodynamics import BarrowmanCalculator
    from net.sf.openrocket.aerodynamics import FlightConditions

    global conf
    global bmc
    global mct
    global conds
    global warnings
    global adc

    opts = sim.getOptions()
    rocket = opts.getRocket()
    conf = rocket.getDefaultConfiguration() # This config seems to give CP results that match OpenRocket GUI more often 
    #conf = sim.getConfiguration() # Or is this the right configuration?
    bmc = BasicMassCalculator()
    mct = MassCalculator.MassCalcType.LAUNCH_MASS

    # Look at updateExtras under https://github.com/openrocket/openrocket/blob/7a9bb436c1ed91335a396693eccfc400ffe236d2/swing/src/main/java/info/openrocket/swing/gui/scalefigure/RocketPanel.java
    adc = BarrowmanCalculator()
    conds = FlightConditions(conf)
    #conds.setMach(cpMach) # how do I get cpMach?
    #conds.setAOA(cpAOA) # how do I get cpAOA?
    #conds.setRollRate(cpRoll) # how do I get cpRoll?
    #conds.setTheta(cpTheta) # how do I get cpTheta?
    warnings = WarningSet()
    warnings.clear()

    if DEBUG:
        print("Mach:", conds.getMach())
        print("AOA:", conds.getAOA())
        print("RollRate:", conds.getRollRate())
        print("Theta:", conds.getTheta())

def simulate_rocket(orh, sim, opts, doc, plt = None):
    """
        Simulate the rocket and return performance information.

        orh     -- Open Rocket helper object
        sim     -- Rocket simulation object
        opts    -- Simulation options
        plt     -- matplotlib instance for plotting. Do not do plotting if this is None.
    """
    global conf
    global bmc
    global mct
    global conds
    global warnings
    global adc

    # Need to know diameter
    rocket = opts.getRocket()
    nose = orh.get_component_named(rocket, 'Nose cone')
    diameter = nose.getAftRadius() * 2

    try:
        # For BC/measures
        cg = bmc.getCG(conf, mct).x
        cp = adc.getCP(conf, conds, warnings).x
    except Exception as e:
        print("Error in CG/CP calculation")
        print("Save error.ork")
        orh.save_doc("error.ork", doc)
        raise e

    stability = (cp - cg) / diameter

    plot_data = list()

    apogees = list()
    wind_speed_increment = (WIND_SPEED_MAX - WIND_SPEED_MIN) / NUM_ROCKET_EVALS
    wind_deviation_increment = (WIND_DEVIATION_MAX - WIND_DEVIATION_MIN) / NUM_ROCKET_EVALS
    wind_speed = WIND_SPEED_MIN
    wind_deviation = WIND_DEVIATION_MIN
    # Evaluate the rocket with different wind conditions and take the average altitude.
    # The fitness goal is to minimize the variance in max altitude.
    for i in range(NUM_ROCKET_EVALS):

        if DEBUG:
            print(f"Wind speed: {wind_speed}")
            print(f"Wind dev  : {wind_deviation}")

        opts.setWindSpeedAverage(wind_speed)
        opts.setWindSpeedDeviation(wind_deviation)
        
        wind_speed += wind_speed_increment
        wind_deviation += wind_deviation_increment

        try:
            orh.run_simulation(sim)
        except Exception as e:
            # This will hopefully never happen in the final version of the code,
            # but I want to see exception details any time that it does.
            print("EXCEPTION")
            e.printStackTrace()
            raise e # Still crash

        data = orh.get_timeseries(sim, [FlightDataType.TYPE_TIME, FlightDataType.TYPE_ALTITUDE, FlightDataType.TYPE_VELOCITY_Z])
        if plt != None: plot_data.append(data)
        events = orh.get_events(sim) 

        events_to_annotate = {
            FlightEvent.APOGEE: 'Apogee' 
        }

        apogee = 0.0 # Default
        index_at = lambda t: (np.abs(data[FlightDataType.TYPE_TIME] - t)).argmin()
        for event, times in events.items():
            if event not in events_to_annotate:
                continue
            for time in times:
                apogee = data[FlightDataType.TYPE_ALTITUDE][index_at(time)]

        apogees.append(apogee)

        if DEBUG: print("Apogee:",apogee)

    average_apogee = statistics.mean(apogees)
    apogee_stdev = statistics.pstdev(apogees)

    if DEBUG:
        print("Average Apogee:",average_apogee)
        print("CG:",cg)
        print("CP:",cp)
        print("Stability:",stability)

    # If this script is being used to evaluate previously evolved rockets, then plot details
    if plt != None:

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        colors = ['b','r','g','c','m','y','tab:orange', 'tab:purple', 'tab:brown', 'tab:pink']

        for i in range(len(plot_data)):
            ax1.plot(plot_data[i][FlightDataType.TYPE_TIME], plot_data[i][FlightDataType.TYPE_ALTITUDE], colors[i % len(colors)])
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Altitude (m)')
        ax1.grid(True)
        plt.show()

    # Fitness is max minus the standard deviation in the max altitude attained.
    # (fitness based on deviation of apogee, stability score, average apogee, type of the nose cone (Java object))
    return (MAX_FITNESS - apogee_stdev, stability, average_apogee, nose.getType())


def row_info(row):
    """
        Get the data from a row separated out into three meaningful pieces of data.
        Parameters:
        row - list of numbers from one row of the file
        Return:
        tuple of (genome list, measure score list, objective score)
    """
    num_measures = 2
    end_count = 1 + num_measures + 1 + 1 # objective, measures, threshold, index
    # Convert the row contents (excluding the first element) to floats
    row_data = [float(item) for item in row[1:]]
    data_len = len(row_data)
    genome = row_data[:(data_len - end_count)]
    objective = row_data[len(genome)]
    measures = row_data[len(genome)+1:(data_len - 2)]
    return (genome, measures, objective)

def extract_row(filename, row_number):
    with open(filename, mode='r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Skip the header row
        for i, row in enumerate(csv_reader):
            if i == row_number:
                return row_info(row)
    return None  # Return None if the row number is out of bounds

def all_rows(filename):
    with open(filename, mode='r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Skip the header row
        rows = []
        for i, row in enumerate(csv_reader):
            (genome,measures,objective) = row_info(row)
            rows.append((i, genome, measures, objective))
        return rows

def highest_stable_fliers(rows, top_count, min_stability, by_nose_type):
    """
        From the rows, filter out rows whose stability value is too low,
        and then from those get the ones with the highest altitudes.

        Parameters:
        rows - data from the saved CSV archive
        top_count - number of top rows to keep
        min_stability - min stability required of the top results
        by_nose_type - whether to split up results by nose type
    """
    # Measures are index 2, and index 0 is either the stability, or a combination of stability and nose type
    if by_nose_type:
        # min_stability does not matter if nose type is factored in, since those archives
        # have their stability/nose values artificially changed to fit into the right bin

        BUFFER = 0.5 # This value is also defined in evolve_rockets.py. Should have one shared value rather than double definition
        MIN_STABILITY = 1.0  # Also doubly defined
        MAX_STABILITY = 3.0  # Also doubly defined

        global NUM_NOSE_TYPES

        range_per_nose_type = (MAX_STABILITY - MIN_STABILITY) + BUFFER
        # There are 6 nose types, and the bottom of each range contains rockets with either the min stability or less.
        # Therefore, any value that is a multiple of range_per_nose_type should be excluded
        epsilon = 0.0001 # To check for approximate floating point equality

        # r[2][0] is the combined stability and nose type value.
        filtered = list(filter(lambda r : abs( (r[2][0]/range_per_nose_type) - int(r[2][0]/range_per_nose_type) ) > epsilon, rows))

        # Change count to allow a few extra, so each nose type is equally represented
        count_per_nose = math.ceil(top_count / NUM_NOSE_TYPES)
       
        top = list()
        for t in range(NUM_NOSE_TYPES):
            lower_bound = t*range_per_nose_type
            upper_bound = lower_bound + (MAX_STABILITY - MIN_STABILITY)

            for_this_nose = filter(lambda r : r[2][0] > lower_bound and r[2][0] <= upper_bound, filtered)
            top_for_nose = sorted(for_this_nose, key=lambda r : -r[2][1])
            
            top = top + top_for_nose[:count_per_nose]

        return top
    else:
        filtered = filter(lambda r : r[2][0] >= min_stability, rows)
        # Measures are index 2, altitude is the measure 1, and negating will sort in descending order
        top = sorted(filtered, key=lambda r : -r[2][1])

        return top[:top_count]

def sigmoid(arr):
    return 1/(1 + np.exp(-arr))

if __name__ == "__main__":
    if len(sys.argv) != 3 and len(sys.argv) != 4 and len(sys.argv) != 2 and len(sys.argv) != 5:
        print("Improper usage. Should be:")
        print("python rocket_evaluate.py <archive file> <row>")
        print("Example:")
        print("python rocket_evaluate.py evolve_rockets_output/map_elites_archive.csv 3")
        print("Optionally, specify an .ork file to save")
        print("python rocket_evaluate.py evolve_rockets_output/map_elites_archive.csv 3 evolved.ork")
        print("Can also skip simulation and only save the .ork file")
        print("python rocket_evaluate.py evolve_rockets_output/map_elites_archive.csv 3 evolved.ork skip")
        print("Can also list top 10 altitude values with this:")
        print("python rocket_evaluate.py <archive file>")
    else:
        DEBUG = True
        global NUM_NOSE_TYPES
        NUM_NOSE_TYPES = 6

        filename = sys.argv[1]

        by_nose_type = "nose" in filename

        # Print top altitude rockets
        top_count = 12
        rows = all_rows(filename)
        top_rows = highest_stable_fliers(rows, top_count, 1.0, by_nose_type) # requires min stability of 1.0
        print("TOP")
        if by_nose_type:
            print(top_count / NUM_NOSE_TYPES, "per nose type")
        for (index, genome, measures, objective) in top_rows:
            print("Index:",index, "Objective:", objective, "Measures:", measures)
        print("END")

        if len(sys.argv) == 2: exit(0)

        row_number = int(sys.argv[2])
        save_file = None
        if len(sys.argv) >= 4:
            save_file = sys.argv[3] # An .ork file to save

        skip = len(sys.argv) == 5 and sys.argv[4] == "skip"

        row_data = extract_row(filename, row_number)
        (genome, measures, objective) = row_data

        print("Genome:",genome)
        print("Stability (and Nose?):",measures[0])
        print("Altitude:",measures[1])
        print("Consistency:",objective)

        import rocket_design as rd
        rd.DEBUG = True
        from rocket_design import GENOME_LENGTH

        import orhelper
        from orhelper import FlightDataType, FlightEvent

        import matplotlib
        import matplotlib.pyplot as plt

        with orhelper.OpenRocketInstance() as instance:
            global orh
            global sim
            global opts
            global rocket

            #from net.sf.openrocket.preset import ComponentPreset
            #body_tube_presets = instance.preset_loader.getDatabase().listForType(ComponentPreset.Type.BODY_TUBE)

            orh = orhelper.Helper(instance)
            doc = orh.load_doc(os.path.join('examples', 'base_15.03.ork')) # File was modified to replace Trapezoidal fin set with Freeform fin set
            sim = doc.getSimulation(0)
            opts = sim.getOptions()
            rocket = opts.getRocket()

            prepare_for_rocket_simulation(sim) # Sets some global variables for rocket evaluation
            nose = orh.get_component_named(rocket, 'Nose cone')
            rd.define_nose_types(nose)
            #rd.define_body_tube_presets(body_tube_presets)

            rocket = opts.getRocket()
            squeezed_genome = sigmoid(np.array(genome))
            rd.apply_genome_to_rocket(orh, rocket, squeezed_genome)

            if not skip:
                result = simulate_rocket(orh, sim, opts, doc, plt)
                print(result)

            if save_file:
                alt = measures[1]
                nose_type = nose.getType()
                full_name = f"Alt_{alt}_Nose_{nose_type}_{save_file}"
                orh.save_doc(full_name, doc)
                print("Saved file:", full_name)
