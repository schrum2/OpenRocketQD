import os

import numpy as np
from matplotlib import pyplot as plt

import orhelper
from orhelper import FlightDataType, FlightEvent

import random
import math

GENOME_INDEX_NOSE_AFT_RADIUS = 0
GENOME_INDEX_NOSE_LENGTH = 1
GENOME_INDEX_NOSE_TYPE = 2
GENOME_INDEX_NOSE_SHAPE = 3
GENOME_INDEX_NOSE_THICKNESS = 4
GENOME_INDEX_BODY_LENGTH = 5
GENOME_INDEX_FIN_COUNT = 6
GENOME_INDEX_FIN_POINT1_X = 7
GENOME_INDEX_FIN_POINT1_Y = 8
GENOME_INDEX_FIN_POINT2_X = 9
GENOME_INDEX_FIN_POINT2_Y = 10
GENOME_INDEX_FIN_POINT3_X = 11

# The number range appropriate for each element of genome
scales = [(0.01, 0.04), # Aft radius
          (0.1, 1.0),   # Nose length
          (0,5),        # Nose type: [0 = nose.Shape.OGIVE,1 = nose.Shape.CONICAL,2 = nose.Shape.ELLIPSOID,3 = nose.Shape.POWER,4 = nose.Shape.PARABOLIC,5 = nose.Shape.HAACK]
          (0.0,1.0),    # Nose shape (only affects some types)
          (0.001,0.009),# Nose thickness
          (0.2,1.0),    # Body length
          (2,5),        # Fin count (integer)
          (0.0,0.3),    # Fin point 1 x-coordinate
          (0.0,0.3),    # Fin point 1 y-coordinate
          (0.0,0.3),    # Fin point 2 x-coordinate
          (0.0,0.3),    # Fin point 2 y-coordinate
          (0.0,0.3)     # Fin point 3 x-coordinate (final y-coordinate must be 0.0)
         ]
# Note: might want to generalize to more fin points later

def random_genome(count):
    """
        Random genome of values from 0.0 to 1.0.

        count -- number of elements in genome

        Return: list of random elements representing the genome
    """

    genome = list()
    for _ in range(count):
        genome.append(random.random())

    return genome

def decode_genome_element_scale(scales, genome, index):
    """
        Convert specific genome element to its properly scaled value.

        scales -- 2-tuples with (min,max) pairs
        genome -- evolved genome of values in [0,1]
        index  -- an index that associates a scale with a genome value

        Return: value from genome index scaled according to same index in scales
    """
    bounds = scales[index]
    lo = bounds[0]
    hi = bounds[1]
    value = genome[index]
    return lo + value * (hi - lo)

def decode_genome_element_discrete(scales, genome, index):
    """
        Convert specific genome element to scaled discrete integer value

        scales -- 2-tuples with (min,max) pairs
        genome -- evolved genome of values in [0,1]
        index  -- an index that associates a scale with a genome value

        Return: value from genome index scaled and rounded down to an int according to same index in scales
    """
    scaled = decode_genome_element_scale(scales, genome, index)
    return int(math.floor(scaled))

def decode_genome_element_nose_type(scales, genome, index):
    """
        Convert specific genome element to a particular nose code type

        scales -- 2-tuples with (min,max) pairs
        genome -- evolved genome of values in [0,1]
        index  -- an index that associates a scale with a genome value

        Return: value from genome index converted to nose cone type according to same index in scales
    """
    type_index = decode_genome_element_discrete(scales, genome, index)
    return NOSE_TYPES[type_index] 

def decode_genome_element_coordinate(scales, genome, x_index, y_index):
    """
        Convert two specific genome elements to a Coordinate point.

        scales -- 2-tuples with (min,max) pairs
        genome -- evolved genome of values in [0,1]
        x_index-- an index that associates a scale with a genome value for an x-coordinate
        y_index-- an index that associates a scale with a genome value for a y-coordinate

        Return: Coordinate resulting from scaling the two genome values to x and y coordinates
    """
    x = decode_genome_element_scale(scales, genome, x_index)
    y = decode_genome_element_scale(scales, genome, y_index)
    return Coordinate(x, y, 0.0) # Coordinates are 3D even when only (x,y) are used

def apply_genome_to_rocket(rocket, genonme):
    """
        Interpret every element of the evolved genome as a parameter for
        the rocket, and apply each parameter setting. There is no return value,
        as the rocket object is modified via side effects.

        rocket -- rocket from Open Rocket that is modified according to the genome
        genome -- evolved genome of values in [0,1] which are interpreted as parameters for rocket design
    """
    nose = orh.get_component_named(rocket, 'Nose cone')
    body = orh.get_component_named(rocket, 'Body tube')
    fins = orh.get_component_named(body, 'Freeform fin set')

    nose.setAftRadius(decode_genome_element_scale(scales, genome, GENOME_INDEX_NOSE_AFT_RADIUS))
    nose.setLength(decode_genome_element_scale(scales, genome, GENOME_INDEX_NOSE_LENGTH))
    nose.setType(decode_genome_element_nose_type(scales, genome, GENOME_INDEX_NOSE_TYPE))
    nose.setShapeParameter(decode_genome_element_scale(scales, genome, GENOME_INDEX_NOSE_SHAPE))
    nose.setThickness(decode_genome_element_scale(scales, genome, GENOME_INDEX_NOSE_THICKNESS))

    body.setLength(decode_genome_element_scale(scales, genome, GENOME_INDEX_BODY_LENGTH))

    fins.setFinCount(decode_genome_element_discrete(scales, genome, GENOME_INDEX_FIN_COUNT))

    fin_points = list()
    fin_points.append( Coordinate(0.0,0.0,0.0) ) # Always start at (0,0,0)
    fin_points.append( decode_genome_element_coordinate(scales, genome, GENOME_INDEX_FIN_POINT1_X, GENOME_INDEX_FIN_POINT1_Y) )
    fin_points.append( decode_genome_element_coordinate(scales, genome, GENOME_INDEX_FIN_POINT2_X, GENOME_INDEX_FIN_POINT2_Y) )
    fin_points.append( Coordinate( decode_genome_element_scale(scales, genome, GENOME_INDEX_FIN_POINT3_X), 0.0, 0.0) ) # Last y-coordinate must be 0

    #print("---------------")
    #for p in fin_points: print(p)

    try:
        fins.setPoints(fin_points)
    except:
        fins.setPoints( [Coordinate(0.0,0.0,0.0), Coordinate(0.025,0.030,0.000), Coordinate(0.075,0.030,0.000), Coordinate(0.05, 0.0, 0.0)] )
        print("Fin point failure: default trapezoid fins")
    
def prepare_for_rocket_simulation():
    #from net.sf.openrocket.util import Coordinate # Once the instance starts, Java classes can be imported using JPype
    #from net.sf.openrocket.masscalc import BasicMassCalculator
    #from net.sf.openrocket.masscalc import MassCalculator
    #from net.sf.openrocket.aerodynamics import WarningSet
    #from net.sf.openrocket.aerodynamics import BarrowmanCalculator
    #from net.sf.openrocket.aerodynamics import FlightConditions

    global conf
    global bmc
    global mct
    global conds
    global warnings
    global adc

    #conf = rocket.getDefaultConfiguration() # Is this the actual simulation config though?
    conf = sim.getConfiguration() # Or is this the right configuration?
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

def simulate_rocket(sim, pts):
    """
        Simulate the rocket and return performance information.

        sim     -- Rocket simulation object
        opts    -- Simulation options
    """
    global conf
    global bmc
    global mct
    global conds
    global warnings
    global adc

    # For BC/measures
    cg = bmc.getCG(conf, mct).x
    cp = adc.getCP(conf, conds, warnings).x

    orh.run_simulation(sim)
    data = orh.get_timeseries(sim, [FlightDataType.TYPE_TIME, FlightDataType.TYPE_ALTITUDE, FlightDataType.TYPE_VELOCITY_Z])
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

    stability = cp - cg

    return (stability, apogee)

# If I want to set a seed
# random.seed(0)

with orhelper.OpenRocketInstance() as instance:
    orh = orhelper.Helper(instance)

    # Load document, run simulation and get data and events

    doc = orh.load_doc(os.path.join('examples', 'modified.ork')) # File was modified to replace Trapezoidal fin set with Freeform fin set
    sim = doc.getSimulation(0)

    # Added code here

    from net.sf.openrocket.util import Coordinate # Once the instance starts, Java classes can be imported using JPype
    from net.sf.openrocket.masscalc import BasicMassCalculator
    from net.sf.openrocket.masscalc import MassCalculator
    from net.sf.openrocket.aerodynamics import WarningSet
    from net.sf.openrocket.aerodynamics import BarrowmanCalculator
    from net.sf.openrocket.aerodynamics import FlightConditions

    opts = sim.getOptions()
    rocket = opts.getRocket()
    nose = orh.get_component_named(rocket, 'Nose cone')
    print('Nose: Aft Radius:', nose.getAftRadius())
    #print('Nose: Fore Radius:', nose.getForeRadius())
    print('Nose: Length:', nose.getLength())
    #print('Nose: Appearance:', nose.getAppearance()) # Superficial? Color
    #print('Nose: CG:', nose.getCG()) # Center of gravity?
    #print('Nose: Name:', nose.getName())
    print('Nose: Type:', nose.getType())
    print('Nose: Shape Parameter:', nose.getShapeParameter())
    print('Nose: Thickness:', nose.getThickness())

    body = orh.get_component_named(rocket, 'Body tube')
    print('Body: Length', body.getLength())

    #fins = orh.get_component_named(body, 'Trapezoidal fin set')
    fins = orh.get_component_named(body, 'Freeform fin set')
    print('Fin: Count:', fins.getFinCount())
    print('Fin: Debug:', fins.toDebugString())

    #new_points = [Coordinate(0.0,0.0,0.0), Coordinate(0.06,0.06,0.0), Coordinate(0.05, 0.0, 0.0)]
    print('Fin: Points Before:')
    printed_points = fins.getFinPoints()
    for p in printed_points: print(p)
    #fins.setPoints(new_points)
    #print('Fin: Points After:')
    #printed_points = fins.getFinPoints()
    #for p in printed_points: print(p)

    print("average wind speed: ", opts.getWindSpeedAverage()) # in m/s
    print("wind speed deviation: ", opts.getWindSpeedDeviation()) # in m/s

    prepare_for_rocket_simulation()

    #conf = rocket.getDefaultConfiguration() # Is this the actual simulation config though?
    #conf = sim.getConfiguration() # Or is this the right configuration?
    #bmc = BasicMassCalculator()
    #mct = MassCalculator.MassCalcType.LAUNCH_MASS

    # Look at updateExtras under https://github.com/openrocket/openrocket/blob/7a9bb436c1ed91335a396693eccfc400ffe236d2/swing/src/main/java/info/openrocket/swing/gui/scalefigure/RocketPanel.java
    #adc = BarrowmanCalculator()
    #conds = FlightConditions(conf)
    #conds.setMach(cpMach) # how do I get cpMach?
    #conds.setAOA(cpAOA) # how do I get cpAOA?
    #conds.setRollRate(cpRoll) # how do I get cpRoll?
    #conds.setTheta(cpTheta) # how do I get cpTheta?
    #warnings = WarningSet()
    #warnings.clear()

    #aft_radii = [0.0125, 0.01, 0.0075, 0.02, 0.03, 0.04]
    #nose_lengths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    NOSE_TYPES = [nose.Shape.OGIVE,nose.Shape.CONICAL,nose.Shape.ELLIPSOID,nose.Shape.POWER,nose.Shape.PARABOLIC,nose.Shape.HAACK]
    #nose_shape = [1.0, 0.9, 0.95, 0.8, 0.85, 0.75]
    #thicknesses = [0.002, 0.001, 0.003, 0.004, 0.005, 0.006]
    #body_lengths = [0.3, 0.4, 0.5, 0.2, 0.6, 0.7]
    #fin_counts = [2,3,4,5,6,7]
    #fin_points = [[Coordinate(0.0,0.0,0.0), Coordinate(0.025,0.030,0.000), Coordinate(0.075,0.030,0.000), Coordinate(0.05, 0.0, 0.0)], 
    #              [Coordinate(0.0,0.0,0.0), Coordinate(0.1,0.08,0.0), Coordinate(0.05, 0.0, 0.0)],
    #              [Coordinate(0.0,0.0,0.0), Coordinate(0.0223,0.0106,0.000), Coordinate(0.0173,0.0318,0.000), Coordinate(0.0427, 0.0172, 0.0), Coordinate(0.0628, 0.0278, 0.0), Coordinate(0.0572, 0.0127, 0.0), Coordinate(0.0747, 0.00714, 0.0), Coordinate(0.05, 0.0, 0.0)]]

    wind_speeds = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    wind_devs = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0]

    data = list()
    events = list()
    #cgs = list()
    #cps = list() # Depends on something called cpTheta, but I'm not sure where that comes from, so ignoring it
    stabilities = list()

    num_rockets = 30

    for i in range(num_rockets):

        #fins.setPoints(fin_points[i])

        genome = random_genome(len(scales))
        apply_genome_to_rocket(rocket, genome)

        #opts.setWindSpeedAverage(wind_speeds[i])
        #opts.setWindSpeedDeviation(wind_devs[i])

        # For BC
        #cgs.append(bmc.getCG(conf, mct).x) 
        #cps.append(adc.getCP(conf, conds, warnings).x)

        result = simulate_rocket(sim, opts)
        print(result)
        stabilities.append(result[0])

        orh.run_simulation(sim)
        data.append( orh.get_timeseries(sim, [FlightDataType.TYPE_TIME, FlightDataType.TYPE_ALTITUDE, FlightDataType.TYPE_VELOCITY_Z]) )
        events.append( orh.get_events(sim) )

    # Make a custom plot of the simulation

    events_to_annotate = {
        #FlightEvent.BURNOUT: 'Motor burnout',
        FlightEvent.APOGEE: 'Apogee' #,
        #FlightEvent.LAUNCHROD: 'Launch rod clearance'
    }

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    colors = ['b','r','g','c','m','y','tab:orange', 'tab:purple', 'tab:brown', 'tab:pink']

    for i in range(len(data)):
        ax1.plot(data[i][FlightDataType.TYPE_TIME], data[i][FlightDataType.TYPE_ALTITUDE], colors[i % len(colors)])
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Altitude (m)')

    apogees = list()

    for i in range(len(data)): 
        index_at = lambda t: (np.abs(data[i][FlightDataType.TYPE_TIME] - t)).argmin()
        for event, times in events[i].items():
            if event not in events_to_annotate:
                continue
            for time in times:
                apogees.append(data[i][FlightDataType.TYPE_ALTITUDE][index_at(time)])
                ax1.annotate(events_to_annotate[event], xy=(time, data[i][FlightDataType.TYPE_ALTITUDE][index_at(time)]),
                             xycoords='data', xytext=(20, 0), textcoords='offset points',
                             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

        if len(apogees) == i: # If a new apogee was not added, the rocket failed and earns an apogee of 0
            apogees.append(0.0)
            print("Failed apogee")

    ax1.grid(True)

    print("Apogees: ", apogees)
    #print("CGs: ", cgs)
    #print("CPs: ", cps)
    # Stability defined as CP - CG
    #stabilities = [cp - cg for cp,cg in zip(cps,cgs)]
    print("Stability: ", stabilities)

    for pair in zip(stabilities, apogees):
        print(pair)

# Leave OpenRocketInstance context before showing plot in order to shutdown JVM first
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(stabilities, apogees)
    
ax1.set_xlabel('Stability')
ax1.set_ylabel('Altitude (m)')
plt.show()
