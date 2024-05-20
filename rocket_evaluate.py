
from orhelper import FlightDataType, FlightEvent
import numpy as np
import statistics

# To test the robustness of the rocket design, it is evaluated at varying wind speeds
WIND_SPEED_MIN = 2.0
WIND_SPEED_MAX = 10.0

WIND_DEVIATION_MIN = 0.2
WIND_DEVIATION_MAX = 4.0

NUM_ROCKET_EVALS = 3

MAX_FITNESS = 10

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

def simulate_rocket(orh, sim, opts):
    """
        Simulate the rocket and return performance information.

        orh     -- Open Rocket helper object
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
    stability = cp - cg

    apogees = list()
    wind_speed_increment = (WIND_SPEED_MAX - WIND_SPEED_MIN) / NUM_ROCKET_EVALS
    wind_deviation_increment = (WIND_DEVIATION_MAX - WIND_DEVIATION_MIN) / NUM_ROCKET_EVALS
    wind_speed = WIND_SPEED_MIN
    wind_deviation = WIND_DEVIATION_MIN
    # Evaluate the rocket with different wind conditions and take the average altitude.
    # The fitness goal is to minimize the variance in max altitude.
    for i in range(NUM_ROCKET_EVALS):

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

    average_apogee = statistics.mean(apogees)
    apogee_stdev = statistics.pstdev(apogees)
    # Fitness is max minus the standard deviation in the max altitude attained
    return (MAX_FITNESS - apogee_stdev, stability, average_apogee)
