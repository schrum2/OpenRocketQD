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

def simulate_rocket(sim, opts):
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
    # For now, use 1 as place holder for fitness
    return (1, stability, apogee)
