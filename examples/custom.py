import os

import numpy as np
from matplotlib import pyplot as plt

import orhelper
from orhelper import FlightDataType, FlightEvent

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

    conf = rocket.getDefaultConfiguration() # Is this the actual simulation config though?
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

    aft_radii = [0.0125, 0.01, 0.0075, 0.02, 0.03, 0.04]
    nose_lengths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    nose_types = [nose.Shape.OGIVE,nose.Shape.CONICAL,nose.Shape.ELLIPSOID,nose.Shape.POWER,nose.Shape.PARABOLIC,nose.Shape.HAACK]
    nose_shape = [1.0, 0.9, 0.95, 0.8, 0.85, 0.75]
    thicknesses = [0.002, 0.001, 0.003, 0.004, 0.005, 0.006]
    body_lengths = [0.3, 0.4, 0.5, 0.2, 0.6, 0.7]
    fin_counts = [2,3,4,5,6,7]
    fin_points = [[Coordinate(0.0,0.0,0.0), Coordinate(0.025,0.030,0.000), Coordinate(0.075,0.030,0.000), Coordinate(0.05, 0.0, 0.0)], 
                  [Coordinate(0.0,0.0,0.0), Coordinate(0.1,0.08,0.0), Coordinate(0.05, 0.0, 0.0)],
                  [Coordinate(0.0,0.0,0.0), Coordinate(0.0223,0.0106,0.000), Coordinate(0.0173,0.0318,0.000), Coordinate(0.0427, 0.0172, 0.0), Coordinate(0.0628, 0.0278, 0.0), Coordinate(0.0572, 0.0127, 0.0), Coordinate(0.0747, 0.00714, 0.0), Coordinate(0.05, 0.0, 0.0)]]

    wind_speeds = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    wind_devs = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0]

    data = list()
    events = list()
    cgs = list()
    cps = list() # Depends on something called cpTheta, but I'm not sure where that comes from, so ignoring it

    for i in range(len(wind_speeds)):
        #nose.setLength(nose_lengths[i])
        #nose.setAftRadius(aft_radii[i])
        #nose.setType(nose_types[i])
        #nose.setShapeParameter(nose_shape[i])
        #nose.setThickness(thicknesses[i])

        #body.setLength(body_lengths[i])

        #fins.setFinCount(fin_counts[i])
        #fins.setPoints(fin_points[i])

        #opts.setWindSpeedAverage(wind_speeds[i])
        opts.setWindSpeedDeviation(wind_devs[i])

        # For BC
        cgs.append(bmc.getCG(conf, mct).x) 
        cps.append(adc.getCP(conf, conds, warnings).x)

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

    colors = ['b','r','g','c','m','y']

    for i in range(len(data)):
        ax1.plot(data[i][FlightDataType.TYPE_TIME], data[i][FlightDataType.TYPE_ALTITUDE], colors[i])
    
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

    ax1.grid(True)

    print("Apogees: ", apogees)
    print("CGs: ", cgs)
    print("CPs: ", cps)
    # Stability defined as CP - CG
    print("Stability: ", [cp - cg for cp,cg in zip(cps,cgs)])

# Leave OpenRocketInstance context before showing plot in order to shutdown JVM first
plt.show()
