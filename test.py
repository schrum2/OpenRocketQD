import orhelper
import os


with orhelper.OpenRocketInstance() as instance:

    orh = orhelper.Helper(instance)
    doc = orh.load_doc(os.path.join('examples', 'modified.ork')) # File was modified to replace Trapezoidal fin set with Freeform fin set
    sim = doc.getSimulation(0)
    opts = sim.getOptions()
    rocket = opts.getRocket()

    nose = orh.get_component_named(rocket, 'Nose cone')
    
    print(nose)
    print(nose.getType())
    print(type(nose.getType()))
    NOSE_TYPES = [nose.Shape.OGIVE,nose.Shape.CONICAL,nose.Shape.ELLIPSOID,nose.Shape.POWER,nose.Shape.PARABOLIC,nose.Shape.HAACK]

    print(NOSE_TYPES[0])
    print(NOSE_TYPES[0] == nose.getType())
    print(NOSE_TYPES[1] == nose.getType())



