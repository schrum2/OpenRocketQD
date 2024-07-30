import random
import math
import rocket_fins as rf

DEBUG = False

#GENOME_INDEX_NOSE_AFT_RADIUS = 0
# Would replace aft radius since it also defines the radius/diameter.
# However, Cody says to just hardcode the diameter.
#GENOME_INDEX_BODY_TUBE_PRESET = 0 

GENOME_INDEX_NOSE_LENGTH = 0
GENOME_INDEX_NOSE_TYPE = 1
GENOME_INDEX_NOSE_SHAPE = 2
GENOME_INDEX_NOSE_THICKNESS = 3
GENOME_INDEX_BODY_LENGTH = 4
GENOME_INDEX_FIN_COUNT = 5
GENOME_INDEX_FIN_POINT1_X = 6
GENOME_INDEX_FIN_POINT1_Y = 7
GENOME_INDEX_FIN_POINT2_X = 8
GENOME_INDEX_FIN_POINT2_Y = 9
GENOME_INDEX_FIN_POINT3_X = 10

# The number range appropriate for each element of genome
SCALES = [#(0.01, 0.04),# Aft radius
          # None,       # Body Tube Present index: Has to be set after list of presets is loaded
          (0.05, 0.3),  # Nose length
          (0,6),        # Nose type: [0 = nose.Shape.OGIVE,1 = nose.Shape.CONICAL,2 = nose.Shape.ELLIPSOID,3 = nose.Shape.POWER,4 = nose.Shape.PARABOLIC,5 = nose.Shape.HAACK]
          (0.0,1.0),    # Nose shape (only affects some types)
          (0.001,0.009),# Nose thickness
          (0.2,1.0),    # Body length
          (2,6),        # Fin count (integer). This only allows values from 2 - 5 (6 excluded)
          (0.0,0.1),    # Fin point 1 x-coordinate
          (0.0,0.1),    # Fin point 1 y-coordinate
          (0.0,0.1),    # Fin point 2 x-coordinate
          (0.0,0.1),    # Fin point 2 y-coordinate
          (0.02,0.1)    # Fin point 3 x-coordinate (final x-coordinate must be far enough from start to create a surface to attach to the rocket)
                        # final y-coordinate must be 0.0
         ]
# Note: might want to generalize to more fin points later

DEFAULT_BODY_TUBE_OUTER_RADIUS = 0.0124 # Cody said to hard code outer diameter to 24.8mm
DEFAULT_BODY_TUBE_INNER_RADIUS = 0.01205 # Cody said to hard code inner diameter to 24.1mm

MINIMUM_FIN_CROSS_SECTION = 0.005 # 5mm

GENOME_LENGTH = len(SCALES)

def define_nose_types(nose):
    global NOSE_TYPES
    NOSE_TYPES = [nose.Shape.OGIVE,nose.Shape.CONICAL,nose.Shape.ELLIPSOID,nose.Shape.POWER,nose.Shape.PARABOLIC,nose.Shape.HAACK]

def nose_type_index(nose_type):
    global NOSE_TYPES
    for i in range(len(NOSE_TYPES)):
        if nose_type == NOSE_TYPES[i]:
            return i

    raise ValueError("Invalid nose_type")

# Do not use
#def define_body_tube_presets(presets):
#    global body_tube_presets
#    global SCALES
#    body_tube_presets = presets
#    SCALES[GENOME_INDEX_BODY_TUBE_PRESET] = (0, len(body_tube_presets) - 1)

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
    if scales[index][1] == scaled:
        scaled -= 1 # For discrete scaling, the ceiling is out of bounds
    return int(math.floor(scaled))

def decode_genome_element_nose_type(scales, genome, index):
    """
        Convert specific genome element to a particular nose cone type

        scales -- 2-tuples with (min,max) pairs
        genome -- evolved genome of values in [0,1]
        index  -- an index that associates a scale with a genome value

        Return: value from genome index converted to nose cone type according to same index in scales
    """
    global NOSE_TYPES
    type_index = decode_genome_element_discrete(scales, genome, index)
    # This will never happen now that I've discovered how to constrain genom values in [0,1]
    if type_index < 0 or type_index >= len(NOSE_TYPES):
        print("Genome:", genome)
        print("Scales:", scales)
        print("index:", index)
        print("type_index:", type_index)
        print("NOSE_TYPES:", NOSE_TYPES)
    return NOSE_TYPES[type_index] 

def decode_genome_element_coordinate(Coordinate, scales, genome, x_index, y_index):
    """
        Convert two specific genome elements to a Coordinate point.

        scales -- 2-tuples with (min,max) pairs
        genome -- evolved genome of values in [0,1]
        x_index-- an index that associates a scale with a genome value for an x-coordinate
        y_index-- an index that associates a scale with a genome value for a y-coordinate

        Return: Coordinate resulting from scaling the two genome values to x and y coordinates
    """
    x = round(decode_genome_element_scale(scales, genome, x_index), 3)
    y = round(decode_genome_element_scale(scales, genome, y_index), 3)
    return Coordinate(x, y, 0.0) # Coordinates are 3D even when only (x,y) are used

def apply_genome_to_rocket(orh, rocket, genome):
    """
        Interpret every element of the evolved genome as a parameter for
        the rocket, and apply each parameter setting. There is no return value,
        as the rocket object is modified via side effects.

        rocket -- rocket from Open Rocket that is modified according to the genome
        genome -- evolved genome of values in [0,1] which are interpreted as parameters for rocket design
    """
        
    from net.sf.openrocket.util import Coordinate # Once the instance starts, Java classes can be imported using JPype

    nose = orh.get_component_named(rocket, 'Nose cone')
    body = orh.get_component_named(rocket, 'Body tube')
    fins = orh.get_component_named(body, 'Freeform fin set')

    #aftRadius = decode_genome_element_scale(SCALES, genome, GENOME_INDEX_NOSE_AFT_RADIUS)
    #if DEBUG: print("Aft Radius:",aftRadius)
    #nose.setAftRadius(aftRadius)

    # Do not select a preset
    #global body_tube_presets
    #body_tube_index = decode_genome_element_discrete(SCALES, genome, GENOME_INDEX_BODY_TUBE_PRESET)
    #if DEBUG: 
    #    print("Body Tube Index:",body_tube_index)
    #    print("Body Tube Preset:",body_tube_presets[body_tube_index])
    #body.loadPreset(body_tube_presets[body_tube_index])
    #nose.setAftRadius(body.getOuterRadius()) # Define Nose Cone Base Radius to match Body Tube

    nose.setAftRadius(DEFAULT_BODY_TUBE_OUTER_RADIUS) # Cody said to hard code 
    body.setOuterRadius(DEFAULT_BODY_TUBE_OUTER_RADIUS) # Cody said to hard code 
    body.setInnerRadius(DEFAULT_BODY_TUBE_INNER_RADIUS) # Cody said to hard code 

    noseLength = decode_genome_element_scale(SCALES, genome, GENOME_INDEX_NOSE_LENGTH)
    if DEBUG: print("Nose Length:",noseLength)
    nose.setLength(noseLength)
    noseType = decode_genome_element_nose_type(SCALES, genome, GENOME_INDEX_NOSE_TYPE)
    if DEBUG: print("Nose Type:",noseType)
    nose.setType(noseType)
    noseShape = decode_genome_element_scale(SCALES, genome, GENOME_INDEX_NOSE_SHAPE)
    if DEBUG: print("Nose Shape:",noseShape)
    nose.setShapeParameter(noseShape)
    noseThickness = decode_genome_element_scale(SCALES, genome, GENOME_INDEX_NOSE_THICKNESS)
    if DEBUG: print("Nose Thickness:",noseThickness)
    nose.setThickness(noseThickness)

    bodyLength = decode_genome_element_scale(SCALES, genome, GENOME_INDEX_BODY_LENGTH)
    if DEBUG: print("Body Length:",bodyLength)
    body.setLength(bodyLength)

    finCount = decode_genome_element_discrete(SCALES, genome, GENOME_INDEX_FIN_COUNT)
    if DEBUG: print("Fin Count:",finCount)
    fins.setFinCount(finCount)

    fin_points = list()
    fin_points.append( Coordinate(0.0,0.0,0.0) ) # Always start at (0,0,0)
    fin_points.append( decode_genome_element_coordinate(Coordinate, SCALES, genome, GENOME_INDEX_FIN_POINT1_X, GENOME_INDEX_FIN_POINT1_Y) )
    fin_points.append( decode_genome_element_coordinate(Coordinate, SCALES, genome, GENOME_INDEX_FIN_POINT2_X, GENOME_INDEX_FIN_POINT2_Y) )
    fin_points.append( Coordinate( round(decode_genome_element_scale(SCALES, genome, GENOME_INDEX_FIN_POINT3_X), 3), 0.0, 0.0) ) # Last y-coordinate must be 0

    if DEBUG:
        print("Fin points")
        for p in fin_points: print(p)

    # At least one of the y-coordinates and one of the x-coordinates must not be 0.0
    non_zero_y = False
    non_zero_x = False
    num_all_zero = 0
    simple_vertices = []
    for p in fin_points:
        simple_vertices.append( (p.x,p.y) )
        if p.y > 0.0: 
            non_zero_y = True
        if p.x > 0.0:
            non_zero_x = True
        if p.x == 0.0 and p.y == 0.0:
            num_all_zero += 1
        elif DEBUG:
            print("Not (0,0):", p.x, ",", p.y)

    duplicate_coordinates = len(fin_points) != len(set(fin_points)) 

    shortest_cross_section_length = rf.shortest_distance_across_fin(simple_vertices)

    if DEBUG:
        print("non zero x:",non_zero_x)
        print("non zero y:",non_zero_y)
        print("num all zero:",num_all_zero)
        print("duplicates:",duplicate_coordinates)

    try:
        if not non_zero_y: raise ValueError("y-coordinates are all zero. Use default fins.")
        if not non_zero_x: raise ValueError("x-coordinates are all zero. Use default fins.")
        if num_all_zero > 1: raise ValueError("There should only be one (0,0) point. Use default fins.") # Seemingly redundant with 0.0 check, but maybe not
        if duplicate_coordinates: raise ValueError("There should be no duplicates. Use default fins.")
        if shortest_cross_section_length < MINIMUM_FIN_CROSS_SECTION: raise ValueError("")
        fins.setPoints(fin_points)
        if DEBUG: print("Use provided fin points")
    except (ValueError,Exception) as e:
        fins.setPoints( [Coordinate(0.0,0.0,0.0), Coordinate(0.025,0.030,0.000), Coordinate(0.075,0.030,0.000), Coordinate(0.05, 0.0, 0.0)] )
        if DEBUG:
            print(e)
            print("Fin point failure: default trapezoid fins")
    
