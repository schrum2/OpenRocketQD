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
SCALES = [(0.01, 0.04), # Aft radius
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

GENOME_LENGTH = len(SCALES)

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
    
