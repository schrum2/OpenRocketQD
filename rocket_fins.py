# Code in this file was generated by ChatGPT

import numpy as np

def point_to_line_distance(point, line_start, line_end):
    """
    Compute the distance from a point to a line segment.
    Args:
    - point: The coordinates of the point (x, y).
    - line_start: The starting coordinates of the line segment (x, y).
    - line_end: The ending coordinates of the line segment (x, y).
    
    Returns:
    - The perpendicular distance from the point to the line segment.
    """
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Line segment length squared
    line_len_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if line_len_sq == 0:
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)  # line_start == line_end case
    
    # Projection of point onto the line segment
    t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_len_sq
    t = max(0, min(1, t))  # Clamp t to the range [0, 1]
    
    # Projection point on the line
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)
    
    # Distance from point to the projection on the line segment
    distance = np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
    return distance

def shortest_distance_to_polygon(point, polygon_vertices):
    """
    Compute the shortest distance from a point to any edge of the polygon.
    Args:
    - point: The coordinates of the point (x, y).
    - polygon_vertices: A list of tuples representing the vertices of the polygon.
    
    Returns:
    - The shortest distance from the point to the polygon.
    """
    min_distance = float('inf')
    num_vertices = len(polygon_vertices)
    
    for i in range(num_vertices):
        line_start = polygon_vertices[i]
        line_end = polygon_vertices[(i + 1) % num_vertices]
        distance = point_to_line_distance(point, line_start, line_end)
        min_distance = min(min_distance, distance)
    
    return min_distance

# Example usage
#polygon = [(0, 0), (4, 0), (4, 3), (0, 3)]
#point = (2, 5)
#distance = shortest_distance_to_polygon(point, polygon)
#print(f"The shortest distance from the point to the polygon is: {distance}")

# Code beneath this point written by me

def shortest_distance_across_fin(vertices):

    # DOES NOT WORK!

    dists = [shortest_distance_to_polygon(p, vertices) for p in vertices]
    return min(dists)

polygon = [(0, 0), (4, 0), (4, 3), (0, 3)]
distance = shortest_distance_across_fin(polygon)
print(f"The shortest distance from the point to the polygon is: {distance}")
