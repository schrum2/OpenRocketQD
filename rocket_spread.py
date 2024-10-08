
import rocket_evaluate as re
import sys
import math

def stable_top_of_range(rows, min_stability, by_nose_type, current_ceiling, top_count):
    """
        From the rows, filter out rows whose stability value is too low,
        or whose altitude is above current ceiling. Then get the highest
        altitude of what remains.

        Parameters:
        rows - data from the saved CSV archive
        min_stability - min stability required of the top results
        by_nose_type - whether to split up results by nose type 
        current_ceiling - do not include altitudes above this
        top_count - number to return
    """
    # Measures are index 2, and index 0 is either the stability, or a combination of stability and nose type.
    # Measure index 1 is the altitude.
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
        filtered = list(filter(lambda r : r[2][1] < current_ceiling and abs( (r[2][0]/range_per_nose_type) - int(r[2][0]/range_per_nose_type) ) > epsilon, rows))

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
        filtered = filter(lambda r : r[2][0] >= min_stability and r[2][1] < current_ceiling, rows)
        
        top = sorted(filtered, key=lambda r : -r[2][1])

        return top[:top_count]

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Improper usage. Should be:")
        print("python rocket_evaluate.py <archive file> <altitude step> <altutude ceiling>")
        print("Example:")
        print("python rocket_spread.py evolve_rockets_output/map_elites_archive.csv 10 100")
    else:
        DEBUG = True
        global NUM_NOSE_TYPES
        NUM_NOSE_TYPES = 6

        filename = sys.argv[1]
        step = float(sys.argv[2])
        ceiling = float(sys.argv[3])

        by_nose_type = "nose" in filename
        top_count = 6

        rows = re.all_rows(filename)
        current_ceiling = step
        while current_ceiling <= ceiling:

            top_rows = stable_top_of_range(rows, 1.0, by_nose_type, current_ceiling, top_count) # requires min stability of 1.0
            print(f"UP TO {current_ceiling}")
            
            #if by_nose_type:
            #    print(top_count / NUM_NOSE_TYPES, "per nose type")
            for (index, genome, measures, objective) in top_rows:
                print("Index:",index, "Objective:", objective, "Measures:", measures)

            print(f"END {current_ceiling}")
            current_ceiling += step

