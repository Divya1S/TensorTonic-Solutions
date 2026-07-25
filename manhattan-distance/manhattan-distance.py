import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    x_array = np.array(x)
    y_array = np.array(y)

    #Vectorized sibstraction absolute value and sum 
    distance = np.sum(np.abs(x_array - y_array)) 

    return float(distance)