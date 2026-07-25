import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    x_arr = np.array(x)
    y_arr = np.array(y)

    #1. Subtract arrays to find differences in each dimension 
    #2. Square the differences (**2)
    #3. Sum them all up (np.sum)
    #4. Take the square root (np.sqrt)
    distance = np.sqrt(np.sum((x_arr - y_arr) ** 2))

    return float(distance)