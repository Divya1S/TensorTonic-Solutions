import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    #Validate 1D shape requirement 
    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")

    #Validate matching lengths 
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("Arrays x and y must be of the same length")

    return float(np.dot(x_arr, y_arr))