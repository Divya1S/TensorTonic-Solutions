import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    #Convert input to a float numpy array and ensure it is at least 1D
    x_arr = np.atleast_1d(np.array(x, dtype=float))

    #Calculate Tanh using the explicit formula: (e^x - e^-x) / (e^x + e^-x)
    e_x = np.exp(x_arr)
    e_minus_x = np.exp(-x_arr)

    return (e_x - e_minus_x) / (e_x + e_minus_x)