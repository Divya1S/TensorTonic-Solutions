import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    x_arr = np.asarray(x, dtype=float)

    return np.maximum(alpha * x_arr, x_arr)