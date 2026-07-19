import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Convert input to a float array to handle scalars, lists, and arrays uniformly
    x = np.asarray(x, dtype=float)

    # Calculate and return the sigmoid using NumPy's vectorized operations
    return 1 / (1 + np.exp(-x))