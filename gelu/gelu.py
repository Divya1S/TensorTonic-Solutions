import numpy as np
import math
from scipy.special import erf

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    x_arr = np.asarray(x, dtype=float)
    
    return 0.5 * x_arr * (1.0 + erf(x_arr / np.sqrt(2.0)))

    
