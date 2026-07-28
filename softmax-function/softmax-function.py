import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    #Find the maximum value along the last axis for numerical stability 
    #keepdims=True ensures the shape matches x for proper broadcasting 
    x_max = np.max(x, axis=-1, keepdims=True)

    #Subtract the max to prevent overflow then exponentiate 
    exp_x = np.exp(x - x_max)

    #Divide by the sum of exponentials along the last exis 
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)