import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    # Convert input to a float numpy array
    x_arr = np.asarray(x, dtype=float)
    
    # Handle scalar edge case: return 1D array with shape (1)
    if x_arr.ndim == 0:
        x_arr = np.atleast_1d(x_arr)
        
    # Initialize an array to hold the sigmoid outputs
    sig = np.empty_like(x_arr)
    
    # Create masks for positive and negative values for numerical stability
    pos_mask = x_arr >= 0
    neg_mask = ~pos_mask
    
    # Stable sigmoid for positive values: 1 / (1 + exp(-x))
    sig[pos_mask] = 1.0 / (1.0 + np.exp(-x_arr[pos_mask]))
    
    # Stable sigmoid for negative values: exp(x) / (1 + exp(x))
    # This prevents overflow in exp(-x) when x is a large negative number
    exp_x_neg = np.exp(x_arr[neg_mask])
    sig[neg_mask] = exp_x_neg / (1.0 + exp_x_neg)
    
    # Swish(x) = x * sigmoid(x)
    return x_arr * sig