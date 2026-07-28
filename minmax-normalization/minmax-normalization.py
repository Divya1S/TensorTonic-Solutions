import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    Return np.ndarray (float).
    """
    X = np.asarray(X, dtype=float)
    
    # Calculate min and max along the specified axis, maintaining dimensions for broadcasting
    min_val = np.min(X, axis=axis, keepdims=True)
    max_val = np.max(X, axis=axis, keepdims=True)
    
    # Calculate the range (denominator)
    range_val = max_val - min_val
    
    # Prevent divide-by-zero by replacing 0s with eps
    range_val = np.where(range_val == 0, eps, range_val)
    
    # Apply the min-max normalization formula
    return (X - min_val) / range_val