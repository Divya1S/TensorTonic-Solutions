import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    # Ensure inputs are NumPy arrays for vectorized operations
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Calculate the absolute error
    abs_error = np.abs(y_true - y_pred)
    
    # Apply the piecewise Huber Loss formula using np.where
    loss = np.where(
        abs_error <= delta,
        0.5 * (abs_error ** 2),               # L2 loss for small errors
        delta * (abs_error - 0.5 * delta)     # L1 loss for large errors
    )
    
    # Return the mean loss across all samples
    return np.mean(loss)