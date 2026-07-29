import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)

    #Determine axes to reduce over and the target shape for parameter broadcasting 
    if x.ndim == 2:
        axis = 0
        param_shape = (1, -1)
    elif x.ndim == 4:
        axis = (0, 2, 3)
        param_shape = (1, -1, 1, 1)
    else:
        raise ValueError(f"Expected 2D or 4D array, got shape {x.shape}")

    #Calculate mean and variance over batch/spatial dimensions 
    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)

    #Standardize features 
    x_hat = (x - mean) / np.sqrt(var + eps)

    #Scale and shift using reshaped gamma and beta 
    return gamma.reshape(param_shape) * x_hat + beta.reshape(param_shape)