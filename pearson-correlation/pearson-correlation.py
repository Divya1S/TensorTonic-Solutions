import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    #Convert input to float numpy array
    try:
        X = np.asarray(X, dtype=np.float64)
    except (ValueError, TypeError):
        return None 

    #Input validation: must be 2D and have N >= 2
    if X.ndim != 2 or X.shape[0] < 2:
        return None

    #Center the dataset by subtracting column means
    X_centered = X - np.mean(X, axis=0, keepdims=True)

    #Unscaled covariance matrix: X_centered ^ T @ X_centered 
    cov = X_centered.T @ X_centered

    #Standard deviations (square root of column variances)
    std = np.sqrt(np.diag(cov))

    #Outer product of standard deviation: std_i * std_j
    std_outer = np.outer(std, std)

    #Compute correlation matrix R = cov / (std_i * std_j)
    with np.errstate(divide='ignore', invalid='ignore'):
        R = cov / std_outer

    # #Numerical stability: clip off-diagonal values to [-1.0, 1.0]
    # R = np.clip(R, -1.0, 1.0)

    # #Ensure exact 1.0 on the main diagonal 
    # np.fill_diagonal(R, 1.0)

    return R