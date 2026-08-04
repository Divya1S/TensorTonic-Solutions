import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    #Convert input to a NumPy array
    X = np.array(X, dtype=float)

    #Requirement: Return None if not 2D
    if X.ndim != 2:
        return None

    N, D = X.shape

    #Requirement: Return None if N < 2 (cannot compute sample covariance)
    if N < 2:
        return None

    #Step 1: Center the Data 
    mu = np.mean(X, axis=0)
    X_centered = X - mu

    #Step 2: Compute Covariance Matrix 
    #Using matrix multiplication (X_centered.T @ X_centered)
    Sigma = (X_centered.T @ X_centered) / (N - 1)

    return Sigma 

    

    