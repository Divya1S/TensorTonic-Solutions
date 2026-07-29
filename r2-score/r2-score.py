import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    #Handle the constant target edge cases 
    if np.all(y_true == y_true[0]):
        if np.all(y_true == y_pred):
            return 1.0
        else:
            return 0.0

    #Calculate Residual Sum of Squares (SSR)
    ss_res = np.sum((y_true - y_pred) ** 2)

    #Calculate Total Sum of Squares (SST)
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)

    #Compute R^2 Score 
    return float(1.0 - (ss_res) / (ss_tot))
    