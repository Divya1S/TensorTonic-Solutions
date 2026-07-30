import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    Computes the Hinge Loss for Binary SVM.
    
    Parameters:
    -----------
    y_true : array-like, 1D array of ground truth labels in {-1, +1}
    y_score : array-like, 1D array of predicted real-valued scores
    margin : float, default=1.0
    reduction : str, "mean" (default) or "sum"
    
    Returns:
    --------
    float: Mean or sum hinge loss scalar
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    # 1. Validate array dimensions and matching shapes
    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("Inputs 'y_true' and 'y_score' must be 1D arrays.")
    if y_true.shape != y_score.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_score.shape}.")
        
    # 2. Validate binary label set {-1, +1}
    if not np.all((y_true == 1) | (y_true == -1)):
        raise ValueError("y_true must contain only values in {-1, +1}.")
        
    # 3. Validate reduction strategy
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be either 'mean' or 'sum'.")
        
    # Fully vectorized loss computation: max(0, margin - y_true * y_score)
    losses = np.maximum(0.0, margin - (y_true * y_score))
    
    # Compute reduction and cast result to a standard Python float
    if reduction == "mean":
        return float(np.mean(losses)) if losses.size > 0 else 0.0
    else:
        return float(np.sum(losses))