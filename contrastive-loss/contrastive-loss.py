import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    y = np.asarray(y)

    #Validate y in {0,1}
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("Lables y must contain only values in {0,1}.")

    #Compute Euclidean distance and squared distance 
    diff = a - b
    sq_dist = np.sum( diff ** 2, axis=-1)
    d = np.sqrt(sq_dist)

    #Contrastive Loss per pair 
    loss = y * sq_dist + (1 - y) * np.maximum(0.0, margin - d) ** 2

    #Reduction 
    if reduction == "mean":
        return float(np.mean(loss))
    elif reduction == "sum":
        return float(np.sum(loss))
    else:
        raise ValueError(f"Invalid reduction method '{reduction}. Use mean or sum'.")