import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    #Convert inputs to numpy arrays for vectorized operations 
    anchor = np.array(anchor) 
    positive = np.array(positive)
    negative = np.array(negative)

    #Compute the squared Euclidean distance: d(x,y) = ||x - y||^2
    #Using axis = -1 makes it work natively for both 1D (D, ) and 2D (N,D) inputs 
    d_ap = np.sum((anchor - positive) ** 2, axis=-1)
    d_an = np.sum((anchor - negative) ** 2, axis=-1)

    #Compute the triplet loss element-wise: max(0, d(a,p) - d(a,n) + margin)
    losses = np.maximum(0, d_ap - d_an + margin)

    #Return the mean loss across the batch as a standard float scalar 
    return float(np.mean(losses))

    