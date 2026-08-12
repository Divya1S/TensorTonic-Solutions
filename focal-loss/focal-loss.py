import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    #Ensure inputs are Numpy arrays for vectorized operations 
    p = np.array(p)
    y = np.array(y)

    #Vectorized focal loss formula 
    loss = - ( (1 - p)**gamma * y * np.log(p) + p**gamma * (1 - y) * np.log(1 - p))

    #Return the scalar mean loss across all samples 
    return np.mean(loss)

    