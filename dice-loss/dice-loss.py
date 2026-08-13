import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    #p: array-like: Predicted probabilities 
    #y : array-like: Ground truth binary mask 
    #eps: float - Smoothing epsilon to prevent division by zero 

    #Returns:
    #float - Dice loss score 
    #Convert inputs to float arrays
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)

    #Compute intersection and union sums 
    intersection = np.sum(p * y)
    union = np.sum(p) + np.sum(y)

    #Calculate Dice coefficient with smoothing epsilon
    dice = (2.0 * intersection + eps) / (union + eps)

    #Return the Dice Loss 
    return 1.0 - dice
    