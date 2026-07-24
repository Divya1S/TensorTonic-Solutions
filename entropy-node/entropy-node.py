import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.asarray(y)

    #Handle empty node edge case 
    if y.size == 0:
        return 0.0

    #Find unique classes and their exact counts 
    _, counts = np.unique(y, return_counts=True)

    #Compute proportions 
    p = counts / y.size

    #Since np.unique only returns counts for classes present in y
    #p > 0 is guaranteed for all elements This naturally avoids log2(0)
    entropy = -np.sum(p * np.log2(p))

    return float(entropy)
    
    