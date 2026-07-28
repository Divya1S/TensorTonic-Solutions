import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    #Calculate the Euclidean (L2) norm of each vector 
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    #Gracefully handle zero vectors to avoid division by zero 
    if norm_a == 0 or norm_b == 0:
        return 0.0

    #Calculate dot product and divide by the product of the norms 
    return float(np.dot(a,b) / (norm_a * norm_b))