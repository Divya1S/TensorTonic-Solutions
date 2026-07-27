import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    #Ensure input is an array and cast to float to prevent dtype mismatches 
    scores = np.array(scores, dtype=float, copy=False)

    #Extract seq len T form the lst dim 
    T = scores.shape[-1]

    mask = np.triu(np.ones((T, T), dtype=bool), k=1)

    masked_scores = np.where(mask, mask_value, scores)

    return masked_scores