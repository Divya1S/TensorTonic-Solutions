import numpy as np

def clip_gradients(g_arr, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    g_arr = np.asarray(g_arr, dtype=float)
    
    if max_norm <= 0:
        return g_arr.copy()
        
    norm = np.linalg.norm(g_arr)
    
    if norm == 0 or norm <= max_norm:
        return g_arr.copy()
        
    scale_factor = max_norm / norm
    return g_arr * scale_factor

    