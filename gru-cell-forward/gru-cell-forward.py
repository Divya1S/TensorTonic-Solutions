import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    # 1. Standardize shapes to 2D
    # Determine feature dimensions based on the last axis
    x = np.asarray(x, dtype=float)
    h_prev = np.asarray(h_prev, dtype=float)
    
    D = x.shape[-1]
    H = h_prev.shape[-1]
    
    x_2d, x_was_1d = _as2d(x, D)
    h_prev_2d, _ = _as2d(h_prev, H)
    
    # 2. Extract parameters for readability
    Wz, Uz, bz = params["Wz"], params["Uz"], params["bz"]
    Wr, Ur, br = params["Wr"], params["Ur"], params["br"]
    Wh, Uh, bh = params["Wh"], params["Uh"], params["bh"]
    
    # 3. Calculate Update Gate (z_t)
    # z_t = σ(x_t * W_z + h_{t-1} * U_z + b_z)
    z_t = _sigmoid(x_2d @ Wz + h_prev_2d @ Uz + bz)
    
    # 4. Calculate Reset Gate (r_t)
    # r_t = σ(x_t * W_r + h_{t-1} * U_r + b_r)
    r_t = _sigmoid(x_2d @ Wr + h_prev_2d @ Ur + br)
    
    # 5. Calculate Candidate Hidden State (h_tilde)
    # h~_t = tanh(x_t * W_h + (r_t ⊙ h_{t-1}) * U_h + b_h)
    h_tilde = np.tanh(x_2d @ Wh + (r_t * h_prev_2d) @ Uh + bh)
    
    # 6. Calculate New Hidden State (h_t)
    # h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h~_t
    h_t = (1.0 - z_t) * h_prev_2d + z_t * h_tilde
    
    # 7. Revert to 1D if the original input was 1D
    if x_was_1d:
        return h_t[0]
        
    return h_t