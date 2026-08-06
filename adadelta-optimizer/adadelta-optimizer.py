import numpy as np

def adadelta_step(w, grad, E_grad_sq, E_update_sq, rho=0.9, eps=1e-6):
    """
    Perform one AdaDelta update step.
    """
    # Convert inputs to NumPy arrays to handle standard lists
    w = np.array(w, dtype=float)
    grad = np.array(grad, dtype=float)
    E_grad_sq = np.array(E_grad_sq, dtype=float)
    E_update_sq = np.array(E_update_sq, dtype=float)
    
    # Step 1: Update Squared Gradient Average
    E_grad_sq = rho * E_grad_sq + (1 - rho) * (grad ** 2)
    
    # Step 2: Compute Parameter Update
    RMS_update = np.sqrt(E_update_sq + eps)
    RMS_grad = np.sqrt(E_grad_sq + eps)
    
    delta_w = - (RMS_update / RMS_grad) * grad
    
    # Step 3: Update Squared Update Average
    E_update_sq = rho * E_update_sq + (1 - rho) * (delta_w ** 2)
    
    # Step 4: Update Parameters
    w = w + delta_w
    
    # Return as lists if the platform expects standard python types, 
    # or leave them as arrays. Usually, arrays are fine if the prompt specified np.ndarray.
    return w, E_grad_sq, E_update_sq