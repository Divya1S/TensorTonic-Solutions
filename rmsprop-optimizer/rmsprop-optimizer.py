import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    # Force inputs into fast NumPy arrays 
    w_np = np.asarray(w, dtype=float)
    g_np = np.asarray(g, dtype=float)
    s_np = np.asarray(s, dtype=float)

    #Step1: Update the running average of squared gradients 
    new_s = beta * s_np + (1.0 - beta) * (g_np ** 2)

    #Step2: Update the parameters 
    new_w = w_np - (lr / np.sqrt(new_s + eps)) * g_np

    return new_w, new_s
    