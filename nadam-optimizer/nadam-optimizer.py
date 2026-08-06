import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Perform one Nadam update step.
    """
    #Convert inputs to NumPy arrays for vectorized applications 
    w = np.asarray(w, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    grad = np.asarray(grad, dtype=float)

    #Step 1: Update First Moment 
    m_new = beta1 * m + (1.0 - beta1) * grad

    #Step 2: Update Second Moment 
    v_new = beta2 * v + (1.0 - beta2) * (grad ** 2)

    #Step 3: Nesterov-Adjusted Update 
    #The numerator applies Nesterov momentum ahead of time 
    m_nesterov = beta1 * m_new + (1.0 - beta1) * grad

    w_new = w - lr * (m_nesterov / (np.sqrt(v_new) + eps))

    #If the original input was a list return lists to match format expectations 
    if isinstance(w, list):
        return w_new.tolist(), m_new.tolist(), v_new.tolist()

    return w_new, m_new, v_new