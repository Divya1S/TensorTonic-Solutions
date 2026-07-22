import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    #Track if the original input was a list to format the output correctly
    is_list = isinstance(param, list) 

    #Convert inputs to NumPy arrays for elementwise vectorized operations 
    param_np = np.asarray(param, dtype=float) 
    grad_np = np.asarray(grad, dtype=float)
    m_np = np.asarray(m, dtype=float)
    v_np = np.asarray(v, dtype=float)

    #Step 1: Update First Moment (Running average of gradients)
    m_new = beta1 * m_np + (1 - beta1) * grad_np

    #Step 2: Update Second Moment (Running average of squared gradients) 
    v_new = beta2 * v_np + (1 - beta2) * (grad_np ** 2)

    #Step 3: Bias Correction 
    m_hat = m_new / (1 - beta1 ** t) 
    v_hat = v_new / (1 - beta2 ** t)

    #Step 4: Parameter Update 
    param_new = param_np - lr * m_hat / (np.sqrt(v_hat) + eps)

    #Return as standard Python lists if thats what was passed in otherwise return arrays
    if is_list:
        return param_new.tolist(), m_new.tolist(), v_new.tolist()
    return param_new, m_new, v_new