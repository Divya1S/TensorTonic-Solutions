import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    #Add eps to q to prevent division by zero or log(0)
    q_safe = q + eps

    #Handle the case where p[i] = 0 (which contributes 0 to the sum)
    #by only computing the logarithm and product where p > 0
    mask = p > 0

    #Compute the element-wise divergence and sum the results 
    divergence = np.sum(p[mask] * np.log(p[mask] / q_safe[mask]))

    return float(divergence)

    