import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    #Convert inputs to numpy arrays 
    Z1 = np.asarray(Z1)
    Z2 = np.asarray(Z2)

    #1. Compute the similarity matrix and scale by temperature 
    #Shape: (N, N)
    S = np.dot(Z1, Z2.T) / temperature

    #2. Apply numerical stability trick (subtract max value per row)
    #This prevents overflow when computing np.exp()
    #Shape of max_S: (N, 1) to allow proper broadcasting 
    max_S = np.max(S, axis=1, keepdims=True)
    S_stable = S - max_S

    #3. Compute the log of the sum of exponentials for the denominator 
    #Shape: (N,)
    log_sum_exp = np.log(np.sum(np.exp(S_stable), axis=1))

    #4. Extract the scaled similarites for the postive pairs 
    #Positive pairs are located on the daigonal of simularity matrix 
    #Shape: (N,)
    pos_pair = np.diag(S_stable)

    #5. Compute the final loss 
    #log(exp(S_ii) / sum(exp(S_ij))) = S_ii - log(sum(exp(S_ij)))
    loss = -np.mean(pos_pair - log_sum_exp)

    return float(loss)
    