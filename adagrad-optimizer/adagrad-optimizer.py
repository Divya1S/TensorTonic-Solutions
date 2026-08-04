import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """

    # Step 1: Convert starndard Python lists to NumPy arrays for math operations
    w = np.array(w, dtype=float)
    g = np.array(g, dtype=float)
    G = np.array(G, dtype=float)

    #Step 2: Accumulate the squared gradients 
    new_G = G + np.square(g)

    #Step 3: Update the paramenters 
    #Using np.sqrt(new_G + eps) applies the stability constant before the square root 
    #preventing division by zero 
    new_w = w - (lr / np.sqrt(new_G + eps)) * g

    #4. Return as a tuple of starndard Python lists 
    return (new_w.tolist(), new_G.tolist())

    