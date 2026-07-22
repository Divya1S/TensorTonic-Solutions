import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    
    #numpy array
    x_np = np.asarray(x, dtype=float)

    #Fast path: If dropout rate is 0, keep everything exactly as is 
    if p == 0.0:
        mask = np.ones_like(x_np)
        return x_np.copy(), mask

    #Set up the random number generator 
    if rng is None:
        rng = np.random.default_rng()

    #Generate an array of random numbers between 0.0 and 1.0 
    rand_vals = rng.uniform(0.0, 1.0, size=x_np.shape)

    #Create the inverted dropout mask (0 for dropped, 1/(1-p) for kept)
    multiplier = 1.0 / (1.0 - p)
    mask = (rand_vals >= p).astype(float) * multiplier

    out = x_np * mask

    return out, mask