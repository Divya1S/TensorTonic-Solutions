import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.asarray(x)
    p = np.asarray(p)

    #Check if shapes match 
    if x.shape != p.shape:
        raise ValueError("The arrays 'x' and 'p' must have the same shape")

    #Verify probabilities sum to 1 within a tolerance of 1e-6
    if not np.isclose(np.sum(p), 1.0, atol=1e-6):
        raise ValueError("Probabilities in 'p' must sum to 1.0.")

    #Compute the expected value: E[X] = sum(x_i * p_i)
    expected_value = np.sum(x * p)

    return float(expected_value)
