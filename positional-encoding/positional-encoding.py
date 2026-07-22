import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Create a blank matrix of the target shape 
    pe = np.zeros((seq_len, d_model))

    # Create a column vector of positions: [[0], [1], [2]....]
    position = np.arange(seq_len)[:, np.newaxis]

    # Compute the divisor terms for the even dimensions: base^(2i / d_model)
    # np.arrange(0, d_model, 2) creates the sequence of 2i values 
    div_term = np.power(base, np.arange(0, d_model, 2) / d_model)

    # Fill even columns (0 2 4 ......) with sine 
    pe[:, 0::2] = np.sin(position/div_term)

    # Fill odd columns with cosine 
    # We slice div_term to perfectly match the number of odd columns
    # which prevents errors if d_model is an odd number 
    pe[:, 1::2] = np.cos(position/div_term[:pe[:, 1::2].shape[1]])

    return pe