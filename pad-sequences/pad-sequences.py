import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    N = len(seqs)
    
    # Determine the target sequence length L
    if max_len is not None:
        L = max_len
    else:
        L = max([len(s) for s in seqs]) if N > 0 else 0
        
    # Create an (N, L) grid filled entirely with the padding value
    padded = np.full((N, L), pad_value)
    
    # Copy the actual sequence values into the grid row by row
    for i, seq in enumerate(seqs):
        # Determine how much of the sequence to copy (truncate if it exceeds L)
        length = min(len(seq), L)
        if length > 0:
            padded[i, :length] = seq[:length]
            
    return padded
        
