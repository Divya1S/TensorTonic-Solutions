import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    
    Parameters:
    - matrix: 2D array-like input
    - axis: 0 (column-wise), 1 (row-wise), or None (entire matrix)
    - norm_type: 'l1', 'l2', or 'max'
    
    Returns:
    - Normalized NumPy array, or None if inputs are invalid.
    """
    # Validate axis parameter
    if axis not in (0, 1, None):
        return None
    
    # Validate norm_type parameter
    if not isinstance(norm_type, str) or norm_type.lower() not in ('l1', 'l2', 'max'):
        return None
    
    norm_type = norm_type.lower()

    # Convert input matrix to 2D numpy float array
    try:
        arr = np.asarray(matrix, dtype=np.float64)
    except (ValueError, TypeError):
        return None

    # Check that input is a non-empty 2D array
    if arr.ndim != 2 or arr.size == 0:
        return None

    # Compute norm based on selected type and axis
    if norm_type == 'l1':
        norm = np.sum(np.abs(arr), axis=axis, keepdims=True) if axis is not None else np.sum(np.abs(arr))
    elif norm_type == 'l2':
        norm = np.sqrt(np.sum(arr ** 2, axis=axis, keepdims=True)) if axis is not None else np.sqrt(np.sum(arr ** 2))
    elif norm_type == 'max':
        norm = np.max(np.abs(arr), axis=axis, keepdims=True) if axis is not None else np.max(np.abs(arr))

    # Divide by norm with safe division for zero vectors/matrices
    if axis is None:
        return arr.copy() if norm == 0 else arr / norm
    else:
        norm = np.where(norm == 0, 1.0, norm)
        return arr / norm