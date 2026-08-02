import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    we will take a matrix: A 2D list or NumPy array representing a square matrix 
    It will return a sorted array of complex eigen values or None if the input is invalid or non-square 
    """
    #Attempt to cast the input to a float array 
    #This automatically catches strings jagged lists or other invalid types 
    try:
        matrix = np.asarray(matrix, dtype=float)
    except (ValueError, TypeError):
        return None

    #Validate the matrix dimensions it must be exactly 2D and square (N x N)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    #Handle the edge case of an emtpy matrix gracefully 
    if matrix.shape[0] == 0:
        return np.array([], dtype=complex)

    #Compute the eigenvalues using NumPys dedicated eigenvalue solver 
    eigenvals = np.linalg.eigvals(matrix)

    #Sort the eigenvalues lexicographically to ensure consistent output 
    #np.lexsort sorts by the last key first so this sorts primarily by
    #real part, and breaks ties using the imaginary part 
    eigenvals = eigenvals[np.lexsort((eigenvals.imag, eigenvals.real))]
    
    return eigenvals

    