import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    #Convert A to a Numpy array just in case it was passed in a standard list 
    A = np.array(A)

    #Figure out the shape of original matrix 
    rows, cols = A.shape

    #Build a blank NumPy gird with dimensions flipped 
    #np.zeros create the grid and fills it with 0s
    #dtype=A.dtype ensures we keep the same data type (like integers or decimals)
    transposed = np.zeros((cols, rows), dtype=A.dtype)

    #Move every item to its new flipped location
    for i in range(rows):
        for j in range(cols):
            transposed[j,i] = A[i,j]

    return transposed
