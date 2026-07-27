import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """
    N, C_in, H, W_in = x.shape
    C_out, _, KH, KW = W.shape
    
    H_out = H - KH + 1
    W_out = W_in - KW + 1
    
    # Extract strides from the input tensor
    sN, sC, sH, sW = x.strides
    
    # Create a strided view of x with shape (N, C_in, H_out, W_out, KH, KW)
    # The strides tell NumPy how many bytes to step to reach the next element 
    # for each dimension, allowing us to "slide" the window virtually.
    x_strided = np.lib.stride_tricks.as_strided(
        x, 
        shape=(N, C_in, H_out, W_out, KH, KW), 
        strides=(sN, sC, sH, sW, sH, sW)
    )
    
    # Compute the convolution using einsum:
    # n: Batch size (N)
    # c: Input channels (C_in)
    # h, w: Spatial output dimensions (H_out, W_out)
    # u, v: Kernel dimensions (KH, KW)
    # o: Output channels (C_out)
    y = np.einsum('nchwuv,ocuv->nohw', x_strided, W)
    
    # Add the bias, reshaping it to broadcast across N, H_out, and W_out
    y += b.reshape(1, C_out, 1, 1)
    
    return y.astype(float)