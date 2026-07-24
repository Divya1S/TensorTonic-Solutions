import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    T = np.asarray(T)
    points = np.asarray(points)
    is_single_point = (points.ndim == 1)
    
    # Force points to be (N, 3) for vectorized operations
    pts_2d = np.atleast_2d(points)
    N = pts_2d.shape[0]
    
    # 1. Convert to homogeneous coordinates by appending 1s
    # Pre-allocating an empty array and assigning is faster than np.hstack
    ph = np.empty((N, 4), dtype=pts_2d.dtype)
    ph[:, :3] = pts_2d
    ph[:, 3] = 1.0
    
    # 2. Apply transform
    # Since our points are row vectors (N, 4), we multiply by T transposed (4, 4)
    ph_transformed = ph @ T.T
    
    # 3. Extract the spatial coordinates (drop the w-coordinate)
    p_transformed = ph_transformed[:, :3]
    
    # Return (3,) if input was (3,), else return (N, 3)
    return p_transformed[0] if is_single_point else p_transformed
    