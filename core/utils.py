
import mlx.core as mx
import numpy as np

class Struct(object):
    """
    A simple dictionary-to-object wrapper for convenient attribute access.
    
    Args:
        **kwargs: Dictionary items to be converted into object attributes.
    """
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

def to_mx(array, dtype=mx.float32):
    """
    Converts various array-like inputs (numpy, scipy sparse, lists) into MLX arrays.
    
    Args:
        array: Input data (numpy array, scipy sparse matrix, list, or existing mlx array).
        dtype: The target MLX data type. Defaults to mx.float32.
        
    Returns:
        mx.array: The input data converted to an MLX array.
    """
    if isinstance(array, mx.array):
        return array.astype(dtype)
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return mx.array(np.array(array), dtype=dtype)

def rot_mat_to_euler(rot_mats):
    """
    Converts a batch of 3x3 rotation matrices into Euler angles (y-axis rotation).
    Primarily used for dynamic head contour landmark selection.
    
    Args:
        rot_mats (mx.array): Batch of rotation matrices of shape (B, 3, 3).
        
    Returns:
        mx.array: Euler angles (rotation around y-axis) of shape (B,).
    """
    # sy = sqrt(r00^2 + r10^2)
    sy = mx.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                 rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return mx.arctan2(-rot_mats[:, 2, 0], sy)
