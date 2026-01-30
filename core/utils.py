
import mlx.core as mx
import numpy as np

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

def to_mx(array, dtype=mx.float32):
    if isinstance(array, mx.array):
        return array.astype(dtype)
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return mx.array(np.array(array), dtype=dtype)

def rot_mat_to_euler(rot_mats):
    """
    Calculates rotation matrix to euler angles (y-axis rotation for head contour)
    """
    # sy = sqrt(r00^2 + r10^2)
    sy = mx.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                 rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return mx.arctan2(-rot_mats[:, 2, 0], sy)
