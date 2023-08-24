import numpy as np


def skew(vector: np.ndarray) -> np.ndarray:
    # vector: 3
    result = np.zeros((3, 3))
    result[0, 1] = -vector[2]
    result[0, 2] = vector[1]
    result[1, 0] = vector[2]
    result[1, 2] = -vector[0]
    result[2, 0] = -vector[1]
    result[2, 1] = vector[0]
    return result

def rotation_matrix_from_axis(axis: np.ndarray, theta: np.ndarray) -> np.ndarray:
    R = np.eye(3) * np.cos(theta)
    R += skew(axis) * np.sin(theta)
    R += (1 - np.cos(theta)) * np.outer(axis, axis)
    return R

def axis2transformation(axis, center, theta):
    rot = rotation_matrix_from_axis(axis, theta)
    translation = - rot.dot(center) + center
    trans_mat = np.eye(4)
    trans_mat[:3, :3] = rot
    trans_mat[:3, 3] = translation
    return trans_mat