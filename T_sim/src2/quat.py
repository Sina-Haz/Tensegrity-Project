import numpy as np

"""
In this file we will define some quaternion operations for utility
"""

def quat_conjugate(quats: np.ndarray):
    """
    Compute the conjugate of an array of quaternions of shape (batch_sz, 4)

    Args:
        quats: array of quaternions, shape = (batch_sz, 4)

    Returns:
        conj: array of quaternion conjugates, shape = (batch_sz, 4)
    """
    conj = np.copy(quats) * -1
    conj[:, 0] *= -1
    return conj


def quat_mul_batch(q1, q2):
    """
    Multiply two batches of quaternions.

    Args:
        q1: numpy array of shape (batch_sz, 4), where each row is a quaternion [w, x, y, z].
        q2: numpy array of shape (batch_sz, 4), where each row is a quaternion [w, x, y, z].

    Returns:
        numpy array of shape (batch_sz, 4), where each row is the resulting quaternion.
    """
    # Extract scalar and vector parts
    s1 = q1[:, 0]  # Shape: (batch_sz,)
    v1 = q1[:, 1:]  # Shape: (batch_sz, 3)
    
    s2 = q2[:, 0]  # Shape: (batch_sz,)
    v2 = q2[:, 1:]  # Shape: (batch_sz, 3)
    
    # Compute the scalar part of the result
    scalar_part = s1 * s2 - np.sum(v1 * v2, axis=1)  # Shape: (batch_sz,)
    
    # Compute the vector part of the result
    vector_part = (
        s1[:, np.newaxis] * v2 +  # s1 * v2
        s2[:, np.newaxis] * v1 +  # s2 * v1
        np.cross(v1, v2)          # v1 x v2
    )  # Shape: (batch_sz, 3)
    
    # Combine scalar and vector parts into the result
    result = np.hstack((scalar_part[:, np.newaxis], vector_part))  # Shape: (batch_sz, 4)
    
    return result

def normalize(vector_arr: np.ndarray):
    """
    Normalize array of vectors to unit length

    Args:
        vectors: numpy array of shape (batch_sz, n), where each row is a vector.

    Returns:
        numpy array of shape (batch_sz, n), where each row is a unit vector.
    """
    # Compute the norm of each vector and keep the dimensionality for broadcasting
    norms = np.linalg.norm(vector_arr, axis=1, keepdims=True)

    # avoid dividing by 0
    norms[norms == 0] = 1

    unit_vecs = vector_arr / norms

    return unit_vecs

def quat_inv(quats: np.ndarray):
    """
    Args:
        quats: Array of quaternions, shape = (batch_sz, 4)

    Returns:
        q_invs: Array of quaternions that are the inverse of inputs, shape = (batch_sz, 4)
    """
    return quat_conjugate(normalize(quats))


def quat_to_matrix(quats: np.ndarray):
    """
    Convert a batch of quaternions to rotation matrices.

    Args:
        quats: numpy array of shape (batch_sz, 4), where each row is a quaternion [w, x, y, z].

    Returns:
        numpy array of shape (batch_sz, 3, 3), where each slice along the first axis is a 3x3 rotation matrix.
    """
    batch_sz = quats.shape[0]
    rot_matrs = np.zeros((batch_sz, 3, 3))

    # Extract quaternion components
    w = quats[:, 0]
    x = quats[:, 1]
    y = quats[:, 2]
    z = quats[:, 3]

    # Compute rotation matrix elements
    rot_matrs[:, 0, 0] = 1 - 2 * y**2 - 2 * z**2
    rot_matrs[:, 0, 1] = 2 * x * y - 2 * w * z
    rot_matrs[:, 0, 2] = 2 * x * z + 2 * w * y

    rot_matrs[:, 1, 0] = 2 * x * y + 2 * w * z
    rot_matrs[:, 1, 1] = 1 - 2 * x**2 - 2 * z**2
    rot_matrs[:, 1, 2] = 2 * y * z - 2 * w * x

    rot_matrs[:, 2, 0] = 2 * x * z - 2 * w * y
    rot_matrs[:, 2, 1] = 2 * y * z + 2 * w * x
    rot_matrs[:, 2, 2] = 1 - 2 * x**2 - 2 * y**2

    return rot_matrs


def quat_exp(quats:np.ndarray):
    """
    Takes an array of quaternions and exponentiates them

    Args:
        quats: array of quaternions, shape = (batch_sz, 4)

    Returns:
        exp_quats: array of exponentiated quaternions, shape = (batch_sz, 4)
    """

    s, v = quats[:, 0], quats[:, 1:]

    norm_v = np.clip(np.linalg.norm(v, axis=1), a_min=1e-8, a_max=np.inf)

    # Compute the scalar part of the result
    scalar_part = np.exp(s) * np.cos(norm_v)  # Shape: (batch_sz,)
    
    # Compute the vector part of the result
    vector_part = (
        np.exp(s)[:, np.newaxis] *  # e^s
        (np.sin(norm_v) / norm_v)[:, np.newaxis] *  # sin(||v||) / ||v||
        v  # v
    )  # Shape: (batch_sz, 3)

    # Combine scalar and vector parts into the result
    result = np.hstack((scalar_part[:, np.newaxis], vector_part))  # Shape: (batch_sz, 4)
    
    return result


def quat_from_endpts(p1, p2) -> np.ndarray:
    '''
    Takes in two endpoints as lists and computes the quaternion rotation
     - Rotation is w.r.t. principal axis z = (0, 0, 1)
     - NOTE: his function is NOT for batch processing

    Do this by computing rotation axis as cross product of z and normalized unit vector b/w endpoints
    '''
    v = p2 - p1
    v_norm = np.linalg.norm(v)

    # Endpoints shouldn't be the same
    assert v_norm != 0

    # Normalize v
    v = v / v_norm
    z = np.array([0,0,1])

    # Rotation axis:
    r = np.cross(z, v)
    r_norm = np.linalg.norm(r)
    theta = np.arccos(np.dot(z, v))

    # Edge cases: (r is parallel to z, no rotation or flipped)
    if r_norm == 0:
        if r == z: return np.array([1,0,0,0]) # Unit quaternion
        else: return np.array([0, 1, 0, 0])

    # Normalize r if r_norm neq to 0
    r = r / r_norm
    w = np.cos(theta / 2)
    xyz = r * np.sin(theta / 2)

    return np.hstack([np.array([w]), xyz])






