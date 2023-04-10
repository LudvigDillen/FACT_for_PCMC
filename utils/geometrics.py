import numpy as np


def rot2quat(R):
    """
    Convert 3x3 rotation matrix to unit quaternion
    NB: Assumes that input R is a rotation matrix
    """
    n = np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])
    # n has length 2sin(phi)
    nn = np.sqrt(np.sum(n.flatten()**2))
    if (nn > 0):
        # Make axis n a unit vector
        n = n/nn        # tr(R) = 1+2cos(phi)
        tn = R[0, 0]+R[1, 1]+R[2, 2]-1
        phi = np.arctan2(nn, tn)
    else:
        phi = 0

    q = np.insert(np.sin(phi/2)*n, 0, np.cos(phi/2))
    return q


def quat2rot(q):
    """
    Convert unit quaternion to rotation matrix.
    NB: Assumes that q is a unit quaternion
    """
    qq = q ** 2
    R = np.array([[qq[0] + qq[1] - qq[2] - qq[3], 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[1]*q[3] + 2*q[0]*q[2]],
                  [2*q[1]*q[2] + 2*q[0]*q[3], qq[0] - qq[1] + qq[2] - qq[3], 2*q[2]*q[3] - 2*q[0]*q[1]],
                  [2*q[1]*q[3] - 2*q[0]*q[2], 2*q[2]*q[3] + 2*q[0]*q[1], qq[0] - qq[1] - qq[2] + qq[3]]])
    return R


def transformation_matrix(quaternion, translation):
    """
    Compute the transformation matrix from a given quaternion and translation vector.

    Parameters:
    quaternion (np.array): A numpy array representing the quaternion [q0, q1, q2, q3].
    translation (np.array): A numpy array representing the translation vector [tx, ty, tz].

    Returns:
    np.array: The 4x4 transformation matrix representing the rotation and translation.
    """
    # Convert the quaternion to a rotation matrix
    # See: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    rotation = quat2rot(quaternion)

    # Create the transformation matrix
    transformation_matrix = np.eye(4)

    # Assign the rotation matrix to the top-left 3x3 sub-matrix of the transformation matrix
    transformation_matrix[:3, :3] = rotation

    # Assign the translation vector to the last column of the transformation matrix
    transformation_matrix[:3, 3] = translation

    return transformation_matrix


def convert_to_same_coordinate_system(T0, T1):
    """
    Convert two 4x4 transformation matrices to the same coordinate system with the first
    transformation matrix in the origin.

    Args:
        T0 (numpy.ndarray): The first 4x4 transformation matrix (base coordinate system).
        T1 (numpy.ndarray): The second 4x4 transformation matrix.

    Returns:
        numpy.ndarray: The transformed second matrix in the same coordinate system as the first one.

    Raises:
        ValueError: If the input matrices are not 4x4 or if the first matrix is not invertible.
    """
    # Check if the matrices are 4x4
    if T0.shape != (4, 4) or T1.shape != (4, 4):
        raise ValueError("Both input matrices must be 4x4")

    T0_inv = np.eye(4)
    T0_inv[:3, :3] = T0[:3, :3].T
    T0_inv[:3, 3] = -T0[:3, :3].T@T0[:3, 3]

    # Compute the inverse of the first matrix
    assert np.allclose(T0_inv, np.linalg.inv(T0)), "Error"

    # Convert the second matrix to the same coordinate system as the first
    transformed_T1 = T0_inv@T1

    return transformed_T1
