import numpy as np
import torch


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


# If I want to have a speed-up I should to this function in batches, but that would require my to
# work read the data in batches as well which would be a new approach which would require some work.
def change_coordinate_system(pc1: torch.Tensor, T0: torch.Tensor, T1: torch.Tensor) -> torch.Tensor:
    """
    Moves a 3D point cloud with with pose T1 to the local coordinate system of another point
    cloud with pose T0.

    :param pc1: A point cloud represented as a Mx3 Tensor (M points x 3 coordinates).
    :type pc1: torch.Tensor

    :param T0: A 4x4 transformation matrix corresponding to the first point cloud.
    :type T0: torch.Tensor

    :param T1: A 4x4 transformation matrix corresponding to the second point cloud.
    :type T1: torch.Tensor

    :return: point cloud in the local coordinate system of pc0 with pose T0
    :rtype: torch.Tensor
    """
    # Ensure input tensors have the correct dimensions
    assert pc1.dim() == 2 and pc1.size(1) == 3, "pc2 must be a Mx3 Tensor"
    assert T0.dim() == 2 and T0.size(0) == 4 and T0.size(1) == 4, "T1 must be a 4x4 Tensor"
    assert T1.dim() == 2 and T1.size(0) == 4 and T1.size(1) == 4, "T2 must be a 4x4 Tensor"

    # Step 1: Compute the inverse of the first transformation matrix
    T0_inv = torch.inverse(T0)

    # Step 2: Compute the transformation matrix to align both point clouds
    T_align = torch.matmul(T0_inv, T1)

    # Step 3: Move the second point cloud to the coordinate system of the first point cloud
    # Convert to homogeneous coordinates
    pc1_homogeneous = torch.cat((pc1, torch.ones(pc1.size(0), 1, dtype=pc1.dtype, device=pc1.device)), dim=1)
    pc1_homogeneous_swap = torch.swapaxes(pc1_homogeneous, 0, 1)

    pc1_CS0_swap = torch.matmul(T_align, pc1_homogeneous_swap)[:3]
    pc1_CS0 = torch.swapaxes(pc1_CS0_swap, 0, 1)

    return pc1_CS0
