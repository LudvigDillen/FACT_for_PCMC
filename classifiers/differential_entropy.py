import torch
import numpy as np


def differential_entropy_metric(PCHandler) -> float:
    """
    Calculate the differential entropy between two point clouds using the method described in the
    paper "CorAl – Are the point clouds Correctly Aligned?".

    Parameters:
    pc0_CS0 (torch.Tensor): The first point cloud as a PyTorch tensor with shape (3, n), where n is the
                            number of points. This point cloud is in the coordinate system of pc0.
    pc1_CS0 (torch.Tensor): The second point cloud as a PyTorch tensor with shape (3, m), where m is the
                            number of points. This point cloud is in the coordinate system of pc0.
    pc1_CS1 (torch.Tensor): The second point cloud as a PyTorch tensor with shape (3, m), where m is the
                            number of points. This point cloud is in the coordinate system of pc1.

    Returns:
    float: The differential entropy between the two point clouds.
    """

    N_union_points = PCHandler.pc_union.shape[1]

    # separate average differential entropy
    H_separate = (differential_entropy(
        PCHandler.pc0_CS0) + differential_entropy(PCHandler.pc1_CS1))/N_union_points
    # joint average differential entropy
    H_joint = differential_entropy_joint(PCHandler)/N_union_points

    metric = H_joint - H_separate  # this is our alignment quality measure for the enitre point cloud
    return metric


def differential_entropy(pc: torch.Tensor) -> float:
    """
    Calculate the differential_entropy of the point cloud using the method described in the
    paper "CorAl – Are the point clouds Correctly Aligned?".

    Parameters:
    pc (torch.Tensor): The first point cloud as a PyTorch tensor with shape (3, n), where n is the
                       number of points.

    Returns:
    float: The entropy the point cloud.
    """
    dim_distribution = pc.shape[0]
    assert dim_distribution == 3, "Expected 3-dim distribution"
    N_points = pc.shape[1]
    # TODO: set it dynamically later (I Put it to 0.1 to make it go faster)
    # but if not run dynamically, it should rather be 0.5 but it goes slower.
    epsilon = torch.tensor(10**(-8))  # Some offset to make sure not taking log of zero
    zero = torch.tensor(0)
    alpha = torch.tensor(1.33)  # vertical angular resolution degrees
    alpha_rad = torch.deg2rad(alpha)
    sin_alpha = torch.sin(alpha_rad)

    scaler = (2*torch.tensor(np.pi)*torch.exp(torch.tensor(1)))**dim_distribution  # (2pi*e)^dim_distribution

    # TODO: Is this necessary? Is cuda available now?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pc = pc.to(device)

    H = torch.zeros(1, device=device)

    # TODO: Can I do this in faster way without the for loop?
    for k in range(N_points):
        pk = torch.unsqueeze(pc[:, k], dim=1)
        d = torch.norm(pk, p=2)
        # Get dynamic radius
        r = get_dynamic_radius(d, sin_alpha)

        # Euclidean distance from key point to all other points in point cloud
        dists = torch.norm(pc-pk, p=2, dim=0)  # this is the slowest operation in the loop it seems
        neighboorhood_points = pc[:, dists < r]
        nk = neighboorhood_points.shape[1]  # the number of neighboring points

        assert nk >= 1, "ERROR"
        if nk == 1:  # there is no neighboring point
            # TODO: How to solve when there is no neighboring point,
            # I cant take the covariance of only 1 point
            hpk = zero
            # print("No neighboring points!")
        else:
            Sigma = torch.cov(neighboorhood_points)
            det = torch.linalg.det(Sigma)
            hpk = 1/2*torch.log(scaler*det + epsilon)
            # alfa 1.33 vertical angular resolution
        H = H + hpk
        if k % 100 == 0:
            print("Iteration: ", k)

    # 2. Reject point with the lowest entropy
    # TODO: Check speed of above when cuda is available. Can we make the above faster? They
    # got it down to 0.246 seconds on a Intel Core i7 processor (but they used voxels and stuff)
    # TODO: Double check! Make look nicer! Did not implement all details.
    # TODO: All separate and joint entropy to one function. Use classes to make this nice.

    return H


def get_dynamic_radius(d, sin_alpha):
    rmin = torch.tensor(0.2)
    rmax = torch.tensor(1.0)
    r = d*sin_alpha
    r_out = r
    if r <= rmin:
        r_out = rmin
    elif r >= rmax:
        r_out = rmax
    return r_out


def differential_entropy_joint(PCHandler) -> float:
    """
    Calculate the differential_entropy of the point cloud using the method described in the
    paper "CorAl – Are the point clouds Correctly Aligned?".

    Parameters:
    pc (torch.Tensor): The first point cloud as a PyTorch tensor with shape (3, n), where n is the
                       number of points.

    Returns:
    float: The entropy the point cloud.
    """
    dim_distribution = PCHandler.pc0_CS0.shape[0]
    assert dim_distribution == 3, "Expected 3-dim distribution"

    # TODO: set it dynamically later (I Put it to 0.1 to make it go faster)
    # but if not run dynamically, it should rather be 0.5 but it goes slower.

    epsilon = torch.tensor(10**(-8))  # Some offset to make sure not taking log of zero
    zero = torch.tensor(0)
    alpha = torch.tensor(1.33)  # vertical angular resolution degrees
    alpha_rad = torch.deg2rad(alpha)
    sin_alpha = torch.sin(alpha_rad)

    N_points = PCHandler.pc_union.shape[1]

    # N0_points = PCHandler.pc0_CS0.shape[1]

    scaler = (2*torch.tensor(np.pi)*torch.exp(torch.tensor(1)))**dim_distribution  # (2pi*e)^dim_distribution

    # pc_union = PCHandler.pc_union

    # TODO: Is this necessary? Is cuda available now?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pc_union = pc_union.to(device)

    H = torch.zeros(1, device=device)

    # TODO: Can I do this in faster way without the for loop?
    for k in range(N_points):
        pk = torch.unsqueeze(PCHandler.pc_union[:, k], dim=1)
        d = PCHandler.union_distances[k]
        r = get_dynamic_radius(d, sin_alpha)

        # Euclidean distance from key point to all other points in point cloud
        # this is the slowest operation in the loop it seems
        dists = torch.norm(PCHandler.pc_union-pk, p=2, dim=0)
        neighboorhood_points = PCHandler.pc_union[:, dists < r]
        nk = neighboorhood_points.shape[1]  # the number of neighboring points

        assert nk >= 1, "ERROR"
        if nk == 1:  # there is no neighboring point
            # TODO: How to solve when there is no neighboring point,
            # I cant take the covariance of only 1 point
            hpk = zero
            # print("No neighboring points!")
        else:
            Sigma = torch.cov(neighboorhood_points)
            det = torch.linalg.det(Sigma)
            hpk = 1/2*torch.log(scaler*det + epsilon)
            # alfa 1.33 vertical angular resolution
        H = H + hpk
        if k % 100 == 0:
            print("Iteration: ", k)

    # 2. Reject point with the lowest entropy
    # TODO: Check speed of above when cuda is available. Can we make the above faster? They
    # got it down to 0.246 seconds on a Intel Core i7 processor (but they used voxels and stuff)
    # TODO: Double check! Make look nicer! Did not implement all details.
    # TODO: All separate and joint entropy to one function. Use classes to make this nice.

    return H
