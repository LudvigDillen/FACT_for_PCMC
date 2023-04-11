import torch
import numpy as np


def differential_entropy_metric(pc0: torch.Tensor, pc1: torch.Tensor) -> float:
    """
    Calculate the differential entropy between two point clouds using the method described in the
    paper "CorAl – Are the point clouds Correctly Aligned?".

    Parameters:
    pc0 (torch.Tensor): The first point cloud as a PyTorch tensor with shape (3, n), where n is the
                        number of points.
    pc1 (torch.Tensor): The second point cloud as a PyTorch tensor with shape (3, m), where m is the
                        number of points.

    Returns:
    float: The differential entropy between the two point clouds.
    """
    N_points0 = pc0.shape[1]  # get number of points in point cloud
    N_points1 = pc1.shape[1]  # get number of points in point cloud
    N_union_points = N_points0 + N_points1  # number of points in the union of the point clouds
    pc_union = torch.cat((pc0, pc1), dim=1)  # Create union of the two point clouds

    # separate average differential entropy
    H_separate = (differential_entropy(pc0) + differential_entropy(pc1))/N_union_points
    # joint average differential entropy
    H_joint = differential_entropy(pc_union)/N_union_points

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
    r = 0.1  # radius

    scaler = (2*torch.tensor(np.pi)*torch.exp(torch.tensor(1)))**dim_distribution  # (2pi*e)^dim_distribution

    # TODO: Is this necessary? Is cuda available now?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pc = pc.to(device)

    H = torch.zeros(1, device=device)

    # TODO: Can I do this in faster way without the for loop?
    for k in range(N_points):
        pk = torch.unsqueeze(pc[:, k], dim=1)
        # Euclidean distance from key point to all other points in point cloud
        dists = torch.norm(pc-pk, p=2, dim=0)  # this is the slowest operation in the loop it seems
        neighboorhood_points = pc[:, dists < r]
        nk = neighboorhood_points.shape[1]  # the number of neighboring points

        assert nk >= 1, "ERROR"
        if nk == 1:  # there is no neighboring point
            # TODO: How to solve when there is no neighboring point,
            # I cant take the covariance of only 1 point
            hpk = torch.tensor(0)
            # print("No neighboring points!")
        else:
            Sigma = torch.cov(neighboorhood_points)
            epsilon = torch.tensor(10**(-8))  # Some offset to make sure not taking log of zero
            det = torch.linalg.det(Sigma)
            hpk = 1/2*torch.log(scaler*det + epsilon)
            # alfa 1.33 vertical angular resolution
        H = H + hpk
        if k % 100 == 0:
            print("Iteration: ", k)

    # TODO: Implement the entropy calculation based on the method described in the paper.
    # 1. Implement r as function of distance to sensor (how to handle multiple sensors,
    #                                                   average distance to the two sensors perhaps)
    # See LaTex
    # 2. Reject point with the lowest entropy
    # TODO: Check speed of above when cuda is available. Can we make the above faster? They
    # got it down to 0.246 seconds on a Intel Core i7 processor (but they used voxels and stuff)
    # TODO: Double check! Make look nicer! Did not implement all details.

    return H
