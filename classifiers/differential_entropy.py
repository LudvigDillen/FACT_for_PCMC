import torch
import numpy as np
from scipy.spatial import KDTree


def differential_entropy_metric(PC0, PC1, PCUnion) -> float:
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

    # separate average differential entropy
    H_separate = (differential_entropy(
        PC0) + differential_entropy(PC1))/PCUnion.N_points
    # joint average differential entropy
    H_joint = differential_entropy(PCUnion)/PCUnion.N_points

    metric = H_joint - H_separate  # this is our alignment quality measure for the enitre point cloud
    return metric


def differential_entropy(PC: torch.Tensor) -> float:
    """
    Calculate the differential_entropy of the point cloud using the method described in the
    paper "CorAl – Are the point clouds Correctly Aligned?".

    Parameters:
    pc (torch.Tensor): The first point cloud as a PyTorch tensor with shape (3, n), where n is the
                       number of points.

    Returns:
    float: The entropy the point cloud.
    """
    dim_distribution = PC.N_dim
    assert dim_distribution == 3, "Expected 3-dim distribution"
    N_points = PC.N_points
    # TODO: set it dynamically later (I Put it to 0.1 to make it go faster)
    # but if not run dynamically, it should rather be 0.5 but it goes slower.
    epsilon = torch.tensor(10**(-8))  # Some offset to make sure not taking log of zero
    zero = torch.tensor(0)

    scaler = (2*torch.tensor(np.pi)*torch.exp(torch.tensor(1)))**dim_distribution  # (2pi*e)^dim_distribution

    # TODO: Is this necessary? Is cuda available now?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PC.pc = PC.pc.to(device)

    # Later for kd-tree
    # import time
    # t1 = time.time()

    # # Implement kd_tree
    # rmax = torch.tensor(1.0)
    # pc_swap = torch.swapaxes(PC.pc, 0, 1)
    # tree = KDTree(pc_swap)
    # indices_potential_neighborhoods = tree.query_ball_tree(tree, r=rmax)
    # neighboorhood_candidates = pc[:, indices_potential_neighborhoods[k]]

    # t2 = time.time()
    # print(t2-t1)

    H = torch.zeros(1, device=device)
    # Get dynamic radius
    radius = get_dynamic_radius(PC.distances_to_origin)

    # pc_swap = torch.swapaxes(PC.pc, 0, 1)
    # dists = torch.cdist(pc_swap, pc_swap)

    # TODO: Can I do this in faster way without the for loop?
    for k in range(N_points):

        pk = torch.unsqueeze(PC.pc[:, k], dim=1)

        # Later for kd-tree
        # # Euclidean distance from key point to all other points in point cloud
        # neighboorhood_candidates = pc[:, indices_potential_neighborhoods[k]]
        # # this is the slowest operation in the loop it seems
        # dists = torch.norm(neighboorhood_candidates-pk, p=2, dim=0)

        # johan stuff
        # el_matriso = torch.cdist(PC,PC)
        # neigh = (el_matriso < r[:,None])

        dists = torch.norm(PC.pc-pk, p=2, dim=0)  # this is the slowest operation in the loop it seems
        neighboorhood_points = PC.pc[:, dists[k] < radius[k]]
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
        H = H + hpk
        if k % 100 == 0:
            print("Iteration: ", k)

    # 2. Reject point with the lowest entropy
    # TODO: Check speed of above when cuda is available. Can we make the above faster? They
    # got it down to 0.246 seconds on a Intel Core i7 processor (but they used voxels and stuff)
    # TODO: Double check! Make look nicer! Did not implement all details.
    # TODO: All separate and joint entropy to one function. Use classes to make this nice.

    return H


def get_dynamic_radius(d):
    alpha = torch.tensor(1.33)  # vertical angular resolution degrees
    alpha_rad = torch.deg2rad(alpha)
    rmin = torch.tensor(0.2)
    rmax = torch.tensor(1.0)

    r = d*torch.sin(alpha_rad)
    r_out = r
    r_out[r <= rmin] = rmin
    r_out[r >= rmax] = rmax
    return r_out
