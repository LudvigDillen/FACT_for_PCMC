import torch
import numpy as np
from scipy.spatial import KDTree


def differential_entropy_metric(PC0, PC1, PCUnion) -> float:
    """
    Calculate the differential entropy between two point clouds using the method described in the
    paper "CorAl – Are the point clouds Correctly Aligned?".

    Parameters:
    PC0 (PC class): The first point cloud as a pointcloud class with shape (n, 3), where n is the
                    number of points. This point cloud is in the coordinate system of pc0.
    PC1 (PC class): The second point cloud as a PyTorch tensor with shape (m, 3), where m is the
                    number of points. This point cloud is in the coordinate system of pc0.
    PCUnion (PC class): The second point cloud as a PyTorch tensor with shape (n+m, 3), where n+m is the
                        number of points. This point cloud is in the coordinate system of pc1.

    Returns:
    float: The differential entropy between the two point clouds.
    """
    E_reject = 0.0  # reject the 20% smallest entropies

    # separate average differential entropy
    H_separate = (differential_entropy(
        PC0, E_reject) + differential_entropy(PC1, E_reject))/(PCUnion.N_points*(1-E_reject))
    # joint average differential entropy
    H_joint = differential_entropy(PCUnion, E_reject)/(PCUnion.N_points*(1-E_reject))

    metric = H_joint - H_separate  # this is our alignment quality measure for the enitre point cloud
    return metric


def differential_entropy(PC: torch.Tensor, E_reject: float) -> float:
    """
    Calculate the differential_entropy of the point cloud using the method described in the
    paper "CorAl – Are the point clouds Correctly Aligned?".

    Parameters:
    PC (PC class): The first point cloud as a PyTorch tensor with shape (n, 3), where n is the
                   number of points.
    E_reject (float): Percentage of lowest entropies hpk to reject.

    Returns:
    float: The entropy the point cloud.
    """
    dim_distribution = PC.N_dim
    assert dim_distribution == 3, "Expected 3-dim distribution"

    epsilon = torch.tensor(10**(-6))  # Some offset to make sure not taking log of zero

    scaler = (2*torch.tensor(np.pi)*torch.exp(torch.tensor(1)))**dim_distribution  # (2pi*e)^dim_distribution

    # TODO: Is this necessary? Is cuda available now?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PC.pc = PC.pc.to(device)

    H = torch.zeros(1, device=device)
    # Get dynamic radius
    radius = get_dynamic_radius(PC.distances_to_origin)
    batch_size = 5000
    pc_batches = torch.split(PC.pc, batch_size, dim=0)
    num_batches = len(pc_batches)
    entropies = torch.zeros(PC.N_points)

    for batch_number, pc_batch in enumerate(pc_batches):
        # Compute the distance matrix between pc_batch and PC.pc
        dists = torch.cdist(pc_batch, PC.pc)

        # Create a mask with True values where the distance is less than the radius
        neighbor_mask = (dists < radius)

        for k in range(pc_batch.shape[0]):
            neighboorhood_points = PC.pc[neighbor_mask[k]]
            nk = neighboorhood_points.shape[0]  # the number of neighboring points

            assert nk >= 1, "ERROR"
            if nk > 1:
                # Calculate covariance, seems slightly faster than torch.cov
                centered_points = neighboorhood_points - neighboorhood_points.mean(dim=0, keepdim=True)
                Sigma = torch.matmul(centered_points.T, centered_points) / (nk - 1)

                det = torch.linalg.det(Sigma)
                j = batch_number*batch_size + k
                entropies[j] = 1/2*torch.log(scaler*det + epsilon)

        print(f"Batch {batch_number + 1} of {num_batches}")
    sorted_entropies = torch.sort(entropies)[0]
    j = round(E_reject*PC.N_points)
    H = sorted_entropies[j:].sum()  # only keep (1-E_reject) of the entropies
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
