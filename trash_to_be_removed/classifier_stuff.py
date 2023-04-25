import torch
import cupy as cp
import numpy as np


def differential_entropy_inner_loop(
        samples_in_batch, pc, neighbor_mask, scaler, epsilon, entropies, entropy_ind):
    for k in range(samples_in_batch):
        neighboorhood_points = pc[neighbor_mask[k]]
        nk = neighboorhood_points.shape[0]  # the number of neighboring points

        # assert nk >= 1, "ERROR"
        if nk > 1:
            # Calculate covariance, seems slightly faster than torch.cov
            centered_points = neighboorhood_points - neighboorhood_points.mean(dim=0, keepdim=True)
            Sigma = torch.matmul(torch.transpose(centered_points, 0, 1), centered_points) / (nk - 1)

            det = torch.linalg.det(Sigma)
            # Remove below to gain speed
            log_argument = scaler*det + epsilon
            if log_argument < 0:
                print(f"problems will occur! Argument value {log_argument}")
            entropies[entropy_ind] = 1/2*torch.log(log_argument)
        entropy_ind += 1
    return entropy_ind, entropies


def get_dynamic_radius_cp(d):
    alpha = np.float32(1.33)  # vertical angular resolution degrees
    alpha_rad = np.deg2rad(alpha)
    rmin = np.float32(0.2)
    rmax = np.float32(1.0)

    r = d*np.sin(alpha_rad)
    r_out = r
    r_out[r <= rmin] = rmin
    r_out[r >= rmax] = rmax
    return r_out


def differential_entropy_cp(PC, E_reject: float) -> float:
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

    pc = cp.asarray(PC.pc)

    # Some offset to make sure not taking log of zero
    epsilon = np.float32(10**(-7))
    scaler = (2*np.pi*np.exp(1))**dim_distribution  # (2pi*e)^dim_distribution

    H = cp.zeros(1)
    # Get dynamic radius
    radius = get_dynamic_radius_cp(cp.asarray(PC.distances_to_origin))

    # Setup batches
    batch_size = 5000
    pc_batches = torch.split(PC.pc, batch_size, dim=0)
    num_batches = len(pc_batches)

    entropies = cp.zeros(PC.N_points)
    entropy_ind = 0

    for batch_number, pc_batch in enumerate(pc_batches):
        # Compute the distance matrix between pc_batch and pc
        dists = cp.asarray(torch.cdist(pc_batch, PC.pc))

        # Create a mask with True values where the distance is less than the radius
        neighbor_mask = (dists < radius)
        # neighbor_mean = (neighbor_mask.T * pc).
        # (pc.T * neighbor_mask * pc).sum(dim = 2)
        for k in range(pc_batch.shape[0]):
            neighboorhood_points = pc[neighbor_mask[k]]
            nk = neighboorhood_points.shape[0]  # the number of neighboring points
            # assert nk >= 1, "ERROR"
            if nk > 1:
                # Calculate covariance, seems slightly faster than torch.cov
                Sigma = cp.cov(neighboorhood_points.T)
                det = np.linalg.det(Sigma.get())
                # Remove below to gain speed
                log_argument = scaler*det + epsilon
                if log_argument < 0:
                    print(f"problems will occur! Argument value {log_argument}")
                entropies[entropy_ind] = 1/2*np.log(log_argument)
            entropy_ind += 1

        print(f"Batch {batch_number + 1} of {num_batches}")
    sorted_entropies = np.sort(entropies)
    j = round(E_reject*PC.N_points)
    H = sorted_entropies[j:].sum()  # only keep (1-E_reject) of the entropies
    return H
