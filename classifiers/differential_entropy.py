import torch
import numpy as np
import time


def get_dynamic_radius(d):
    alpha = torch.tensor(1.33, dtype=torch.float32)  # vertical angular resolution degrees
    alpha_rad = torch.deg2rad(alpha)
    rmin = torch.tensor(0.2, dtype=torch.float32)
    rmax = torch.tensor(1.0, dtype=torch.float32)

    r = d*torch.sin(alpha_rad)
    r_out = r
    r_out[r <= rmin] = rmin
    r_out[r >= rmax] = rmax
    return r_out


@torch.no_grad()
def differential_entropy(PC, E_reject: float) -> float:
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pc = PC.pc.to(device)
    n_points = PC.N_points
    # Get dynamic radius
    radius = get_dynamic_radius(PC.distances_to_origin).to(device)
    del PC

    # Some offset to make sure not taking log of zero
    epsilon = torch.tensor(10**(-7), dtype=torch.float64).to(device)
    scaler = ((2*torch.tensor(np.pi, dtype=torch.float64)*torch.exp(torch.tensor(1, dtype=torch.float64))
               )**dim_distribution).to(device)  # (2pi*e)^dim_distribution

    # Setup batches
    batch_size = 8**3
    pc_batches = torch.split(pc, batch_size, dim=0)
    num_batches = len(pc_batches)

    # We will remove the lowest entropies later, so we want something low initially
    entropies = -100*torch.ones(n_points, device=device)
    from_ind = 0
    to_ind = 0

    # Timing
    dists_time = 0
    neighborhood_times = 0
    mu_times = 0
    center_data_times = 0
    covariances_einsum_times = 0
    entropies_times = 0

    for batch_number, pc_batch in enumerate(pc_batches):
        # Compute the distance matrix between pc_batch and pc

        t1 = time.perf_counter()
        dists = torch.cdist(pc_batch, pc)
        del pc_batch
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        dists_time += t2 - t1

        # Create a mask with True values where the distance is less than the radius
        neighbor_mask = (dists < radius)
        del dists

        n_neighbors_per_point_in_batch = torch.sum(neighbor_mask, dim=1)
        # keep only neighborboods with more than one point
        inds_to_valid_neighborhoods = (n_neighbors_per_point_in_batch > 1)
        filtered_neighbor_mask = neighbor_mask[inds_to_valid_neighborhoods]
        del neighbor_mask

        filtered_n_neighbors = n_neighbors_per_point_in_batch[inds_to_valid_neighborhoods].unsqueeze(dim=1)

        del inds_to_valid_neighborhoods
        del n_neighbors_per_point_in_batch

        masked_pc_batch = (pc.unsqueeze(dim=0))*(filtered_neighbor_mask.unsqueeze(dim=2))
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        neighborhood_times += t3 - t2

        mu = torch.sum(masked_pc_batch, dim=1) / filtered_n_neighbors
        torch.cuda.synchronize()
        t31 = time.perf_counter()
        mu_times += t31-t3

        centered_data = (masked_pc_batch - mu.unsqueeze(dim=1))*(filtered_neighbor_mask.unsqueeze(dim=2))
        torch.cuda.synchronize()
        t32 = time.perf_counter()
        center_data_times += t32 - t31

        del filtered_neighbor_mask
        del masked_pc_batch
        del mu
        n_points_in_batch = filtered_n_neighbors.shape[0]
        covariances = (centered_data[..., None] * centered_data[..., None, :]
                       ).sum(dim=1) / (filtered_n_neighbors.unsqueeze(dim=2) - 1)

        torch.cuda.synchronize()
        t4 = time.perf_counter()
        covariances_einsum_times += t4 - t32
        del centered_data
        del filtered_n_neighbors
        determinants = torch.linalg.det(covariances)
        del covariances
        to_ind += n_points_in_batch
        entropies[from_ind:to_ind] = 1/2*torch.log(scaler*determinants + epsilon)
        del determinants
        from_ind = to_ind
        torch.cuda.synchronize()
        t5 = time.perf_counter()
        entropies_times += t5 - t4
        del n_points_in_batch

        decimals = 4
        print(f"Batch {batch_number + 1} of {num_batches}\n",
              f"dists_time                  {np.around(dists_time, decimals)}\n",
              f"neighborhood_times          {np.around(neighborhood_times, decimals)}\n",
              f"mu_times                    {np.around(mu_times, decimals)}\n",
              f"center_data_times           {np.around(center_data_times, decimals)}\n",
              f"covariances_einsum_times    {np.around(covariances_einsum_times, decimals)}\n",
              f"entropies_times             {np.around(entropies_times, decimals)}\n")

    sorted_entropies = torch.sort(entropies)[0]
    j = round(E_reject*n_points)
    H = sorted_entropies[j:].sum()  # only keep (1-E_reject) of the entropies
    return H


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
    E_reject = 0.20  # reject the 20% smallest entropies

    # separate average differential entropy
    H_separate = (differential_entropy(
        PC0, E_reject) + differential_entropy(PC1, E_reject))/(PCUnion.N_points*(1-E_reject))
    # joint average differential entropy
    H_joint = differential_entropy(PCUnion, E_reject)/(PCUnion.N_points*(1-E_reject))

    metric = H_joint - H_separate  # this is our alignment quality measure for the enitre point cloud
    return metric
