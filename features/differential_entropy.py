import torch
import numpy as np
import time

from features.feature_utils import get_dynamic_radii


def differential_entropy(PC, params: dict) -> float:
    """
    Calculate the differential_entropy of the point cloud using the method described in the
    paper "CorAl – Are the point clouds Correctly Aligned?".

    Parameters:
    PC (PC class): The first point cloud as a PyTorch tensor with shape (n, 3), where n is the
                   number of points.
    params (dict): Parameter dictionary.

    Returns:
    float: The entropy the point cloud.
    """
    device = PC.device
    assert PC.N_dim == 3, "Expected 3-dim distribution"
    # Get dynamic radii
    radii = get_dynamic_radii(PC.distances_to_origin, params)

    # Some offset to make sure not taking log of zero
    epsilon = torch.exp(torch.tensor(params.args.diff_entropy.log_epsilon))
    scaler = (2 * np.pi * np.exp(1)) ** PC.N_dim  # (2pi*e)^dim_distribution

    # Setup batches for the points of interest
    batch_size = params.args.batch_size_feature_extraction
    pc_batches = torch.split(PC.pc[PC.fps_inds], batch_size, dim=0)
    radii_batches = torch.split(radii[PC.fps_inds], batch_size, dim=0)

    # We will remove the lowest entropies later, so we want something low initially
    entropies = (
        1
        / 2
        * torch.log(epsilon)
        * torch.ones(PC.N_fps_points, device=device, dtype=PC.dtype)
    )
    from_ind = 0
    for pc_batch, radii_batch in zip(pc_batches, radii_batches):
        # Compute the distance matrix between pc_batch and pc
        dists = torch.cdist(pc_batch, PC.pc)

        # Create a mask with True values where the distance is less than the radius
        neighbor_mask = dists < radii_batch[:, None]
        del dists

        # Count the number of neighbors for each point in the batch
        n_neighbors_per_point_in_batch = torch.sum(neighbor_mask, dim=1)
        # Filter out neighborhoods with only one point
        bool_mask_valid_neighborhood = n_neighbors_per_point_in_batch > 1
        inds_to_valid_neighborhood = torch.nonzero(
            bool_mask_valid_neighborhood
        ).squeeze(dim=1)
        if inds_to_valid_neighborhood.numel() == 0:
            print(
                "Found no valid neighborhoods when calculating differential entropy for any of the"
                + " batches (must be > 1 points to calculate covariance in a neighborhood)"
            )
            current_batch_size = radii_batch.shape[0]
            from_ind += current_batch_size
            continue

        filtered_neighbor_mask = neighbor_mask[inds_to_valid_neighborhood]
        del neighbor_mask

        # Update neighbor count for valid neighborhoods
        filtered_n_neighbors = n_neighbors_per_point_in_batch[
            inds_to_valid_neighborhood
        ].unsqueeze(dim=1)

        # Apply the filtered neighbor mask to the point cloud batch
        masked_pc_batch = (PC.pc.unsqueeze(dim=0)) * (
            filtered_neighbor_mask.unsqueeze(dim=2)
        )

        # Compute the mean of the masked point cloud batch
        mu = torch.sum(masked_pc_batch, dim=1) / filtered_n_neighbors

        # Center the data by subtracting the mean
        centered_data = (masked_pc_batch - mu.unsqueeze(dim=1)) * (
            filtered_neighbor_mask.unsqueeze(dim=2)
        )

        del filtered_neighbor_mask, masked_pc_batch

        # Calculate covariance matrices
        covariances = (centered_data[..., None] * centered_data[..., None, :]).sum(
            dim=1
        ) / (filtered_n_neighbors.unsqueeze(dim=2) - 1)
        del centered_data

        # Compute determinants of covariance matrices
        determinants = torch.linalg.det(covariances)
        entropies[inds_to_valid_neighborhood + from_ind] = (
            1 / 2 * torch.log(scaler * determinants + epsilon)
        )
        current_batch_size = radii_batch.shape[0]
        from_ind += current_batch_size

    return entropies


def extract_differential_entropy(
    PC,
    n_neighbors_per_point_in_batch,
    neighbor_mask,
    inds_to_valid_neighborhood,
    params,
):
    # init batch entropies
    epsilon = torch.exp(torch.tensor(params.args.diff_entropy.log_epsilon))
    batch_entropies = (
        1
        / 2
        * torch.log(epsilon)
        * torch.ones(neighbor_mask.shape[0], device=PC.device, dtype=PC.pc.dtype)
    )

    if inds_to_valid_neighborhood.numel() == 0:
        print(
            "Found no valid neighborhoods when calculating differential entropy for any of the"
            + " batches (must be > 1 points to calculate covariance in a neighborhood)"
        )
        return batch_entropies

    filtered_neighbor_mask = neighbor_mask[inds_to_valid_neighborhood]
    #del neighbor_mask
    # Update neighbor count for valid neighborhoods
    filtered_n_neighbors = n_neighbors_per_point_in_batch[
        inds_to_valid_neighborhood
    ].unsqueeze(dim=1)

    # Apply the filtered neighbor mask to the point cloud batch
    masked_pc_batch = (PC.pc.unsqueeze(dim=0)) * (
        filtered_neighbor_mask.unsqueeze(dim=2)
    )

    # Compute the mean of the masked point cloud batch
    mu = torch.sum(masked_pc_batch, dim=1) / filtered_n_neighbors

    # Center the data by subtracting the mean
    centered_data = (masked_pc_batch - mu.unsqueeze(dim=1)) * (
        filtered_neighbor_mask.unsqueeze(dim=2)
    )
    covariances = (centered_data[..., None] * centered_data[..., None, :]).sum(
        dim=1
    ) / (filtered_n_neighbors.unsqueeze(dim=2) - 1)
    #del centered_data

    # Compute determinants of covariance matrices
    determinants = torch.linalg.det(covariances)
    # The determinant is only below zero due to numerical errors. In theory, it cannot be lower
    # than zero since the covariance matrix is symmetric positive semi-definite. To avoid nans
    # when taking the log, it thus becomes important to set the negative values to zero.
    determinants[determinants < 0] = 0

    scaler = (2 * np.pi * np.exp(1)) ** PC.N_dim
    batch_entropies[inds_to_valid_neighborhood] = (
        1 / 2 * torch.log(scaler * determinants + epsilon)
    )
    return batch_entropies


def filter_and_sum_entropies(entropies, params):
    n_points = entropies.shape[0]
    sorted_entropies = torch.sort(entropies)[0]
    keep_inds = round(params.args.diff_entropy.E_reject * n_points)
    H = torch.sum(
        sorted_entropies[keep_inds:]
    )  # only keep (1-E_reject) of the entropies
    return H.cpu().numpy()


def get_overlap_share(PC0, PC1, params):
    """
    We get the overlap share from the perspective of PC0.
    return: overlap_share
        the share of points in PC0 that has at least one point from PC1 in its neighborhood.
    """
    device = PC0.device
    pc0 = PC0.pc
    pc1 = PC1.pc
    radii0 = (
        get_dynamic_radii(PC0.distances_to_origin, params).to(device).unsqueeze(dim=1)
    )
    n_points = PC0.N_points
    del PC0, PC1

    batch_size = 8**4
    pc0_batches = torch.split(pc0, batch_size, dim=0)
    radii0_batches = torch.split(radii0, batch_size, dim=0)
    del pc0, radii0

    n_non_empty_neighborhoods = 0
    for pc0_batch, radii0_batch in zip(pc0_batches, radii0_batches):
        dists = torch.cdist(pc0_batch, pc1)
        lower_elements = dists < radii0_batch
        n_non_empty_neighborhoods += torch.sum(lower_elements.any(dim=1)).item()

    del radii0_batches, dists, pc1, pc0_batches, lower_elements
    overlap_share = n_non_empty_neighborhoods / n_points
    return overlap_share


def differential_entropy_metric(PC0, PC1, PCUnion, misaligned, params) -> float:
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
    overlap_misaligned_thresh = 0.10  # TODO: Possibly add to .cls file
    overlap_share = get_overlap_share(PC0, PC1, params)
    if overlap_share < overlap_misaligned_thresh:
        # return some large number which imply misalignment
        H_separate = np.array([0])
        H_joint = np.array([1e6])
        metric = H_joint - H_separate
        return metric, H_joint, H_separate

    # separate differential entropies
    entropies_PC0 = differential_entropy(PC0, params)
    entropies_PC1 = differential_entropy(PC1, params)
    # joint differential entropies
    entropies_joint = differential_entropy(PCUnion, params)

    H_PC0 = filter_and_sum_entropies(entropies_PC0, params)
    H_PC1 = filter_and_sum_entropies(entropies_PC1, params)
    N_pts_used = PCUnion.N_points * (1 - params.args.diff_entropy.E_reject)
    H_separate = (H_PC0 + H_PC1) / (N_pts_used)
    H_joint = filter_and_sum_entropies(entropies_joint, params) / N_pts_used

    # this is our alignment quality measure for the entire point cloud
    metric = H_joint - H_separate

    # display result
    if params.verbose:
        print(
            "[joint|sep|metric|misaligned]:",
            f"[{H_joint:.3f}|{H_separate:.3f}|{metric:.3f}|{misaligned}]",
            flush=True,
        )
    return metric, H_joint, H_separate


def differential_entropy_dataset(PC_scenes, params):
    metrics_aligned = []
    metrics_misaligned = []

    input_data = []
    labels = []

    for PC_scene in PC_scenes:
        for PC_pair in PC_scene:
            t1 = time.time()
            result, H_joint, H_separate = differential_entropy_metric(
                PC_pair.PC0,
                PC_pair.PC1,
                PC_pair.PCUnion,
                PC_pair.class_category,
                params,
            )
            input_data.append([H_joint, H_separate])
            labels.append(PC_pair.class_category)

            if PC_pair.class_category:
                metrics_misaligned.append(result)
            else:
                metrics_aligned.append(result)
            if params.verbose:
                if len(metrics_aligned) > 0:
                    print(
                        f"Mean abs metric aligned    {np.mean(np.abs(metrics_aligned)):.4f}",
                        f"(N = {len(metrics_aligned)})",
                    )
                if len(metrics_misaligned) > 0:
                    print(
                        f"Mean abs metric misaligned {np.mean(np.abs(metrics_misaligned)):.4f}",
                        f"(N = {len(metrics_misaligned)})",
                    )
                print(f"Execution time: {time.time() - t1:.3f} sec", flush=True)

    input_data = np.array(input_data)
    labels = np.array(labels)
    return input_data, labels
