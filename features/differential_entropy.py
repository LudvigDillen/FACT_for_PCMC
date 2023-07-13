import torch
import numpy as np
import time

from classifiers.regression import perform_logistic_regression


def get_dynamic_radii(d, params):
    # Unpack parameters
    rmin = torch.tensor(params["rmin"])
    rmax = torch.tensor(params["rmax"])
    alpha = torch.tensor(params["alpha"])
    #
    alpha_rad = torch.deg2rad(alpha)
    r = d*torch.sin(alpha_rad)
    r_out = r
    r_out[r < rmin] = rmin
    r_out[r > rmax] = rmax
    return r_out


@torch.no_grad()
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
    radii = get_dynamic_radii(PC.distances_to_origin, params).to(device)

    # Some offset to make sure not taking log of zero
    epsilon = torch.exp(torch.tensor(params["log_epsilon"]))
    scaler = ((2*torch.tensor(np.pi, dtype=torch.float64)*torch.exp(torch.tensor(1, dtype=torch.float64))
               )**PC.N_dim).to(device)  # (2pi*e)^dim_distribution

    # Setup batches for the points of interest
    batch_size = 8**3
    pc_batches = torch.split(PC.pc[PC.fps_inds], batch_size, dim=0)
    radii_batches = torch.split(radii[PC.fps_inds], batch_size, dim=0)
    N_fps_points = len(PC.fps_inds)

    # We will remove the lowest entropies later, so we want something low initially
    entropies = 1/2*torch.log(epsilon)*torch.ones(N_fps_points, device=device)
    from_ind = 0
    to_ind = 0

    for pc_batch, radii_batch in zip(pc_batches, radii_batches):
        # Compute the distance matrix between pc_batch and pc
        dists = torch.cdist(pc_batch, PC.pc)
        del pc_batch

        # Create a mask with True values where the distance is less than the radius
        neighbor_mask = (dists < radii_batch[:, None])
        del dists

        # Count the number of neighbors for each point in the batch
        n_neighbors_per_point_in_batch = torch.sum(neighbor_mask, dim=1)
        # Filter out neighborhoods with only one point
        inds_to_valid_neighborhoods = (n_neighbors_per_point_in_batch > 1)
        filtered_neighbor_mask = neighbor_mask[inds_to_valid_neighborhoods]
        del neighbor_mask

        # Update neighbor count for valid neighborhoods
        filtered_n_neighbors = n_neighbors_per_point_in_batch[inds_to_valid_neighborhoods].unsqueeze(dim=1)
        del inds_to_valid_neighborhoods
        del n_neighbors_per_point_in_batch

        # Apply the filtered neighbor mask to the point cloud batch
        masked_pc_batch = (PC.pc.unsqueeze(dim=0))*(filtered_neighbor_mask.unsqueeze(dim=2))

        # Compute the mean of the masked point cloud batch
        mu = torch.sum(masked_pc_batch, dim=1) / filtered_n_neighbors

        # Center the data by subtracting the mean
        centered_data = (masked_pc_batch - mu.unsqueeze(dim=1))*(filtered_neighbor_mask.unsqueeze(dim=2))

        del filtered_neighbor_mask, masked_pc_batch, mu

        # Calculate covariance matrices
        covariances = (centered_data[..., None] * centered_data[..., None, :]
                       ).sum(dim=1) / (filtered_n_neighbors.unsqueeze(dim=2) - 1)
        del centered_data

        # Get the number of points in the batch
        n_points_in_batch = filtered_n_neighbors.shape[0]
        del filtered_n_neighbors

        # Compute determinants of covariance matrices
        determinants = torch.linalg.det(covariances)
        del covariances
        to_ind += n_points_in_batch
        entropies[from_ind:to_ind] = 1/2*torch.log(scaler*determinants + epsilon)
        del determinants
        from_ind = to_ind
        del n_points_in_batch

    return entropies


def filter_and_sum_entropies(entropies, params):
    n_points = entropies.shape[0]
    sorted_entropies = torch.sort(entropies)[0]
    keep_inds = round(params["E_reject"]*n_points)
    H = torch.sum(sorted_entropies[keep_inds:])  # only keep (1-E_reject) of the entropies
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
    radii0 = get_dynamic_radii(PC0.distances_to_origin, params).to(device).unsqueeze(dim=1)
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


def differential_entropy_metric(PC0, PC1, PCUnion, misaligned, params, verbose=True) -> float:
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
    overlap_misaligned_thresh = 0.10
    overlap_share = get_overlap_share(PC0, PC1, params)
    if overlap_share < overlap_misaligned_thresh:
        # return some large number which imply misalignment
        H_separate = np.array([0])
        H_joint = np.array([1e10])
        metric = H_joint - H_separate
        return metric, H_joint, H_separate

    # separate differential entropies
    entropies_PC0 = differential_entropy(PC0, params)
    entropies_PC1 = differential_entropy(PC1, params)
    # joint differential entropies
    entropies_joint = differential_entropy(PCUnion, params)

    H_PC0 = filter_and_sum_entropies(entropies_PC0, params)
    H_PC1 = filter_and_sum_entropies(entropies_PC1, params)
    N_pts_used = PCUnion.N_points*(1-params["E_reject"])
    H_separate = (H_PC0 + H_PC1)/(N_pts_used)
    H_joint = filter_and_sum_entropies(entropies_joint, params)/(N_pts_used)
    metric = H_joint - H_separate  # this is our alignment quality measure for the enitre point cloud

    # display result
    if verbose:
        print("[joint|sep|metric|misaligned]:",
              f"[{np.round(H_joint, 3)}|{np.round(H_separate, 3)}|{np.round(metric, 3)}|{misaligned}]",
              flush=True)
    return metric, H_joint, H_separate


def differential_entropy_pointwise(PC_scenes, params):
    N_scenes = PC_scenes.shape[0]
    N_samples_per_scene = PC_scenes.shape[1]
    for scene_number in range(N_scenes):
        for sample_number in range(N_samples_per_scene):
            PC_pair = PC_scenes[scene_number][sample_number]
            # separate differential entropies
            entropies_PC0 = differential_entropy(PC_pair.PC0, params)
            entropies_PC1 = differential_entropy(PC_pair.PC1, params)
            # joint differential entropies
            entropies_joint = differential_entropy(PC_pair.PCUnion, params)

            entropies_joint_from_PC0 = entropies_joint[:PC_pair.PC0.N_points]
            entropies_joint_from_PC1 = entropies_joint[PC_pair.PC0.N_points:]
            PC_pair.PC0.set_joint_diff_entropy(entropies_joint_from_PC0)
            PC_pair.PC1.set_joint_diff_entropy(entropies_joint_from_PC1)
            PC_pair.PC0.set_sep_diff_entropy(entropies_PC0)
            PC_pair.PC1.set_sep_diff_entropy(entropies_PC1)
            PC_scenes[scene_number][sample_number] = PC_pair
    return PC_scenes


def differential_entropy_dataset(PC_scenes, params, verbose=True):
    metrics_aligned = []
    metrics_misaligned = []

    input_data = []
    labels = []

    for PC_scene in PC_scenes:
        for PC_pair in PC_scene:
            t1 = time.time()
            result, H_joint, H_separate = differential_entropy_metric(
                PC_pair.PC0, PC_pair.PC1, PC_pair.PCUnion, PC_pair.misaligned, params, verbose)
            input_data.append([H_joint, H_separate])
            labels.append(PC_pair.misaligned)

            if PC_pair.misaligned:
                metrics_misaligned.append(result)
            else:
                metrics_aligned.append(result)
            if verbose:
                if len(metrics_aligned) > 0:
                    print(f"Mean abs metric aligned    {np.around(np.mean(np.abs(metrics_aligned)), 4)}",
                          f"(N = {len(metrics_aligned)})")
                if len(metrics_misaligned) > 0:
                    print(f"Mean abs metric misaligned {np.around(np.mean(np.abs(metrics_misaligned)), 4)}",
                          f"(N = {len(metrics_misaligned)})")
                print(f"Execution time: {round(time.time() - t1, 3)} sec", flush=True)

    input_data = np.array(input_data)
    labels = np.array(labels)
    return input_data, labels


def differential_entropy_test_accuracy(params, PC_scenes_training, PC_scenes_test, verbose=False):
    print("\nGetting training data\n")
    X_train, y_train = differential_entropy_dataset(PC_scenes_training, params, verbose=verbose)
    print("\nGetting test data\n")
    X_test, y_test = differential_entropy_dataset(PC_scenes_test, params, verbose=verbose)
    print("\nPerform logistic regression\n")
    model, accuracy_test = perform_logistic_regression(X_train, X_test, y_train, y_test, verbose=verbose)
    print(f"Accuracy: {accuracy_test} with parameters\n {params}", flush=True)
    return accuracy_test
