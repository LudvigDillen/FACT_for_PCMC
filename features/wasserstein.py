from geomloss import SamplesLoss
import torch
import time


COMPUTATION_THRESHOLD = 2.5*1e7


# TODO: Make this function easier to comprehend. Split it into several functions.
def sinkhorn_divergence(PC_pair, neighbor_mask_0, neighbor_mask_1, params_sinkhorn):
    current_batch_size = neighbor_mask_0.shape[0]
    device = PC_pair.PC0.device
    dtype = PC_pair.PC0.pc.dtype

    # start = time.time()
    # Mask the point clouds with NaN for non-neighboring points
    masked_tensor0 = torch.where(neighbor_mask_0, neighbor_mask_0.to(dtype),
                                 torch.tensor(float('nan')))
    masked_tensor1 = torch.where(neighbor_mask_1, neighbor_mask_1.to(dtype),
                                 torch.tensor(float('nan')))

    # Apply the mask to the point clouds
    batch_neighborhood_pc0 = (PC_pair.PC0.pc.unsqueeze(dim=0))*(masked_tensor0.unsqueeze(dim=2))
    batch_neighborhood_pc1 = (PC_pair.PC1.pc.unsqueeze(dim=0))*(masked_tensor1.unsqueeze(dim=2))

    # If computational load is too high. Do sequential computations.
    if current_batch_size == 1:
        del masked_tensor0, masked_tensor1
        dists = sequential_sinkhorn_computations(batch_neighborhood_pc0, batch_neighborhood_pc1,
                                                 params_sinkhorn)
        return dists

    # Create a mask where True indicates the point is not nan
    not_nan_mask_pc0 = ~torch.isnan(batch_neighborhood_pc0).any(dim=2)
    not_nan_mask_pc1 = ~torch.isnan(batch_neighborhood_pc1).any(dim=2)

    # Find the maximum number of non-NaN points in the point clouds across the batch
    neighbors_per_batch0 = not_nan_mask_pc0.sum(dim=1)
    neighbors_per_batch1 = not_nan_mask_pc1.sum(dim=1)
    max_neighbors_size = max(neighbors_per_batch0.max().item(), neighbors_per_batch1.max().item())

    # The computational complexity of Sinkhorn is O(M X N) where M and N are the sizes of the point clouds
    # Then, when batching it will become O(B X M X N).
    computation_size = current_batch_size*max_neighbors_size**2
    # This intimidating looking if-statement with recursion just make sure that we do not overuse
    # the GPU. So, if a computation threshold is reached, we decrease the batch_size by a factor 2.
    # This threshold could probably be tuned more carefully.
    if computation_size > COMPUTATION_THRESHOLD:
        old_batch_size = current_batch_size
        current_batch_size = (current_batch_size + 1)//2
        neighbor_mask_batches_0 = torch.split(neighbor_mask_0, current_batch_size)
        neighbor_mask_batches_1 = torch.split(neighbor_mask_1, current_batch_size)
        del neighbor_mask_0, neighbor_mask_1
        dists = torch.empty((old_batch_size), dtype=dtype, device=device)
        ind = 0
        for nm_0, nm_1 in zip(neighbor_mask_batches_0, neighbor_mask_batches_1):
            small_batch_size = nm_0.shape[0]
            dists[ind:ind + small_batch_size] = sinkhorn_divergence(PC_pair, nm_0, nm_1,
                                                                    params_sinkhorn)
            ind += small_batch_size
        return dists
    del neighbors_per_batch0, neighbors_per_batch1, masked_tensor0, masked_tensor1

    # 1. Sort the Points
    # Use torch.sort to sort the mask along dimension 1 (points dimension).
    # This will give a mask where False (nan points) come first.
    sorted_indices_pc0 = not_nan_mask_pc0.cpu().sort(dim=1, descending=True).indices.to(device)
    sorted_indices_pc1 = not_nan_mask_pc1.cpu().sort(dim=1, descending=True).indices.to(device)
    del not_nan_mask_pc0, not_nan_mask_pc1

    # Use the sorted indices to gather the points in the desired sorted order
    sorted_pc0 = torch.gather(batch_neighborhood_pc0, 1,
                              sorted_indices_pc0.unsqueeze(2).expand(-1, -1, 3))
    sorted_pc1 = torch.gather(batch_neighborhood_pc1, 1,
                              sorted_indices_pc1.unsqueeze(2).expand(-1, -1, 3))
    del sorted_indices_pc0, sorted_indices_pc1
    del batch_neighborhood_pc0, batch_neighborhood_pc1

    # 2. Truncate or Slice the Tensor
    truncated_pc0 = sorted_pc0[:, :max_neighbors_size, :].contiguous()
    truncated_pc1 = sorted_pc1[:, :max_neighbors_size, :].contiguous()
    del sorted_pc0, sorted_pc1

    nan_inds0 = torch.isnan(truncated_pc0)
    nan_inds1 = torch.isnan(truncated_pc1)
    truncated_pc0[nan_inds0] = 0.0
    truncated_pc1[nan_inds1] = 0.0

    # Check which batches have valid points in both point clouds

    # 2. Creating a Mask:
    weights_pc0 = (~nan_inds0).any(dim=2).float()
    weights_pc1 = (~nan_inds1).any(dim=2).float()
    del nan_inds0, nan_inds1

    # Find valid distances (must be corresp. points in both points clouds,
    # at least one point in each)
    valid_batches = (weights_pc0.any(dim=1)) & (weights_pc1.any(dim=1))
    if not valid_batches.any().item():
        batch_distances = torch.full((current_batch_size,), params_sinkhorn.max_dist,
                                     dtype=dtype, device=device)
        print(f"Found no valid distance for any of the {current_batch_size} batches!")
        return batch_distances
    valid_weights_pc0 = weights_pc0[valid_batches]
    valid_weights_pc1 = weights_pc1[valid_batches]
    del weights_pc0, weights_pc1

    # Normalize the weights (important for Sinkhorn to ensure equal total masses)S
    valid_weights_pc0 /= valid_weights_pc0.sum(dim=1, keepdim=True)
    valid_weights_pc1 /= valid_weights_pc1.sum(dim=1, keepdim=True)

    # 3. Computing the Sinkhorn Distances:
    loss_fn = SamplesLoss(loss="sinkhorn", p=params_sinkhorn.p, blur=params_sinkhorn.blur)

    # Only compute distances for valid batches
    valid_truncated_pc0 = truncated_pc0[valid_batches].float()
    valid_truncated_pc1 = truncated_pc1[valid_batches].float()
    del truncated_pc0, truncated_pc1

    res_distances = loss_fn(
        valid_weights_pc0, valid_truncated_pc0, valid_weights_pc1, valid_truncated_pc1)

    # Some distances invalid, set these to the max distance calculated
    inds_nan_distances = torch.isnan(res_distances)
    valid_distances = res_distances[~inds_nan_distances].to(dtype)
    max_distance = params_sinkhorn.max_dist
    valid_distances[inds_nan_distances] = max_distance
    del inds_nan_distances

    # Initialize batch_distances tensor with the max_distance
    batch_distances = torch.full((current_batch_size,), max_distance, dtype=dtype, device=device)

    # Place valid distances in the appropriate positions
    batch_distances[valid_batches] = valid_distances

    # end = time.time()
    # print(f"Time {np.around(end - start, 4)}")
    return batch_distances


def sequential_sinkhorn_computations(batch_neighborhood_pc0, batch_neighborhood_pc1,
                                     params_sinkhorn):
    print("Perform sequential Sinkhorn computations")
    dtype = batch_neighborhood_pc0.dtype
    device = batch_neighborhood_pc0.device
    # Cannot have blur 0 as the method takes the log of the blur variable.
    loss = SamplesLoss(loss="sinkhorn", p=params_sinkhorn.p, blur=params_sinkhorn.blur)

    # Initialize tensor to store distances
    N_neighborhoods = batch_neighborhood_pc0.shape[0]
    distances = torch.zeros(N_neighborhoods, dtype=dtype, device=device)

    # Compute the loss for each pair of point clouds
    for i in range(N_neighborhoods):
        pc0 = batch_neighborhood_pc0[i]
        pc1 = batch_neighborhood_pc1[i]

        # Exclude zero points
        pc0 = pc0[~torch.isnan(pc0).any(dim=1)]
        pc1 = pc1[~torch.isnan(pc1).any(dim=1)]

        # Compute the loss
        if pc0.shape[0] != 0 and pc1.shape[0] != 0:
            distances[i] = loss(pc0, pc1)
        elif distances.max() > 0:
            distances[i] = distances.max()
        else:
            distances[i] = params_sinkhorn.max_dist
    return distances
