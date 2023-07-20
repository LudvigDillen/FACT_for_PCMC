from geomloss import SamplesLoss
import torch
import time
import numpy as np


COMPUTATION_THRESHOLD = 2.5*1e7


def sinkhorn_distance(PC_pair, neighbor_mask_0, neighbor_mask_1):
    start = time.time()
    device = PC_pair.PC0.device
    dtype = PC_pair.PC0.pc.dtype
    current_batch_size = neighbor_mask_0.shape[0]

    # Mask the point clouds with NaN for non-neighboring points
    masked_tensor0 = torch.where(neighbor_mask_0, neighbor_mask_0.float(), torch.tensor(float('nan')))
    masked_tensor1 = torch.where(neighbor_mask_1, neighbor_mask_1.float(), torch.tensor(float('nan')))

    # Apply the mask to the point clouds
    batch_neighborhood_pc0 = (PC_pair.PC0.pc.unsqueeze(dim=0))*(masked_tensor0.unsqueeze(dim=2))
    batch_neighborhood_pc1 = (PC_pair.PC1.pc.unsqueeze(dim=0))*(masked_tensor1.unsqueeze(dim=2))

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
    # TODO: RE-run and make sure that computations does not crash ... I'll decrease COMPUTATION_THRES a bit ..
    # Use debug so you can check sizes if it crashes

    if computation_size > COMPUTATION_THRESHOLD:
        torch.cuda.empty_cache()
        current_batch_size = current_batch_size//2
        assert (current_batch_size >= 1), "ERROR: Batch size can't be lower than 1!"
        neighbor_mask_0 = torch.split(neighbor_mask_0, current_batch_size)
        neighbor_mask_1 = torch.split(neighbor_mask_1, current_batch_size)
        N_batches = len(neighbor_mask_0)
        dists = torch.empty((N_batches, current_batch_size), dtype=dtype, device=device)
        for i, (small_mask0, small_mask1) in enumerate(zip(neighbor_mask_0, neighbor_mask_1)):
            dists[i] = sinkhorn_distance(PC_pair, small_mask0, small_mask1)
        return_dists = torch.reshape(dists, (current_batch_size*N_batches, 1)).squeeze()
        return return_dists
    del neighbors_per_batch0, neighbors_per_batch1, neighbor_mask_0, neighbor_mask_1
    del masked_tensor0, masked_tensor1

    # 1. Sort the Points
    # Use torch.sort to sort the mask along dimension 1 (points dimension).
    # This will give a mask where False (nan points) come first.
    sorted_indices_pc0 = not_nan_mask_pc0.cpu().sort(dim=1, descending=True).indices.to(device)
    sorted_indices_pc1 = not_nan_mask_pc1.cpu().sort(dim=1, descending=True).indices.to(device)
    del not_nan_mask_pc0, not_nan_mask_pc1

    # Use the sorted indices to gather the points in the desired sorted order
    sorted_pc0 = torch.gather(batch_neighborhood_pc0, 1, sorted_indices_pc0.unsqueeze(2).expand(-1, -1, 3))
    sorted_pc1 = torch.gather(batch_neighborhood_pc1, 1, sorted_indices_pc1.unsqueeze(2).expand(-1, -1, 3))
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

    # 2. Creating a Mask:
    weights_pc0 = (~nan_inds0).any(dim=2).to(dtype)
    weights_pc1 = (~nan_inds1).any(dim=2).to(dtype)
    del nan_inds0, nan_inds1

    # Normalize the weights (important for Sinkhorn to ensure equal total masses)
    weights_pc0 /= weights_pc0.sum(dim=1, keepdim=True)
    weights_pc1 /= weights_pc1.sum(dim=1, keepdim=True)

    # 3. Computing the Sinkhorn Distances:
    loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
    torch.cuda.empty_cache()
    weights_pc0, weights_pc1 = weights_pc0.float(), weights_pc1.float()
    truncated_pc0, truncated_pc1 = truncated_pc0.float(), truncated_pc1.float()
    batch_distances = loss_fn(weights_pc0, truncated_pc0, weights_pc1, truncated_pc1).to(dtype)

    nan_distances = torch.isnan(batch_distances)
    # Some distances invalid, set these to the max distance calculated
    all_distances_non_nan = batch_distances[~nan_distances]
    if all_distances_non_nan.numel() > 0:
        max_distance = all_distances_non_nan.max()
    else:
        # Handle the case where the tensor is empty. Maybe set max_distance to a default value or raise a
        # specific error.
        max_distance = 1e3  # TODO what to do otherwise, it is unclear.

    batch_distances[nan_distances] = max_distance
    end = time.time()
    # print(f"Time {np.around(end - start, 4)}")
    return batch_distances


# CODE for sequential Sinkhorn computations ...
# start0 = time.time()
# # Cannot have blur 0 as the method takes the log of the blur variable.
# loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)  # this becomes very close to wasserstein

# # Initialize tensor to store distances
# N_neighborhoods = neighbor_mask_0.shape[0]
# distances1 = torch.zeros(N_neighborhoods, dtype=dtype, device=device)

# # Compute the loss for each pair of point clouds
# for i in range(N_neighborhoods):
#     pc0 = batch_neighborhood_pc0[i]
#     pc1 = batch_neighborhood_pc1[i]

#     # Exclude zero points
#     pc0 = pc0[~torch.isnan(pc0).any(dim=1)]
#     pc1 = pc1[~torch.isnan(pc1).any(dim=1)]

#     # Compute the loss
#     if pc0.shape[0] != 0 and pc1.shape[0] != 0:
#         distances1[i] = loss(pc0, pc1)
#     else:
#         distances1[i] = max_distance  # TODO: Change this or select something more appropriate later

# end0 = time.time()