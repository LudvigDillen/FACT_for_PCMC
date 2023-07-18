import torch


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


def divide_into_batches(tensor, N, max_batch_size=256):
    # Calculate the number of full batches needed
    full_batches = N // max_batch_size
    remaining = N % max_batch_size

    # Create batch sizes
    batch_sizes = [max_batch_size] * full_batches
    if remaining > 0:  # Only append remaining samples if remaining > 0
        batch_sizes.append(remaining)

    # Split the tensor into batches
    batches = []
    start = 0
    for size in batch_sizes:
        end = start + size
        batch = tensor[start:end]
        batches.append(batch)
        start = end

    return batches


def get_data_batches(PC, params):
    # Assuming PC_joint.fps_inds is a tensor, create an indices tensor
    indices = torch.arange(PC.fps_inds.shape[0])
    # Get the total number of samples
    N = indices.shape[0]
    # Now split the indices tensor into batches
    index_batches = divide_into_batches(indices, N)

    # Similarly, for pc_batches
    pc_batches = divide_into_batches(PC.pc[PC.fps_inds], N)

    # Calculate radii
    radii = get_dynamic_radii(PC.distances_to_origin, params.params_diff_entropy).to(PC.device)
    # Split radii into batches
    radii_batches = divide_into_batches(radii[PC.fps_inds], N)
    return index_batches, pc_batches, radii_batches
