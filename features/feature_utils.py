import sys
import torch
import numpy as np

from utils.pointnet_util import pc_normalize_batch, normalize_features_batch


def get_dynamic_radii(d, params):
    rmin = params.args.neighborhood.rmin
    rmax = params.args.neighborhood.rmax
    r = d * params.sin_alpha
    r[r < rmin] = rmin
    r[r > rmax] = rmax
    return r


def get_dynamic_radii_joint(PC_pair, params):
    PC_joint = PC_pair.PCUnion

    pc_joint_fps_lcs0 = PC_joint.pc[PC_joint.fps_inds]
    dists_to_pose0 = torch.norm(pc_joint_fps_lcs0, dim=1)

    T_lcs0_to_lcs1 = torch.matmul(torch.inverse(PC_pair.pose1), PC_pair.pose0)
    R = T_lcs0_to_lcs1[:3, :3]
    t = T_lcs0_to_lcs1[:3, 3]
    pc_joint_fps_lcs1 = torch.matmul(pc_joint_fps_lcs0, R.T) + t
    dists_to_pose1 = torch.norm(pc_joint_fps_lcs1, dim=1)

    if params.args.neighborhood.k == "joint":
        d = (dists_to_pose0 + dists_to_pose1) / 2
    elif params.args.neighborhood.k == "adaptive":
        d = (
            np.sqrt(2)
            * dists_to_pose0
            * dists_to_pose1
            / torch.sqrt(dists_to_pose0**2 + dists_to_pose1**2)
        )
    else:
        sys.exit("ERROR: Neighborhood k-parameter not in [joint, adaptive]")
    r_out = get_dynamic_radii(d, params)
    return r_out


def divide_into_even_batches(N, max_batch_size):
    N_batches = N // max_batch_size
    if N_batches % 2 == 1:
        N_batches += 1
    elif N_batches == 0:
        N_batches = 2

    samples_counted = 0
    N_batches_per_pc = int(N_batches / 2)
    one_pc_batch_sizes = np.empty(N_batches_per_pc, dtype=int)
    for i in range(N_batches_per_pc):
        if samples_counted + 2 * max_batch_size <= N:
            samples_to_append = max_batch_size
        else:
            samples_left = N - samples_counted
            assert (
                samples_left % 2 == 0
            ), "There should be an even number of samples left"
            samples_to_append = int(samples_left / 2)
        one_pc_batch_sizes[i] = samples_to_append
        samples_counted += 2 * samples_to_append
    both_pc_batch_sizes = np.append(one_pc_batch_sizes, one_pc_batch_sizes)
    return both_pc_batch_sizes


def divide_into_batches(pc, max_batch_size):
    N = pc.shape[0]
    batch_sizes = divide_into_even_batches(N, max_batch_size)

    # Split the tensor into batches
    batches = []
    start = 0
    for size in batch_sizes:
        end = start + size
        batch = pc[start:end]
        batches.append(batch)
        start = end

    return batches


def get_data_batches(PC_pair, params):
    PC = PC_pair.PCUnion
    batch_size = params.batch_size_feature_extraction
    # Assuming PC_joint.fps_inds is a tensor, create an indices tensor
    indices = torch.arange(PC.fps_inds.shape[0])
    # TODO Maybe handle if this scenario appears: if PC.N_points < PC.fps_inds.shape[0]:
    # It is quite unlikely if we do not take too many fps points, but it can happen
    # Now split the indices tensor into batches
    index_batches = divide_into_batches(indices, batch_size)

    # Similarly, for pc_batches
    pc_batches = divide_into_batches(PC.pc[PC.fps_inds], batch_size)

    # Calculate radii
    if params.args.neighborhood.k == "normal":
        radii = get_dynamic_radii(PC.distances_to_origin[PC.fps_inds], params)
    elif params.args.neighborhood.k in ["joint", "adaptive"]:
        radii = get_dynamic_radii_joint(PC_pair, params)
    else:
        sys.exit("ERROR: Neighborhood k-parameter not in [normal, joint, adaptive]")
    # Split radii into batches
    # TODO: Check that the radii is on cuda ...
    radii_batches = divide_into_batches(radii, batch_size)

    return index_batches, pc_batches, radii_batches


def process_features(args):
    """
    Sets up stuff related to args and configs.
    """
    args.n_samples = args.n_scenes * args.n_samples_per_scene

    # The order of the keys matters. Ensure both dictionaries have the same order.
    keys = list(args.features_to_use.keys())

    if set(keys) != set(args.features_to_create.keys()):
        raise ValueError("Both dictionaries should have the same set of keys.")

    result = []
    for feature_ind, key in enumerate(keys):
        use_val = args.features_to_use[key]
        create_val = args.features_to_create[key]

        if use_val and create_val:
            result.append(1)
        elif not use_val and create_val:
            result.append(0)
        elif use_val and not create_val:
            raise ValueError(
                f"'use' is True for {key} but 'create' is False. This is prohibited."
            )
        # If both are False, we don't append anything and continue to the next iteration.

        if use_val:
            if key == "use_jde":
                args.inds_features.ind_joint_de = feature_ind
            elif key == "use_sde":
                args.inds_features.ind_sep_de = feature_ind
            elif key == "use_sd":
                args.inds_features.ind_sink_div = feature_ind

    args.feature_filter = result
    return args


def number_of_features(args):
    if args.ablation:
        n_features = int(sum(args.ablation_feature_filter))
    else:
        n_features = int(sum(args.feature_filter))
    return n_features, args


# def append_spatial_features(args, points):
#     if args.xyz_features.use_xyz:
#         points = torch.cat((points, points[..., :3].clone()), dim=-1)
#     if args.xyz_features.use_z:
#         points = torch.cat((points, points[..., 2, None].clone()), dim=-1)
#     if args.xyz_features.use_norm_xyz:
#         xyz_norm = torch.norm(points[..., :3], dim=-1, keepdim=True)
#         points = torch.cat((points, xyz_norm), dim=-1)
#     return points


def normalize_data_on_condition(args, points):
    """
    Normalize point cloud data based on the given conditions.

    Parameters:
    - args: Argument object containing normalization settings.
    - points: The point cloud data to normalize.

    Returns:
    - Normalized point cloud data.
    """
    B, N, C = points.shape
    if args.normalization.pos_enc:
        if args.normalization.pos_enc_batch:
            xyz_flatten = points[..., :3].reshape(1, B * N, 3)
            xyz_flatten_norm = pc_normalize_batch(xyz_flatten)
            points[..., :3] = xyz_flatten_norm.reshape(B, N, 3)
        else:
            points[..., :3] = pc_normalize_batch(points[..., :3])
    if args.normalization.features:
        if args.normalization.features_batch:
            features_flatten = points[..., 3:].reshape(1, B * N, C - 3)
            features_flatten_norm = normalize_features_batch(features_flatten, args)
            points[..., 3:] = features_flatten_norm.reshape(B, N, C - 3)
        else:
            points[..., 3:] = normalize_features_batch(points[..., 3:], args)
    return points


def augment_data(args, points):
    if args.aug.dropout:
        points = random_point_dropout(points)
    if args.aug.scale:
        points[:, :, 0:3] = random_scale_point_cloud(points[:, :, 0:3])
    if args.aug.shift:
        points[:, :, 0:3] = random_shift_point_cloud(points[:, :, 0:3])
    if args.aug.rotate:
        points[:, :, 0:3] = random_rotate_point_cloud(points[:, :, 0:3])
    if args.aug.jitter:
        points[:, :, 0:3] = random_jitter_point_cloud(
            points[:, :, 0:3],
            sigma=args.aug.jitter_settings.sigma,
            clip=args.aug.jitter_settings.clip,
        )
    return points


def random_shift_point_cloud(batch_data, shift_range=0.1):
    """Randomly shift point cloud. Shift is per point cloud.
    Input:
      BxNx3 tensor, original batch of point clouds
    Return:
      BxNx3 tensor, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    batch_max_sample = torch.max(torch.norm(batch_data, dim=2), dim=1, keepdim=True)[
        0
    ]  # Bx1
    shifts = (
        torch.rand((B, 3), device=batch_data.device) * 2 * shift_range - shift_range
    )
    shifts_scaled = batch_max_sample * shifts  # Bx3
    batch_data_shifted = batch_data + shifts_scaled.unsqueeze(1)  # BxNx3
    return batch_data_shifted


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """Randomly scale the point cloud. Scale is per point cloud.
    Input:
        BxNx3 tensor, original batch of point clouds
    Return:
        BxNx3 tensor, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = (
        torch.rand(B, device=batch_data.device) * (scale_high - scale_low) + scale_low
    )
    scaled_batch_data = batch_data * scales[:, None, None]
    return scaled_batch_data


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    """
    Perform random point dropout on a batch of point clouds.

    Parameters:
    - batch_pc (torch.Tensor): Input tensor of shape BxNxF.
    - max_dropout_ratio (float): Maximum ratio of points to dropout.

    Returns:
    - torch.Tensor: Tensor after dropout of shape BxNxF.
    """
    B, N, C = batch_pc.shape
    dropout_ratios = torch.rand(B, 1, device=batch_pc.device) * max_dropout_ratio  # Bx1
    drop_mask = torch.rand(B, N, device=batch_pc.device) < dropout_ratios  # BxN
    first_points = batch_pc[:, 0, :]  # BxF
    # where drop_mask is True, return first point of batch, otherwise, return batch_pc
    dropout_batch_pc = torch.where(
        drop_mask[..., None], first_points[:, None, :], batch_pc
    )  # BxNxF
    return dropout_batch_pc


def random_rotate_point_cloud(batch_pc):
    """
    This function a point cloud perturb it with an angular and translational offset.

    :param pc: point cloud
    :param angular_offset: float, angular offset in radians around the sensor's vertical axis
    :return: perturbed point cloud
    """
    B, N, C = batch_pc.shape
    device = batch_pc.device
    dtype = batch_pc.dtype
    angular_offsets = (
        torch.rand(B, 1, 1, device=device, dtype=dtype) * 2 * torch.pi
    )  # B

    # Define rotation with angle "angular_offset" around the up-vector
    cos_off = torch.cos(angular_offsets)  # B x 1 x 1
    sin_off = torch.sin(angular_offsets)  # B x 1 x 1

    # Batching R_peturb
    zeros_tensor = torch.zeros(B, 1, 1, device=device, dtype=dtype)
    ones_tensor = torch.ones(B, 1, 1, device=device, dtype=dtype)

    R_peturb = torch.cat(
        (
            torch.cat((cos_off, -sin_off, zeros_tensor), dim=2),
            torch.cat((sin_off, cos_off, zeros_tensor), dim=2),
            torch.cat((zeros_tensor, zeros_tensor, ones_tensor), dim=2),
        ),
        dim=1,
    )  # Bx3x3
    batch_pc_rotated = torch.matmul(batch_pc, R_peturb.transpose(1, 2))  # BxNx3

    return batch_pc_rotated


# Jittering
def random_jitter_point_cloud(batch_pc, sigma=0.01, clip=0.05):
    device = batch_pc.device
    dtype = batch_pc.dtype
    # Default vals give: Jitter points at most +- 5cm. The average absolute offset is approx. 8mm.
    batch_pc_jittered = batch_pc + torch.clamp(
        sigma * torch.randn(batch_pc.shape, device=device, dtype=dtype), -clip, clip
    )
    return batch_pc_jittered
