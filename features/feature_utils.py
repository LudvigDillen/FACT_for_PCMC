import sys
import torch
import numpy as np


from visualization.classifications import plot_accuracies_ablation


def get_dynamic_radii(d, params):
    rmin = params.args.neighborhood.rmin
    rmax = params.args.neighborhood.rmax
    r = d*params.sin_alpha
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
        d = (dists_to_pose0 + dists_to_pose1)/2
    elif params.args.neighborhood.k == "adaptive":
        d = np.sqrt(2)*dists_to_pose0*dists_to_pose1/torch.sqrt(dists_to_pose0**2 + dists_to_pose1**2)
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
    N_batches_per_pc = int(N_batches/2)
    one_pc_batch_sizes = np.empty(N_batches_per_pc, dtype=int)
    for i in range(N_batches_per_pc):
        if samples_counted + 2*max_batch_size <= N:
            samples_to_append = max_batch_size
        else:
            samples_left = N - samples_counted
            assert samples_left % 2 == 0, "There should be an even number of samples left"
            samples_to_append = int(samples_left / 2)
        one_pc_batch_sizes[i] = samples_to_append
        samples_counted += 2*samples_to_append
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


def process_features(features_to_use, features_to_create):
    # The order of the keys matters. Ensure both dictionaries have the same order.
    keys = list(features_to_use.keys())

    if set(keys) != set(features_to_create.keys()):
        raise ValueError("Both dictionaries should have the same set of keys.")

    result = []
    for key in keys:
        use_val = features_to_use[key]
        create_val = features_to_create[key]

        if use_val and create_val:
            result.append(1)
        elif not use_val and create_val:
            result.append(0)
        elif use_val and not create_val:
            raise ValueError(f"'use' is True for {key}, but 'create' is False. This is not allowed.")
        # If both are False, we don't append anything and continue to the next iteration.

    return result


def run_ablation_features(args, logger):
    N_features = args.feature_filter.shape[0]
    from classifiers.PointTransformers.train_cls import run_cls
    all_train_accuracies = np.empty((N_features, args.epoch))
    all_val_accuracies = np.empty((N_features, args.epoch))
    true_keys = [key for key, value in args.features_to_create.items() if value]

    for i in range(N_features):
        print(f"Start feature {true_keys[i]} ({np.around(100*i/N_features, 2)}%)", flush=True)
        ablation_feature_filter = np.zeros(N_features, dtype=int)
        ablation_feature_filter[i] = 1
        # TODO: If running this function, I have to handle the ablation_feature_filter input ...
        train_accuracies, val_accuracies = run_cls(ablation_feature_filter, args, logger,
                                                   pretrained=False)
        all_train_accuracies[i] = train_accuracies
        all_val_accuracies[i] = val_accuracies
    plot_accuracies_ablation(all_train_accuracies, all_val_accuracies, true_keys,
                             plot_train=args.plot_train_acc)
    return None


def number_of_features(feature_filter):
    # xyz is always used (=> 3)
    N_features = int(3 + sum(feature_filter))
    return N_features
