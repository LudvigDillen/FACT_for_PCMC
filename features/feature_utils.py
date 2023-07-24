import torch
import numpy as np

from visualization.classifications import plot_accuracies_ablation


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


def get_data_batches(PC, params):
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
    radii = get_dynamic_radii(PC.distances_to_origin, params.params_diff_entropy).to(PC.device)
    # Split radii into batches
    radii_batches = divide_into_batches(radii[PC.fps_inds], batch_size)

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

    return np.array(result)


def run_ablation_features(n_samples, feature_filter, args, logger):
    N_features = feature_filter.shape[0]
    from classifiers.PointTransformers.train_cls import run_cls
    all_train_accuracies = np.empty((N_features, args.epoch))
    all_val_accuracies = np.empty((N_features, args.epoch))
    true_keys = [key for key, value in args.features_to_create.items() if value]

    for i in range(N_features):
        print(f"Start feature {true_keys[i]} ({np.around(100*i/N_features, 2)}%)", flush=True)
        ablation_feature_filter = np.zeros(N_features, dtype=int)
        ablation_feature_filter[i] = 1
        train_accuracies, val_accuracies = run_cls(n_samples, ablation_feature_filter, args, logger,
                                                   pretrained=False)
        all_train_accuracies[i] = train_accuracies
        all_val_accuracies[i] = val_accuracies
    plot_accuracies_ablation(all_train_accuracies, all_val_accuracies, true_keys, plot_train=False)
    return None


def number_of_features(feature_filter):
    # xyz is always used (=> 3)
    N_features = int(3 + sum(feature_filter))
    return N_features
