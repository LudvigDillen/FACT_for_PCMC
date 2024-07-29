import os
import sys
import numpy as np
import time
import torch
import importlib.util

from features.differential_entropy import differential_entropy_dataset
from utils.nuscenes_handling import read_nuscenes_data
from utils.pointclouds import farthest_point_sample_PC_scenes


def sample_from_scene(scene, samples):
    """
    Sample from scene with distant samples.
    """
    if samples == 0:
        return []
    n_samples_in_scene = len(scene)
    step_length = int(n_samples_in_scene / samples)
    sampled_scene = []
    for i in range(int(step_length / 2), n_samples_in_scene, step_length):
        sampled_scene.append(scene[i])
    return sampled_scene


def split_data(PC_scenes, scenes_training):
    n_scenes = len(PC_scenes)
    PC_scenes_training = PC_scenes[0:scenes_training]
    PC_scenes_test = PC_scenes[scenes_training:n_scenes]
    return PC_scenes_training, PC_scenes_test


def gather_data(PC_scenes, samples_training, samples_test):
    """
    Divide scenes into training and test scenes. Furthermore, we take
    a given number of samples for training and for testing, as spread out as
    possible (almost at least).
    """
    N_scenes = len(PC_scenes)
    assert N_scenes > 1, (
        "ERROR: We can't do division into training and test"
        "data if we do not have at least 2 scenes"
    )
    total_samples = samples_training + samples_test
    scenes_training = round(N_scenes * samples_training / total_samples)
    scenes_test = round(N_scenes * samples_test / total_samples)
    assert scenes_training > 0, "ERROR: We must have at least 1 training scene"
    assert scenes_test > 0, "ERROR: We must have at least 1 test scene"

    # Becomes necessary if both are rounded down from X.5 to X.0.
    if scenes_training + scenes_test != N_scenes:
        scenes_training += 1
    assert (
        scenes_training + scenes_test == N_scenes
    ), "Error in division of data between training and test"

    samples_per_scene_training = list_of_samples_per_scene(
        samples_training, scenes_training
    )
    samples_per_scene_test = list_of_samples_per_scene(samples_test, scenes_test)

    PC_scenes_training = []
    PC_scenes_test = []
    for i in range(scenes_training):
        scene = PC_scenes[i]
        PC_scenes_training.append(
            sample_from_scene(scene, samples_per_scene_training[i])
        )
    for i in range(scenes_test):
        scene = PC_scenes[scenes_training + i]
        PC_scenes_test.append(sample_from_scene(scene, samples_per_scene_test[i]))
    return PC_scenes_training, PC_scenes_test


def list_of_samples_per_scene(N, M):
    """
    Given a total number of samples N and a number of scenes M, calculate and return a list that indicates
    how many samples belong to each scene. The samples are distributed as evenly as possible across the
    scenes. If N is not evenly divisible by M, the extra samples are distributed one by one to the scenes,
    starting from the first.

    Parameters:
    N (int): The total number of samples.
    M (int): The number of scenes.

    Returns:
    list of int: A list of size M where each element indicates the number of samples in the corresponding
                 scene.

    Raises:
    AssertionError: If the sum of the elements in the returned list is not equal to N, an AssertionError is
                    raised.
    """
    base = N // M
    remainder = N % M

    result = [base] * M
    for i in range(remainder):
        result[i] += 1
    assert sum(result) == N, "ERROR: Division of data gone wrong"
    return result


def calculate_sample_gaps(N, M):
    """
    Given a total number of samples N and a number of selected samples M, calculate and return a list of the
    number of samples between every two consecutive selected samples. The samples are selected such that they
    are distributed as evenly as possible across the total range.

    If M == 1, the function returns [0], as there is only one sample and therefore no gaps between samples.

    Parameters:
    N (int): The total number of samples.
    M (int): The number of samples to select.

    Returns:
    list of int: A list of the number of samples between each pair of consecutive selected samples.
                 The first element is always 0, representing the number of samples before the first
                 selected sample.

    Raises:
    ValueError: If M > N, a ValueError is raised, as it's not possible to select more samples than are
                available.
    """
    if M > N:
        raise ValueError("M must be less than or equal to N")

    if M == 1:
        return [0]

    interval = (N - 1) / (M - 1)

    samples = [round(i * interval) for i in range(M)]
    distances = [0] + [samples[i + 1] - samples[i] - 1 for i in range(len(samples) - 1)]

    return distances


def generate_class_names_file(folder, filename, n_classes, class_names):
    # Check if folder exists, if not, create it
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Construct the full path to the output file
    file_path = os.path.join(folder, filename)

    # Open the file for writing
    with open(file_path, "w") as f:
        # Write each class category to the file
        for i in range(n_classes):
            class_name = class_names[i] + "\n"
            class_name = class_name.replace(".", "_")
            f.write(class_name)


def classes_to_txt(PC_scenes, class_counts, file, class_names, scene_counter):
    for i, PC_scene in enumerate(PC_scenes):
        for PC_pair in PC_scene:
            class_counts[PC_pair.class_category] += 1
            message = (
                f"scene_{scene_counter+i}_"
                + class_names[PC_pair.class_category]
                + "_"
                + str(class_counts[PC_pair.class_category]).zfill(4)
            )
            PC_pair.set_name(message)
            file.write(message + "\n")
    return class_counts, PC_scenes


def display_progress(scene_counter, n_scenes, mode, start, train_ratio, val_ratio):
    amount_loaded = scene_counter / n_scenes
    max_train_scenes = round(train_ratio * n_scenes)
    max_val_scenes = round(val_ratio * n_scenes)

    if mode == "train":
        print(f"Have loaded {scene_counter} of {max_train_scenes} {mode} scenes ")
    elif mode == "validation":
        val_scenes_loaded = scene_counter - max_train_scenes
        print(f"Have loaded {val_scenes_loaded} of {max_val_scenes} {mode} scenes ")
    else:
        test_scenes_loaded = scene_counter - max_train_scenes - max_val_scenes
        tot_test_scenes = n_scenes - max_train_scenes - max_val_scenes
        print(f"Have loaded {test_scenes_loaded} of {tot_test_scenes} {mode} scenes ")

    print(
        f"Have loaded {scene_counter} of {n_scenes} scenes "
        + f"({np.around(100*amount_loaded, 1)}% of train, validation, and test scenes)",
        flush=True,
    )
    if amount_loaded != 0:
        time_check = time.time()
        time_gone = time_check - start

        print(f"Time gone: {np.around(time_gone / 3600, 2)} hours")
        estimated_time_left = time_gone / amount_loaded - time_gone
        print(
            f"Estimated time left: {np.around(estimated_time_left / 3600, 2)} hours\n",
            flush=True,
        )


def features_to_txt_files(
    scenes_lower,
    scenes_upper,
    n_scenes_per_loop,
    params,
    mode,
    class_counts,
    start,
    write_mode,
):
    """
    Extract differential entropy features from a range of scenes.

    Parameters:
    scenes_lower (int): The starting scene index.
    scenes_upper (int): The ending scene index.
    n_scenes_per_loop (int): The number of scenes to process in each loop.
    params (see the class Params in utils.parameters)
    """
    # Prepare storage for features
    assert mode in [
        "train",
        "validation",
        "test",
    ], f"Specified mode ({mode}) is not known!"

    from features.feature_extractor import (
        write_features_to_txt_files,
        feature_extraction,
    )

    data_folder = params.args.feature_folder
    file_name = "PCAC_data_" + mode + ".txt"
    data_file = os.path.join(data_folder, file_name)

    with open(data_file, write_mode) as file:
        for scene_counter in range(scenes_lower, scenes_upper, n_scenes_per_loop):
            # Display data loading progress
            display_progress(
                scene_counter,
                params.n_scenes,
                mode,
                start,
                params.train_ratio,
                params.args.val_ratio,
            )

            # Determine the number of scenes to read in this loop
            if scene_counter + n_scenes_per_loop > scenes_upper:
                read_n_scenes = scenes_upper - scene_counter
            else:
                read_n_scenes = n_scenes_per_loop
            read_n_samples = read_n_scenes * params.n_samples_per_scene

            # Load data sequentially (can't read all at once, too high memory requirement)
            # TODO: Probably I can read KITTI instead or use dataloader or something ... Perhaps I create a completely new function for this ...
            PC_scenes = read_nuscenes_data(
                params,
                mode,
                n_samples=read_n_samples,
                n_scenes=read_n_scenes,
                scene_counter=scene_counter,
            )

            if params.args.fps.do_fps:
                farthest_point_sample_PC_scenes(PC_scenes, params.N_fps_points)

            # Feature extraction.
            PC_scenes_with_features = feature_extraction(PC_scenes, params)

            # Write class names to text file
            class_counts, PC_scenes_named = classes_to_txt(
                PC_scenes_with_features,
                class_counts,
                file,
                params.class_names,
                scene_counter,
            )
            del PC_scenes_with_features

            write_features_to_txt_files(PC_scenes_named, params, data_folder)
    return class_counts


def geotrans_features_to_txt_files(
    scenes_lower,
    scenes_upper,
    n_scenes_per_loop,
    params,
    mode,
    class_counts,
    start,
    write_mode,
):
    """
    Extract differential entropy features from a range of scenes.

    Parameters:
    scenes_lower (int): The starting scene index.
    scenes_upper (int): The ending scene index.
    n_scenes_per_loop (int): The number of scenes to process in each loop.
    params (see the class Params in utils.parameters)
    """
    # Prepare storage for features
    assert mode in [
        "train",
        "validation",
        "test",
    ], f"Specified mode ({mode}) is not known!"

    #region Load some GeoTransformer functions/modules
    from registration.geotransformer_handling import to_PC_format
    from GeoTransformer_202407.geotransformer.utils.torch import to_cuda

    # Specify the full path to the config.py file
    config_path = 'GeoTransformer_202407/experiments/geotransformer.kitti.stage5.gse.k3.max.oacl.stage2.sinkhorn/config.py'
    dataset_path = 'GeoTransformer_202407/experiments/geotransformer.kitti.stage5.gse.k3.max.oacl.stage2.sinkhorn/dataset.py'
    model_path = 'GeoTransformer_202407/experiments/geotransformer.kitti.stage5.gse.k3.max.oacl.stage2.sinkhorn/model.py'
    # Get the module name (you can name it anything that doesn't conflict with existing module names)
    config_name = 'config_module'
    dataset_name = 'dataset_module'
    model_name = 'model_module'

    # Handle some path complications
    original_cwd = os.getcwd()

    # Change to the parent directory (e.g., back up two levels)
    os.chdir("../../../")

    geo_path = os.path.abspath('GeoTransformer_202407/experiments/geotransformer.kitti.stage5.gse.k3.max.oacl.stage2.sinkhorn')
    sys.path.append(geo_path)

    # Load the module
    config_spec = importlib.util.spec_from_file_location(config_name, config_path)
    dataset_spec = importlib.util.spec_from_file_location(dataset_name, dataset_path)
    model_spec = importlib.util.spec_from_file_location(model_name, model_path)

    config_module = importlib.util.module_from_spec(config_spec)
    dataset_module = importlib.util.module_from_spec(dataset_spec)
    model_module = importlib.util.module_from_spec(model_spec)

    config_spec.loader.exec_module(config_module)
    dataset_spec.loader.exec_module(dataset_module)
    model_spec.loader.exec_module(model_module)

    make_cfg = getattr(config_module, 'make_cfg', None)
    train_valid_data_loader = getattr(dataset_module, 'train_valid_data_loader', None)
    test_data_loader = getattr(dataset_module, 'test_data_loader', None)
    create_model = getattr(model_module, 'create_model', None)
    #endregion

    cfg = make_cfg()
    model = create_model(cfg).cuda()
    snapshot = 'GeoTransformer_202407/weights/geotransformer-kitti.pth.tar'
    # Load the snapshot
    print('Loading from "{}".'.format(snapshot))
    state_dict = torch.load(snapshot, map_location=torch.device('cpu'))
    assert 'model' in state_dict, 'No model can be loaded.'
    model.load_state_dict(state_dict['model'], strict=True)
    print('Model has been loaded.')
    model.eval()
    torch.set_grad_enabled(False)

    # Revert back to the original working directory
    os.chdir(original_cwd)

    from features.feature_extractor import (
        write_features_to_txt_files,
        feature_extraction,
    )

    data_folder = params.args.feature_folder
    file_name = "PCAC_data_" + mode + ".txt"
    data_file = os.path.join(data_folder, file_name)
    scene_counter = scenes_lower
    with open(data_file, write_mode) as file:
        if mode == "train":
            train_loader, _, _ = train_valid_data_loader(cfg, False)  # TODO: Should I augment train data or not
            loader = train_loader
        elif mode == "validation":
            _, val_loader, _ = train_valid_data_loader(cfg, False)  # TODO: Should I augment train data or not
            loader = val_loader
        elif mode == "test":
            test_loader, _ = test_data_loader(cfg)
            loader = test_loader
        else:
            raise ValueError("Unknown mode")

        PC_scenes = []
        counter = 0
        for iteration, data_dict in enumerate(loader):  # TODO: I cannot loop through a full epoch ...
            data_dict = to_cuda(data_dict)
            # GeoTransformer registration done
            output_dict = model(data_dict)

            # Extract data from GeoTransformer output
            src = output_dict['src_points']
            ref = output_dict['ref_points']
            gt_trans = data_dict['transform']
            est_trans = output_dict['estimated_transform']
            del output_dict, data_dict
            torch.cuda.empty_cache()

            # TODO: Load more than one scene at a time ...
            # TODO: I currently do HPR operator things like in nuscenes handling,
            #       but geotransformer is not trained on HPR operator filtered data, so it might be unfair ...
            #       I could however compare the difference in performance between the two ...
            # TODO I think I perhaps should remove the closest points after the registration ...
            PC_scene = to_PC_format(src, ref, params, est_trans, gt_trans, geo_args=None)
            PC_scenes.append(PC_scene)
            counter += 1
            if (iteration + 1) % n_scenes_per_loop == 0 or scene_counter + counter == scenes_upper:
                scene_counter += counter
                # Display data loading progress
                display_progress(
                    scene_counter,
                    params.n_scenes,
                    mode,
                    start,
                    params.train_ratio,
                    params.args.val_ratio,
                )
                PC_scenes = np.array(PC_scenes)

                if params.args.fps.do_fps:
                    farthest_point_sample_PC_scenes(PC_scenes, params.N_fps_points)

                # Feature extraction.
                PC_scenes_with_features = feature_extraction(PC_scenes, params)

                # Write class names to text file
                class_counts, PC_scenes_named = classes_to_txt(
                    PC_scenes_with_features,
                    class_counts,
                    file,
                    params.class_names,
                    scene_counter,
                )
                del PC_scenes_with_features

                write_features_to_txt_files(PC_scenes_named, params, data_folder)
                PC_scenes = []
                counter = 0
                if scene_counter + counter == scenes_upper:
                    break
        return class_counts


def get_diff_entropy_features(
    scenes_lower, scenes_upper, n_scenes_per_loop, params, mode, start
):
    """
    Extract differential entropy features from a range of scenes.

    Parameters:
    scenes_lower (int): The starting scene index.
    scenes_upper (int): The ending scene index.
    n_scenes_per_loop (int): The number of scenes to process in each loop.
    params (see the class Params in utils.parameters)

    Returns:
    X (np.array): The differential entropy features extracted from the scenes.
    y (np.array): The corresponding labels.
    """
    # Prepare storage for features
    X = np.empty((0, 2))
    y = np.empty((0))
    for scene_counter in range(scenes_lower, scenes_upper, n_scenes_per_loop):
        display_progress(
            scene_counter,
            params.n_scenes,
            mode,
            start,
            params.train_ratio,
            params.args.val_ratio,
        )
        # Determine the number of scenes to read in this loop
        if scene_counter + n_scenes_per_loop > scenes_upper:
            read_n_scenes = scenes_upper - scene_counter
        else:
            read_n_scenes = n_scenes_per_loop
        read_n_samples = read_n_scenes * params.n_samples_per_scene
        # Load data sequentially
        PC_scenes = read_nuscenes_data(
            params,
            mode="test",
            n_samples=read_n_samples,
            n_scenes=read_n_scenes,
            scene_counter=scene_counter,
        )

        if params.args.fps.do_fps:
            # Compute the differential entropy features for the scenes
            farthest_point_sample_PC_scenes(PC_scenes, params.N_fps_points)
        # Compute the differential entropy features for the scenes
        X_loop, y_loop = differential_entropy_dataset(PC_scenes, params)
        # Append the features and labels to the storage arrays
        X = np.concatenate((X, X_loop), axis=0)
        y = np.concatenate((y, y_loop), axis=0)
    return X, y


def get_n_scenes_per_loop(n_samples_per_scene, n_training_scenes, n_scenes):
    """
    Determine the number of scenes to process in each loop.

    Parameters:
    n_samples_per_scene (int): The number of samples to take from each scene.
    n_training_scenes (int): The number of scenes allocated for training.
    n_scenes (int): The total number of scenes.

    Returns:
    n_scenes_per_loop (int): The number of scenes to process in each loop.
    """
    n_test_scenes = n_scenes - n_training_scenes
    smallest_loop = min(n_test_scenes, n_training_scenes)
    # TODO: Maybe try to increase from 40, could give faster computations ...
    n_scenes_per_loop = max(round(100 / n_samples_per_scene), 1)
    n_scenes_per_loop = min(n_scenes_per_loop, smallest_loop)
    return n_scenes_per_loop


def run_differential_entropy_on_dataset(params, logger):
    """
    Run differential entropy feature extraction on a dataset.

    Parameters:
    params (see the class Params in utils.parameters)

    Returns:
    X_train (np.array): The differential entropy features extracted from the training scenes.
    y_train (np.array): The corresponding labels for the training data.
    X_test (np.array): The differential entropy features extracted from the test scenes.
    y_test (np.array): The corresponding labels for the test data.
    """
    n_training_scenes = round(params.train_ratio * params.n_scenes)
    n_val_scenes = round(params.args.val_ratio * params.n_scenes)

    scenes_lower_train = 0
    scenes_lower_val = n_training_scenes
    scenes_lower_test = n_training_scenes + n_val_scenes
    start = time.time()

    # Determine the number of scenes to process in each loop
    n_scenes_per_loop = get_n_scenes_per_loop(
        params.n_samples_per_scene, n_training_scenes, params.n_scenes
    )

    logger.info("Start extracting training samples")
    # Extract features from the training scenes
    X_train, y_train = get_diff_entropy_features(
        scenes_lower=scenes_lower_train,
        scenes_upper=n_training_scenes,
        n_scenes_per_loop=n_scenes_per_loop,
        params=params,
        mode="train",
        start=start,
    )

    logger.info("Start extracting validation samples")
    # Extract features from the validation scenes
    X_val, y_val = get_diff_entropy_features(
        scenes_lower=scenes_lower_val,
        scenes_upper=n_training_scenes + n_val_scenes,
        n_scenes_per_loop=n_scenes_per_loop,
        params=params,
        mode="validation",
        start=start,
    )

    logger.info("Start extracting test samples")
    # Extract features from the test scenes
    X_test, y_test = get_diff_entropy_features(
        scenes_lower=scenes_lower_test,
        scenes_upper=params.n_scenes,
        n_scenes_per_loop=n_scenes_per_loop,
        params=params,
        mode="test",
        start=start,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def setup_inputs_to_dnn(params):
    """
    Run differential entropy feature extraction on a dataset.

    Parameters:
    params (see the class Params in utils.parameters)

    Returns:
    """
    n_training_scenes = round(params.train_ratio * params.n_scenes)
    n_val_scenes = round(params.args.val_ratio * params.n_scenes)
    # Determine the number of scenes to process in each loop
    n_scenes_per_loop = get_n_scenes_per_loop(
        params.n_samples_per_scene, n_training_scenes, params.n_scenes
    )

    scenes_lower_train = 0
    scenes_lower_val = n_training_scenes
    scenes_lower_test = n_training_scenes + n_val_scenes
    write_mode_train = "w"
    write_mode_val = "w"
    write_mode_test = "w"

    continue_training_extraction = True
    continue_val_extraction = True
    if n_val_scenes == 0:
        continue_val_extraction = False  # This means do not use validation dataset
    start = time.time()

    if params.args.rerun_crash:
        assert (
            params.args.scenes_finished < params.n_scenes
        ), "Cannot extract data, already extracted all"
        if params.args.scenes_finished < n_training_scenes:
            scenes_lower_train = params.args.scenes_finished
            continue_training_extraction = True
            filename = "PCAC_data_train.txt"
            write_mode_train = "a"  # append, not write
        elif params.args.scenes_finished < n_training_scenes + n_val_scenes:
            scenes_lower_val = params.args.scenes_finished
            continue_training_extraction = False
            filename = "PCAC_data_validation.txt"
            write_mode_val = "a"  # append, not write
        else:
            scenes_lower_test = params.args.scenes_finished
            continue_training_extraction = False
            continue_val_extraction = False
            filename = "PCAC_data_test.txt"
            write_mode_test = "a"  # append, not write
        path_folder = os.path.join(params.args.feature_folder, filename)
        class_counts = extract_max_values_from_end(params.args, path_folder)
        start = start - 3600 * params.args.time_gone
    else:
        class_counts = np.zeros(params.args.perturb_settings.n_classes, dtype=int)
        # Extract features from the training scenes

    if continue_training_extraction:
        if params.args.general_dataset == "nuscenes":
            class_counts = features_to_txt_files(
                scenes_lower=scenes_lower_train,
                scenes_upper=n_training_scenes,
                n_scenes_per_loop=n_scenes_per_loop,
                params=params,
                mode="train",
                class_counts=class_counts,
                start=start,
                write_mode=write_mode_train,
            )
        elif params.args.general_dataset == "kitti":
            class_counts = geotrans_features_to_txt_files(
                scenes_lower=scenes_lower_train,
                scenes_upper=n_training_scenes,
                n_scenes_per_loop=n_scenes_per_loop,
                params=params,
                mode="train",
                class_counts=class_counts,
                start=start,
                write_mode=write_mode_train,
            )
        else:
            raise ValueError("Unknown dataset")
    # Extract features from the validation scenes
    if continue_val_extraction:
        if params.args.general_dataset == "nuscenes":
            class_counts = features_to_txt_files(
                scenes_lower=scenes_lower_val,
                scenes_upper=n_training_scenes + n_val_scenes,
                n_scenes_per_loop=n_scenes_per_loop,
                params=params,
                mode="validation",
                class_counts=class_counts,
                start=start,
                write_mode=write_mode_val,
            )
        elif params.args.general_dataset == "kitti":
            class_counts = geotrans_features_to_txt_files(
                scenes_lower=scenes_lower_val,
                scenes_upper=n_training_scenes + n_val_scenes,
                n_scenes_per_loop=n_scenes_per_loop,
                params=params,
                mode="validation",
                class_counts=class_counts,
                start=start,
                write_mode=write_mode_val,
            )
        else:
            raise ValueError("Unknown dataset")

    # Extract features from the test scenes
    if params.args.general_dataset == "nuscenes":
        class_counts = features_to_txt_files(
            scenes_lower=scenes_lower_test,
            scenes_upper=params.n_scenes,
            n_scenes_per_loop=n_scenes_per_loop,
            params=params,
            mode="test",
            class_counts=class_counts,
            start=start,
            write_mode=write_mode_test,
        )
    elif params.args.general_dataset == "kitti":
        class_counts = geotrans_features_to_txt_files(
            scenes_lower=scenes_lower_test,
            scenes_upper=params.n_scenes,
            n_scenes_per_loop=n_scenes_per_loop,
            params=params,
            mode="test",
            class_counts=class_counts,
            start=start,
            write_mode=write_mode_test,
        )
    else:
        raise ValueError("Unknown dataset")


def extract_max_values_from_end(args, file_path):
    if not os.path.exists(file_path):
        max_values = np.zeros(args.perturb_settings.n_classes, dtype=int)
        # If the file doesn't exist, create it by opening it in write mode
        with open(file_path, "w") as f:
            pass  # No need to write anything, just creating the file
        return max_values

    with open(file_path, "r") as f:
        data = f.read()

    lines = data.strip().split("\n")
    # Use -1 as the initial value to indicate "not found"
    max_values = np.zeros(args.perturb_settings.n_classes, dtype=int) - 1
    found_categories = set()  # To track which categories we've already found

    for line in reversed(lines):  # Start from the end of the file
        class_category = int(line.split("_")[-10])
        if class_category not in found_categories:
            value = int(line.split("_")[-1])
            max_values[class_category] = value
            found_categories.add(class_category)

        # If we've found all categories, we can break out of the loop
        if len(found_categories) == 10:
            break

    # Check if all categories were found and replace unfound ones with an error message
    error_indices = np.where(max_values == -1)
    for idx in error_indices[0]:
        print(f"Error: No entry found for class_category_{idx}")

    return max_values
