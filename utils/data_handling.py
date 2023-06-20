import numpy as np

from utils.nuscenes_handling import NuscenesHandling
from classifiers.differential_entropy import differential_entropy_dataset, differential_entropy_pointwise


def read_nuscenes_data(nusc, n_samples, downsample_factor=1, n_scenes='all',
                       T_close_thresh=0, lidar_token=None, scene_counter=0,
                       verbose=True):
    # Read data
    PCHandler = NuscenesHandling(nusc, downsample_factor=downsample_factor,
                                 T_close_thresh=T_close_thresh, lidar_token=lidar_token,
                                 scene_counter=scene_counter, verbose=verbose)
    PC_scenes = PCHandler.sample_from_scenes(n_samples=n_samples, n_scenes=n_scenes)
    return PC_scenes


def sample_from_scene(scene, samples):
    """
    Sample from scene with distant samples.
    """
    if samples == 0:
        return []
    N_samples_in_scene = len(scene)
    step_length = int(N_samples_in_scene/samples)
    sampled_scene = []
    for i in range(int(step_length/2), N_samples_in_scene, step_length):
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
    assert N_scenes > 1, ("ERROR: We can't do division into training and test"
                          "data if we do not have at least 2 scenes")
    total_samples = samples_training + samples_test
    scenes_training = round(N_scenes*samples_training/total_samples)
    scenes_test = round(N_scenes*samples_test/total_samples)
    assert scenes_training > 0, "ERROR: We must have at least 1 training scene"
    assert scenes_test > 0, "ERROR: We must have at least 1 test scene"

    # Becomes necessary if both are rounded down from X.5 to X.0.
    if scenes_training + scenes_test != N_scenes:
        scenes_training += 1
    assert scenes_training + scenes_test == N_scenes, "Error in division of data between training and test"

    samples_per_scene_training = list_of_samples_per_scene(samples_training, scenes_training)
    samples_per_scene_test = list_of_samples_per_scene(samples_test, scenes_test)

    PC_scenes_training = []
    PC_scenes_test = []
    for i in range(scenes_training):
        scene = PC_scenes[i]
        PC_scenes_training.append(sample_from_scene(scene, samples_per_scene_training[i]))
    for i in range(scenes_test):
        scene = PC_scenes[scenes_training+i]
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
    distances = [0] + [samples[i+1] - samples[i] - 1 for i in range(len(samples)-1)]

    return distances


def feature_extraction(PC_scenes, params):
    PC_scenes_with_features = differential_entropy_pointwise(
        PC_scenes, params.params_diff_entropy, hpr_radius=params.hpr_radius, preprocess=params.preprocess)
    # TODO: Extract more features ...
    return PC_scenes_with_features


def run_dnn(scenes_lower, scenes_upper, n_scenes_per_loop, params):
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
    first_iter = True
    for scene_counter in range(scenes_lower, scenes_upper, n_scenes_per_loop):
        # Determine the number of scenes to read in this loop
        if scene_counter + n_scenes_per_loop > scenes_upper:
            read_n_scenes = scenes_upper - scene_counter
        else:
            read_n_scenes = n_scenes_per_loop
        read_n_samples = read_n_scenes*params.n_samples_per_scene
        # Load data sequentially
        PC_scenes = read_nuscenes_data(
            params.nusc, n_scenes=read_n_scenes, n_samples=read_n_samples,
            downsample_factor=params.downsample_factor, T_close_thresh=params.T_close_thresh,
            scene_counter=scene_counter, verbose=params.verbose)
        # Compute the differential entropy features for the scenes
        # Feature extraction
        PC_scenes_with_features = feature_extraction(PC_scenes, params)
        if first_iter:
            all_PC_scenes = PC_scenes_with_features
            first_iter = False
        else:
            all_PC_scenes = np.vstack((all_PC_scenes, PC_scenes_with_features))
    return all_PC_scenes


def get_diff_entropy_features(scenes_lower, scenes_upper, n_scenes_per_loop, params):
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
        # Determine the number of scenes to read in this loop
        if scene_counter + n_scenes_per_loop > scenes_upper:
            read_n_scenes = scenes_upper - scene_counter
        else:
            read_n_scenes = n_scenes_per_loop
        read_n_samples = read_n_scenes*params.n_samples_per_scene
        # Load data sequentially
        PC_scenes = read_nuscenes_data(
            params.nusc, n_scenes=read_n_scenes, n_samples=read_n_samples,
            downsample_factor=params.downsample_factor, T_close_thresh=params.T_close_thresh,
            scene_counter=scene_counter, verbose=params.verbose)
        # Compute the differential entropy features for the scenes
        X_loop, y_loop = differential_entropy_dataset(
            PC_scenes, params.params_diff_entropy, verbose=params.verbose, hpr_radius=params.hpr_radius,
            preprocess=params.preprocess)
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

    n_scenes_per_loop = max(round(100/n_samples_per_scene), 1)
    n_scenes_per_loop = min(n_scenes_per_loop, smallest_loop)
    return n_scenes_per_loop


def run_differential_entropy_on_dataset(params):
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
    n_training_scenes = round(params.train_ratio*params.n_scenes)
    # Determine the number of scenes to process in each loop
    n_scenes_per_loop = get_n_scenes_per_loop(params.n_samples_per_scene, n_training_scenes, params.n_scenes)

    # Extract features from the training scenes
    X_train, y_train = get_diff_entropy_features(
        scenes_lower=0, scenes_upper=n_training_scenes, n_scenes_per_loop=n_scenes_per_loop, params=params)

    # Extract features from the test scenes
    X_test, y_test = get_diff_entropy_features(
        scenes_lower=n_training_scenes, scenes_upper=params.n_scenes, n_scenes_per_loop=n_scenes_per_loop,
        params=params)
    return X_train, y_train, X_test, y_test


def setup_inputs_to_dnn(params):
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
    n_training_scenes = round(params.train_ratio*params.n_scenes)
    # Determine the number of scenes to process in each loop
    n_scenes_per_loop = get_n_scenes_per_loop(params.n_samples_per_scene, n_training_scenes, params.n_scenes)

    # Extract features from the training scenes
    all_PC_scenes_train = run_dnn(
        scenes_lower=0, scenes_upper=n_training_scenes, n_scenes_per_loop=n_scenes_per_loop, params=params)

    # Extract features from the test scenes
    all_PC_scenes_test = run_dnn(
        scenes_lower=n_training_scenes, scenes_upper=params.n_scenes, n_scenes_per_loop=n_scenes_per_loop,
        params=params)
    return all_PC_scenes_train, all_PC_scenes_test
