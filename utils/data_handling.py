import nuscenes as ns

from utils.nuscenes_handling import NuscenesHandling


def read_nuscenes_data(n_samples, data_folder='data/nuscenes/', version='v1.0-trainval',
                       downsample_factor=1, n_scenes='all'):
    # Initialize NuScenes object
    nusc = ns.nuscenes.NuScenes(version=version, dataroot=data_folder, verbose=True)
    PCHandler = NuscenesHandling(nusc, downsample_factor=downsample_factor)
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
