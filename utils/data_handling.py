import nuscenes as ns

from utils.nuscenes_handling import NuscenesHandling


def read_nuscenes_data(data_folder='data/nuscenes/mini/', version='v1.0-mini', n_scenes='all',
                       downsample_factor=1):
    # Initialize NuScenes object
    nusc = ns.nuscenes.NuScenes(version=version, dataroot=data_folder, verbose=True)
    PCHandler = NuscenesHandling(nusc, downsample_factor=downsample_factor)

    PC_scenes = PCHandler.get_entire_sub_dataset(n_scenes=n_scenes)
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


def gather_data(PC_scenes, samples_training, samples_test):
    """
    Divide scenes into training and test scenes. Furthermore, we take
    a given number of samples for training and for testing, as spread out as
    possible (almost at least).
    """
    N_scenes = len(PC_scenes)
    assert N_scenes > 1, "ERROR: We can't do division into training and test data if we do not have at least 2 scenes"
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
    base = N // M
    remainder = N % M

    result = [base] * M
    for i in range(remainder):
        result[i] += 1
    assert sum(result) == N, "ERROR: Division of data gone wrong"
    return result
