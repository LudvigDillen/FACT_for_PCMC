import nuscenes as ns

from utils.nuscenes_handling import NuscenesHandling


def read_nuscenes_data(data_folder='data/nuscenes/mini/', version='v1.0-mini', n_scenes=1,
                       downsample_factor=1):
    # Initialize NuScenes object
    nusc = ns.nuscenes.NuScenes(version=version, dataroot=data_folder, verbose=True)
    PCHandler = NuscenesHandling(nusc, downsample_factor=downsample_factor)

    PC_scenes = PCHandler.get_entire_sub_dataset(n_scenes=n_scenes)
    return PC_scenes
