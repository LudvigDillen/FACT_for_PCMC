import torch

import numpy as np
from utils.other import start_debug
from utils.geometrics import transformation_matrix, convert_to_same_coordinate_system
from classifiers.differential_entropy import differential_entropy_metric

import nuscenes as ns
from nuscenes.utils.data_classes import LidarPointCloud


def main():
    # Initialize NuScenes object
    nusc = ns.nuscenes.NuScenes(version='v1.0-mini', dataroot='data/nuscenes/mini/', verbose=True)

    # Get the token to the first sample in the first scene
    token0 = nusc.scene[0]['first_sample_token']
    print("cuda available:", torch.cuda.is_available())

    # Get the token for the LIDAR_TOP sensor in the first sample
    lidar_token0 = nusc.get('sample', token0)['data']['LIDAR_TOP']

    # Get LIDAR data dictionary
    lidar_dict = nusc.get('sample_data', lidar_token0)

    # Get the path to the LIDAR binary file
    path_to_lidar_bin_file = nusc.get_sample_data_path(lidar_token0)

    # Load the point cloud from the LIDAR binary file
    pc0 = LidarPointCloud.from_file(path_to_lidar_bin_file)
    first_pc_xyz = pc0.points[:3]

    # Get ego_pose for the first point cloud
    ego_pose_pc0 = nusc.get('ego_pose', lidar_dict['ego_pose_token'])
    rot_pc0 = np.array(ego_pose_pc0['rotation'])
    trans_pc0 = np.array(ego_pose_pc0['translation'])

    print(rot_pc0, trans_pc0)

    # Get the token for the next LIDAR sample
    lidar_token1 = lidar_dict['next']

    # Get LIDAR data dictionary for the next sample
    lidar_dict = nusc.get('sample_data', lidar_token1)

    # Get the path to the LIDAR binary file for the next sample
    path_to_lidar_bin_file = nusc.get_sample_data_path(lidar_token1)

    # Load the point cloud from the LIDAR binary file for the next sample
    pc1 = LidarPointCloud.from_file(path_to_lidar_bin_file)
    second_pc_xyz = pc1.points[:3]

    # Get ego_pose for the second point cloud
    ego_pose_pc1 = nusc.get('ego_pose', lidar_dict['ego_pose_token'])
    rot_pc1 = np.array(ego_pose_pc1['rotation'])
    trans_pc1 = np.array(ego_pose_pc1['translation'])

    print(rot_pc1, trans_pc1)

    print(second_pc_xyz.shape)
    print(np.linalg.norm(first_pc_xyz - second_pc_xyz))

    # Convert NumPy arrays to PyTorch tensors
    pc0 = torch.from_numpy(first_pc_xyz)
    pc1 = torch.from_numpy(second_pc_xyz)

    # Calculate differential entropy
    T0 = transformation_matrix(rot_pc0, trans_pc0)
    T1 = transformation_matrix(rot_pc1, trans_pc1)

    result = differential_entropy_metric(pc0, pc1)
    print(f"Differential entropy before alignment: {result}")

    T1_upd = convert_to_same_coordinate_system(T0, T1)
    # TODO: Not sure I want this T1_upd above. I just want to align the two point clouds I have
    # with the know true transformation.
    # TODO: Now transform point cloud 1 to point cloud 2:s coordinate system. (Perfect alignment)
    # TODO: Compare the differential entropy metric before and after the alignment.
    # result = differential_entropy_metric(pc0, pc1)
    # print(f"Differential entropy after alignment: {result}")

    print("Finito!")


if __name__ == "__main__":
    start_debug()
    main()
