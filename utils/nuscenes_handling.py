import numpy as np
import torch
from utils.geometrics import transformation_matrix

from nuscenes.utils.data_classes import LidarPointCloud


class NuscenesHandling:
    def __init__(self, nusc, lidar_token=None):
        self.nusc = nusc
        if lidar_token is None:
            self.first_token = self.get_first_token()
            self.lidar_token0 = self.get_lidar_token(self.first_token)
        else:
            self.lidar_token0 = lidar_token

        # Set info PC0
        self.pc0_CS0 = self.get_point_cloud(self.lidar_token0)
        self.lidar_pose0 = self.get_sensor_pose_in_WCS(self.lidar_token0)
        self.point_distances0 = self.get_point_distances_to_origin(self.pc0_CS0)

        # Set info PC1
        self.lidar_token1 = self.get_next_lidar_token(self.lidar_token0)
        self.pc1_CS1 = self.get_point_cloud(self.lidar_token1)
        self.lidar_pose1 = self.get_sensor_pose_in_WCS(self.lidar_token1)
        self.point_distances1 = self.get_point_distances_to_origin(self.pc1_CS1)

    def get_first_token(self):
        # Get the token to the first sample in the first scene
        return self.nusc.scene[0]['first_sample_token']

    def get_lidar_token(self, token):
        # Get the token for the LIDAR_TOP sensor in the first sample
        return self.nusc.get('sample', token)['data']['LIDAR_TOP']

    def get_lidar_dict(self, lidar_token):
        return self.nusc.get('sample_data', lidar_token)

    def get_point_cloud(self, lidar_token):
        # Load the point cloud xyz coordinates from the LIDAR binary file.
        # It is the sensor coordinate system.
        path_to_lidar_bin_file = self.nusc.get_sample_data_path(lidar_token)
        pc = LidarPointCloud.from_file(path_to_lidar_bin_file)
        pc_xyz = torch.from_numpy(pc.points[:3]).to(torch.float64)
        return pc_xyz

    def get_point_distances_to_origin(self, pc):
        return torch.norm(pc, p=2, dim=0)

    def get_sensor_pose_in_WCS(self, lidar_token):
        lidar_dict = self.get_lidar_dict(lidar_token)
        # Load ego vehicle pose (in WCS)
        ego_pose = self.nusc.get('ego_pose', lidar_dict['ego_pose_token'])
        # Get translation and rotation in of ego vehicle in WCS
        rotation_ego_vehicle = np.array(ego_pose['rotation'])
        translation_ego_vehicle = np.array(ego_pose['translation'])
        # Get transformation matrix of ego vehicle (WCS)
        Tv = transformation_matrix(rotation_ego_vehicle, translation_ego_vehicle)

        # Get lidar pose in ego vehicle CS
        calibrated_sensor_dict = self.nusc.get(
            'calibrated_sensor', lidar_dict['calibrated_sensor_token'])
        lidar_rotation_CS_ego_vehicle = np.array(calibrated_sensor_dict['rotation'])
        lidar_translation_CS_ego_vehicle = np.array(calibrated_sensor_dict['translation'])
        # Get transforomation matrix of lidar sensor (ego vechicle CS)
        Ts_ego_vehicle = transformation_matrix(
            lidar_rotation_CS_ego_vehicle, lidar_translation_CS_ego_vehicle)

        # Get transformation matrix of lidar in WCS
        Ts = Tv@Ts_ego_vehicle
        Ts_out = torch.from_numpy(Ts).to(torch.float64)
        return Ts_out

    def get_next_lidar_token(self, lidar_token):
        lidar_dict = self.get_lidar_dict(lidar_token)
        return lidar_dict['next']
