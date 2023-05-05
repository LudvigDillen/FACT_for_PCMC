import numpy as np
import torch

from utils.geometrics import transformation_matrix
from utils.pointclouds import PC, PCPair
from nuscenes.utils.data_classes import LidarPointCloud


class NuscenesHandling:
    def __init__(self, nusc, downsample_factor=1, lidar_token=None):
        self.nusc = nusc
        self.scene_counter = 0
        self.number_of_scenes_in_dataset = len(self.nusc.scene)
        self.dataset_read = False
        self.scene_read = False
        # Randomly downsample point clouds
        self.downsample_factor = downsample_factor
        self.setup_new_scene_data(lidar_token)

    def downsample_both_point_clouds(self):
        N_samples_before = self.pc0_CS0.shape[0]  # Assuming both point clouds have equally many points
        N_samples_after = round(N_samples_before/self.downsample_factor)
        samples_to_keep = np.random.choice(N_samples_before, size=N_samples_after)
        self.samples_to_keep = samples_to_keep

        self.pc0_CS0 = self.pc0_CS0[samples_to_keep]  # Downsample point cloud
        self.pc1_CS1 = self.pc1_CS1[samples_to_keep]  # Downsample point cloud

    def downsample_second_point_cloud(self):
        self.pc1_CS1 = self.pc1_CS1[self.samples_to_keep]  # Downsample point cloud

    def setup_new_scene_data(self, lidar_token=None):
        # Set info PC0
        if lidar_token is None:
            self.first_sample_token = self.get_first_sample_token()
            self.lidar_token0 = self.get_lidar_token(self.first_sample_token)
        else:
            self.lidar_token0 = lidar_token
        self.lidar_pose0 = self.get_sensor_pose_in_WCS(self.lidar_token0)
        self.pc0_CS0 = self.get_point_cloud(self.lidar_token0)

        # Set info PC1
        self.lidar_token1 = self.get_next_lidar_token(self.lidar_token0)
        self.lidar_pose1 = self.get_sensor_pose_in_WCS(self.lidar_token1)
        self.pc1_CS1 = self.get_point_cloud(self.lidar_token1)

        # Randomly downsample point clouds
        self.downsample_factor = self.downsample_factor
        if self.downsample_factor > 1:
            self.downsample_both_point_clouds()

        self.point_distances0 = self.get_point_distances_to_origin(self.pc0_CS0)
        self.point_distances1 = self.get_point_distances_to_origin(self.pc1_CS1)

    def get_first_sample_token(self):
        # Get the sample_token to the first sample in the scene
        return self.nusc.scene[self.scene_counter]['first_sample_token']

    def get_sample_token(self, lidar_token):
        # From lidar_token get sample token
        return self.nusc.get('sample_data', lidar_token)['sample_token']

    def get_scene_token(self, lidar_token):
        """
        From lidar_token get scene_token
        """
        # From lidar_token get sample_token
        sample_token = self.nusc.get('sample_data', lidar_token)['sample_token']
        # From sample_token get scene_token
        return self.nusc.get('sample', sample_token)['scene_token']

    def get_lidar_token(self, sample_token):
        # Get the lidar_token for the LIDAR_TOP sensor from sample_token
        return self.nusc.get('sample', sample_token)['data']['LIDAR_TOP']

    def get_lidar_dict(self, lidar_token):
        return self.nusc.get('sample_data', lidar_token)

    def get_point_cloud(self, lidar_token):
        # Load the point cloud xyz coordinates from the LIDAR binary file.
        # It is the sensor coordinate system.
        path_to_lidar_bin_file = self.nusc.get_sample_data_path(lidar_token)
        pc = LidarPointCloud.from_file(path_to_lidar_bin_file)
        pc_xyz = torch.swapaxes(torch.from_numpy(pc.points[:3]).to(torch.float64), 0, 1)
        return pc_xyz

    def get_point_distances_to_origin(self, pc):
        return torch.norm(pc, p=2, dim=1)

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

    def set_next_point_cloud_pair(self):
        """
        Iteratate to next point cloud in the dataset. If the scene is finished, we go on to
        the next scene. If all as scenes are read, then we have read the entire dataset. Then,
        we have collected all the data we need/want.
        """
        # Move to next point cloud pair in the scene
        self.lidar_token0 = self.lidar_token1
        self.lidar_pose0 = self.lidar_pose1
        self.pc0_CS0 = self.pc1_CS1

        # Set info PC1
        self.lidar_token1 = self.get_next_lidar_token(self.lidar_token0)
        end_of_scene = (self.lidar_token1 == '')
        if end_of_scene:
            self.scene_read = True
            self.scene_counter += 1
            end_of_dataset = (self.scene_counter >= self.number_of_scenes_in_dataset)
            if end_of_dataset:
                self.dataset_read = True
            else:
                self.setup_new_scene_data(lidar_token=None)
            return None
        self.lidar_pose1 = self.get_sensor_pose_in_WCS(self.lidar_token1)
        self.pc1_CS1 = self.get_point_cloud(self.lidar_token1)

        # Randomly downsample point clouds
        if self.downsample_factor > 1:
            self.downsample_second_point_cloud()

        self.point_distances1 = self.get_point_distances_to_origin(self.pc1_CS1)

    def get_number_of_samples_in_scenes(self, PC_scenes):
        n_samples = 0
        for PC_scene in PC_scenes:
            n_samples += len(PC_scene)
        return n_samples

    def get_entire_sub_dataset(self):
        """
        Here we return the entire sub-dataset. E.g. the mini-dataset or part1-dataset of
        the Nuscenes dataset.

        Data format:
            A list of scenes. Each scenes correspond to roughly 20s recoreded lidar data
            at 20 Hz. This means that each scene contains around 400 point clouds. We have
            divided in so that the list consists of the class PCPairs which contains the
            one point cloud pair and the corresponding union point cloud.
        """
        PC_scenes = []
        PC_scene = []
        PC0 = PC(self.pc0_CS0, self.point_distances0)
        count = 0
        while True:
            if count % 2000 == 0 and count != 0:
                print(f"We have collected {count} number of samples")

            # Load in second point cloud
            PC1 = PC(self.pc1_CS1, self.point_distances1)
            # Set point cloud pair and their union, and perform possible perturbation
            currentPCPair = PCPair(PC0, PC1, self, perturb_probability=0.5)
            # Append pair to list
            PC_scene.append(currentPCPair)
            # Iterate to next pair in scene
            self.set_next_point_cloud_pair()

            if self.scene_read == True:
                # Append scene to list of scenes
                PC_scenes.append(PC_scene)
                PC_scene = []
                self.scene_read = False
                print(f"We have collected {self.scene_counter} scenes")
            if self.dataset_read == True:
                print("We have collected all data from the dataset")
                print(f"Total number of samples: {self.get_number_of_samples_in_scenes(PC_scenes)}")
                print(f"Total number of scenes:  {len(PC_scenes)}")
                break

            # Old PC1 is the new PC0
            PC0 = PC1
            count += 1
        return PC_scenes
