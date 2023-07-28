import numpy as np
import torch

from utils.geometrics import transformation_matrix
from utils.pointclouds import PC, PCPair
from utils.visibility import keep_covisible_points
from nuscenes.utils.data_classes import LidarPointCloud


DTYPE = torch.float64


class NuscenesHandling:
    def __init__(self, nusc, downsample_factor=1, lidar_token=None, T_close_thresh=0,
                 scene_counter=0, verbose=True, preprocess=True, perturb_settings=None):
        self.nusc = nusc
        self.scene_counter_init = scene_counter
        self.scene_counter = scene_counter
        self.number_of_scenes_in_dataset = len(self.nusc.scene)
        self.dataset_read = False
        self.scene_read = False

        # Set settings
        self.T_close_thresh = T_close_thresh
        self.downsample_factor = downsample_factor  # Randomly downsample point clouds
        self.verbose = verbose
        self.preprocess = preprocess
        self.perturb_settings = perturb_settings

        # Setup scene data
        self.setup_new_scene_data(lidar_token)

    def downsample_both_point_clouds(self):
        N_samples_before = self.pc0_CS0.shape[0]
        N_samples_after = round(N_samples_before/self.downsample_factor)
        samples_to_keep = np.random.choice(N_samples_before, size=N_samples_after)
        self.pc0_CS0 = self.pc0_CS0[samples_to_keep]  # Downsample point cloud

        N_samples_before = self.pc1_CS1.shape[0]
        N_samples_after = round(N_samples_before/self.downsample_factor)
        samples_to_keep = np.random.choice(N_samples_before, size=N_samples_after)
        self.pc1_CS1 = self.pc1_CS1[samples_to_keep]  # Downsample point cloud

    def downsample_second_point_cloud(self):
        N_samples_before = self.pc1_CS1.shape[0]
        N_samples_after = round(N_samples_before/self.downsample_factor)
        samples_to_keep = np.random.choice(N_samples_before, size=N_samples_after)
        self.pc1_CS1 = self.pc1_CS1[samples_to_keep]  # Downsample point cloud

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
        pc_xyz = torch.swapaxes(torch.from_numpy(pc.points[:3]).to(DTYPE), 0, 1)
        if self.T_close_thresh != 0:
            pc_xyz_filtered = self.remove_close_points_from_pc(pc_xyz, self.T_close_thresh)
        return pc_xyz_filtered

    def remove_close_points_from_pc(self, pc, T_close_thresh=1.5):
        # Pre-Process data
        pc_filtered = pc[torch.norm(pc, dim=1) >= T_close_thresh]
        return pc_filtered

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
        Ts_out = torch.from_numpy(Ts).to(DTYPE)
        return Ts_out

    def get_next_lidar_token(self, lidar_token):
        lidar_dict = self.get_lidar_dict(lidar_token)
        return lidar_dict['next']

    def check_end_of_scene(self, lidar_token):
        return (lidar_token == '')

    def check_end_of_dataset(self):
        return (self.scene_counter >= self.number_of_scenes_in_dataset)

    def count_scene(self):
        self.scene_read = True
        self.scene_counter += 1

    def update_dataset_status(self):
        self.count_scene()
        if self.check_end_of_dataset():
            self.dataset_read = True
        else:
            self.setup_new_scene_data(lidar_token=None)

    def set_next_point_cloud_pair(self, n_samples_jump=0):
        """
        Iteratate to next point cloud in the dataset. If the scene is finished, we go on to
        the next scene. If all as scenes are read, then we have read the entire dataset. Then,
        we have collected all the data we need/want.
        """
        # Move to next point cloud pair in the scene
        assert n_samples_jump >= 0, "ERROR: n_samples_jump>=0. Can't skip a negative amount of samples"
        assert isinstance(n_samples_jump, int), "ERROR: n_samples_jump must be an integer"

        if n_samples_jump <= 1:
            self.lidar_token0 = self.lidar_token1
            self.lidar_pose0 = self.lidar_pose1
            self.pc0_CS0 = self.pc1_CS1
            self.point_distances0 = self.point_distances1

        # Set info PC1
        self.lidar_token1 = self.get_next_lidar_token(self.lidar_token1)

        if self.check_end_of_scene(self.lidar_token1):
            self.update_dataset_status()
            return None

        if n_samples_jump <= 1:
            self.lidar_pose1 = self.get_sensor_pose_in_WCS(self.lidar_token1)
            self.pc1_CS1 = self.get_point_cloud(self.lidar_token1)

            # Randomly downsample point clouds
            if self.downsample_factor > 1:
                self.downsample_second_point_cloud()

            self.point_distances1 = self.get_point_distances_to_origin(self.pc1_CS1)

        if n_samples_jump > 0:
            self.set_next_point_cloud_pair(n_samples_jump=n_samples_jump-1)

    def get_number_of_samples_in_scenes(self, PC_scenes):
        n_samples = 0
        for PC_scene in PC_scenes:
            n_samples += len(PC_scene)
        return n_samples

    def get_number_lidar_samples_in_scene(self):
        scene = self.nusc.scene[self.scene_counter]
        # Get the first and last sample in the scene
        first_sample = self.nusc.get('sample', scene['first_sample_token'])
        last_sample = self.nusc.get('sample', scene['last_sample_token'])

        # Get the timestamps of the first and last sample
        first_timestamp = first_sample['timestamp']
        last_timestamp = last_sample['timestamp']

        # Get the Lidar data from the scene
        lidar_data = [d for d in self.nusc.sample_data if d['sensor_modality'] == 'lidar' and
                      first_timestamp <= d['timestamp'] <= last_timestamp]

        # Get the number of Lidar sweeps in the scene
        num_lidar_sweeps = len(lidar_data)
        return num_lidar_sweeps

    def sample_from_scenes(self, n_samples, cov_params, use_c, n_scenes='all',
                           batch_size=256):
        """
        Here we return the sample from the dataset s.t. we evenly distribute the sample of the number
        of scenes we want to utilize.

        Data format:
            A list of scenes. Each scenes correspond to roughly 20s recoreded lidar data
            at 20 Hz. This means that each scene contains around 400 point clouds. We have
            divided in so that the list consists of the class PCPairs which contains the
            one point cloud pair and the corresponding union point cloud.
        """
        PC_scenes = []
        PC_scene = []
        count = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # HACK: Avoid circular imports
        from utils.data_handling import list_of_samples_per_scene, calculate_sample_gaps

        samples_per_scene = list_of_samples_per_scene(n_samples, n_scenes)

        skip_sample_index = 0
        skip_samples_list = calculate_sample_gaps(
            self.get_number_lidar_samples_in_scene(),
            samples_per_scene[self.scene_counter-self.scene_counter_init])
        n_skip_samples = len(skip_samples_list)
        while True:
            if count % 200 == 0 and count != 0:
                print(f"We have collected {count} number of samples")

            # Load in first point cloud
            PC0 = PC(self.pc0_CS0, self.point_distances0, label=0, device=device)
            # Load in second point cloud
            PC1 = PC(self.pc1_CS1, self.point_distances1, label=1, device=device)
            # Set point cloud pair and their union, and perform possible perturbation
            currentPCPair = PCPair(PC0, PC1, device=device, PCHandler=self,
                                   perturb_settings=self.perturb_settings)

            if self.preprocess:
                # Calculate the co-visible points
                PC0_cov, PC1_cov, PCUnion_cov = keep_covisible_points(
                    PC0, PC1, currentPCPair.PCUnion, currentPCPair.pose0, currentPCPair.pose1,
                    compute_weights=use_c, hpr_radius=cov_params.hpr_radius,
                    gamma=cov_params.gamma, inversion_kernel=cov_params.inversion_kernel,
                    batch_size=batch_size)
                currentPCPair.set_new_PC(PC0_cov, PC1_cov, PCUnion_cov)
            # Append pair to list
            PC_scene.append(currentPCPair)
            del PC0, PC1, currentPCPair
            # Iterate to next pair in scene
            self.set_next_point_cloud_pair(n_samples_jump=skip_samples_list[skip_sample_index])
            skip_sample_index += 1
            if skip_sample_index == n_skip_samples and skip_samples_list[skip_sample_index-1] == 0:
                self.update_dataset_status()

            if self.scene_read:
                # Append scene to list of scenes
                PC_scenes.append(PC_scene)
                PC_scene = []
                self.scene_read = False
                if self.verbose:
                    print(f"We have collected {self.scene_counter-self.scene_counter_init} scenes")
                # We might just want to read a few scenes
                if n_scenes != 'all':
                    if self.scene_counter-self.scene_counter_init >= n_scenes:
                        break
                if self.dataset_read is False:
                    skip_sample_index = 0
                    skip_samples_list = calculate_sample_gaps(
                        self.get_number_lidar_samples_in_scene(),
                        samples_per_scene[self.scene_counter-self.scene_counter_init])
                    n_skip_samples = len(skip_samples_list)

            if self.dataset_read:
                print("We have collected all data from the dataset")
                break

            count += 1

        PC_scenes = np.array(PC_scenes)
        if self.verbose:
            print(f"Total number of samples: {self.get_number_of_samples_in_scenes(PC_scenes)}")
            print(f"Total number of scenes:  {len(PC_scenes)}")
        return PC_scenes


def read_nuscenes_data(nusc, n_samples, perturb_settings, cov_params, use_c, downsample_factor=1, n_scenes='all',
                       T_close_thresh=0, lidar_token=None, scene_counter=0,
                       verbose=True, preprocess=True, batch_size=256):
    # Read data
    PCHandler = NuscenesHandling(nusc, downsample_factor=downsample_factor,
                                 T_close_thresh=T_close_thresh, lidar_token=lidar_token,
                                 scene_counter=scene_counter, verbose=verbose, preprocess=preprocess,
                                 perturb_settings=perturb_settings)
    PC_scenes = PCHandler.sample_from_scenes(n_samples=n_samples, cov_params=cov_params, n_scenes=n_scenes,
                                             use_c=use_c, batch_size=batch_size)
    del PCHandler
    return PC_scenes
