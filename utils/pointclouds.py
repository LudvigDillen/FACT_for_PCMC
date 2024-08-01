import os
import sys
import torch
import numpy as np

from utils.pointnet_util import pad_point_clouds, farthest_point_sample_paddded
from utils.geometrics import change_coordinate_system
import registration.registration_utils as ru
from visualization.point_clouds import vis_2pcs


class PCAC_dataset(torch.utils.data.Dataset):
    def __init__(self, args, split="train", cache_size=1000):
        self.root = args.feature_folder
        self.args = args
        self.catfile = os.path.join(self.root, "PCAC_data_class_names.txt")

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        class_ids = {}
        modes = ["train", "validation", "test"]
        total_samples_available = 0
        for mode in modes:
            mode_file = "PCAC_data_" + mode + ".txt"
            class_ids[mode] = [
                line.rstrip() for line in open(os.path.join(self.root, mode_file))
            ]
            total_samples_available += len(class_ids[mode])

        self.n_samples = self.args.n_samples
        if total_samples_available != self.args.n_samples:
            class_ids = self._change_data_usage(class_ids, total_samples_available)

        assert split in modes, "Error valid split not given!"
        # class_names = ['_'.join(x.split('_')[0:-1]) for x in class_ids[split]]  # old one
        class_names = ["_".join(x.split("_")[2:-1]) for x in class_ids[split]]
        # list of (class_name, class_txt_file_path) tuple
        self.datapath = [
            (
                class_names[i],
                os.path.join(self.root, class_names[i], class_ids[split][i]) + ".txt",
            )
            for i in range(len(class_ids[split]))
        ]
        print("The size of %s data is %d" % (split, len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

        if args.ablation.run_ablation:
            feature_filter_with_xyz = np.concatenate(
                (np.ones(3, dtype=int), np.array(args.ablation_feature_filter))
            )
        else:
            feature_filter_with_xyz = np.concatenate(
                (np.ones(3, dtype=int), np.array(args.feature_filter))
            )
        self.feature_channels = np.where(feature_filter_with_xyz == 1)[0]

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls, scene_number = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=",").astype(np.float32)
            # Remove some of the features which we do not want to use
            point_set = point_set[:, self.feature_channels]
            scene_number = int(fn[1].split("/")[-1].split("_")[1])

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, scene_number)

        return point_set, cls, scene_number

    def __getitem__(self, index):
        return self._get_item(index)

    def _change_data_usage(self, class_ids, total_samples_available):
        n_samples = self.args.n_samples
        if n_samples > total_samples_available:
            print(f"Only {total_samples_available} are available. Using all!")
        elif n_samples < total_samples_available:
            n_train_samples = round(n_samples * self.args.train_ratio)
            n_val_samples = round(n_samples * self.args.val_ratio)
            n_test_samples = round(
                n_samples * (1 - self.args.train_ratio - self.args.val_ratio)
            )
            class_ids["train"] = class_ids["train"][:n_train_samples]
            class_ids["validation"] = class_ids["validation"][:n_val_samples]
            class_ids["test"] = class_ids["test"][:n_test_samples]
        return class_ids


class PC:
    def __init__(self, pc, distances, label, device):
        self.device = device
        self.pc = pc.to(device)
        self.dtype = pc.dtype
        self.distances_to_origin = distances.to(device)
        self.N_points = pc.shape[0]
        self.N_fps_points = self.N_points  # Can be overwritten later
        self.N_dim = pc.shape[1]

        self.weight_c = None

        self.set_label(label)
        # init the fps_inds tensor as the inds to all pc points (i.e. that means no downsampling)
        self.fps_inds = torch.arange(0, self.N_points, dtype=torch.int).to(device)

    def set_label(self, label):
        """
        For the label we have that:
            0 means PC0
            1 means PC1
            2 means PCUnion
        """
        self.label = label

    def init_features(self):
        # TODO: Do I need to clone here?
        empty_feature = torch.empty(self.N_fps_points, dtype=self.pc.dtype).to(
            self.device
        )
        self.metric_jde = empty_feature.clone()
        self.metric_sde = empty_feature.clone()
        self.metric_wd = empty_feature.clone()
        # TODO: perhaps do this in a more clean way ...
        if self.label == 2:
            if self.weight_c is not None:
                self.weight_c = self.weight_c[self.fps_inds]
        else:
            self.weight_c = empty_feature.clone()
        self.weight_s = empty_feature.clone()
        self.weight_cj = empty_feature.clone()
        self.weight_cs = empty_feature.clone()
        self.weight_csj = empty_feature.clone()

    def set_joint_diff_entropy(self, value):
        self.metric_jde = value

    def set_sep_diff_entropy(self, value):
        self.metric_sde = value

    def set_wasserstein_dist(self, value):
        self.metric_wd = value

    def set_covisibility_weight(self, weight):
        self.weight_c = weight

    def set_static_point_weight(self, weight):
        self.weight_s = weight

    def set_cardinality_ratio_joint_weight(self, ratio):
        self.weight_cj = ratio

    def set_cardinality_ratio_sep_weight(self, ratio):
        self.weight_cs = ratio

    def set_cardinality_ratio_sep_and_joint_weight(self, ratio):
        self.weight_csj = ratio

    def set_fps_inds(self, fps_inds):
        self.fps_inds = fps_inds
        self.N_fps_points = len(fps_inds)
        # initiate features
        self.init_features()

def subsample_point_cloud(point_cloud, fraction=0.20):
    """
    Subsamples the point cloud by retaining only a fraction of the points.
    
    :param point_cloud: Nx3 array of points in the point cloud.
    :param fraction: Fraction of points to retain.
    :return: Subsampled point cloud.
    """
    num_points = point_cloud.shape[0]
    num_subsampled_points = int(num_points * fraction)
    
    # Randomly select indices to retain
    indices = np.random.choice(num_points, num_subsampled_points, replace=False)
    
    return point_cloud[indices]


# TODO: I do not need to calculate the distance from the origin for all points. It suffices with
# the point we choose to select after FPS. This might be true for other things we do as well. Like
# co-visibility score. Potential speed-up possible there.
class PCPair:
    def __init__(self, PC0, PC1, device, PCHandler, perturb_settings, change_of_pose_C1=False,
                 perturbation_method="m_classes", pc_reg_dist=1, reg_method="p2l", geo_args=None,
                 est_pc1_to_pc0=None, gt_pc1_to_pc0=None, geotrans_dataset="kitti"):
        # Set point cloud pair
        self.PC0 = PC0
        self.PC1 = PC1

        if est_pc1_to_pc0 is not None:
            # This below is correct, I have checked around 15 samples and they all look good (
            # except one which I think has error in the ground truth pose).
            self.pose0 = torch.eye(4, dtype=PC0.dtype, device=device)
            self.pose1 = gt_pc1_to_pc0
            self.already_registered = True
            reg_method = "geotransformer"
            self.est_pc1_to_pc0 = est_pc1_to_pc0
            plot = False
            if plot:
                updated_pc = change_coordinate_system(PC1.pc, self.pose0, self.pose1)
                vis_2pcs(self.PC0.pc.cpu(), updated_pc.cpu(), title="GT aligned")
        else:
            self.pose0 = PCHandler.lidar_pose0.to(device)
            self.pose1 = PCHandler.lidar_pose1.to(device)
            self.already_registered = False

        self.device = device

        # Setup class
        self.N_classes = perturb_settings.n_classes
        self.R_bin = perturb_settings.r_bin
        self.t_bin = perturb_settings.t_bin
        self.change_of_pose_C1 = change_of_pose_C1
        self.perturbation_method = perturbation_method
        self.reg_method = reg_method
        self.geo_args = geo_args
        # 1 if point clouds follow each other, 2 if there is one point cloud in between and so on.
        self.pc_reg_dist = pc_reg_dist
        # Draw random value between 0 and 1 if we should perturb or not perturb the point cloud.
        # if perturb_settings.class_distribution == 'uniform': ... I have variant currently ...
        if perturbation_method == "m_classes":
            self.class_category = np.random.choice(np.arange(self.N_classes))

        self.set_union_of_point_clouds()

    def set_new_PC(self, PC0, PC1, PCUnion):
        self.PC0 = PC0
        self.PC1 = PC1
        self.PCUnion = PCUnion
        # TODO: Remove line this later ... There was a bugg here previously where we had this line:
        # self.pc1_CS0 = change_coordinate_system(PC1.pc, self.pose0, self.pose1)
        # instead of the one below
        self.pc1_CS0 = PCUnion.pc[PC0.pc.shape[0] :]

    def set_name(self, name):
        self.name = name

    def set_union_of_point_clouds(self):
        pc_union_dists = torch.cat(
            (self.PC0.distances_to_origin, self.PC1.distances_to_origin)
        )
        if self.change_of_pose_C1 is False:
            self.pc1_CS0 = self.PC1.pc.clone()
        elif self.perturbation_method == "m_classes":  # discrete
            self.pc1_CS0 = change_coordinate_system(self.PC1.pc, self.pose0, self.pose1)
            # if class_category == 0, do not perturb anything ...
            self.R_offset = self.R_bin * self.class_category
            self.t_offset = self.t_bin * self.class_category
            if self.class_category != 0:
                self.pc1_CS0 = self.perform_random_perturbation_CorAl(
                    self.pc1_CS0,
                    angular_offset=self.R_offset,
                    translational_offset=self.t_offset,
                )
        elif self.perturbation_method in ["GMM", "registration"]:  # continuous
            # CODE FOR GMM is not checked and might not work
            # if self.perturbation_method == "GMM":
            #     s = np.random.rand([0, 1])
            #     if s:
            #         self.t_offset = np.random.normal(loc=0, scale=0.05)  # t_tight
            #     else:
            #         self.t_offset = np.random.normal(loc=0, scale=0.75)  # t_wide
            #     s = np.random.rand([0, 1])
            #     if s:
            #         self.R_offset = np.random.normal(loc=0, scale=0.001)  # r_tight
            #     else:
            #         self.R_offset = np.random.normal(loc=0, scale=0.015)  # r_wide
            #     self.pc1_CS0 = change_coordinate_system(self.PC1.pc, self.pose0, self.pose1)
            #     # TODO: Potentially, I want to rewrite perform_random_perturbation_CorAl a bit.
            #     self.pc1_CS0 = perform_random_perturbation_CorAl(
            #         self.pc1_CS0,
            #         angular_offset=self.R_offset,
            #         translational_offset=self.t_offset,
            #     )
            gt_pose = torch.matmul(torch.linalg.inv(self.pose0), self.pose1)
            
            if self.already_registered:
                rel_pose = self.est_pc1_to_pc0
            else:
                source = ru.from_tensor_to_pcd(self.PC1.pc)
                target = ru.from_tensor_to_pcd(self.PC0.pc)
                rel_pose = ru.register_pair(source, target, method=self.reg_method,
                                            gt_pose=gt_pose, geo_args=self.geo_args)
            self.est_rel_pose = rel_pose
            self.gt_pose = gt_pose
            self.pc1_CS0 = ru.align_pair(self, rel_pose, plot=False)

            version = "average distance est to gt pose"  # ["average distance est to gt pose", "binary"]
            if version == "binary":
                self.R_offset, self.t_offset = ru.get_transformation_error(rel_pose, gt_pose)
                if self.t_offset < 0.1 and self.R_offset < 0.002:
                    self.class_category = 0
                else:
                    self.class_category = 1
            elif version == "average distance est to gt pose":
                pc1_CS0_gt = ru.align_pair(self, gt_pose, plot=False)
                error = torch.linalg.norm(self.pc1_CS0 - pc1_CS0_gt, dim=1).mean()
                self.class_category = ru.get_error_class(error, self.reg_method)
        else:
            sys.exit(f"Perturbation method ({self.perturbation_method}) not known!")

        # pc_union is in the coordinate system of pc0
        # print(f"Error {error:.3f} and class {self.class_category}")
        pc_union = torch.cat((self.PC0.pc, self.pc1_CS0), dim=0)
        self.PCUnion = PC(pc_union, pc_union_dists, label=2, device=self.device)
        return None


    # TODO: If I want to a speed up I should do this in batches,
    # TODO: Why didn't I perturb T1 instead of PC1?
    def perform_random_perturbation_CorAl(
        self, pc, angular_offset=0.01, translational_offset=0.1
    ):
        """
        This function a point cloud perturb it with an angular and translational offset.

        :param pc: point cloud
        :param angular_offset: float, angular offset in radians around the sensor's vertical axis
                                (default: 0.01 rad)
        :param translational_offset: float, distance of random translational offset (x,y)-coord in meters
                                        (default: 0.1 m)
        :return: perturbed point cloud
        """
        # Define rotation with angle "angular_offset" around the up-vector
        cos_off = torch.cos(torch.tensor(angular_offset))
        sin_off = torch.sin(torch.tensor(angular_offset))
        R_peturb = torch.tensor([[cos_off, -sin_off, 0], [sin_off, cos_off, 0], [0, 0, 1]])

        # Define random translation offset of 0.1m in (x,y)-plane
        random_xy_offset = torch.rand((2, 1))
        scaled_random_xy_offset = (
            translational_offset * random_xy_offset / torch.norm(random_xy_offset)
        )
        z_offset = torch.tensor(0)  # there should be no offset in y-direction
        t_peturb = torch.vstack((scaled_random_xy_offset, z_offset)).squeeze()

        # Define rigid transformation matrix in homogeneuous coordinates
        T_peturb = torch.eye(4, dtype=pc.dtype, device=pc.device)
        T_peturb[:3, :3] = R_peturb
        T_peturb[:3, 3] = t_peturb
        self.est_rel_pose = T_peturb

        # Peturb point cloud
        n_points = pc.shape[0]
        homog_ones = torch.ones(n_points, device=pc.device)
        pc_homog_swapped = torch.vstack((torch.swapaxes(pc, 0, 1), homog_ones))
        perturbed_point_cloud = torch.swapaxes(
            torch.matmul(T_peturb, pc_homog_swapped)[:3], 0, 1
        )
        return perturbed_point_cloud


def farthest_point_sample_PC_scenes(PC_scenes, fps_N_points):
    device = PC_scenes[0][0].PC0.device
    # Find the least points of the current pcs
    list_pcs = []
    for PC_scene in PC_scenes:
        for PC_sample in PC_scene:
            for j in range(2):
                if j == 0:
                    PC = PC_sample.PC0
                else:
                    PC = PC_sample.PC1
                list_pcs.append(PC.pc)
    padded_PC_scenes = pad_point_clouds(list_pcs)
    batch_fps_inds = farthest_point_sample_paddded(padded_PC_scenes, fps_N_points)

    # Set new indices
    N_scenes, N_samples_per_scenes = PC_scenes.shape
    k = 0
    for i in range(N_scenes):
        for j in range(N_samples_per_scenes):
            PC_scenes[i][j].PC0.set_fps_inds(batch_fps_inds[k])
            k += 1
            PC_scenes[i][j].PC1.set_fps_inds(batch_fps_inds[k])
            k += 1
            fps_inds_in_union_pc = torch.cat(
                (
                    batch_fps_inds[k - 2],
                    batch_fps_inds[k - 1] + PC_scenes[i][j].PC0.N_points,
                )
            ).to(device)
            PC_scenes[i][j].PCUnion.set_fps_inds(fps_inds_in_union_pc)
    # when we e.g. evaluate differential_entropy, the whole
    # neighborhood is considered from the full point cloud, but only points that are
    # downsampled with furthest point sampling will be the points we consider the
    # neighborhoods from
    return None
