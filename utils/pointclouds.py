import torch
import random
import os
import numpy as np
from torch.utils.data import Dataset
import json

from PointTransformers.pointnet_util import farthest_point_sample, pc_normalize
from utils.geometrics import change_coordinate_system

###### Things to remember
# Only extract features for 'npoint' of the points. Select the npoints using farthest point sampling. (Speed)

# Fill up PCAC_data_train/test.txt with info in the data loading ..


class PCAC_dataset(torch.utils.data.Dataset):
    def __init__(self, root='/home/luddi824/thesis/PCAC/data/PCAC_data', npoint=1024, split='train',
                 uniform=False, normal_channel=True, cache_size=15000):
        self.root = root
        self.npoint = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'PCAC_data_class_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'PCAC_data_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'PCAC_data_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)

    def __len__(self):
        # This method should return the total size/length of the dataset.
        # For example, if you're reading from an array, you'd do "return len(self.my_array)"
        return len(self.dataset)


class PC:
    def __init__(self, pc, distances, label):
        self.pc = pc
        self.distances_to_origin = distances
        self.N_points = pc.shape[0]
        self.N_dim = pc.shape[1]
        """
        For the label we have that:
            0 means PC0
            1 means PC1
            2 means PCUnion
        """
        self.label = label

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

    def set_cardinality_joint_weight(self, ratio):
        self.weight_cj = ratio

    def set_cardinality_sep_weight(self, ratio):
        self.weight_cs = ratio


class PCPair:
    def __init__(self, PC0, PC1, PCHandler=None, perturb_probability=0.5):
        # Set point cloud pair
        self.PC0 = PC0
        self.pose0 = PCHandler.lidar_pose0
        self.PC1 = PC1
        self.pose1 = PCHandler.lidar_pose1

        # Draw random value between 0 and 1 if we should perturb or not perturb the point cloud.
        peturb_point_cloud = (random.random() < perturb_probability)
        self.misaligned = peturb_point_cloud  # The ground truth if the point cloud pair is aligned or not
        # If CorAl is implemented we also store the union point cloud
        self.set_union_of_point_clouds(PCHandler)

    def set_union_of_point_clouds(self, PCHandler):
        assert (PCHandler is not None), ("ERROR: We assume that the differential entropy method is being"
                                         "implemented! This can be changed, but in that case functionality",
                                         "for that has to be added. Th union of the point clouds should not",
                                         "be necessary in that case.")

        pc_union_dists = torch.cat((self.PC0.distances_to_origin, self.PC1.distances_to_origin))
        self.pc1_CS0 = change_coordinate_system(PCHandler.pc1_CS1, PCHandler.lidar_pose0,
                                                PCHandler.lidar_pose1)
        if self.misaligned:
            self.pc1_CS0 = self.perform_random_perturbation_CorAl(self.pc1_CS0, angular_offset=0.03,
                                                                  translational_offset=0.3)

        # pc_union may be the concatenation of either two aligned point clouds or two misaligned point clouds.
        # This will depend on if we randomly peturb one of the aligned point cloud or not.
        # This happen with the peturb probality handed to the constructor of the class.
        pc_union = torch.cat((PCHandler.pc0_CS0, self.pc1_CS0), dim=0)
        self.PCUnion = PC(pc_union, pc_union_dists, label=2)

    def perform_random_perturbation_CorAl(self, pc, angular_offset=0.01, translational_offset=0.1):
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
        R_peturb = torch.tensor([[cos_off,  -sin_off,   0],
                                 [sin_off,  cos_off,    0],
                                 [0,        0,          1]])

        # Define random translation offset of 0.1m in (x,y)-plane
        random_xy_offset = torch.rand((2, 1))
        scaled_random_xy_offset = translational_offset*random_xy_offset/torch.norm(random_xy_offset)
        z_offset = torch.tensor(0)  # there should be no offset in y-direction
        t_peturb = torch.vstack((scaled_random_xy_offset, z_offset)).squeeze()

        # Define rigid transformation matrix in homogeneuous coordinates
        T_peturb = torch.eye(4, dtype=pc.dtype)
        T_peturb[:3, :3] = R_peturb
        T_peturb[:3, 3] = t_peturb

        # Peturb point cloud
        n_points = pc.shape[0]
        homog_ones = torch.ones(n_points)
        pc_homog_swapped = torch.vstack((torch.swapaxes(pc, 0, 1), homog_ones))
        perturbed_point_cloud = torch.swapaxes(torch.matmul(T_peturb, pc_homog_swapped)[:3], 0, 1)

        return perturbed_point_cloud


def farthest_point_sample_PC_scenes(PC_scenes, n_points):
    n_point_clouds = 2*PC_scenes.size  # PC0 and PC1 for every point cloud
    # Find the least points of the current pcs
    min_points = int(1e10)
    for i in range(PC_scenes.shape[0]):
        for j in range(PC_scenes.shape[1]):
            pc0_points = PC_scenes[i][j].PC0.pc.shape[0]
            if pc0_points < min_points:
                min_points = pc0_points
            pc1_points = PC_scenes[i][j].PC1.pc.shape[0]
            if pc1_points < min_points:
                min_points = pc0_points
    # Go on with creating a downsampling each point cloud so it has least points
    # Farthest point sample every point cloud down to n_points
    # Add some index which keeps track of the farthest point sampled points
    # Return something s.t. when we e.g. evaluate differential_entropy, the whole
    # neighborhood is considered from the full point cloud, but only points that are
    # downsampled with furthest point sampling will be the point we consider the 
    # neighborhood from