import torch
import random
import os
import numpy as np

from utils.pointnet_util import pc_normalize, pad_point_clouds, farthest_point_sample_paddded
from utils.geometrics import change_coordinate_system


class PCAC_dataset(torch.utils.data.Dataset):
    def __init__(self, root='/home/luddi824/thesis/PCAC/data/PCAC_data', split='train', cache_size=3000):
        self.root = root
        self.catfile = os.path.join(self.root, 'PCAC_data_class_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        class_ids = {}
        class_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'PCAC_data_train.txt'))]
        class_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'PCAC_data_test.txt'))]

        assert (split == 'train' or split == 'test')
        class_names = ['_'.join(x.split('_')[0:-1]) for x in class_ids[split]]
        # list of (class_name, class_txt_file_path) tuple
        self.datapath = [(class_names[i],
                          os.path.join(self.root, class_names[i], class_ids[split][i]) + '.txt')
                         for i in range(len(class_ids[split]))]
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
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)


class PC:
    def __init__(self, pc, distances, label, device):
        self.device = device

        self.pc = pc.to(device)
        self.distances_to_origin = distances.to(device)
        self.N_points = pc.shape[0]
        self.N_dim = pc.shape[1]

        self.set_label(label)
        # initiate the fps_inds tensor as all points in the point cloud (i.e. that means no downsampling)
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
        # TODO: DO I need to clone here?
        empty_feature = torch.empty(self.N_fps_points, dtype=self.pc.dtype).to(self.device)
        self.metric_jde = empty_feature.clone()
        self.metric_sde = empty_feature.clone()
        self.metric_wd = empty_feature.clone()
        self.weight_c = empty_feature.clone()
        self.weight_s = empty_feature.clone()
        self.weight_cj = empty_feature.clone()
        self.weight_cs = empty_feature.clone()

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

    def set_fps_inds(self, fps_inds):
        self.fps_inds = fps_inds
        self.N_fps_points = len(fps_inds)
        # initiate features
        self.init_features()


class PCPair:
    def __init__(self, PC0, PC1, device, PCHandler, perturb_probability=0.5):
        # Set point cloud pair
        self.PC0 = PC0
        self.pose0 = PCHandler.lidar_pose0.to(device)
        self.PC1 = PC1
        self.pose1 = PCHandler.lidar_pose1.to(device)

        self.device = device

        # Draw random value between 0 and 1 if we should perturb or not perturb the point cloud.
        peturb_point_cloud = (random.random() < perturb_probability)
        self.misaligned = peturb_point_cloud  # The ground truth if the point cloud pair is aligned or not
        # If CorAl is implemented we also store the union point cloud
        self.set_union_of_point_clouds()

    def set_union_of_point_clouds(self):
        self.PCUnion, self.pc1_CS0 = get_union_of_point_clouds(self.PC0, self.PC1, self.pose0,
                                                               self.pose1, self.misaligned, self.device)

    def set_new_PC(self, PC0, PC1, PCUnion):
        self.PC0 = PC0
        self.PC1 = PC1
        self.PCUnion = PCUnion
        self.pc1_CS0 = change_coordinate_system(PC1.pc, self.pose0, self.pose1)

    def set_name(self, name):
        self.name = name


def perform_random_perturbation_CorAl(pc, angular_offset=0.01, translational_offset=0.1):
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
    T_peturb = torch.eye(4, dtype=pc.dtype, device=pc.device)
    T_peturb[:3, :3] = R_peturb
    T_peturb[:3, 3] = t_peturb

    # Peturb point cloud
    n_points = pc.shape[0]
    homog_ones = torch.ones(n_points, device=pc.device)
    pc_homog_swapped = torch.vstack((torch.swapaxes(pc, 0, 1), homog_ones))
    perturbed_point_cloud = torch.swapaxes(torch.matmul(T_peturb, pc_homog_swapped)[:3], 0, 1)

    return perturbed_point_cloud


def get_union_of_point_clouds(PC0, PC1, pose0, pose1, misaligned, device):
    pc_union_dists = torch.cat((PC0.distances_to_origin, PC1.distances_to_origin))
    pc1_CS0 = change_coordinate_system(PC1.pc, pose0, pose1)
    if misaligned:
        pc1_CS0 = perform_random_perturbation_CorAl(pc1_CS0, angular_offset=0.03, translational_offset=0.3)

    # pc_union may be the concatenation of either two aligned point clouds or two misaligned point clouds.
    # This will depend on if we randomly peturb one of the aligned point cloud or not.
    # This happen with the peturb probality handed to the constructor of the class.
    pc_union = torch.cat((PC0.pc, pc1_CS0), dim=0)
    PCUnion = PC(pc_union, pc_union_dists, label=2, device=device)
    return PCUnion,  pc1_CS0


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
            fps_inds_in_union_pc = torch.cat((batch_fps_inds[k-2],
                                              batch_fps_inds[k-1]+PC_scenes[i][j].PC0.N_points)).to(device)
            PC_scenes[i][j].PCUnion.set_fps_inds(fps_inds_in_union_pc)
    # when we e.g. evaluate differential_entropy, the whole
    # neighborhood is considered from the full point cloud, but only points that are
    # downsampled with furthest point sampling will be the points we consider the
    # neighborhoods from
    return None
