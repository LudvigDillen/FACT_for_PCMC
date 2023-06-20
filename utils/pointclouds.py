import torch
import random
from utils.geometrics import change_coordinate_system


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
