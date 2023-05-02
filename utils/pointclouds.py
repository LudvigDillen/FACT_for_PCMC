import torch
import random
from utils.geometrics import align_point_clouds


class PC:
    def __init__(self, pc, distances):
        self.pc = pc
        self.distances_to_origin = distances
        self.N_points = pc.shape[0]
        self.N_dim = pc.shape[1]


class PCPair:
    def __init__(self, PC0, PC1, PCHandler=None, perturb_probability=0.5):
        # Set point cloud pair
        self.PC0 = PC0
        self.PC1 = PC1

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
        pc1_CS0 = align_point_clouds(PCHandler.pc1_CS1, PCHandler.lidar_pose0, PCHandler.lidar_pose1)
        if self.misaligned:
            pc1_CS0 = self.perform_random_perturbation_CorAl(pc1_CS0)
        pc_union_after_alignment = torch.cat((PCHandler.pc0_CS0, pc1_CS0), dim=0)
        PCUnion_after_alignment = PC(pc_union_after_alignment, pc_union_dists)
        self.PCUnion = PCUnion_after_alignment

    def perform_random_perturbation_CorAl(self, pc, angular_offset=0.01, translational_offset=0.1):
        """
        This function a point cloud perturb it with an angular and translational offset.

        :param pc: point cloud
        :param angular_offset: float, angular offset in radians around the sensor's vertical axis (default: 0.01 rad)
        :param translational_offset: float, distance of random translational offset in meters (default: 0.1 m)
        :return: perturbed point cloud
        """
        # TODO: Perturb point cloud around sensor's vertical axis
        print(pc.shape)
        print("Continue implementation here (utils/pointclouds.py)")
        # TODO: Perturb point cloud with a distance 0.1 meter from the ground truth
        perturbed_point_cloud = pc
        return perturbed_point_cloud
