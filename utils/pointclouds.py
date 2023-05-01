import torch
from utils.geometrics import align_point_clouds


class PC:
    def __init__(self, pc, distances):
        self.pc = pc
        self.distances_to_origin = distances
        self.N_points = pc.shape[0]
        self.N_dim = pc.shape[1]


class PCPair:
    def __init__(self, PC0, PC1, PCHandler=None):
        # Set point cloud pair
        self.PC0 = PC0
        self.PC1 = PC1

        # If CorAl is implemented we also store the union point cloud
        if PCHandler is not None:
            pc_union_dists = torch.cat((PC0.distances_to_origin, PC1.distances_to_origin))
            pc1_CS0 = align_point_clouds(PCHandler.pc1_CS1, PCHandler.lidar_pose0, PCHandler.lidar_pose1)
            pc_union_after_alignment = torch.cat((PCHandler.pc0_CS0, pc1_CS0), dim=0)
            PCUnion_after_alignment = PC(pc_union_after_alignment, pc_union_dists)
            self.PCUnion = PCUnion_after_alignment
        else:
            self.PCUnion = None
