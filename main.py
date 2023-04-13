import torch
from utils.other import start_debug
from utils.geometrics import align_point_clouds
from utils.nuscenes_handling import NuscenesHandling
from utils.pointclouds import PC
from classifiers.differential_entropy import differential_entropy_metric

import nuscenes as ns


def main():
    print("cuda available:", torch.cuda.is_available())

    # Initialize NuScenes object
    nusc = ns.nuscenes.NuScenes(version='v1.0-mini', dataroot='data/nuscenes/mini/', verbose=True)
    PCHandler = NuscenesHandling(nusc)
    PC0 = PC(PCHandler.pc0_CS0, PCHandler.point_distances0)
    PC1 = PC(PCHandler.pc1_CS1, PCHandler.point_distances1)
    pc_union_dists = torch.cat((PC0.distances_to_origin, PC1.distances_to_origin))

    # Calculate differential entropy
    pc_union_before_alignment = torch.cat((PCHandler.pc0_CS0, PCHandler.pc1_CS1), dim=1)
    PCUnion_before_alignment = PC(pc_union_before_alignment, pc_union_dists)

    # result = differential_entropy_metric(PC0, PC1, PCUnion_before_alignment)
    # print(f"Differential entropy before alignment: {result}")

    pc1_CS0 = align_point_clouds(PCHandler.pc1_CS1, PCHandler.lidar_pose0, PCHandler.lidar_pose1)
    pc_union_after_alignment = torch.cat((PCHandler.pc0_CS0, pc1_CS0), dim=1)
    PCUnion_after_alignment = PC(pc_union_after_alignment, pc_union_dists)

    # TODO: Clean up the code structure. It is currently a mess
    # TODO: Compare the differential entropy metric before and after the alignment.
    # TODO: Try out kd-trees for real
    result = differential_entropy_metric(PC0, PC1, PCUnion_after_alignment)
    print(f"Differential entropy after alignment: {result}")

    print("Finito!")


if __name__ == "__main__":
    start_debug()
    main()
