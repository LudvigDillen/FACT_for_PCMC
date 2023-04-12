import torch
from utils.other import start_debug
from utils.geometrics import align_point_clouds
from utils.nuscenes_handling import NuscenesHandling
from classifiers.differential_entropy import differential_entropy_metric

import nuscenes as ns


def main():
    print("cuda available:", torch.cuda.is_available())

    # Initialize NuScenes object
    nusc = ns.nuscenes.NuScenes(version='v1.0-mini', dataroot='data/nuscenes/mini/', verbose=True)
    PCHandler = NuscenesHandling(nusc)

    # Calculate transformation matrices
    T0 = PCHandler.lidar_pose0
    T1 = PCHandler.lidar_pose1

    pc0_CS0 = PCHandler.pc0_CS0
    pc1_CS1 = PCHandler.pc1_CS1

    # Calculate differential entropy

    # result = differential_entropy_metric(pc0, pc1)
    # print(f"Differential entropy before alignment: {result}")

    pc1_CS0 = align_point_clouds(pc1_CS1, T0, T1)
    PCHandler.set_pc1_CS0(pc1_CS0)
    PCHandler.set_union_distances()
    PCHandler.set_pc_union()

    # TODO: Clean up the code structure. It is currently a mess
    # TODO: Compare the differential entropy metric before and after the alignment.
    result = differential_entropy_metric(PCHandler)
    print(f"Differential entropy after alignment: {result}")

    print("Finito!")


if __name__ == "__main__":
    start_debug()
    main()
