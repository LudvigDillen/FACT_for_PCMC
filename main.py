import torch
import time
import nuscenes as ns

from utils.other import start_debug
from utils.geometrics import align_point_clouds
from utils.nuscenes_handling import NuscenesHandling
from utils.pointclouds import PC
from classifiers.differential_entropy import differential_entropy_metric


def main():
    print("cuda available:", torch.cuda.is_available())

    # Initialize NuScenes object
    nusc = ns.nuscenes.NuScenes(version='v1.0-mini', dataroot='data/nuscenes/mini/', verbose=True)
    PCHandler = NuscenesHandling(nusc, downsample_factor=1)
    PC0 = PC(PCHandler.pc0_CS0, PCHandler.point_distances0)
    PC1 = PC(PCHandler.pc1_CS1, PCHandler.point_distances1)

    pc_union_dists = torch.cat((PC0.distances_to_origin, PC1.distances_to_origin))

    # Calculate differential entropy
    # Seems to work!
    pc1_CS0 = align_point_clouds(PCHandler.pc1_CS1, PCHandler.lidar_pose0, PCHandler.lidar_pose1)
    pc_union_after_alignment = torch.cat((PCHandler.pc0_CS0, pc1_CS0), dim=0)
    PCUnion_after_alignment = PC(pc_union_after_alignment, pc_union_dists)
    t1 = time.time()
    result = differential_entropy_metric(PC0, PC1, PCUnion_after_alignment)
    print(f"Differential entropy after alignment: {result}")
    print(f"Execution time (sec): {round(time.time() - t1, 3)}")
    print("Finito!")
    # TODO: Compare the differential entropy metric before and after the alignment.
    # TODO: Maybe try out kd-trees.
    # TODO: Is the usage of epsilon reasonable ...
    # TODO: Check covariance of A vs 2A. Have checked, it is not the same ....
    # TODO: Must H_joint > H_sep? (probably not after adding epsilon, but maybe)
    # TODO: E_reject, reject points with lowest entropy. Does this include negative entropies ...
    # TODO: They seemed to forgot N in their implementation of the diff entropy.
    # About -0.118 in metric with same PC (takes 34 sec without cuda)
    #  0.55 for different PCs but aligned (33 sec)
    #  0.16 for different PCs but non-aligned (34 sec) (not good ...)


if __name__ == "__main__":
    start_debug()
    main()
