import torch
import time
import nuscenes as ns
import numpy as np
from scipy.spatial.transform import Rotation as Rotation

from utils.other import start_debug
from utils.geometrics import align_point_clouds
from utils.nuscenes_handling import NuscenesHandling
from utils.pointclouds import PC
from classifiers.differential_entropy import differential_entropy_metric
from visualization.point_clouds import vis_pc, vis_2pcs_aligned_vs_misaligned


def main():
    print("cuda available:", torch.cuda.is_available())
    # TODO: 2023-04-27 (Gather data into lists e.g.)
    # 1. Gather data into lists (DONE!)
    # 2. Split data up into training and test set
    # 3. Perform random perturbations or register data with ICP e.g.
    # 3.1 With ICP we can compare the estimate transformation by with the ground truth to classify
    #     the data as either aligned or misaligned. This data is then quite realistic and can be
    #     used to train our classifier (select suitable parameters).
    # 3.2 Align point clouds with ground truth and then apply a small offset and then register
    #     with ICP. This can lead to misalignments which are harder to detect as the relative
    #     error transformation is smaller. Thus, we might challenge the model a bit more which is
    #     good. Furthermore, ICP is a very interesting registration model to use for three reasons,
    #       1. Implementations of ICP are easily accesible thorugh e.g. Open3d
    #       2. It easily get stuck in local minima which is the realistic/common point when registration
    #          methods stop to iterate.
    #       3. It is common and well known, and useful as a benchmark registration method in that sense.

    # TODO: How to choose model parameters
    # 1. Use some optimization in PyTorch to backpropagate the parameters of the logistic regression
    #    model (beta0, beta1, beta2).
    # 2. Optimize of the other parameters alpha (even though it is given by the dataset), epsilon,
    #    and E_reject by optimizing over a grid of values (grid search).

    # Initialize NuScenes object
    nusc = ns.nuscenes.NuScenes(version='v1.0-mini', dataroot='data/nuscenes/mini/', verbose=True)
    PCHandler = NuscenesHandling(nusc, downsample_factor=1)

    PC_scenes = PCHandler.get_entire_sub_dataset()

    exit()
    pc_union_dists = torch.cat((PC0.distances_to_origin, PC1.distances_to_origin))

    # Calculate differential entropy
    # Seems to work! I also checked by plotting and it seems to be aligned.
    pc1_CS0 = align_point_clouds(PCHandler.pc1_CS1, PCHandler.lidar_pose0, PCHandler.lidar_pose1)

    # Some visualization: We can see that the point clouds are better aligned after the transformation
    # vis_2pcs_aligned_vs_misaligned(PC0.pc, PC1.pc, pc1_CS0)
    ##
    # exit()

    pc_union_after_alignment = torch.cat((PCHandler.pc0_CS0, pc1_CS0), dim=0)
    PCUnion_after_alignment = PC(pc_union_after_alignment, pc_union_dists)
    t1 = time.time()
    result = differential_entropy_metric(PC0, PC1, PCUnion_after_alignment)
    print(f"Differential entropy after alignment: {np.around(result.cpu().numpy(), decimals=3)}")
    print(f"Execution time (sec): {round(time.time() - t1, 3)}")
    print("Finito!")
    # TODO: Compare the differential entropy metric before and after the alignment.
    # TODO: Check covariance of A vs 2A. Have checked, it is not the same ....
    # TODO: Must H_joint > H_sep? (probably not after adding epsilon, but maybe)


if __name__ == "__main__":
    start_debug()
    main()
