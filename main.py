import torch
import time
import nuscenes as ns
import numpy as np
from scipy.spatial.transform import Rotation as Rotation
# rom sklearn import train_test_split


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
    # (HAVE NOW AUTOMATICALLY PETURBED AS THEY DO IN CORAL)
    # 2. Perform random perturbations or register data with ICP e.g.
    # 2.1 With ICP we can compare the estimate transformation by with the ground truth to classify
    #     the data as either aligned or misaligned. This data is then quite realistic and can be
    #     used to train our classifier (select suitable parameters).
    # 2.2 Align point clouds with ground truth and then apply a small offset and then register
    #     with ICP. This can lead to misalignments which are harder to detect as the relative
    #     error transformation is smaller. Thus, we might challenge the model a bit more which is
    #     good. Furthermore, ICP is a very interesting registration model to use for three reasons,
    #       1. Implementations of ICP are easily accesible thorugh e.g. Open3d
    #       2. It easily get stuck in local minima which is the realistic/common point when registration
    #          methods stop to iterate.
    #       3. It is common and well known, and useful as a benchmark registration method in that sense.
    # 3. Split data up into training and test set

    # TODO: How to choose model parameters
    # 1. Use some optimization in PyTorch to backpropagate the parameters of the logistic regression
    #    model (beta0, beta1, beta2).
    # 2. Optimize of the other parameters alpha (even though it is given by the dataset), epsilon,
    #    and E_reject by optimizing over a grid of values (grid search).

    # Initialize NuScenes object
    nusc = ns.nuscenes.NuScenes(version='v1.0-mini', dataroot='data/nuscenes/mini/', verbose=True)
    PCHandler = NuscenesHandling(nusc, downsample_factor=1)

    PC_scenes = PCHandler.get_entire_sub_dataset(n_scenes=1)

    # Some visualization: We can see that the point clouds are better aligned after the transformation
    # vis_2pcs_aligned_vs_misaligned(PC0.pc, PC1.pc, pc1_CS0)
    ##

    metrics_aligned = []
    metrics_misaligned = []

    for PC_scene in PC_scenes:
        for count, PC_pair in enumerate(reversed(PC_scene)):
            # if count >= 10:
            #     print("I dont have the time to run this on the entire dataset")
            #     break
            t1 = time.time()
            result = differential_entropy_metric(
                PC_pair.PC0, PC_pair.PC1, PC_pair.PCUnion, PC_pair.misaligned)

            if PC_pair.misaligned:
                metrics_misaligned.append(result)
            else:
                metrics_aligned.append(result)
            if len(metrics_aligned) > 0:
                print(f"Mean abs metric aligned    {np.around(np.mean(np.abs(metrics_aligned)), 4)}",
                      f"(N = {len(metrics_aligned)})")
            if len(metrics_misaligned) > 0:
                print(f"Mean abs metric misaligned {np.around(np.mean(np.abs(metrics_misaligned)), 4)}",
                      f"(N = {len(metrics_misaligned)})")
            print(f"Execution time (sec): {round(time.time() - t1, 3)}")
    print("Finito!")
    # TODO: Add plot of misalignment vs alignment metric lists ... to see if we can discriminate

    # TODO: Compare the differential entropy metric before and after the alignment.
    # TODO: Check covariance of A vs 2A. Have checked, it is not the same ....
    # TODO: Must H_joint > H_sep? (probably not after adding epsilon, but maybe)


if __name__ == "__main__":
    start_debug()
    main()
