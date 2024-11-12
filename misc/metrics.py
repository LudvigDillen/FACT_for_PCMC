import torch
import hydra
import nuscenes as ns
import numpy as np
from geomloss import SamplesLoss
import time
import matplotlib.pyplot as plt
from datetime import datetime

from utils.parameters import Params
from utils.nuscenes_handling import read_nuscenes_data, NuscenesHandling
from utils.experiment_utils import setup_experiment
import visualization.registration as vr
import registration.registration_utils as ru


def estimate_transformation_scene(PC_scene, gt_poses, method="p2l", voxelize=False, plot=False,
                                  geo_args=None, get_icp_residuals=False):
    if get_icp_residuals:
        assert method in ["ICP-p2l", "icp-p2l", "p2l"], "ICP residuals only available for p2l."
    trans_init = np.eye(4)
    rel_poses_est = torch.zeros((len(PC_scene), 4, 4), device='cuda')
    fitness = np.zeros(len(PC_scene))
    rmse = np.zeros(len(PC_scene))
    for j, (pair, gt_pose) in enumerate(zip(PC_scene, gt_poses)):
        # We want to find T from CS1 to CS0
        source = ru.from_tensor_to_pcd(pair.PC1.pc)
        target = ru.from_tensor_to_pcd(pair.PC0.pc)

        if get_icp_residuals:
            rel_poses_est[j], fitness[j], rmse[j] = ru.register_pair(
                source, target, method=method, trans_init=trans_init, voxelize=voxelize,
                gt_pose=gt_pose, geo_args=geo_args, get_icp_residuals=get_icp_residuals)
        else:
            rel_poses_est[j] = ru.register_pair(
                source, target, method=method, trans_init=trans_init, voxelize=voxelize,
                gt_pose=gt_pose, geo_args=geo_args)

        if plot:
            vr.draw_registration_result(source, target, rel_poses_est[j].cpu().numpy(),
                                        title=f"{method}")
    if get_icp_residuals:
        return rel_poses_est, fitness, rmse
    return rel_poses_est


def get_hausdorff_distance(pc0, pc1):
    # Compute pairwise distances
    dist_pc0_pc1 = torch.cdist(pc0, pc1)  # Shape: [N, M]

    # Minimum distance from each point in pc0 to any point in pc1
    min_dist_pc0_pc1 = dist_pc0_pc1.min(dim=1).values

    # Minimum distance from each point in pc1 to any point in pc0
    min_dist_pc1_pc0 = dist_pc0_pc1.min(dim=0).values
    del dist_pc0_pc1

    # Hausdorff distance is the maximum of these minimum distances in both directions
    hausdorff_dist = torch.max(min_dist_pc0_pc1.max(), min_dist_pc1_pc0.max())
    return hausdorff_dist


def get_chamfer_distance(pc0, pc1):
    # Compute pairwise distances
    dist_pc0_pc1 = torch.cdist(pc0, pc1)  # Shape: [N, M]

    # Minimum distance from each point in pc0 to any point in pc1
    min_dist_pc0_pc1 = dist_pc0_pc1.min(dim=1).values

    # Minimum distance from each point in pc1 to any point in pc0
    min_dist_pc1_pc0 = dist_pc0_pc1.min(dim=0).values
    del dist_pc0_pc1

    # Chamfer distance is the sum of minimum distances
    chamfer_dist = min_dist_pc0_pc1.mean() + min_dist_pc1_pc0.mean()
    return chamfer_dist


def compute_sinkhorn(pc0, pc1, blur=0.05):
    loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=blur)
    return loss_fn(pc0, pc1).item()


@hydra.main(config_path="../classifiers/PointTransformers/config", config_name="cls_metrics")
def metrics_vs_gt_class(args):
    ### SETUP
    # Init Nusc object
    nusc = ns.nuscenes.NuScenes(version=args.dataset, dataroot=args.data_folder, verbose=False)

    # args.n_scenes = int(scene_ind)
    # Extra settings:
    n_samples_per_scene = args.n_samples_per_scene
    DO_REG = True
    CHANGE_OF_POSE_C1 = True  # TODO: Perhaps change back to True
    REG_METHOD = "p2l"  # [p2l, geotrans]
    GET_ICP_RESIDUALS = False
    args["get_icp_residuals"] = GET_ICP_RESIDUALS

    # Some settings if I use geotrans. Mode in [kitti, 3dmatch]
    geo_args = ru.get_geo_config(mode="kitti") if REG_METHOD == "geotrans" else None
    #
    # test_start_scene = args.n_scenes - 1 if args.one_scene else 0
    test_start_scene = 0
    # Set to 638 if we want to only test at test data
    n_scenes = args.n_scenes - test_start_scene


    args, logger = setup_experiment(args, do_reg=DO_REG, change_of_pose_C1=CHANGE_OF_POSE_C1,
                                    reg_method=REG_METHOD)
    params = Params(nusc=nusc, args=args, pointwise=True)
    if args.one_scene:
        # Read data
        PCHandler = NuscenesHandling(params, mode="test", lidar_token=None,
                                     scene_counter=test_start_scene)
        max_samples = PCHandler.get_number_lidar_samples_in_scene()
        if n_samples_per_scene >= max_samples:
            print(f"There are not {n_samples_per_scene} samples in scene {n_scenes}." +
                  f" Let's use the max number of samples: {max_samples}.")
            n_samples_per_scene = max_samples - 1

    errors_scenes = torch.zeros((n_scenes, n_samples_per_scene), device='cuda')
    params.set_which_features_to_use(args.features_to_create)
    gts = np.zeros((n_scenes, n_samples_per_scene))
    poses_est_scenes = torch.zeros((n_scenes, n_samples_per_scene, 4, 4), device='cuda')
    poses_gt_scenes = torch.zeros((n_scenes, n_samples_per_scene, 4, 4), device='cuda')
    fitness = np.zeros((n_scenes, n_samples_per_scene))
    rmse = np.zeros((n_scenes, n_samples_per_scene))
    hausdorff_distances = torch.zeros((n_scenes, n_samples_per_scene), device='cuda')
    chamfer_distances = torch.zeros((n_scenes, n_samples_per_scene), device='cuda')
    sinkhorn_distances_0_05 = torch.zeros((n_scenes, n_samples_per_scene), device='cuda')
    sinkhorn_distances_1em9 = torch.zeros((n_scenes, n_samples_per_scene), device='cuda')
    
    for i in range(n_scenes):
        if args.one_scene:
            PC_scene = PCHandler.sample_from_scenes(params, n_samples=n_samples_per_scene, n_scenes=1,
                                                    geo_args=geo_args)[0]
        else:
            PC_scene = read_nuscenes_data(params, mode="test", n_samples=n_samples_per_scene,
                                          n_scenes=1, scene_counter=i+test_start_scene)[0]
        for j in range(n_samples_per_scene):
            pc0 = PC_scene[j].PC0.pc.float()
            pc1 = PC_scene[j].PC1.pc.float()
            t1 = time.time()
            hausdorff_distances[i, j] = get_hausdorff_distance(pc0, pc1)
            t2 = time.time()
            chamfer_distances[i, j] = get_chamfer_distance(pc0, pc1)
            t3 = time.time()
            sinkhorn_distances_0_05[i, j] = compute_sinkhorn(pc0, pc1, blur=0.05)
            t4 = time.time()
            sinkhorn_distances_1em9[i, j] = compute_sinkhorn(pc0, pc1, blur=1e-9)
            t5 = time.time()
            #print(f"Time: Hausdorff: {t2-t1:.3f}, Chamfer: {t3-t2:.3f}, Sinkhorn 0.05: {t4-t3:.3f}, Sinkhorn 1e-9: {t5-t4:.3f}")

        if i % 5 == 0:
            print(f"Scene {i}")
        # Register scene
        poses_gt_scenes[i] = ru.get_gt_poses(PC_scene)
        if CHANGE_OF_POSE_C1:
            if GET_ICP_RESIDUALS:
                poses_est_scenes[i], fitness[i], rmse[i] = ru.get_est_rel_poses(
                    PC_scene, get_icp_residuals=GET_ICP_RESIDUALS)
            else:
                poses_est_scenes[i] = ru.get_est_rel_poses(PC_scene)
        elif GET_ICP_RESIDUALS:
            poses_est_scenes[i], fitness[i], rmse[i] = estimate_transformation_scene(
                PC_scene, gt_poses=poses_gt_scenes[i], method=REG_METHOD, geo_args=geo_args,
                get_icp_residuals=GET_ICP_RESIDUALS)
        else:
            poses_est_scenes[i] = estimate_transformation_scene(
                PC_scene, gt_poses=poses_gt_scenes[i], method=REG_METHOD, geo_args=geo_args)
        # Calculate errors
        errors_scenes[i] = ru.get_mean_point_error(PC_scene, poses_est_scenes[i], poses_gt_scenes[i])
        gts[i] = ru.get_gt_classes(errors_scenes[i])
        pass
  
    reg_error = errors_scenes.cpu().numpy()  # Replace errors_scenes with your actual registration error data

    # Convert metrics to numpy for plotting
    reg_error_np = reg_error.flatten()
    hausdorff_distances_np = hausdorff_distances.cpu().numpy().flatten()
    chamfer_distances_np = chamfer_distances.cpu().numpy().flatten()
    sinkhorn_distances_0_05_np = sinkhorn_distances_0_05.cpu().numpy().flatten()
    sinkhorn_distances_1em9_np = sinkhorn_distances_1em9.cpu().numpy().flatten()

    data_to_save = np.column_stack((
        reg_error_np,
        hausdorff_distances_np,
        chamfer_distances_np,
        sinkhorn_distances_0_05_np,
        sinkhorn_distances_1em9_np,
        gts.flatten()  # Assuming gts is defined as your labels array
    ))
    n_samples = reg_error_np.shape[0]
    # Get the current time and format it as a string
    now = datetime.now()
    time_string = now.strftime("%Y%m%d_%H%M%S")
    directory = f"{args.visualization_folder}/metrics_vs_gt_class"


    # Save to a text file
    np.savetxt("metrics_data.txt", data_to_save, fmt="%.6f", delimiter="\t", 
               header="reg_error\tHausdorff\tChamfer\tSinkhorn_0.05\tSinkhorn_1e-9\tGT")
    file_name = f"metrics_data_{time_string}_{args.model_identifier}_n_samples{n_samples}"
    np.savetxt(f"{directory}/{file_name}.txt", data_to_save, fmt="%.6f", delimiter="\t",
               header="reg_error\tHausdorff\tChamfer\tSinkhorn_0.05\tSinkhorn_1e-9\tGT")

    # Create 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot Hausdorff distance vs. registration error
    axs[0, 0].scatter(reg_error, hausdorff_distances_np, alpha=0.5)
    axs[0, 0].set_title("Hausdorff Distance vs Registration Error")
    axs[0, 0].set_xlabel("Registration Error")
    axs[0, 0].set_ylabel("Hausdorff Distance")

    # Plot Chamfer distance vs. registration error
    axs[0, 1].scatter(reg_error, chamfer_distances_np, alpha=0.5, color='orange')
    axs[0, 1].set_title("Chamfer Distance vs Registration Error")
    axs[0, 1].set_xlabel("Registration Error")
    axs[0, 1].set_ylabel("Chamfer Distance")

    # Plot Sinkhorn distance (blur=0.05) vs. registration error
    axs[1, 0].scatter(reg_error, sinkhorn_distances_0_05_np, alpha=0.5, color='green')
    axs[1, 0].set_title("Sinkhorn Distance (blur=0.05) vs Registration Error")
    axs[1, 0].set_xlabel("Registration Error")
    axs[1, 0].set_ylabel("Sinkhorn Distance (blur=0.05)")

    # Plot Sinkhorn distance (blur=1e-9) vs. registration error
    axs[1, 1].scatter(reg_error, sinkhorn_distances_1em9_np, alpha=0.5, color='purple')
    axs[1, 1].set_title("Sinkhorn Distance (blur=1e-9) vs Registration Error")
    axs[1, 1].set_xlabel("Registration Error")
    axs[1, 1].set_ylabel("Sinkhorn Distance (blur=1e-9)")

    # Adjust layout
    plt.tight_layout()
    #plt.show()
    plt.savefig("metrics_vs_registration_error.png")




    file_name = f"metrics_vs_registration_error_{time_string}_{args.model_identifier}"

    # Save the figure to an .eps file
    fig.savefig(f"{directory}/{file_name}.eps", format="eps", bbox_inches='tight')
    # Save the figure to an .jpg file
    fig.savefig(f"{directory}/{file_name}.jpg", format="jpg", bbox_inches='tight')

    plt.close()  # Close the figure
    
    return None