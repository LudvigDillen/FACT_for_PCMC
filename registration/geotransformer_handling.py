import subprocess
import os
import hydra
import torch
import numpy as np
import copy

from utils.parameters import Params
from utils.pointclouds import farthest_point_sample_PC_scenes
from utils.experiment_utils import setup_experiment
from classifiers.PointTransformers.classify import classify_pairs
from classifiers.PointTransformers.train_cls import load_best_model
from features.feature_extractor import feature_extraction, create_feature_map
from features.feature_utils import normalize_data_on_condition

from utils.pointclouds import PC, PCPair
from utils.visibility import keep_covisible_points
from utils.geometrics import change_coordinate_system
from visualization.point_clouds import vis_2pcs


#@hydra.main(config_path="../classifiers/PointTransformers/config", config_name="cls_registration")
@hydra.main(config_path="../classifiers/PointTransformers/config", config_name="cls_adaptive")
def geotransformer_with_fact(args):
    # Set the environment variable
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Define the path to the test script and the argument
    script_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '../', 
        'GeoTransformer_202407', 
        'experiments', 
        'geotransformer.kitti.stage5.gse.k3.max.oacl.stage2.sinkhorn', 
        'test.py'
    ))

    # Define the relative path to the snapshot file
    snapshot_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '../',
        'GeoTransformer_202407',
        'weights', 
        'geotransformer-kitti.pth.tar'
    ))

    # Call the test script with the argument
    subprocess.run(['python', script_path, f'--snapshot={snapshot_path}', f'--fact_args={args}'])


def setup_params_for_fact(args):
    DO_REG = True
    CHANGE_OF_POSE_C1 = True
    REG_METHOD = "..."  # [p2l, geotrans] TODO: I register elsewhere so what to set this to ...

    # TODO: I should not create params many times, just once ...

    params = Params(nusc=None, args=args, pointwise=True)
    args, logger = setup_experiment(args, do_reg=DO_REG, change_of_pose_C1=CHANGE_OF_POSE_C1,
                                    reg_method=REG_METHOD)

    params.set_which_features_to_use(args.features_to_create)

    #region Handle some path complications
    original_cwd = os.getcwd()
    try:
        # Change to the parent directory (e.g., back up two levels)
        os.chdir("../../../")
        model_fact, _, args = load_best_model(args, logger)
    finally:
        # Revert back to the original working directory
        os.chdir(original_cwd)
    #endregion
    return params, args, model_fact


def fact_prediction(args, params, model_fact, src_points, ref_points, est_trans, gt_trans):
    # TODO: If things does not work, I could double-check that these points are on the same
    # scale as the nuscenes points and that the density is somewhat similar.
    # TODO: If thing does not work, I could also retrain FACT on KITTI.

    PC_scene = to_PC_format(src_points, ref_points, params, est_trans, gt_trans, geo_args=None)
    PC_scene_reg = copy.deepcopy(PC_scene)  # TODO: Perhaps not necessary

    # EXTRACT FEATURE DATA
    # FPS on scenes
    if params.args.fps.do_fps:
        farthest_point_sample_PC_scenes(PC_scene_reg[None, ...], params.N_fps_points)
    # Feature extraction. [0]: Go from scenes to scene
    PC_scene_with_features = feature_extraction(PC_scene_reg[None, ...], params)[0]

    feature_maps = create_feature_map(PC_scene_with_features[0], params)[None, ...]
    # B, N, D (batch_size, N_pts, feature_dim)
    points = torch.from_numpy(feature_maps).to(params.device).float()
    del PC_scene, PC_scene_reg, PC_scene_with_features, feature_maps

    # Note that since we just normalize so that xyz lie within the unit ball,
    # with the farthest point on the ball => It does not matter which batch size we use.
    points = normalize_data_on_condition(args, points)

    # Get the absolute path from the new working directory including extra path specification
    fact_error_class = classify_pairs(model_fact, points)
    # Free up GPU memory used by points
    del points
    torch.cuda.empty_cache()  # Free up GPU memory again

    return fact_error_class


def to_PC_format(src_pc, ref_pc, params, est_trans, gt_trans, geo_args=None):
    src_pt_dists = torch.norm(src_pc, p=2, dim=1)
    ref_pt_dists = torch.norm(ref_pc, p=2, dim=1)

    # Load in first point cloud
    PC1 = PC(src_pc, src_pt_dists, label=1, device=params.device)
    # Load in second point cloud
    PC0 = PC(ref_pc, ref_pt_dists, label=0, device=params.device)

    # Set point cloud pair and their union, and perform possible perturbation
    currentPCPair = PCPair(
        PC0, PC1, device=params.device, PCHandler=None,
        perturb_settings=params.args.perturb_settings, change_of_pose_C1=params.args.change_of_pose_C1,
        perturbation_method=params.args.perturb_settings.perturbation_method,
        reg_method = params.args.reg_method, geo_args=geo_args, est_pc1_to_pc0=est_trans,
        gt_pc1_to_pc0=gt_trans, geotrans_dataset=params.args.general_dataset
    )


    #updated_pc1 = change_coordinate_system(PC1.pc, currentPCPair.pose0, currentPCPair.pose1)
    #v'is_2pcs(PC0.pc.cpu(), updated_pc1.cpu())
    if params.args.preprocessing.apply_hpr_operator:
        # Calculate the co-visible points
        PC0_cov, PC1_cov, PCUnion_cov = keep_covisible_points(
            PC0, PC1, currentPCPair.PCUnion, currentPCPair.pose0, currentPCPair.pose1,
            compute_weights=params.use_c, hpr_radius=params.args.covisibility.hpr_radius,
            gamma=params.args.covisibility.gamma,
            inversion_kernel=params.args.covisibility.inversion_kernel,
            batch_size=params.batch_size_feature_extraction,
        )
        currentPCPair.set_new_PC(PC0_cov, PC1_cov, PCUnion_cov)

    if params.args.general_dataset == "kitti":
        # Remove points close to each lidar's sensor
        est_location_pose1 = currentPCPair.est_pc1_to_pc0[:3, 3]
        mask0 = torch.norm(PC0_cov.pc - est_location_pose1, dim=1) > 6
        mask1 = torch.norm(currentPCPair.pc1_CS0, dim=1) > 6
        # pc0_removed = PC0_cov.pc[mask0]
        # pc1_removed = currentPCPair.pc1_CS0[mask1]
        # vis_2pcs(pc0_removed.cpu(), pc1_removed.cpu(), title="est aligned (co-visible points) with removed close neighborhoods")
        inds_to_keep = torch.hstack((mask0, mask1))
        device = PC0.device
        PC_union_filt = PC(currentPCPair.PCUnion.pc[inds_to_keep],
                        currentPCPair.PCUnion.distances_to_origin[inds_to_keep], label=2,
                        device=device)

        PC0_filt = PC(currentPCPair.PC0.pc[mask0], currentPCPair.PC0.distances_to_origin[mask0],
                    label=0, device=device)
        PC1_filt = PC(currentPCPair.PC1.pc[mask1], currentPCPair.PC1.distances_to_origin[mask1],
                    label=1, device=device)
        if params.use_c:
            PC_union_filt.weight_c = currentPCPair.PCUnion.weight_c[inds_to_keep]
        currentPCPair.set_new_PC(PC0_filt, PC1_filt, PC_union_filt)


    PC_scene = np.array([currentPCPair])  # this is just the format that the code expects

    #updated_cov_pc1 = change_coordinate_system(PC1_cov.pc, currentPCPair.pose0, currentPCPair.pose1)
    #vis_2pcs(PC0_cov.pc.cpu(), updated_cov_pc1.cpu(), title="Gt aligned co-visible points")

    #far_points0 = PC0_cov.pc[torch.norm(PC0_cov.pc, dim=1) > 10]
    #far_points1 = updated_cov_pc1[torch.norm(updated_cov_pc1, dim=1) > 10]

    #vis_2pcs(PC0_cov.pc.cpu(), updated_cov_pc1.cpu(), title="Gt aligned co-visible points (colored)",
    #         cmap=PCUnion_cov.weight_c.cpu().numpy())
    return PC_scene
