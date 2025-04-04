import torch
import hydra
import nuscenes as ns
import numpy as np
import copy

from utils.parameters import Params
from utils.nuscenes_handling import read_nuscenes_data, NuscenesHandling
from utils.pointclouds import farthest_point_sample_PC_scenes
from utils.experiment_utils import setup_experiment
from classifiers.PointTransformers.classify import classify_pairs
from classifiers.PointTransformers.train_cls import load_best_model
from visualization.classifications import store_confusion_matrix
from visualization.residuals import analyze_relationship
import visualization.registration as vr
from features.feature_extractor import feature_extraction, create_feature_map
from features.feature_utils import normalize_data_on_condition
import registration.registration_utils as ru
from visualization.presentation import visualize_and_save



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


def align_scene(poses_est_scene, PC_scene, plot=False):
    PC_scene_registered = copy.deepcopy(PC_scene)
    for j, pose_est in enumerate(poses_est_scene):
        pair = PC_scene[j]
        if plot:
            ru.plot_pc_pair(pair, "Unregistered")
        pc1_CS0_reg = ru.align_pair(pair, pose_est)
        PCUnion_registered = copy.deepcopy(pair.PCUnion)
        PCUnion_registered.pc[pair.PC0.N_points:] = pc1_CS0_reg
        PC_scene_registered[j].set_new_PC(pair.PC0, pair.PC1, PCUnion_registered)
        if plot:
            ru.plot_pc_pair(PC_scene_registered[j], "After alignment")
    return PC_scene_registered


@hydra.main(config_path="../classifiers/PointTransformers/config", config_name="cls_registration")
def reg_mpe(args):
    print("If you want to plot point clouds. Consider not applying the HPR operator")
    ### SETUP
    # Init Nusc object
    nusc = ns.nuscenes.NuScenes(version=args.dataset, dataroot=args.data_folder, verbose=False)

    # args.n_scenes = int(scene_ind)
    # Extra settings:
    n_samples_per_scene = args.n_samples_per_scene
    n_classes = args.perturb_settings.n_classes
    DO_REG = True
    CHANGE_OF_POSE_C1 = True
    VISUALIZE_RESULT = True
    REPAIR_METHOD = "gt"  # [gt, p2l]
    REG_METHOD = "p2l"  # [p2l]
    GET_ICP_RESIDUALS = False
    args["get_icp_residuals"] = GET_ICP_RESIDUALS

    # Some settings if I use geotrans. Mode in [kitti, 3dmatch]
    geo_args = ru.get_geo_config(mode="kitti") if REG_METHOD == "geotrans" else None
    #
    test_start_scene = args.n_scenes - 1 if args.one_scene else 0
    n_scenes = 1
    #test_start_scene = 707
    # Set to 638 if we want to only test at test data
    #n_scenes = args.n_scenes - test_start_scene


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
    pcac_model, _, args = load_best_model(args, logger)
    preds = np.zeros((n_scenes, n_samples_per_scene))
    gts = np.zeros((n_scenes, n_samples_per_scene))
    poses_est_scenes = torch.zeros((n_scenes, n_samples_per_scene, 4, 4), device='cuda')
    poses_gt_scenes = torch.zeros((n_scenes, n_samples_per_scene, 4, 4), device='cuda')
    fitness = np.zeros((n_scenes, n_samples_per_scene))
    rmse = np.zeros((n_scenes, n_samples_per_scene))
    for i in range(n_scenes):
        if args.one_scene:
            PC_scene = PCHandler.sample_from_scenes(params, n_samples=n_samples_per_scene, n_scenes=1,
                                                    geo_args=geo_args)[0]
        else:
            PC_scene = read_nuscenes_data(params, mode="test", n_samples=n_samples_per_scene,
                                          n_scenes=1, scene_counter=i+test_start_scene)[0]

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

        if DO_REG:
            PC_scene_reg = align_scene(poses_est_scenes[i], PC_scene, plot=False)
        else:
            PC_scene_reg = copy.deepcopy(PC_scene)
        # EXTRACT FEATURE DATA
        # FPS on scenes
        if params.args.fps.do_fps:
            farthest_point_sample_PC_scenes(PC_scene_reg[None, ...], params.N_fps_points)
        # Feature extraction. [0]: Go from scenes to scene
        PC_scene_with_features = feature_extraction(PC_scene_reg[None, ...], params)[0]
        feature_maps = create_feature_map(PC_scene_with_features[0], params)[None, ...]
        for pair in PC_scene_with_features[1:]:
            feature_maps = np.concatenate((feature_maps,
                                           create_feature_map(pair, params)[None, ...]), axis=0)
        # B, N, D (batch_size, N_pts, feature_dim)
        points = torch.from_numpy(feature_maps).to(params.device).float()
        # Note that since we just normalize so that xyz lie within the unit ball,
        # with the farthest point on the ball => It does not matter which batch size we use.
        points = normalize_data_on_condition(args, points)
        batch_size = 8  # Doesn't matter which bs we use in test, as store lots, lets keep it low.
        points_batches = torch.split(points, batch_size)
        for j, ps in enumerate(points_batches):
            if j == 0:
                p, _ = classify_pairs(pcac_model, ps, regression=args.regression, reg_method=args.reg_method)
            else:
                p  = torch.cat((p, classify_pairs(pcac_model, ps, regression=args.regression, reg_method=args.reg_method)[0]))
        preds[i] = p.cpu().numpy()
    # Ravel data
    if GET_ICP_RESIDUALS:
        preds_flat = preds.ravel()
        fitness_flat = fitness.ravel()
        rmse_flat = rmse.ravel()
        gts_flat = gts.ravel()
        analyze_relationship(preds_flat, fitness_flat, rmse_flat, gts_flat, args)
        for i in range(n_scenes*n_samples_per_scene):
            print(f"preds: {preds_flat[i]}; fitness: {fitness_flat[i]:.3f}; rmse: {rmse_flat[i]:.3f} gts: {gts_flat[i]}")
        # for i in range(n_scenes):
        #     print("preds: ", preds[i], "fitness", fitness[i], "rmse", rmse[i], "gts: ", gts[i])

    # store_confusion_matrix(y_pred=preds.ravel(), y_true=gts.ravel(), N_classes=n_classes,
    #                        logger=logger, args=args, accumulate=True)
    if VISUALIZE_RESULT:
        #region Visualize
        def _align_pc(pc, pose):
            R_new = pose[:3, :3]
            t_new = pose[:3, 3]
            return torch.matmul(pc, R_new.T) + t_new[None, :]
        
        eye = torch.eye(4, dtype=PC_scene[0].PC0.pc.dtype, device=PC_scene[0].device)
        to_CS0 = eye.clone()
        to_CSO_repaired = eye.clone()
        to_CSO_gt = eye.clone()

        for i, (PC_pair, pose, pose_gt) in enumerate(zip(PC_scene, poses_est_scenes[0], poses_gt_scenes[0])):
            if i == 0:
                complete_pc = _align_pc(PC_pair.PC0.pc, to_CS0)
                complete_pc_repaired = _align_pc(PC_pair.PC0.pc, to_CSO_repaired)
                complete_pc_gt = _align_pc(PC_pair.PC0.pc, to_CSO_gt)

            to_CS0 = to_CS0 @ pose.double()
            pc_new = _align_pc(PC_pair.PC1.pc, to_CS0)
            complete_pc = torch.cat((complete_pc, pc_new), dim=0)

            to_CSO_gt = to_CSO_gt @ pose_gt.double()
            pc_new = _align_pc(PC_pair.PC1.pc, to_CSO_gt)
            complete_pc_gt = torch.cat((complete_pc_gt, pc_new), dim=0)
    
            predicted_misalignment = (preds[0, i] == 3 or preds[0, i] == 4)
            if predicted_misalignment:
                if REPAIR_METHOD == "gt":
                    pose = pose_gt
                elif REPAIR_METHOD == "p2l":
                    # We want to find T from CS1 to CS0
                    source = ru.from_tensor_to_pcd(PC_pair.PC1.pc)
                    target = ru.from_tensor_to_pcd(PC_pair.PC0.pc)
                    if i > 0:
                        new_init = poses_est_scenes[0][i-1].cpu().numpy()
                    else:
                        new_init = np.eye(4)
                    pose = ru.register_pair(source, target, method="p2l",
                                            trans_init=new_init)
                else:
                    print(f"Repair method {REPAIR_METHOD} is not implemented. Using p2l.")
            to_CSO_repaired = to_CSO_repaired @ pose.double()
            pc_new = _align_pc(PC_pair.PC1.pc, to_CSO_repaired)

            complete_pc_repaired = torch.cat((complete_pc_repaired, pc_new), dim=0)

        visualize_and_save(complete_pc, title=f"Point clouds 0 to {i}")
        visualize_and_save(complete_pc_repaired, title=f"Point clouds 0 to {i} repaired")
        visualize_and_save(complete_pc_gt, title=f"Point clouds 0 to {i} gt")
        #endregion

    return None
