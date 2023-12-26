import torch
import hydra
import nuscenes as ns
import numpy as np
import copy
import open3d as o3d
import open3d.pipelines.registration as treg

from utils.parameters import Params
from utils.nuscenes_handling import read_nuscenes_data
from utils.pointclouds import farthest_point_sample_PC_scenes
from utils.experiment_utils import setup_experiment
from classifiers.PointTransformers.classify import classify_pairs
from classifiers.PointTransformers.train_cls import load_best_model
import visualization.registration as vr
from features.feature_extractor import feature_extraction, create_feature_map
from features.feature_utils import normalize_data_on_condition
import registration.registration_utils as ru


def estimate_transformation_scene(PC_scene, method="p2l", voxelize=False, verbose=False, plot=False):
    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])
    rel_poses_est = torch.zeros((len(PC_scene), 4, 4), device='cuda')
    for j, pair in enumerate(PC_scene):
        # We want to find T from CS1 to CS0
        source = ru.from_tensor_to_pcd(pair.PC1.pc)
        target = ru.from_tensor_to_pcd(pair.PC0.pc)

        rel_poses_est[j], reg_res = ru.register_pair(source, target, method=method,
                                                     trans_init=trans_init, voxelize=voxelize)
        if verbose and method != "init":
            print(f"Method {method} with res\n {reg_res}")
            print(f"Transformation is: {reg_res.transformation}")

        if plot:
            vr.draw_registration_result(source, target, rel_poses_est[j].cpu().numpy(),
                                        title=f"{method}")
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


@hydra.main(config_path="../classifiers/PointTransformers/config", config_name="cls")
def reg(args):
    ### SETUP
    # Extra settings:
    n_samples_per_scene = args.n_samples_per_scene
    n_scenes = args.n_scenes
    DO_REG = True

    # Init Nusc object
    nusc = ns.nuscenes.NuScenes(version=args.dataset, dataroot=args.data_folder, verbose=False)
    R_errors_scenes = torch.zeros((n_scenes, n_samples_per_scene), device='cuda')
    t_errors_scenes = torch.zeros((n_scenes, n_samples_per_scene), device='cuda')

    args, logger = setup_experiment(args, do_reg=DO_REG)
    params = Params(nusc=nusc, args=args, pointwise=True)
    params.set_which_features_to_use(args.features_to_create)
    pcac_model, _, args = load_best_model(args, logger)

    poses_est_scenes = torch.zeros((n_scenes, n_samples_per_scene, 4, 4), device='cuda')
    poses_gt_scenes = torch.zeros((n_scenes, n_samples_per_scene, 4, 4), device='cuda')
    # for gamma in np.arange(0.01, 0.10, 0.01):
    #     gd = ru.rot_offset_to_geodesic_distance(gamma)
    for i in range(n_scenes):
        PC_scene = read_nuscenes_data(params, n_samples=n_samples_per_scene,
                                      n_scenes=1, scene_counter=i)[0]
        if i % 5 == 0:
            print(f"Scene {i}")
        # Register scene
        # TODO: GT pose is not correct since we do not account for a potential perturbation.ö
        poses_gt_scenes[i] = ru.get_gt_poses(PC_scene)
        poses_est_scenes[i] = estimate_transformation_scene(PC_scene, method="p2l")
        # Convert point clouds to the same coordinates system with our estimated relative pose
        # poses_est_scenes[i] = torch.eye(4, device='cuda')
        # Calculate errors
        R_errors_scenes[i], t_errors_scenes[i] = ru.get_transformation_errors(poses_est_scenes[i],
                                                                              poses_gt_scenes[i])
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
        preds = classify_pairs(pcac_model, points)
        # TODO: Compare prediction with pose error (R_errors_scenes[i], t_errors_scenes[i])
        for j, (R_error, t_error, pred) in enumerate(zip(R_errors_scenes[i], t_errors_scenes[i], preds)):
            # gt = 1 if R_error > 0.03 or t_error > 0.3 else 0
            gt = np.zeros(2)
            for k in range(10):
                if R_error >= 0.01*k:
                    gt[0] = k
                if t_error >= 0.1*k:
                    gt[1] = k
            if DO_REG:
                # plot_pc_pair(PC_scene_reg[j], f"Pred. class: {pred}, gt pose class (R, t): {gt} (Error {R_error:.4f}, {t_error:.2f})")
                print(f"Pred. class: {pred}, gt pose class (R, t): {gt} (Error {R_error:.4f}, {t_error:.2f})")
            else:  # For prediction of manual perturbed data         # 
                # plot_pc_pair(PC_scene_reg[j], f"Pred. class: {pred}, misalignment class: {PC_scene[j].class_category}")
                print(f"Pred. class: {pred}, misalignment class: {PC_scene[j].class_category}")

    vr.plot_reg_error_hist(R_errors_scenes.flatten().cpu(), t_errors_scenes.flatten().cpu())
    return None
# TODO: Voxelize data before ICP (based on https://ispc-group.github.io/pages/files/HRegNet/HRegNet.pdf)
# 1. Voxels of width 0.3m
# 2. Select 8192 pts randomly. I think they mean that we should take it randomly over the voxels?

# TODO: Test sample so that we have equally many pts for both point clouds.

# TODO: QUESTION: Determine what is an threshold for aligned and misaligned point clouds
# Maybe I should have three classes (aligned, remove_class, misaligned). I perhaps want the classes
# aligned and misaligned to be seperated with a margin and not be continuously together.

# TODO: QUESTION: Can we map the rotation error (geodesic error) to an actual class that I had previously.
# Is the error in radians?


@hydra.main(config_path="../classifiers/PointTransformers/config", config_name="cls")
def compare_reg_methods(args):
    ### SETUP
    # Extra settings:
    SEED = True
    n_samples_per_scene = args.n_samples_per_scene
    n_scenes = args.n_scenes
    n_samples = n_samples_per_scene*n_scenes

    # Init Nusc object
    nusc = ns.nuscenes.NuScenes(
        version=args.dataset, dataroot=args.data_folder, verbose=False
    )

    R_errors_init = torch.zeros((n_scenes, n_samples_per_scene), device='cuda')
    t_errors_init = torch.zeros((n_scenes, n_samples_per_scene), device='cuda')
    R_errors_p2p = torch.zeros((n_scenes, n_samples_per_scene), device='cuda')
    t_errors_p2p = torch.zeros((n_scenes, n_samples_per_scene), device='cuda')
    R_errors_p2l = torch.zeros((n_scenes, n_samples_per_scene), device='cuda')
    t_errors_p2l = torch.zeros((n_scenes, n_samples_per_scene), device='cuda')
    R_errors_p2p_vox = torch.zeros((n_scenes, n_samples_per_scene), device='cuda')
    t_errors_p2p_vox = torch.zeros((n_scenes, n_samples_per_scene), device='cuda')
    R_errors_p2l_vox = torch.zeros((n_scenes, n_samples_per_scene), device='cuda')
    t_errors_p2l_vox = torch.zeros((n_scenes, n_samples_per_scene), device='cuda')

    if SEED:
        np.random.seed(1)
        # Set the seed for PyTorch
        torch.manual_seed(1)
        # If you are using CUDA (PyTorch with GPU support)
        torch.cuda.manual_seed(1)

    args, logger = setup_experiment(args, do_reg=True)
    params = Params(nusc=nusc, args=args, pointwise=True)
    params.set_which_features_to_use(args.features_to_create)

    PC_scenes = read_nuscenes_data(
        params,
        n_samples=n_samples,
        n_scenes=n_scenes,
        scene_counter=0,
    )

    for i, PC_scene in enumerate(PC_scenes):
        if i % 50 == 0: 
            print(i)
        for j, pair in enumerate(PC_scene):
            trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0]])
            source = ru.from_tensor_to_pcd(pair.PC1.pc)
            target = ru.from_tensor_to_pcd(pair.PC0.pc)
            T_gt = torch.matmul(torch.linalg.inv(pair.pose0), pair.pose1)

            ### PLOT
            # o3d.visualization.draw_geometries([source], window_name="Before voxel")
            source_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(source, voxel_size=0.01)
            target_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(target, voxel_size=0.01)
            source_voxelized = ru.voxel_grid_to_pcd(source_voxel_grid)
            target_voxelized = ru.voxel_grid_to_pcd(target_voxel_grid)
        
            # ### ICP
            # print("Apply point-to-point ICP")
            threshold = 1
            reg_p2p = treg.registration_icp(
                source, target, threshold, trans_init,
                treg.TransformationEstimationPointToPoint(),
                treg.ICPConvergenceCriteria(max_iteration=1000))
            reg_p2p_vox = treg.registration_icp(
                source_voxelized, target_voxelized, threshold, trans_init,
                treg.TransformationEstimationPointToPoint(),
                treg.ICPConvergenceCriteria(max_iteration=1000))

            # Compute normals for the target point cloud
            target.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=175)
            )
            target_voxelized.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=175)
            )
            threshold = 0.1
            reg_p2l = treg.registration_icp(
                source, target, threshold, trans_init,
                treg.TransformationEstimationPointToPlane(),
                treg.ICPConvergenceCriteria(max_iteration=1000))
            reg_p2l_vox = treg.registration_icp(
                source_voxelized, target_voxelized, threshold, trans_init,
                treg.TransformationEstimationPointToPlane(),
                treg.ICPConvergenceCriteria(max_iteration=1000))

            R_errors_init[i, j], t_errors_init[i, j] = ru.get_transformation_error(torch.from_numpy(trans_init.copy()).to('cuda'), T_gt)
            R_errors_p2p[i, j], t_errors_p2p[i, j] = ru.get_transformation_error(torch.from_numpy(reg_p2p.transformation.copy()).to('cuda'), T_gt)
            R_errors_p2l[i, j], t_errors_p2l[i, j] = ru.get_transformation_error(torch.from_numpy(reg_p2l.transformation.copy()).to('cuda'), T_gt)
            R_errors_p2p_vox[i, j], t_errors_p2p_vox[i, j] = ru.get_transformation_error(torch.from_numpy(reg_p2p_vox.transformation.copy()).to('cuda'), T_gt)
            R_errors_p2l_vox[i, j], t_errors_p2l_vox[i, j] = ru.get_transformation_error(torch.from_numpy(reg_p2l_vox.transformation.copy()).to('cuda'), T_gt)

    print(f"\navr. Init:     R_error {torch.mean(R_errors_init):.4f}, t_error {torch.mean(t_errors_init):.2f}")
    print(f"avr. p2p:      R_error {torch.mean(R_errors_p2p):.4f}, t_error {torch.mean(t_errors_p2p):.2f}")
    print(f"avr. p2l:      R_error {torch.mean(R_errors_p2l):.4f}, t_error {torch.mean(t_errors_p2l):.2f}")
    print(f"avr. p2p_vox:  R_error {torch.mean(R_errors_p2p_vox):.4f}, t_error {torch.mean(t_errors_p2p_vox):.2f}")
    print(f"avr. p2l_vox:  R_error {torch.mean(R_errors_p2l_vox):.4f}, t_error {torch.mean(t_errors_p2l_vox):.2f}")

    vr.plot_reg_error_over_samples(R_errors_init, R_errors_p2p, R_errors_p2l, n_samples,
                                        t_errors_init, t_errors_p2p, t_errors_p2l)
    vr.plot_reg_error_hists(R_errors_init, R_errors_p2p, R_errors_p2l, n_samples,
                                 t_errors_init, t_errors_p2p, t_errors_p2l)
    return None