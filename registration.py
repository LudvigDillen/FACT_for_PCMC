import torch
import hydra
import nuscenes as ns
import numpy as np
import copy
import open3d as o3d
import sys
import open3d.pipelines.registration as treg

from utils.parameters import Params
from utils.nuscenes_handling import read_nuscenes_data
from classifiers.PointTransformers.classify import classify_pairs
from classifiers.PointTransformers.train_cls import load_best_model
from utils.experiment_utils import setup_experiment
import visualization.registration as vis_reg


# From open3d: http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html#Point-to-point-ICP
def draw_registration_result(source, target, transformation, title):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996],
                                      window_name=title)
    return None


def from_tensor_to_pcd(a):
    if a.device.type == 'cuda':
        b = a.cpu().numpy()
    else:
        b = a.numpy()

    pcd = o3d.geometry.PointCloud()
    # Assign the points to the point cloud
    pcd.points = o3d.utility.Vector3dVector(b)
    return pcd


def voxel_grid_to_pcd(voxel_grid):
    # Assuming voxel_grid is your voxelized point cloud of type 'open3d.geometry.VoxelGrid'
    voxel_centers = voxel_grid.get_voxels()
    points = [voxel.grid_index for voxel in voxel_centers]
    # Create a new PointCloud object from voxel centers
    voxelized_point_cloud = o3d.geometry.PointCloud()
    voxelized_point_cloud.points = o3d.utility.Vector3dVector(points)
    return voxelized_point_cloud


def calc_rotation_distance(R1, R2):
    # This is the geodesic rotation distance
    distance_arg = torch.linalg.norm((R1 - R2)/np.sqrt(8), ord='fro')
    limited_angle_distance = torch.arcsin(distance_arg)  # only -pi/2 to pi/2
    return 2*limited_angle_distance  # -pi to pi


def calc_translation_distance(t1, t2):
    dt = t1 - t2
    return torch.linalg.norm(dt, ord=2)


def get_transformation_error(T_est, T_gt):
    # Assumes input is torch, cuda
    # Normalize if this is not already done
    T_est_n = T_est/T_est[3, 3]
    T_gt_n = T_gt/T_gt[3, 3]

    # Gather components
    R_est = T_est_n[:3, :3]
    t_est = T_est_n[:3, 3]
    R_gt = T_gt_n[:3, :3]
    t_gt = T_gt_n[:3, 3]

    # Translation error
    R_error = calc_rotation_distance(R_est, R_gt)
    t_error = calc_translation_distance(t_est, t_gt)
    return R_error, t_error


def get_transformation_errors(poses_est_scene, poses_gt_scene):
    assert len(poses_est_scene) == len(poses_gt_scene)
    n_samples_in_scene = len(poses_est_scene)
    R_errors = torch.zeros((n_samples_in_scene), device='cuda')
    t_errors = torch.zeros((n_samples_in_scene), device='cuda')
    for i, (pose_est, pose_gt) in enumerate(zip(poses_est_scene, poses_gt_scene)):
        R_errors[i], t_errors[i] = get_transformation_error(pose_est, pose_gt)
    return R_errors, t_errors


def register_scene(PC_scene, method="p2l", voxelize=False, verbose=False, plot=False):
    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])
    rel_poses_est = torch.zeros((len(PC_scene), 4, 4), device='cuda')
    for j, pair in enumerate(PC_scene):
        source = from_tensor_to_pcd(pair.PC0.pc)
        target = from_tensor_to_pcd(pair.PC1.pc)

        ### PLOT
        if voxelize:
            source_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(source, voxel_size=0.01)
            target_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(target, voxel_size=0.01)
            source = voxel_grid_to_pcd(source_voxel_grid)
            target = voxel_grid_to_pcd(target_voxel_grid)

        # ### ICP
        # print("Apply point-to-point ICP")
        if method in ["ICP-p2p", "icp-p2p", "p2p"]:
            threshold = 1
            reg_res = treg.registration_icp(
                source, target, threshold, trans_init,
                treg.TransformationEstimationPointToPoint(),
                treg.ICPConvergenceCriteria(max_iteration=1000))
            rel_poses_est[j] = torch.from_numpy(reg_res.transformation.copy()).to('cuda')
        elif method in ["ICP-p2l", "icp-p2l", "p2l"]:
            # Compute normals for the target point cloud
            target.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=175)
            )
            threshold = 0.1
            reg_res = treg.registration_icp(
                source, target, threshold, trans_init,
                treg.TransformationEstimationPointToPlane(),
                treg.ICPConvergenceCriteria(max_iteration=1000))
            rel_poses_est[j] = torch.from_numpy(reg_res.transformation.copy()).to('cuda')
        elif method == "init":
            rel_poses_est[j] = torch.from_numpy(trans_init.copy()).to('cuda')
        else:
            sys.exit("Have no other method")
        
        if verbose and method != "init":
            print(f"Method {method} with res\n {reg_res}")
            print(f"Transformation is: {reg_res.transformation}")

        if plot:
            draw_registration_result(source, target, rel_poses_est[j].cpu().numpy(), title=f"{method}")
    return rel_poses_est


def get_gt_poses(PC_scene):
    rel_poses_gt = torch.zeros((len(PC_scene), 4, 4), device='cuda')
    for j, pair in enumerate(PC_scene):
        rel_poses_gt[j] = torch.matmul(torch.linalg.inv(pair.pose1), pair.pose0)
    return rel_poses_gt


@hydra.main(config_path="classifiers/PointTransformers/config", config_name="cls")
def reg(args):
    ### SETUP
    # Extra settings:
    SEED = True
    n_samples_per_scene = args.n_samples_per_scene
    n_scenes = args.n_scenes
    n_samples = n_samples_per_scene*n_scenes

    # Init Nusc object
    nusc = ns.nuscenes.NuScenes(version=args.dataset, dataroot=args.data_folder, verbose=False)
    R_errors_scenes = torch.zeros((n_scenes, n_samples_per_scene), device='cuda')
    t_errors_scenes = torch.zeros((n_scenes, n_samples_per_scene), device='cuda')

    if SEED:
        np.random.seed(1)
        # Set the seed for PyTorch
        torch.manual_seed(1)
        # If you are using CUDA (PyTorch with GPU support)
        torch.cuda.manual_seed(1)

    params = Params(nusc=nusc, args=args, pointwise=True)
    poses_est_scenes = torch.zeros((n_scenes, n_samples_per_scene, 4, 4), device='cuda')
    poses_gt_scenes = torch.zeros((n_scenes, n_samples_per_scene, 4, 4), device='cuda')
    args, logger = setup_experiment(args)
    pcac_model = load_best_model(args, logger, pretrained=True)

    for i in range(n_scenes):
        PC_scene = read_nuscenes_data(params, n_samples=n_samples_per_scene,
                                      n_scenes=1, scene_counter=i)[0]
        if i % 5 == 0:
            print(f"Scene {i}")
        # Register scene
        poses_gt_scenes[i] = get_gt_poses(PC_scene)
        poses_est_scenes[i] = register_scene(PC_scene, method="p2l")
        # Calculate errors
        R_errors_scenes[i], t_errors_scenes[i] = get_transformation_errors(poses_est_scenes[i], poses_gt_scenes[i])

        # TODO: Extract feature data
        # FPS on scenes 
        # if params.args.fps.do_fps:
        #     farthest_point_sample_PC_scenes(PC_scenes, params.N_fps_points)

        # # Feature extraction.
        # PC_scenes_with_features = feature_extraction(PC_scenes, params)
        # TODO: Go from PC_scenes_with_features to points (see below)
        # points shape: B, N, D (batch_size, N_pts, feature_dim)
        points = None  # This should be the feature data

        # TODO: Do classification
        # classify_pairs(model, ...)

    vis_reg.plot_reg_error_hist(R_errors_scenes.flatten().cpu(), t_errors_scenes.flatten().cpu())
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


@hydra.main(config_path="classifiers/PointTransformers/config", config_name="cls")
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

    params = Params(nusc=nusc, args=args, pointwise=True)
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
            source = from_tensor_to_pcd(pair.PC0.pc)
            target = from_tensor_to_pcd(pair.PC1.pc)
            T_gt = torch.matmul(torch.linalg.inv(pair.pose1), pair.pose0)

            ### PLOT
            # o3d.visualization.draw_geometries([source], window_name="Before voxel")
            source_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(source, voxel_size=0.01)
            target_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(target, voxel_size=0.01)
            source_voxelized = voxel_grid_to_pcd(source_voxel_grid)
            target_voxelized = voxel_grid_to_pcd(target_voxel_grid)
        
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

            R_errors_init[i, j], t_errors_init[i, j] = get_transformation_error(torch.from_numpy(trans_init.copy()).to('cuda'), T_gt)
            R_errors_p2p[i, j], t_errors_p2p[i, j] = get_transformation_error(torch.from_numpy(reg_p2p.transformation.copy()).to('cuda'), T_gt)
            R_errors_p2l[i, j], t_errors_p2l[i, j] = get_transformation_error(torch.from_numpy(reg_p2l.transformation.copy()).to('cuda'), T_gt)
            R_errors_p2p_vox[i, j], t_errors_p2p_vox[i, j] = get_transformation_error(torch.from_numpy(reg_p2p_vox.transformation.copy()).to('cuda'), T_gt)
            R_errors_p2l_vox[i, j], t_errors_p2l_vox[i, j] = get_transformation_error(torch.from_numpy(reg_p2l_vox.transformation.copy()).to('cuda'), T_gt)

    print(f"\navr. Init:     R_error {torch.mean(R_errors_init):.4f}, t_error {torch.mean(t_errors_init):.2f}")
    print(f"avr. p2p:      R_error {torch.mean(R_errors_p2p):.4f}, t_error {torch.mean(t_errors_p2p):.2f}")
    print(f"avr. p2l:      R_error {torch.mean(R_errors_p2l):.4f}, t_error {torch.mean(t_errors_p2l):.2f}")
    print(f"avr. p2p_vox:  R_error {torch.mean(R_errors_p2p_vox):.4f}, t_error {torch.mean(t_errors_p2p_vox):.2f}")
    print(f"avr. p2l_vox:  R_error {torch.mean(R_errors_p2l_vox):.4f}, t_error {torch.mean(t_errors_p2l_vox):.2f}")

    vis_reg.plot_reg_error_over_samples(R_errors_init, R_errors_p2p, R_errors_p2l, n_samples,
                                        t_errors_init, t_errors_p2p, t_errors_p2l)
    vis_reg.plot_reg_error_hists(R_errors_init, R_errors_p2p, R_errors_p2l, n_samples,
                                 t_errors_init, t_errors_p2p, t_errors_p2l)
    return None