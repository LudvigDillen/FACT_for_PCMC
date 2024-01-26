import torch
import numpy as np
import open3d as o3d
import sys
import open3d.pipelines.registration as treg
from utils.pointnet_util import farthest_point_sample

from utils.geometrics import change_coordinate_system
import visualization.registration as vr
import registration.overlap_predator as op


def rot_offset_to_geodesic_distance(gamma):
    a, b = 1 - np.cos(gamma), np.sin(gamma)
    m = np.array([[a,  b, 0],
                  [-b, a, 0],
                  [0,  0, 0]])
    gd = 2*np.arcsin(1/np.sqrt(8)*np.linalg.norm(m, ord='fro'))
    assert np.allclose(gd, gamma)
    return gd


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
    limited_angle_distance = torch.arcsin(distance_arg)  # only 0 to pi/2 (since dist >= 0)
    return 2*limited_angle_distance  # 0 to pi


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


def get_mean_point_error(PC_scene, poses_est_scene, poses_gt_scene):
    assert len(poses_est_scene) == len(poses_gt_scene)
    n_samples_in_scene = len(poses_est_scene)
    errors = torch.zeros((n_samples_in_scene), device='cuda')
    for i, (pose_est, pose_gt) in enumerate(zip(poses_est_scene, poses_gt_scene)):
        est_pc = align_pair(PC_scene[i], pose_est)
        gt_pc = align_pair(PC_scene[i], pose_gt)
        errors[i] = torch.linalg.norm(est_pc - gt_pc, dim=1).mean()
    return errors


def get_gt_poses(PC_scene):
    rel_poses_gt = torch.zeros((len(PC_scene), 4, 4), device='cuda')
    for j, pair in enumerate(PC_scene):
        rel_poses_gt[j] = torch.matmul(torch.linalg.inv(pair.pose0), pair.pose1)
    return rel_poses_gt


def plot_pc_pair(pair, title):
    pc0_npoints = pair.PC0.N_points
    source = from_tensor_to_pcd(pair.PCUnion.pc[:pc0_npoints])
    target = from_tensor_to_pcd(pair.PCUnion.pc[pc0_npoints:])
    vr.draw_registration_result(source, target, np.eye(4), title=title)


def align_pair(pair, rel_pose):
    pose0_CS0 = torch.eye(4).to(pair.device).to(torch.float64)
    pc1_CS0_reg = change_coordinate_system(pair.PC1.pc, pose0_CS0, rel_pose.to(torch.float64))
    return pc1_CS0_reg


def register_pair(source, target, method="p2l", trans_init=None, voxelize=False):
    if trans_init is None:
        trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0]])

    if voxelize:
        source_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(source,
                                                                            voxel_size=0.01)
        target_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(target,
                                                                            voxel_size=0.01)
        source = voxel_grid_to_pcd(source_voxel_grid)
        target = voxel_grid_to_pcd(target_voxel_grid)

    # ### ICP
    if method in ["ICP-p2p", "icp-p2p", "p2p"]:
        threshold = 1
        reg_res = treg.registration_icp(
            source, target, threshold, trans_init,
            treg.TransformationEstimationPointToPoint(),
            treg.ICPConvergenceCriteria(max_iteration=1000))
        rel_pose = reg_res.transformation.copy()
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
        rel_pose = reg_res.transformation.copy()
    elif method == "init":
        rel_pose = trans_init.copy()
    elif method == "predator":
        config = op.setup_op_registration()
        neighborhood_limits = None
        s = 0.1  # scale_factor
        src_pc, tgt_pc = s*np.asarray(source.points), s*np.asarray(target.points)
        dir = '/home/lu2277di/Projects/FACT/data/point_clouds/'
        src_pth = dir + 'pc0.pth'
        tgt_pth = dir + 'pc1.pth'
        torch.save(tgt_pc.squeeze(), src_pth)
        torch.save(src_pc.squeeze(), tgt_pth)
        demo_loader = op.get_pair_loader(config, neighborhood_limits, src_pth, tgt_pth)
        rel_pose = op.main(config, demo_loader).copy()
    else:
        sys.exit("Have no other method")
    rel_pose = torch.from_numpy(rel_pose).cuda()
    # print(f"T_Est:\n {rel_pose}")
    return rel_pose


def get_error_class(error):
    if error < 0.03:
        error_class = 0
    elif error < 0.10:
        error_class = 1
    elif error < 0.25:
        error_class = 2
    elif error < 0.5:
        error_class = 3
    else:
        error_class = 4
    return error_class


def get_gt_classes(errors_scene):
    n_samples = len(errors_scene)
    gt_scene = np.zeros(n_samples)
    for i, error in enumerate(errors_scene):
        gt_scene[i] = get_error_class(error)
    return gt_scene
