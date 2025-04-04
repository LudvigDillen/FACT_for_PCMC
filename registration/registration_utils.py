import torch
import numpy as np
import open3d as o3d
import sys
import open3d.pipelines.registration as treg

import os
#ABS_PTH = os.path.abspath('GeoTransformer_202407')
#sys.path.append(ABS_PTH)
# from experiments.kitti.config import make_cfg as make_cfg_kitti
# from experiments.kitti.model import create_model as create_model_kitti
# from experiments.kitti.dataset import test_data_loader as test_data_loader_kitti
# from experiments.threedmatch.config import make_cfg as make_cfg_3dmatch
# from experiments.threedmatch.model import create_model as create_model_3dmatch

# from geotransformer.utils.data import registration_collate_fn_stack_mode
# from geotransformer.utils.torch import to_cuda, release_cuda

from utils.geometrics import change_coordinate_system
import visualization.registration as vr
from visualization.point_clouds import vis_2pcs


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


def get_est_rel_poses(PC_scene, get_icp_residuals=False):
    rel_poses_est = torch.zeros((len(PC_scene), 4, 4), device='cuda')
    for j, pair in enumerate(PC_scene):
        rel_poses_est[j] = pair.est_rel_pose
    if get_icp_residuals:
        fitness = np.zeros((len(PC_scene)))
        inlier_rmse = np.zeros((len(PC_scene)))
        for j, pair in enumerate(PC_scene):
            fitness[j] = pair.fitness
            inlier_rmse[j] = pair.inlier_rmse
        return rel_poses_est, fitness, inlier_rmse
    return rel_poses_est


def plot_pc_pair(pair, title):
    pc0_npoints = pair.PC0.N_points
    source = from_tensor_to_pcd(pair.PCUnion.pc[:pc0_npoints])
    target = from_tensor_to_pcd(pair.PCUnion.pc[pc0_npoints:])
    vr.draw_registration_result(source, target, np.eye(4), title=title)


def align_pair(pair, rel_pose, plot=False):
    if plot:
        vis_2pcs(pair.PC0.pc.cpu(), pair.PC1.pc.cpu(), title="Before registration")
    pose0_CS0 = torch.eye(4).to(pair.device).to(pair.PC1.dtype)
    pc1_CS0_reg = change_coordinate_system(pair.PC1.pc, pose0_CS0, rel_pose.to(pair.PC1.dtype))
    if plot:
        vis_2pcs(pair.PC0.pc.cpu(), pc1_CS0_reg.cpu(), title="After registration")
    return pc1_CS0_reg


def register_pair(source, target, method="p2l", trans_init=None, voxelize=False,
                  gt_pose=None, geo_args=None, get_icp_residuals=False):
    if get_icp_residuals:
        assert method in ["ICP-p2l", "icp-p2l", "p2l"], "Only ICP-p2l is supported for now"
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
        fitness = reg_res.fitness
        inlier_rmse = reg_res.inlier_rmse
        rel_pose = reg_res.transformation.copy()
    elif method == "init":
        rel_pose = trans_init.copy()
    elif method == "geotrans":
        s = 1
        if geo_args["mode"] == "3dmatch":
            s = 0.05
        src_pc, tgt_pc = s*np.asarray(source.points), s*np.asarray(target.points)
        gt_pose[:3, 3] = s*gt_pose[:3, 3]
        output_dict, _ = reg_with_geo(src_pc, tgt_pc, gt_pose.cpu().numpy(), geo_args)
        rel_pose = output_dict["estimated_transform"]
        rel_pose[:3, 3]= rel_pose[:3, 3] * 1/s
    else:
        sys.exit("Have no other method")
    rel_pose = torch.from_numpy(rel_pose).cuda()
    # print(f"T_Est:\n {rel_pose}")
    if get_icp_residuals:
        return rel_pose, fitness, inlier_rmse
    return rel_pose


def get_error_class(error, reg_method="p2l"):
    if reg_method == "p2l":
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
    elif reg_method == "geotransformer":
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
    else:
        sys.exit(f"Registration method {reg_method} not recognized")
    return error_class


def get_gt_classes(errors_scene):
    n_samples = len(errors_scene)
    gt_scene = np.zeros(n_samples)
    for i, error in enumerate(errors_scene):
        gt_scene[i] = get_error_class(error)
    return gt_scene


def load_data(src_points, ref_points, gt_pose):
    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
    }
    data_dict["transform"] = gt_pose.astype(np.float32)
    return data_dict


def reg_with_geo(pc0, pc1, gt_pose, geo_args):
    cfg = geo_args["cfg"]

    data_dict = load_data(pc0, pc1, gt_pose)
    data_dict = registration_collate_fn_stack_mode(
        [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius, geo_args["neighbor_limits"]
    )

    # prepare model
    if geo_args["mode"] == "kitti":
        model = create_model_kitti(cfg).cuda()
    elif geo_args["mode"] == "3dmatch":
        model = create_model_3dmatch(cfg).cuda()
    else:
        print(f"Mode {geo_args['mode']} not recognized. Using kitti instead.")
        model = create_model_kitti(cfg).cuda()

    state_dict = torch.load(str(ABS_PTH) + geo_args["weight_pth"])
    model.load_state_dict(state_dict["model"])

    # prediction
    data_dict = to_cuda(data_dict)
    output_dict = model(data_dict)
    data_dict = release_cuda(data_dict)
    output_dict = release_cuda(output_dict)
    return output_dict, data_dict


def get_geo_config(mode):
    if mode == "kitti":
        cfg = make_cfg_kitti()
        _, neighbor_limits = test_data_loader_kitti(cfg)
    elif mode == "3dmatch":
        cfg = make_cfg_3dmatch()
        neighbor_limits = [38, 36, 36, 38]  # default setting in 3DMatch
    else:
        sys.exit(f"Mode {mode} does not exist. Choose from kitti and 3dmatch")

    geo_args = {
        "mode": mode,
        "cfg": cfg,
        "neighbor_limits": neighbor_limits,
        "weight_pth": f"/weights/geotransformer-{mode}.pth.tar"
    }
    return geo_args
