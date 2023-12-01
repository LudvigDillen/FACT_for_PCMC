# Load data
import torch
import hydra
import nuscenes as ns
import numpy as np
import copy
import open3d as o3d

from utils.parameters import Params
from utils.nuscenes_handling import read_nuscenes_data


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


@hydra.main(config_path="classifiers/PointTransformers/config", config_name="cls")
def reg(args):
    ### SETUP
    # Extra settings:
    SEED = True
    threshold = 0.02

    # Init Nusc object
    nusc = ns.nuscenes.NuScenes(
        version=args.dataset, dataroot=args.data_folder, verbose=False
    )

    if SEED:
        np.random.seed(1)
        # Set the seed for PyTorch
        torch.manual_seed(1)
        # If you are using CUDA (PyTorch with GPU support)
        torch.cuda.manual_seed(1)

    params = Params(nusc=nusc, args=args, pointwise=True)
    for i in np.arange(0, 10):
        PC_scenes = read_nuscenes_data(
            params,
            n_samples=2,
            n_scenes=1,
            scene_counter=i,
        )

        for PC_scene in PC_scenes:
            for pair in PC_scene:


                print(f"\nClass category {pair.class_category}")
                trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0]])
                source = from_tensor_to_pcd(pair.PC0.pc)
                target = from_tensor_to_pcd(pair.PC1.pc)
                T_gt = torch.matmul(torch.linalg.inv(pair.pose1), pair.pose0)
                T_gt_np = T_gt.cpu().numpy()
                ### PLOT
                
                # draw_registration_result(source, target, trans_init, title="Unregistered")
                # draw_registration_result(source, target, T_gt_np, title="GT registered")

                # ### ICP
                # print("Apply point-to-point ICP")
                threshold = 0.1
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    source, target, threshold, trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

                # print(reg_p2p)
                # print("Transformation is:")
                # print(reg_p2p.transformation)
                # draw_registration_result(source, target, reg_p2p.transformation, title="ICP-p2p registered")

                # print("Apply point-to-plane ICP")
                # Compute normals for the target point cloud
                target.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=300)
                )
                threshold = 0.01
                reg_p2l = o3d.pipelines.registration.registration_icp(
                    source, target, threshold, trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
                # print(reg_p2l)
                # print("Transformation is:")
                # print(reg_p2l.transformation)
                # draw_registration_result(source, target, reg_p2l.transformation, title="ICP-p2l registered")

                R_error_init, t_error_init = get_transformation_error(torch.from_numpy(trans_init.copy()).to('cuda'), T_gt)
                R_error_p2p, t_error_p2p = get_transformation_error(torch.from_numpy(reg_p2p.transformation.copy()).to('cuda'), T_gt)
                R_error_p2l, t_error_p2l = get_transformation_error(torch.from_numpy(reg_p2l.transformation.copy()).to('cuda'), T_gt)
                print(f"Init: R_error {R_error_init:.4f}, t_error {t_error_init:.2f}")
                print(f"p2p:  R_error {R_error_p2p:.4f}, t_error {t_error_p2p:.2f}")
                print(f"p2l:  R_error {R_error_p2l:.4f}, t_error {t_error_p2l:.2f}")

    return None
