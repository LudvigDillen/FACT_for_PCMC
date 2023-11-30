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
    R_error = 2*torch.arcsin(torch.linalg.norm(R_est - R_gt) / torch.sqrt(8))  # TODO: double check this formula
    t_error = torch.linalg.norm(t_est - t_gt)
    return R_error, t_error


@hydra.main(config_path="classifiers/PointTransformers/config", config_name="cls")
def reg(args):
    ### SETUP
    # Extra settings:
    SEED = True
    threshold = 0.2

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
    for i in np.arange(0, 1):
        PC_scenes = read_nuscenes_data(
            params,
            n_samples=1,
            n_scenes=1,
            scene_counter=i,
        )

        for PC_scene in PC_scenes:
            for pair in PC_scene:


                print(f"Class category {pair.class_category}")
                trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0]])
                source = from_tensor_to_pcd(pair.PC0.pc)
                target = from_tensor_to_pcd(pair.PC1.pc)
                T_gt = torch.matmul(torch.linalg.inv(pair.pose1), pair.pose0)
                T_gt_np = T_gt.cpu().numpy()
                ### PLOT
                
                draw_registration_result(source, target, trans_init, title="Unregistered")
                draw_registration_result(source, target, T_gt_np, title="GT registered")

                # ### ICP
                print("Apply point-to-point ICP")
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    source, target, threshold, trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

                print(reg_p2p)
                print("Transformation is:")
                print(reg_p2p.transformation)
                draw_registration_result(source, target, reg_p2p.transformation, title="ICP-p2p registered")


                # print("Apply point-to-plane ICP")
                # # Compute normals for the target point cloud
                # target.estimate_normals(
                #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
                # )

                # reg_p2l = o3d.pipelines.registration.registration_icp(
                #     source, target, threshold, trans_init,
                #     o3d.pipelines.registration.TransformationEstimationPointToPlane())
                # print(reg_p2l)
                # print("Transformation is:")
                # print(reg_p2l.transformation)
                # draw_registration_result(source, target, reg_p2l.transformation, title="ICP-p2l registered")


    return None
