# Load data
import torch
import hydra
import nuscenes as ns
import numpy as np
import copy
import open3d as o3d
import open3d.pipelines.registration as treg
import matplotlib.pyplot as plt


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


@hydra.main(config_path="classifiers/PointTransformers/config", config_name="cls")
def reg(args):
    ### SETUP
    # Extra settings:
    SEED = True
    n_samples_per_scene = 1
    n_scenes = 200
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
            # o3d.visualization.draw_geometries([source_voxelized], window_name="After voxel")

            # draw_registration_result(source, target, trans_init, title="Unregistered")
            # draw_registration_result(source, target, T_gt.cpu().numpy(), title="GT registered")

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

            # print(reg_p2p)
            # print("Transformation is:")
            # print(reg_p2p.transformation)
            # draw_registration_result(source, target, reg_p2p.transformation, title="ICP-p2p registered")

            # print("Apply point-to-plane ICP")
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
            # print(reg_p2l)
            # print("Transformation is:")
            # print(reg_p2l.transformation)
            # draw_registration_result(source, target, reg_p2l.transformation, title="ICP-p2l registered")


            # Generalized ICP
            # threshold = 0.1
            # reg_ICP_gen = treg.registration_icp(
            #     source, target, threshold, trans_init,
            #     treg.TransformationEstimationForGeneralizedICP(),
            #     treg.ICPConvergenceCriteria(max_iteration=300))

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

      # # Create a figure and a set of subplots
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns, optional figure size

    # # First subplot for R_errors
    # ax1.plot(x_values, R_errors_init.reshape(n_samples).cpu(), label='R_errors_init')
    # ax1.plot(x_values, R_errors_p2p.reshape(n_samples).cpu(), label='R_errors_p2p')
    # ax1.plot(x_values, R_errors_p2l.reshape(n_samples).cpu(), label='R_errors_p2l')
    # ax1.set_title('R_errors')
    # ax1.set_xlabel('Sample Index')
    # ax1.set_ylabel('Magnitude')
    # ax1.legend()

    # # Second subplot for t_errors (assuming t_errors_init, t_errors_p2p, t_errors_p2l are defined)
    # ax2.plot(x_values, t_errors_init.reshape(n_samples).cpu(), label='t_errors_init')
    # ax2.plot(x_values, t_errors_p2p.reshape(n_samples).cpu(), label='t_errors_p2p')
    # ax2.plot(x_values, t_errors_p2l.reshape(n_samples).cpu(), label='t_errors_p2l')
    # ax2.set_title('t_errors')
    # ax2.set_xlabel('Sample Index')
    # ax2.set_ylabel('Magnitude')
    # ax2.legend()

    # # Adjust the layout
    # plt.tight_layout()

    # # Show the plot
    # plt.show()_values = range(n_samples)  # This will create a range from 0 to n_samples-1


    R_labels = ['identity', 'icp-p2p', 'icp-p2l']
    t_labels = ['zero', 'icp-p2p', 'icp-p2l']

    # Bin boundaries
    r_bin_edges = np.linspace(0, 0.020, num=10).tolist() + [np.inf]
    t_bin_edges = np.linspace(0, 0.9, num=10).tolist() + [np.inf]

    # Create a figure and a set of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns, figure size
    R_bar_width = (r_bin_edges[1] - r_bin_edges[0]) / 6  # Divide by the number of datasets plus some spacing
    t_bar_width = (t_bin_edges[1] - t_bin_edges[0]) / 6  # Divide by the number of datasets plus some spacing

    # Histogram for R_errors
    for i, (data, label) in enumerate(zip([R_errors_init, R_errors_p2p, R_errors_p2l], R_labels)):
        ax1.hist(data.reshape(n_samples).cpu(), bins=r_bin_edges, alpha=0.5,
                 label=label, width=R_bar_width, edgecolor='black')

    # # Adjust the bar positions to be side by side
    for i, rect in enumerate(ax1.patches):
        rect.set_x(rect.get_x() + i // 10 * R_bar_width)

    # Histogram for t_errors
    for i, (data, label) in enumerate(zip([t_errors_init, t_errors_p2p, t_errors_p2l], t_labels)):
        ax2.hist(data.reshape(n_samples).cpu(), bins=t_bin_edges, alpha=0.5,
                 label=label, width=t_bar_width, edgecolor='black')

    # # Adjust the bar positions to be side by side
    for i, rect in enumerate(ax2.patches):
        rect.set_x(rect.get_x() + i // 10 * t_bar_width)

    ax1.set_title('Histogram of rotation errors')
    ax1.set_xlabel('Geodesic distance [m]')
    ax1.set_ylabel('Frequency')
    ax1.legend()

    ax2.set_title('Histogram of translation errors')
    ax2.set_xlabel('Translation error [m]')
    ax2.set_ylabel('Frequency')
    ax2.legend()

    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    plt.show()


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