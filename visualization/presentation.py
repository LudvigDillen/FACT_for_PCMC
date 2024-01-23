import torch
import hydra
import omegaconf
import nuscenes as ns
from utils.other import start_debug
from utils.nuscenes_handling import read_nuscenes_data
from utils.experiment_utils import setup_experiment
from utils.parameters import Params
import open3d as o3d
import time


def visualize_and_save(point_cloud, title="Point Cloud"):
    pcd = o3d.geometry.PointCloud()
    points_np = point_cloud.detach().cpu().numpy()

    # Translate point cloud to center it around the origin
    # centroid = points_np.mean(axis=0)
    # points_centered = points_np - centroid
    mask = points_np[:, 2] > -1
    pcd.points = o3d.utility.Vector3dVector(points_np[mask])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, visible=False)
    o3d.visualization.draw_geometries([pcd])


def setup_presentation_mode(args):
    for feat in args.features_to_create:
        args.features_to_create[feat] = False
    print("Soon")
    args.perturb_settings.n_classes = 1
    args.fps.do_fps = False
    args.downsample_factor = 1
    return args


@hydra.main(config_path="../classifiers/PointTransformers/config", config_name="cls")
def main(args):
    t0 = time.time()
    if args.debug:
        start_debug()
    omegaconf.OmegaConf.set_struct(args, False)
    # Init Nusc object
    nusc = ns.nuscenes.NuScenes(
        version=args.dataset, dataroot=args.data_folder, verbose=False
    )

    args, logger = setup_experiment(args, do_reg=True)
    args = setup_presentation_mode(args)

    params = Params(nusc=nusc, args=args, pointwise=True)
    # Set which features to use
    params.set_which_features_to_use(args.features_to_create)
    n_samples = 10
    for j in range(1):
        # 382
        PC_scenes = read_nuscenes_data(
            params, n_samples=n_samples, n_scenes=1, lidar_token=None, scene_counter=j, mode="test"
        )
    
        complete_point_cloud = PC_scenes[0][0].PC0.pc.clone()
        pose_CS0 = PC_scenes[0][0].pose0.clone()
        WCS_to_CS0 = torch.inverse(pose_CS0)

        image_paths = []

    #     # No errors:
    #     for i, PC_pair in enumerate(PC_scenes[0]):
    #         if i == 380:
    #             image_path = f"frame_{i}.png"
    #             image_paths.append(image_path)

    #             visualize_and_save(
    #                 complete_point_cloud,
    #                 title=f"Point clouds 0 to {i}",
    #                 save_path=image_path,
    #             )

    #         pose_new_CS0 = torch.matmul(WCS_to_CS0, PC_pair.pose1)
    #         R_new_CS0 = pose_new_CS0[:3, :3]
    #         t_new_CS0 = pose_new_CS0[:3, 3]
    #         pc_CSnew = PC_pair.PC1.pc
    #         pc_new_CS0 = torch.matmul(pc_CSnew, R_new_CS0.T) + t_new_CS0[None, :]

    #         complete_point_cloud = torch.cat((complete_point_cloud, pc_new_CS0), dim=0)
    t1 = time.time()
    print(f"Time for point cloud extraction: {t1 -t0:.3f}")


    for i, PC_pair in enumerate(PC_scenes[0]):
        if i == n_samples-2:
            image_path = f"frame_{i}.png"
            image_paths.append(image_path)
            t2 = time.time()
            print(f"Time for visualization: {t2 -t1:.3f}")

            visualize_and_save(
                complete_point_cloud,
                title=f"Point clouds 0 to {i}",
                save_path=image_path,
            )

        pose_new_CS0 = torch.matmul(WCS_to_CS0, PC_pair.pose1)
        R_new_CS0 = pose_new_CS0[:3, :3]
        t_new_CS0 = pose_new_CS0[:3, 3]

        # if i == 100 or i == 200 or i == 300:
        #     R_new_CS0 = torch.rand((3, 3), dtype=R_new_CS0.dtype)
        #     t_new_CS0 = torch.rand((3), dtype=t_new_CS0.dtype)

        pc_CSnew = PC_pair.PC1.pc
        pc_new_CS0 = torch.matmul(pc_CSnew, R_new_CS0.T) + t_new_CS0[None, :]

        complete_point_cloud = torch.cat((complete_point_cloud, pc_new_CS0), dim=0)
