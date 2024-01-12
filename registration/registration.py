import torch
import hydra
import nuscenes as ns
import numpy as np
import copy

from utils.parameters import Params
from utils.nuscenes_handling import read_nuscenes_data
from utils.pointclouds import farthest_point_sample_PC_scenes
from utils.experiment_utils import setup_experiment
from classifiers.PointTransformers.classify import classify_pairs
from classifiers.PointTransformers.train_cls import load_best_model
from visualization.classifications import store_confusion_matrix
import visualization.registration as vr
from features.feature_extractor import feature_extraction, create_feature_map
from features.feature_utils import normalize_data_on_condition
import registration.registration_utils as ru


def estimate_transformation_scene(PC_scene, method="p2l", voxelize=False, plot=False):
    trans_init = np.eye(4)
    rel_poses_est = torch.zeros((len(PC_scene), 4, 4), device='cuda')
    for j, pair in enumerate(PC_scene):
        # We want to find T from CS1 to CS0
        source = ru.from_tensor_to_pcd(pair.PC1.pc)
        target = ru.from_tensor_to_pcd(pair.PC0.pc)

        rel_poses_est[j] = ru.register_pair(source, target, method=method,
                                            trans_init=trans_init, voxelize=voxelize)

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
def reg_mpe(args):
    ### SETUP
    # Extra settings:
    n_samples_per_scene = args.n_samples_per_scene
    n_classes = args.perturb_settings.n_classes
    DO_REG = True
    test_start_scene = 638  # Set to 630 if we want to only test at test data
    n_scenes = args.n_scenes - test_start_scene

    # Init Nusc object
    nusc = ns.nuscenes.NuScenes(version=args.dataset, dataroot=args.data_folder, verbose=False)
    errors_scenes = torch.zeros((n_scenes, n_samples_per_scene), device='cuda')

    args, logger = setup_experiment(args, do_reg=DO_REG)
    params = Params(nusc=nusc, args=args, pointwise=True)
    params.set_which_features_to_use(args.features_to_create)
    pcac_model, _, args = load_best_model(args, logger)
    preds = np.zeros((n_scenes, n_samples_per_scene))
    gts = np.zeros((n_scenes, n_samples_per_scene))
    poses_est_scenes = torch.zeros((n_scenes, n_samples_per_scene, 4, 4), device='cuda')
    poses_gt_scenes = torch.zeros((n_scenes, n_samples_per_scene, 4, 4), device='cuda')
    for i in range(n_scenes):
        PC_scene = read_nuscenes_data(params, mode="test", n_samples=n_samples_per_scene,
                                      n_scenes=1, scene_counter=i+test_start_scene)[0]
        if i % 5 == 0:
            print(f"Scene {i}")
        # Register scene
        # TODO: GT pose is not correct since we do not account for a potential perturbation.
        poses_gt_scenes[i] = ru.get_gt_poses(PC_scene)
        poses_est_scenes[i] = estimate_transformation_scene(PC_scene, method="p2l")
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
        preds[i] = classify_pairs(pcac_model, points).cpu().numpy()
    # TODO: Set the acc_col and acc_row to true and solve to get the sum next to the conf matrix.
    store_confusion_matrix(y_pred=preds.ravel(), y_true=gts.ravel(), N_classes=n_classes,
                           logger=logger, args=args, accumulate=True)
    return None