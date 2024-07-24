import subprocess
import os
import hydra
import torch
import numpy as np
import copy

from utils.parameters import Params
from utils.nuscenes_handling import read_nuscenes_data, NuscenesHandling
from utils.pointclouds import farthest_point_sample_PC_scenes
from utils.experiment_utils import setup_experiment
from classifiers.PointTransformers.classify import classify_pairs
from classifiers.PointTransformers.train_cls import load_best_model
from visualization.classifications import store_confusion_matrix
import visualization.registration as vr
from features.feature_extractor import feature_extraction, create_feature_map
from features.feature_utils import normalize_data_on_condition
import registration.registration_utils as ru
from visualization.presentation import visualize_and_save

from utils.geometrics import transformation_matrix
from utils.pointclouds import PC, PCPair
from utils.visibility import keep_covisible_points


@hydra.main(config_path="../classifiers/PointTransformers/config", config_name="cls_registration")
def geotransformer_with_fact(args):
    # Set the environment variable
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Define the path to the test script and the argument
    script_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '../', 
        'GeoTransformer_202407', 
        'experiments', 
        'geotransformer.kitti.stage5.gse.k3.max.oacl.stage2.sinkhorn', 
        'test.py'
    ))

    # Define the relative path to the snapshot file
    snapshot_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '../',
        'GeoTransformer_202407',
        'weights', 
        'geotransformer-kitti.pth.tar'
    ))

    # Call the test script with the argument
    subprocess.run(['python', script_path, f'--snapshot={snapshot_path}', f'--fact_args={args}'])


def get_input_data_fact(args, src_points, ref_points, gt_trans, est_trans):
    # TODO: If things does not work, I could double-check that these points are on the same
    # scale as the nuscenes points and that the density is somewhat similar.
    # TODO: If thing does not work, I could also retrain FACT on KITTI.
    DO_REG = True
    CHANGE_OF_POSE_C1 = True
    VISUALIZE_RESULT = False
    REPAIR_METHOD = "gt"  # [gt, p2l]
    REG_METHOD = "..."  # [p2l, geotrans] TODO: I register elsewhere so what to set this to ...

    # TODO: I should not create params many times, just once ...

    params = Params(nusc=None, args=args, pointwise=True)
    args, logger = setup_experiment(args, do_reg=DO_REG, change_of_pose_C1=CHANGE_OF_POSE_C1,
                                    reg_method=REG_METHOD)

    params.set_which_features_to_use(args.features_to_create)

    PC_scene = to_PC_format(src_points, ref_points, params, est_trans, gt_trans, geo_args=None)

    # errors_scenes = torch.zeros((n_scenes, n_samples_per_scene), device='cuda')
    # pcac_model, _, args = load_best_model(args, logger)
    # preds = np.zeros((n_scenes, n_samples_per_scene))
    # gts = np.zeros((n_scenes, n_samples_per_scene))
    # poses_est_scenes = torch.zeros((n_scenes, n_samples_per_scene, 4, 4), device='cuda')
    # poses_gt_scenes = torch.zeros((n_scenes, n_samples_per_scene, 4, 4), device='cuda')

    # if args.one_scene:
    #     PC_scene = PCHandler.sample_from_scenes(params, n_samples=n_samples_per_scene, n_scenes=1,
    #                                             geo_args=geo_args)[0]
    # else:
    #     PC_scene = read_nuscenes_data(params, mode="test", n_samples=n_samples_per_scene,
    #                                     n_scenes=1, scene_counter=i+test_start_scene)[0]
    # Register scene
    # Calculate errors

    # if DO_REG:
    #     PC_scene_reg = align_scene(poses_est_scenes[i], PC_scene, plot=False)
    # else:
    #     PC_scene_reg = copy.deepcopy(PC_scene)
    # # EXTRACT FEATURE DATA
    # # FPS on scenes
    # if params.args.fps.do_fps:
    #     farthest_point_sample_PC_scenes(PC_scene_reg[None, ...], params.N_fps_points)
    # # Feature extraction. [0]: Go from scenes to scene
    # PC_scene_with_features = feature_extraction(PC_scene_reg[None, ...], params)[0]
    # feature_maps = create_feature_map(PC_scene_with_features[0], params)[None, ...]
    # for pair in PC_scene_with_features[1:]:
    #     feature_maps = np.concatenate((feature_maps,
    #                                     create_feature_map(pair, params)[None, ...]), axis=0)
    # # B, N, D (batch_size, N_pts, feature_dim)
    # points = torch.from_numpy(feature_maps).to(params.device).float()
    # # Note that since we just normalize so that xyz lie within the unit ball,
    # # with the farthest point on the ball => It does not matter which batch size we use.
    # points = normalize_data_on_condition(args, points)
    # batch_size = 8  # Doesn't matter which bs we use in test, as store lots, lets keep it low.
    # points_batches = torch.split(points, batch_size)
    # for j, ps in enumerate(points_batches):
    #     if j == 0:
    #         p = classify_pairs(pcac_model, ps)
    #     else:
    #         p  = torch.cat((p, classify_pairs(pcac_model, ps)))
    # preds[i] = p.cpu().numpy()    
    # print("hej")

    return None


def to_PC_format(src_pc, ref_pc, params, est_trans, gt_trans, geo_args=None):
    src_pt_dists = torch.norm(src_pc, p=2, dim=1)
    ref_pt_dists = torch.norm(ref_pc, p=2, dim=1)

    # Load in first point cloud
    PC1 = PC(src_pc, src_pt_dists, label=1, device=params.device)
    # Load in second point cloud
    PC0 = PC(ref_pc, ref_pt_dists, label=0, device=params.device)

    # Set point cloud pair and their union, and perform possible perturbation
    currentPCPair = PCPair(
        PC0, PC1, device=params.device, PCHandler=None,
        perturb_settings=params.args.perturb_settings, change_of_pose_C1=params.args.change_of_pose_C1,
        perturbation_method=params.args.perturb_settings.perturbation_method,
        reg_method = params.args.reg_method, geo_args=geo_args, est_pc1_to_pc0=est_trans,
        gt_pc1_to_pc0=gt_trans,
    )

    # TODO: Cont'd here
    if self.apply_hpr_operator:
        # Calculate the co-visible points
        PC0_cov, PC1_cov, PCUnion_cov = keep_covisible_points(
            PC0, PC1, currentPCPair.PCUnion, currentPCPair.pose0, currentPCPair.pose1,
            compute_weights=params.use_c, hpr_radius=cov_params.hpr_radius,
            gamma=cov_params.gamma, inversion_kernel=cov_params.inversion_kernel,
            batch_size=params.batch_size_feature_extraction,
        )
        currentPCPair.set_new_PC(PC0_cov, PC1_cov, PCUnion_cov)