import torch
import numpy as np

from utils.data_handling import setup_inputs_to_dnn
from utils.parameters import Params
from features.differential_entropy import differential_entropy_pointwise


def number_of_features(features):
    # xyz is always used (=> 3)
    N_features = 3 + sum(features.values())
    if features.use_de:
        N_features += 1  # differential entropy entails two features
    return N_features


def feature_extraction(PC_scenes, params):
    # TODO: Extract the features better, I wrote about this on Trello
    PC_scenes_with_features = differential_entropy_pointwise(PC_scenes, params.params_diff_entropy)
    # TODO: Extract more features ...
    return PC_scenes_with_features


def write_features_to_txt_files(flat_PC_scenes, data_folder, params):
    # TODO: The feature files are saved for the next run and not deleted. This should not cause any
    # problems really.
    for PC_pair in flat_PC_scenes:
        if PC_pair.misaligned:
            subfolder = '/misaligned/'
        else:
            subfolder = '/aligned/'
        save_file = data_folder + subfolder + PC_pair.name + ".txt"
        PC0 = PC_pair.PC0
        PC1 = PC_pair.PC1

        # Set xyz feature channels
        xyz_channels = np.vstack((PC0.pc[PC0.fps_inds].cpu().numpy(), PC1.pc[PC1.fps_inds].cpu().numpy()))
        feature_map = xyz_channels
        if params.use_label:
            # Set label feature (which point cloud the point belongs to)
            label_channel = np.vstack((np.zeros((PC0.N_fps_points, 1)), np.ones((PC1.N_fps_points, 1))))
            feature_map = np.concatenate((feature_map, label_channel), axis=1)
        if params.use_de:
            # Set joint differential entropy channel
            jde_channel = np.vstack((PC0.metric_jde.cpu().numpy()[:, np.newaxis],
                                    PC1.metric_jde.cpu().numpy()[:, np.newaxis]))
            # Set separate differential entropy channel
            sde_channel = np.vstack((PC0.metric_sde.cpu().numpy()[:, np.newaxis],
                                    PC1.metric_sde.cpu().numpy()[:, np.newaxis]))
            feature_map = np.concatenate((feature_map, jde_channel, sde_channel), axis=1)
        if params.use_wd:
            print("Wasserstein Distance feature not implemented yet")
        if params.use_c:
            print("Co-visibility weight feature not implemented yet")
        if params.use_s:
            print("Static point weight feature not implemented yet")
        if params.use_cj:
            # Set joint number of neighbors ratio
            cj_channel = np.vstack((PC0.weight_cj.cpu().numpy()[:, np.newaxis],
                                    PC1.weight_cj.cpu().numpy()[:, np.newaxis]))
            feature_map = np.concatenate((feature_map, cj_channel), axis=1)
        if params.use_cs:
            # Set separate number of neighbors ratio
            cs_channel = np.vstack((PC0.weight_cs.cpu().numpy()[:, np.newaxis],
                                    PC1.weight_cs.cpu().numpy()[:, np.newaxis]))
            feature_map = np.concatenate((feature_map, cs_channel), axis=1)
        np.savetxt(save_file, feature_map, delimiter=',', fmt='%.6f')


def extract_features_to_txt_files(nusc, features, n_scenes=10, n_samples_per_scene=1, train_ratio=0.60,
                                  N_fps_points=1024):
    # TODO: Add some assertions that we do not use more scenes than we actually have
    # TODO: Find suitable parameters. Although, I think these are rather ok
    # Set parameters
    scale_factor = 5
    params_diff_entropy = {
        "rmin": 0.2*scale_factor,
        "rmax": 1*scale_factor,
        "log_epsilon": -18.0,
        "alpha": 1.33*scale_factor,
        "E_reject": 0.20
    }
    T_close_thresh = 1.5
    downsample_factor = 1
    verbose = False
    hpr_radius = 3.25
    preprocess = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = Params(nusc=nusc, n_scenes=n_scenes, n_samples_per_scene=n_samples_per_scene,
                    train_ratio=train_ratio, downsample_factor=downsample_factor,
                    T_close_thresh=T_close_thresh, params_diff_entropy=params_diff_entropy,
                    verbose=verbose, hpr_radius=hpr_radius, preprocess=preprocess, pointwise=True,
                    do_fps=True, N_fps_points=N_fps_points, device=device)
    # Set which features to use
    params.set_which_features_to_use(features)
    setup_inputs_to_dnn(params)
