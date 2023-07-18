import torch
import numpy as np
import os

from utils.data_handling import setup_inputs_to_dnn
from utils.parameters import Params
from features.differential_entropy import extract_differential_entropy
from features.feature_utils import get_data_batches


def get_neighborhood(PC_pair, current_pc_is_0, pc_batch, radii_batch, params):
    if params.use_de or params.use_cs:
        # Note that everything is in the coordinate system of PC0
        dists_pc0 = torch.cdist(pc_batch, PC_pair.PC0.pc)
        dists_pc1 = torch.cdist(pc_batch, PC_pair.pc1_CS0)

        if current_pc_is_0:
            neighbor_mask_sep = (dists_pc0 < radii_batch[:, None])
        else:
            neighbor_mask_sep = (dists_pc1 < radii_batch[:, None])

        dists_joint = torch.concatenate((dists_pc0, dists_pc1), dim=1)
        del dists_pc0, dists_pc1
        neighbor_mask_joint = (dists_joint < radii_batch[:, None])
        return neighbor_mask_joint, neighbor_mask_sep

    dists_joint = torch.cdist(pc_batch, PC_pair.PCUnion.pc)
    neighbor_mask_joint = (dists_joint < radii_batch[:, None])
    return neighbor_mask_joint, None


def feature_extraction(PC_scenes, params):
    # TODO: Extract the features better, I wrote about this on Trello
    N_scenes, N_samples_per_scene = PC_scenes.shape
    for scene_number in range(N_scenes):
        for sample_number in range(N_samples_per_scene):
            PC_pair = PC_scenes[scene_number][sample_number]
            PC_joint = PC_pair.PCUnion
            if params.study_neighborhoods:
                # Setup batches for the points of interest
                index_batches, pc_batches, radii_batches = get_data_batches(PC_joint, params)

                N_batches = len(index_batches)
                assert N_batches % 2 == 0, f"{N_batches} is not an even number"
                for i, (pc, index, radii) in enumerate(zip(pc_batches, index_batches, radii_batches)):
                    # Keep track if which PC this batch belongs to
                    if i < N_batches/2:
                        current_pc_is_0 = True
                        PC_sep = PC_pair.PC0
                    else:
                        current_pc_is_0 = False
                        PC_sep = PC_pair.PC1

                    # Find a mask for the neighborhood of the points of interest
                    neighbor_mask_j, neighbor_mask_s = get_neighborhood(PC_pair, current_pc_is_0,
                                                                        pc, radii, params)
                    # EXTRACT NEIGHBOR FEATURES
                    # Count the number of neighbors in the joint/sep pc for each point in the batch
                    if params.calc_joint_neighbors:
                        n_neighbors_per_point_in_batch_j = torch.sum(neighbor_mask_j, dim=1,
                                                                     dtype=PC_joint.weight_cj.dtype)
                        if params.use_cj:
                            # SET JOINT NEIGHBORHOOD CARDINALITY RATIO
                            PC_joint.weight_cj[index] = n_neighbors_per_point_in_batch_j/PC_joint.N_points
                    if params.calc_sep_neighbors:
                        n_neighbors_per_point_in_batch_s = torch.sum(neighbor_mask_s, dim=1,
                                                                     dtype=PC_joint.weight_cs.dtype)
                        if params.use_cs:
                            # SET SEPARATE NEIGHBORHOOD CARDINALITY RATIO
                            if current_pc_is_0:
                                PC_joint.weight_cs[index] = \
                                    n_neighbors_per_point_in_batch_s/PC_pair.PC0.N_points
                            else:
                                PC_joint.weight_cs[index] = \
                                    n_neighbors_per_point_in_batch_s/PC_pair.PC1.N_points

                    if params.use_de:
                        # EXTRACT JOINT ENTROPY FEATURE
                        # Filter out neighborhoods with only one point
                        bool_mask_valid_neighborhood_j = (n_neighbors_per_point_in_batch_j > 1)
                        inds_to_valid_neighborhood_j = torch.nonzero(bool_mask_valid_neighborhood_j).squeeze()

                        entropies_batch_j = extract_differential_entropy(
                            PC_joint, n_neighbors_per_point_in_batch_j, neighbor_mask_j,
                            inds_to_valid_neighborhood_j, params)
                        PC_joint.metric_jde[index] = entropies_batch_j
                        del n_neighbors_per_point_in_batch_j, neighbor_mask_j, inds_to_valid_neighborhood_j

                        # EXTRACT SEPARATE ENTROPY FEATURE
                        # Filter out neighborhoods with only one point
                        bool_mask_valid_neighborhood_s = (n_neighbors_per_point_in_batch_s > 1)
                        inds_to_valid_neighborhood_s = torch.nonzero(bool_mask_valid_neighborhood_s).squeeze()

                        entropies_batch_s = extract_differential_entropy(
                            PC_sep, n_neighbors_per_point_in_batch_s, neighbor_mask_s,
                            inds_to_valid_neighborhood_s, params)
                        PC_joint.metric_sde[index] = entropies_batch_s
                        del n_neighbors_per_point_in_batch_s, neighbor_mask_s, inds_to_valid_neighborhood_s

            PC_scenes[scene_number][sample_number].PCUnion = PC_joint

    # TODO: Extract more features ...
    return PC_scenes


def number_of_features(features):
    # xyz is always used (=> 3)
    N_features = 3 + sum(features.values())
    if features.use_de:
        N_features += 1  # differential entropy entails two features
    return N_features


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
            jde_channel = PC_pair.PCUnion.metric_jde.cpu().numpy()[:, np.newaxis]
            # Set separate differential entropy channel
            sde_channel = PC_pair.PCUnion.metric_sde.cpu().numpy()[:, np.newaxis]
            feature_map = np.concatenate((feature_map, jde_channel, sde_channel), axis=1)
        if params.use_wd:
            print("Wasserstein Distance feature not implemented yet")
        if params.use_c:
            print("Co-visibility weight feature not implemented yet")
        if params.use_s:
            print("Static point weight feature not implemented yet")
        if params.use_cj:
            # Set joint number of neighbors ratio
            cj_channel = PC_pair.PCUnion.weight_cj.cpu().numpy()[:, np.newaxis]
            feature_map = np.concatenate((feature_map, cj_channel), axis=1)
        if params.use_cs:
            # Set separate number of neighbors ratio
            cs_channel = PC_pair.PCUnion.weight_cs.cpu().numpy()[:, np.newaxis]
            feature_map = np.concatenate((feature_map, cs_channel), axis=1)

        # Get the directory name from the save_file path
        dir_name = os.path.dirname(save_file)
        # Check if the directory exists
        if not os.path.exists(dir_name):
            # If the directory doesn't exist, create it
            os.makedirs(dir_name)
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
