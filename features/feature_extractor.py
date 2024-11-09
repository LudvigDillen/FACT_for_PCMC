import os
import torch
import numpy as np
import time

from utils.data_handling import setup_inputs_to_dnn, generate_class_names_file
from utils.parameters import Params
from features.differential_entropy import extract_differential_entropy
from features.feature_utils import get_data_batches
from features.wasserstein import sinkhorn_divergence


def get_neighborhoods(PC_pair, pc_batch, radii_batch):
    """
    PC_pair.PC0.pc Nx3
    PC_pair.pc1_CS0 Mx3
    pc_batch Bx3
    radii_batch B
    """
    # Note that everything is in the coordinate system of PC0
    dists_pc0 = torch.cdist(pc_batch, PC_pair.PC0.pc)  # BxN
    neighbor_mask_0 = dists_pc0 < radii_batch[:, None]  # BxN
    del dists_pc0

    # I solved a bug so that PC_pair.pc1_CS0 is actually correct ...
    dists_pc1 = torch.cdist(pc_batch, PC_pair.pc1_CS0)  # BxM
    neighbor_mask_1 = dists_pc1 < radii_batch[:, None]  # BxM
    return neighbor_mask_0, neighbor_mask_1


def feature_extraction(PC_scenes, params):
    N_scenes, N_samples_per_scene = PC_scenes.shape
    for scene_number in range(N_scenes):
        torch.cuda.empty_cache()
        for sample_number in range(N_samples_per_scene):
            PC_pair = PC_scenes[scene_number][sample_number]
            PC_joint = PC_pair.PCUnion
            if params.study_neighborhoods:
                # Setup batches for the points of interest
                index_batches, pc_batches, radii_batches = get_data_batches(
                    PC_pair, params
                )

                N_batches = len(index_batches)
                assert N_batches % 2 == 0, f"{N_batches} is not an even number"
                for i, (pc, index, radii) in enumerate(
                    zip(pc_batches, index_batches, radii_batches)
                ):
                    # Keep track if which PC this batch belongs to
                    if i < N_batches / 2:
                        current_pc_is_0 = True
                    else:
                        current_pc_is_0 = False

                    # Find a mask for the neighborhood of the points of interest
                    neighbor_mask_0, neighbor_mask_1 = get_neighborhoods(
                        PC_pair, pc, radii
                    )

                    # EXTRACT NEIGHBOR FEATURES
                    # Count the number of neighbors in the joint/sep pc for each point in the batch
                    if params.calc_joint_neighbors:
                        neighbor_mask_j = torch.cat(
                            (neighbor_mask_0, neighbor_mask_1), dim=1
                        )  # BxN
                        n_neighbors_per_point_in_batch_j = torch.sum(
                            neighbor_mask_j, dim=1, dtype=PC_joint.weight_cj.dtype
                        )  # B

                        # EXTRACT JOINT NEIGHBORHOOD CARDINALITY RATIO FEATURE
                        if params.use_cj:
                            PC_joint.weight_cj[index] = (
                                n_neighbors_per_point_in_batch_j / PC_joint.N_points
                            )

                    if params.calc_sep_neighbors:
                        if current_pc_is_0:
                            neighbor_mask_s = neighbor_mask_0
                        else:
                            neighbor_mask_s = neighbor_mask_1
                        n_neighbors_per_point_in_batch_s = torch.sum(
                            neighbor_mask_s, dim=1, dtype=PC_joint.weight_cs.dtype
                        )

                        # EXTRACT SEPARATE NEIGHBORHOOD CARDINALITY RATIO FEATURE
                        if params.use_cs:
                            if current_pc_is_0:
                                PC_joint.weight_cs[index] = (
                                    n_neighbors_per_point_in_batch_s
                                    / PC_pair.PC0.N_points
                                )
                            else:
                                PC_joint.weight_cs[index] = (
                                    n_neighbors_per_point_in_batch_s
                                    / PC_pair.PC1.N_points
                                )

                    # EXTRACT CARDINALITY RATIO SEPARATE AND JOINT NEIGHBORHOOD
                    if params.use_csj:
                        assert (
                            n_neighbors_per_point_in_batch_j > 0
                        ).all(), "Can't have any joint neighborhood without a point ..."
                        PC_joint.weight_csj[index] = (
                            n_neighbors_per_point_in_batch_s
                            / n_neighbors_per_point_in_batch_j
                        )

                    # EXTRACT JOINT ENTROPY FEATURE
                    if params.use_jde:
                        # Filter out neighborhoods with only one point
                        bool_mask_valid_neighborhood_j = (
                            n_neighbors_per_point_in_batch_j > 1
                        )
                        inds_to_valid_neighborhood_j = torch.nonzero(
                            bool_mask_valid_neighborhood_j
                        ).squeeze(dim=1)

                        entropies_batch_j = extract_differential_entropy(
                            PC_joint,
                            n_neighbors_per_point_in_batch_j,
                            neighbor_mask_j,
                            inds_to_valid_neighborhood_j,
                            params,
                        )
                        PC_joint.metric_jde[index] = entropies_batch_j
                        del n_neighbors_per_point_in_batch_j, neighbor_mask_j
                        del inds_to_valid_neighborhood_j

                    # EXTRACT SEPARATE ENTROPY FEATURE
                    if params.use_sde:
                        # Filter out neighborhoods with only one point
                        bool_mask_valid_neighborhood_s = (
                            n_neighbors_per_point_in_batch_s > 1
                        )
                        inds_to_valid_neighborhood_s = torch.nonzero(
                            bool_mask_valid_neighborhood_s
                        ).squeeze(dim=1)

                        if current_pc_is_0:
                            PC_sep = PC_pair.PC0
                        else:
                            PC_sep = PC_pair.PC1

                        entropies_batch_s = extract_differential_entropy(
                            PC_sep,
                            n_neighbors_per_point_in_batch_s,
                            neighbor_mask_s,
                            inds_to_valid_neighborhood_s,
                            params,
                        )
                        PC_joint.metric_sde[index] = entropies_batch_s
                        del n_neighbors_per_point_in_batch_s, neighbor_mask_s
                        del inds_to_valid_neighborhood_s

                    # EXTRACT SINKHORN DIVERGENCE FEATURE
                    if params.use_sd:
                        dists = sinkhorn_divergence(
                            PC_pair,
                            neighbor_mask_0,
                            neighbor_mask_1,
                            params.args.sinkhorn_div,
                        )
                        PC_joint.metric_wd[index] = dists

            PC_scenes[scene_number][sample_number].PCUnion = PC_joint
    return PC_scenes


def create_feature_map(PC_pair, params):
    PC0 = PC_pair.PC0
    PC1 = PC_pair.PC1
    # Set xyz feature channels
    xyz_channels = np.vstack(
        (PC0.pc[PC0.fps_inds].cpu().numpy(), PC1.pc[PC1.fps_inds].cpu().numpy())
    )
    feature_map = xyz_channels
    if params.use_label:
        # Set label feature (which point cloud the point belongs to)
        label_channel = np.vstack(
            (np.zeros((PC0.N_fps_points, 1)), np.ones((PC1.N_fps_points, 1)))
        )
        feature_map = np.concatenate((feature_map, label_channel), axis=1)
    if params.use_jde:
        # Set joint differential entropy channel
        jde_channel = PC_pair.PCUnion.metric_jde.cpu().numpy()[:, np.newaxis]
        feature_map = np.concatenate((feature_map, jde_channel), axis=1)
    if params.use_sde:
        # Set separate differential entropy channel
        sde_channel = PC_pair.PCUnion.metric_sde.cpu().numpy()[:, np.newaxis]
        feature_map = np.concatenate((feature_map, sde_channel), axis=1)
    if params.use_sd:
        wd_channel = PC_pair.PCUnion.metric_wd.cpu().numpy()[:, np.newaxis]
        feature_map = np.concatenate((feature_map, wd_channel), axis=1)
    if params.use_c:
        c_channel = PC_pair.PCUnion.weight_c.cpu().numpy()[:, np.newaxis]
        feature_map = np.concatenate((feature_map, c_channel), axis=1)
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
    if params.use_csj:
        # Set cardinality ratio sep and joint neighborhood
        csj_channel = PC_pair.PCUnion.weight_csj.cpu().numpy()[:, np.newaxis]
        feature_map = np.concatenate((feature_map, csj_channel), axis=1)
    if params.args.features_to_create.use_xyz:
        feature_map = np.concatenate((feature_map, xyz_channels), axis=1)
    if params.args.features_to_create.use_z:
        if params.args.features_to_create.use_xyz:
            print(
                f">The xyz-channels were already created. We will not add the z-channel as well"
            )
            params.args.features_to_create.use_z = False
        else:
            z_channel = xyz_channels[:, 2][:, None]
            feature_map = np.concatenate((feature_map, z_channel), axis=1)
    if params.args.features_to_create.use_norm_xyz:
        norm_channel = np.linalg.norm(xyz_channels, axis=1)[:, None]
        feature_map = np.concatenate((feature_map, norm_channel), axis=1)
    return feature_map


def write_features_to_txt_files(PC_scenes, params, data_folder):
    for PC_scene in PC_scenes:
        for PC_pair in PC_scene:
            # Get feature map
            feature_map = create_feature_map(PC_pair, params)
            # Output to txt file
            category_folder = os.path.join(
                data_folder, params.class_names[PC_pair.class_category]
            )
            save_file = os.path.join(category_folder, PC_pair.name + ".txt")
            # Get the directory name from the save_file path
            dir_name = os.path.dirname(save_file)
            # Check if the directory exists
            if not os.path.exists(dir_name):
                # If the directory doesn't exist, create it
                os.makedirs(dir_name)
            # If the file already contains content, we overwrite it
            reg_error = PC_pair.reg_error.item()
            s = f"GT registration error: {reg_error}\n"
            # Open the file in write mode
            with open(save_file, 'w') as f:
                # Write the registration error line
                f.write(s)
                # Now save the feature map data below the registration error line
                np.savetxt(f, feature_map, delimiter=",", fmt="%.6f")


def write_features_to_txt_files_error(PC_scenes, params):
    data_folder = params.args.feature_folder
    for PC_scene in PC_scenes:
        for PC_pair in PC_scene:
            # Get feature map
            feature_map = create_feature_map(PC_pair, params)
            # Output to txt file
            category_folder = os.path.join(
                data_folder, params.class_names[PC_pair.class_category]
            )
            save_file = os.path.join(category_folder, PC_pair.name + ".txt")
            # Get the directory name from the save_file path
            dir_name = os.path.dirname(save_file)
            # Check if the directory exists
            if not os.path.exists(dir_name):
                # If the directory doesn't exist, create it
                os.makedirs(dir_name)
            # If the file already contains content, we overwrite it
            np.savetxt(save_file, feature_map, delimiter=",", fmt="%.6f")


def extract_features_to_txt_files(nusc, args):
    params = Params(nusc=nusc, args=args, pointwise=True)
    # Set which features to use
    params.set_which_features_to_use(args.features_to_create)
    # We do not have to recreate the file if we are re-running after crash
    if not params.args.rerun_crash:
        generate_class_names_file(
            folder=args.feature_folder,
            filename="PCAC_data_class_names.txt",
            n_classes=args.perturb_settings.n_classes,
            class_names=params.class_names,
        )
    setup_inputs_to_dnn(params)
