import torch
import numpy as np

from utils.data_handling import setup_inputs_to_dnn
from utils.parameters import Params


def get_N_feature_channels(flat_PC_scenes):
    for PC_pair in flat_PC_scenes:
        PC0 = PC_pair.PC0
        PC1 = PC_pair.PC1
        assert (PC0.N_feature_channels == PC1.N_feature_channels), "ERROR: Different many feature channels!"
        return PC0.N_feature_channels


def write_features_to_txt_files(flat_PC_scenes, data_folder):
    # TODO: The feature files are saved for the next run and not deleted. This should not cause any
    # problems really.
    for PC_pair in flat_PC_scenes:
        if PC_pair.misaligned:
            subfolder = '/misaligned/'
        else:
            subfolder = '/aligned/'
        file = data_folder + subfolder + PC_pair.name + ".txt"
        PC0 = PC_pair.PC0
        PC1 = PC_pair.PC1

        # Set xyz feature channels
        xyz_channels = np.vstack((PC0.pc[PC0.fps_inds].cpu().numpy(), PC1.pc[PC1.fps_inds].cpu().numpy()))
        # Set label feature (which point cloud the point belongs to)
        label_channel = np.vstack((np.zeros((PC0.N_fps_points, 1)), np.ones((PC1.N_fps_points, 1))))
        # Set joint differential entropy channel
        jde_channel = np.vstack((PC0.metric_jde.cpu().numpy()[:, np.newaxis],
                                 PC1.metric_jde.cpu().numpy()[:, np.newaxis]))
        # Set separate differential entropy channel
        sde_channel = np.vstack((PC0.metric_sde.cpu().numpy()[:, np.newaxis],
                                 PC1.metric_sde.cpu().numpy()[:, np.newaxis]))
        feature_map = np.concatenate((xyz_channels, label_channel, jde_channel, sde_channel), axis=1)
        np.savetxt(file, feature_map, delimiter=',', fmt='%.6f')


def classes_to_txt(data_file, flat_PC_scenes, aligned_samples, misaligned_samples):
    with open(data_file, 'w') as file:
        for PC_pair in flat_PC_scenes:
            if PC_pair.misaligned:
                misaligned_samples += 1
                message = 'misaligned' + "_" + str(misaligned_samples).zfill(4)
            else:
                aligned_samples += 1
                message = 'aligned' + "_" + str(aligned_samples).zfill(4)
            PC_pair.set_name(message)
            file.write(message + '\n')
    return aligned_samples, misaligned_samples, flat_PC_scenes


def write_classes_to_txt_files(
        flat_PC_scenes_train, flat_PC_scenes_test,
        train_data_file="/home/luddi824/thesis/PCAC/data/PCAC_data/PCAC_data_train.txt",
        test_data_file="/home/luddi824/thesis/PCAC/data/PCAC_data/PCAC_data_test.txt"):
    # We count the test and train samples together
    aligned_samples = 0
    misaligned_samples = 0

    aligned_samples, misaligned_samples, flat_PC_scenes_train_named = classes_to_txt(
        train_data_file, flat_PC_scenes_train, aligned_samples, misaligned_samples)

    aligned_samples, misaligned_samples, flat_PC_scenes_test_named = classes_to_txt(
        test_data_file, flat_PC_scenes_test, aligned_samples, misaligned_samples)
    return flat_PC_scenes_train_named, flat_PC_scenes_test_named


def sort_PC_scenes(all_PC_scenes):
    flat_PC_scenes = []
    for scene in all_PC_scenes:
        for PC_pair in scene:
            if PC_pair.misaligned:
                flat_PC_scenes.append(PC_pair)
            else:
                flat_PC_scenes.insert(0, PC_pair)
    return np.array(flat_PC_scenes)


def extract_features_to_txt_files(nusc, n_scenes=10, n_samples_per_scene=1, train_ratio=0.60,
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
    ##
    print(f"Radius running: {hpr_radius}")
    params = Params(nusc=nusc, n_scenes=n_scenes, n_samples_per_scene=n_samples_per_scene,
                    train_ratio=train_ratio, downsample_factor=downsample_factor,
                    T_close_thresh=T_close_thresh, params_diff_entropy=params_diff_entropy,
                    verbose=verbose, hpr_radius=hpr_radius, preprocess=preprocess, pointwise=True,
                    do_fps=True, N_fps_points=N_fps_points, device=device)

    all_PC_scenes_train, all_PC_scenes_test = setup_inputs_to_dnn(params)
    print("Features Extracted!")
    flat_PC_scenes_train = sort_PC_scenes(all_PC_scenes_train)
    flat_PC_scenes_test = sort_PC_scenes(all_PC_scenes_test)
    print("Data sorted")
    PC_train, PC_test = write_classes_to_txt_files(flat_PC_scenes_train, flat_PC_scenes_test)
    print("Classes written to txt files")
    N_feature_channels = get_N_feature_channels(PC_train)  # could look at test data instead ofc
    write_features_to_txt_files(PC_train, data_folder="/home/luddi824/thesis/PCAC/data/PCAC_data")
    write_features_to_txt_files(PC_test, data_folder="/home/luddi824/thesis/PCAC/data/PCAC_data")
    print("Features written to txt files")
    print("\nPCAC data is now extracted!\n")
    return N_feature_channels
