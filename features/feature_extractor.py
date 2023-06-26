from utils.data_handling import setup_inputs_to_dnn
from utils.parameters import Params


def extract_features(nusc, n_scenes=10, n_samples_per_scene=1, train_ratio=0.60, n_points=1024):
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
    downsample_factor = 8
    verbose = False
    hpr_radius = 3.25
    preprocess = True
    pointwise = True
    ##
    print(f"Radius running: {hpr_radius}")
    params = Params(nusc=nusc, n_scenes=n_scenes, n_samples_per_scene=n_samples_per_scene,
                    train_ratio=train_ratio, downsample_factor=downsample_factor,
                    T_close_thresh=T_close_thresh, params_diff_entropy=params_diff_entropy,
                    hpr_radius=hpr_radius, preprocess=preprocess, pointwise=pointwise,
                    n_points=n_points)

    if pointwise:
        all_PC_scenes_train, all_PC_scenes_test = setup_inputs_to_dnn(params)
        print("Features Extracted!")
        return all_PC_scenes_train, all_PC_scenes_test
    return None
