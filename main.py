import torch


from utils.other import start_debug
from features.differential_entropy import differential_entropy_dataset
from utils.optimize_parameters import optimize_with_ax
from utils.nuscenes_handling import read_nuscenes_data
from utils.data_handling import gather_data
from classifiers.regression import perform_logistic_regression

from classifiers.PointTransformers.train_cls import main as PCAC


def main():
    print("cuda available:", torch.cuda.is_available())
    # TODO: Test run differential entropy on the entire point cloud instead 
    #       of neighborhoods. Should go quite much fast I would guess.

    # Some visualization: We can see that the point clouds are better aligned after the transformation
    # vis_2pcs_aligned_vs_misaligned(PC0.pc, PC1.pc, pc1_CS0)
    ##
    PC_scenes = read_nuscenes_data(downsample_factor=16)
    PC_scenes_training, PC_scenes_test = gather_data(PC_scenes, samples_training=200, samples_test=25)
    del PC_scenes
    scale_factor = 10
    params = {
        "rmin": 0.2*scale_factor,
        "rmax": 1*scale_factor,
        "log_epsilon": -18.0,
        "alpha": 1.33*scale_factor,
        "E_reject": 0.20
    }
    X_train, y_train = differential_entropy_dataset(PC_scenes_training, params)
    del PC_scenes_training
    X_test, y_test = differential_entropy_dataset(PC_scenes_test, params)
    del PC_scenes_test
    model, accuracy_test = perform_logistic_regression(X_train, X_test, y_train, y_test, verbose=True)
    print(f"Accuracy: {accuracy_test} with parameters\n {params}")
    print("Finito!")
    # TODO: Compare the differential entropy metric before and after the alignment
    #       with per point entropy to see how well the method works (maybe, I should
    #       maybe not put to much time on this).
    # TODO: Must H_joint > H_sep? No, it mustn't, but in practice it usually is.
    return None


if __name__ == "__main__":
    # start_debug()
    PCAC()
    # optimize_with_ax(samples_training=100, samples_test=40, verbose=True, total_trials=40)
