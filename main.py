import torch


from utils.other import start_debug
from classifiers.differential_entropy import differential_entropy_dataset
from utils.optimize_parameters import optimize_with_ax
from utils.data_handling import read_nuscenes_data, gather_data
from classifiers.regression import perform_logistic_regression


def main():
    print("cuda available:", torch.cuda.is_available())
    # TODO: Test run differential entropy on the entire point cloud instead 
    #       of neighborhoods. Should go quite much fast I would guess.

    # TODO: 2023-04-27 (Gather data into lists e.g.)
    # 1. Gather data into lists (DONE!)
    # (HAVE NOW AUTOMATICALLY PETURBED AS THEY DO IN CORAL)
    # 2. Perform random perturbations or register data with ICP e.g.
    # 2.1 With ICP we can compare the estimate transformation by with the ground truth to classify
    #     the data as either aligned or misaligned. This data is then quite realistic and can be
    #     used to train our classifier (select suitable parameters).
    # 2.2 Align point clouds with ground truth and then apply a small offset and then register
    #     with ICP. This can lead to misalignments which are harder to detect as the relative
    #     error transformation is smaller. Thus, we might challenge the model a bit more which is
    #     good. Furthermore, ICP is a very interesting registration model to use for three reasons,
    #       1. Implementations of ICP are easily accesible thorugh e.g. Open3d
    #       2. It easily get stuck in local minima which is the realistic/common point when registration
    #          methods stop to iterate.
    #       3. It is common and well known, and useful as a benchmark registration method in that sense.
    # 3. Split data up into training and test set (DONE!)

    # TODO: How to choose model parameters
    # 1. Use some optimization in PyTorch to backpropagate the parameters of the logistic regression
    #    model (beta0, beta1, beta2). (DONE!)
    # 1.1 When I start implementing this, I start by fixating all parameters except the betas. I might
    #     later add a few of them to the loss function, but we'll see. (DONE!)
    # 2. Optimize of the other parameters alpha (even though it is given by the dataset), epsilon,
    #    and E_reject by optimizing over a grid of values (grid search). (DONE!)

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
    #start_debug()
    main()
    #optimize_with_ax(samples_training=100, samples_test=40, verbose=True, total_trials=40)
