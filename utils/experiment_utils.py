import os
import numpy as np
import omegaconf
import logging
import sys

from utils.other import start_debug
import visualization.classifications as vis_cls
from features.feature_utils import process_features


def run_ablation_features(args, logger):
    from classifiers.PointTransformers.train_cls import run_cls, run_test

    N_features = sum(args.feature_filter) - len(args.ablation.remove_keys)
    if args.plot_train_acc:
        all_train_accuracies = np.empty((N_features, args.epoch))
    all_val_accuracies = np.empty((N_features, args.epoch))
    feature_keys = [key for key, value in args.features_to_create.items() if value]
    use_feature_keys = [key for key, value in args.features_to_use.items() if value]
    ablation_use_feature_keys = [
        key for key in use_feature_keys if key not in args.ablation.remove_keys
    ]
    i = -1
    for j, feature_key in enumerate(feature_keys):
        # TODO: REMOVE
        # Here we skip the use_label, use_jde, use_sd, and use_c part
        if feature_key in args.ablation.remove_keys:
            continue
        if not args.features_to_use[feature_key]:
            continue
        i = i + 1
        args.model_identifier = f"ablation_without_{feature_key}"
        logger.info(f"Remove feature {feature_key} ({np.around(100*i/N_features, 2)}%)")
        logger.info(f"Ablation features {ablation_use_feature_keys}")
        logger.info(f"Use features {args.features_to_use}")
        args.ablation_feature_filter = args.feature_filter.copy()
        args.ablation_feature_filter[j] = 0
        train_accuracies, val_accuracies, classifier = run_cls(args, logger)

        _, _, y_true, y_pred = run_test(args, logger, classifier)
        vis_cls.plot_accuracies(
            train_accuracies,
            val_accuracies,
            args=args,
        )
        vis_cls.store_confusion_matrix(
            y_pred,
            y_true,
            N_classes=args.perturb_settings.n_classes,
            logger=logger,
            args=args,
        )
        all_val_accuracies[i] = val_accuracies
        if args.plot_train_acc:
            all_train_accuracies[i] = train_accuracies
    if args.plot_train_acc:
        vis_cls.plot_accuracies_ablation(
            all_val_accuracies,
            ablation_use_feature_keys,
            args=args,
            all_train_accuracies=all_train_accuracies,
            model_identifier="ablation_all",
        )
    else:
        vis_cls.plot_accuracies_ablation(
            all_val_accuracies,
            ablation_use_feature_keys,
            args=args,
            model_identifier="ablation_all",
        )
    return None


def setup_coral_args(args, logger):
    logger.info("Changing args to conform with coral settings")
    args.fps.do_fps = False
    args.neighborhood.rmin = 1
    args.neighborhood.rmax = 5
    args.neighborhood.k = "normal"
    args.preprocessing.apply_hpr_operator = False
    args.ablation.run_ablation = False
    return args


def setup_fact_args(args, logger):
    logger.info("Changing args to conform with fact settings")
    args.fps.do_fps = True
    args.neighborhood.rmin = 0.5
    args.neighborhood.rmax = 7.5
    args.neighborhood.k = "adaptive"
    args.preprocessing.apply_hpr_operator = True
    args = process_features(args)
    return args


def setup_experiment(args, do_reg, change_of_pose_C1=True, reg_method="p2l"):
    if args.debug:
        start_debug()
    omegaconf.OmegaConf.set_struct(args, False)
    logger = logging.getLogger(__name__)
    assert args.classifier in [
        "CorAl",
        "FACT",
    ], "ERROR: Did not get a valid classifier!"

    # NOTICE THAT THE BELOW OVERWRITES SOME SETTINGS
    if args.classifier == "CorAl":
        args = setup_coral_args(args, logger)
    elif args.classifier == "FACT" and args.general_dataset == "nuscenes":
        args = setup_fact_args(args, logger)
    elif args.general_dataset in ["kitti", "3dmatch", "3dlomatch", "modelnet"]:
        args = process_features(args)
    else:
        sys.exit(f"ERROR: Did not get a valid dataset: {args.general_dataset}")
    args.do_reg = do_reg
    args.change_of_pose_C1 = change_of_pose_C1
    args.reg_method = reg_method
    directory = args.visualization_folder

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    return args, logger


def setup_args_for_iteration(i, args):
    if i == 0:
        args.model_identifier = "FACT_best_network_optimal"
        args.lambda_lf = 0.5
        args.re_use_data = False
    elif i == 1:
        args.model_identifier = "FACT_best_network_optimal_lambda0"
        args.lambda_lf = 0
        args.re_use_data = True
    elif i == 2:
        args.model_identifier = "FACT_best_network_optimal_lambda1"
        args.lambda_lf = 1
        args.re_use_data = True
    # High Performance
    else:
        sys.exit(f"Not supposed to be more than {args.running_iterations} runs")
    args = process_features(args)
    return args
