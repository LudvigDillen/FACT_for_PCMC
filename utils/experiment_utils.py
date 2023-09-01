import numpy as np

from visualization.classifications import (
    plot_accuracies_ablation,
    store_confusion_matrix,
    plot_accuracies,
)


def run_ablation_features(args, logger):
    from classifiers.PointTransformers.train_cls import run_cls, run_test

    N_features = len(args.feature_filter)
    if args.plot_train_acc:
        all_train_accuracies = np.empty((N_features, args.epoch))
    all_val_accuracies = np.empty((N_features, args.epoch))
    feature_keys = [key for key, value in args.features_to_create.items() if value]

    for i, feature_key in enumerate(feature_keys):
        args.model_identifier = f"ablation_without_{feature_key}"
        logger.info(f"Remove feature {feature_key} ({np.around(100*i/N_features, 2)}%)")
        args.ablation_feature_filter = args.feature_filter.copy()
        args.ablation_feature_filter[i] = 0
        train_accuracies, val_accuracies, classifier = run_cls(args, logger)
        _, _, y_true, y_pred = run_test(args, logger, classifier)
        plot_accuracies(
            train_accuracies,
            val_accuracies,
            plot_train_acc=args.plot_train_acc,
            default_tick_step=1,
            model_identifier=args.model_identifier,
        )
        store_confusion_matrix(
            y_pred,
            y_true,
            N_classes=args.perturb_settings.n_classes,
            logger=logger,
            model_identifier=args.model_identifier,
        )
        all_val_accuracies[i] = val_accuracies
        if args.plot_train_acc:
            all_train_accuracies[i] = train_accuracies
    if args.plot_train_acc:
        plot_accuracies_ablation(
            all_val_accuracies,
            feature_keys,
            all_train_accuracies=all_train_accuracies,
            default_tick_step=1,
            model_identifier="ablation_all",
        )
    else:
        plot_accuracies_ablation(
            all_val_accuracies,
            feature_keys,
            default_tick_step=1,
            model_identifier="ablation_all",
        )
    return None


def setup_coral_args(args, logger):
    logger.info("Changing args to conform with coral settings")
    args.fps.do_fps = False
    args.perturb_settings.n_classes = 2
    args.neighborhood.rmin = 1
    args.neighborhood.rmax = 5
    args.neighborhood.k = 'normal'
    args.preprocessing.apply_hpr_operator = False
    args.ablation = False
    return args
