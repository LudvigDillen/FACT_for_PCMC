"""
Author: Benny
Date: Nov 2019
"""
import sys
import numpy as np
import torch
import logging
from tqdm import tqdm
import importlib
import shutil
import hydra
import omegaconf

import nuscenes as ns
from features.feature_extractor import extract_features_to_txt_files
from utils.experiment_utils import (
    run_ablation_features,
    setup_coral_args,
    setup_fact_args,
)
from features.feature_utils import (
    process_features,
    number_of_features,
    augment_data,
    normalize_data_on_condition,
)
from utils.other import start_debug
from utils.pointclouds import PCAC_dataset
from visualization.classifications import (
    plot_accuracies,
    store_confusion_matrix,
    model_plot,
)
from classifiers.loss_functions import get_loss
from classifiers.coral import (
    get_coral_features,
    perform_coral_training,
    perform_coral_inference,
)


def get_mean_acc(class_acc):
    if class_acc[:, 1].any() == 0:
        print("No valid class accuracy! All samples not encountered")
        return 0
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    mean_acc = np.mean(class_acc[:, 2])
    return mean_acc


def inference_loop(data, args, model, class_acc, mean_correct):
    points, target, scene_numbers = data
    # same_scene = (scene_numbers == scene_numbers[0]).sum() == len(scene_numbers)
    # assert same_scene, "All samples does not come from the same scene"

    if torch.cuda.is_available():
        points, target = points.cuda(), target[:, 0].cuda()
    else:
        target = target[:, 0]

    # points = append_spatial_features(args, points)
    points = normalize_data_on_condition(args, points)

    classifier = model.eval()
    pred = classifier(
        points, inference=True
    )  # [B, n_classes], here we have a score for each class
    pred_choice = pred.data.max(1)[1]  # highest score wins

    for cat in np.unique(target.cpu()):
        classacc = (
            pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
        )
        class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
        class_acc[cat, 1] += 1
    correct = pred_choice.eq(target.long().data).cpu().sum()
    mean_correct.append(correct.item() / float(points.size()[0]))
    return class_acc, mean_correct, target, pred_choice


def track_accuracy(args, model, loader):
    mean_correct = []
    class_acc = np.zeros((args.num_class, 3))
    for data in tqdm(loader, total=len(loader)):
        class_acc, mean_correct, _, _ = inference_loop(
            data, args, model, class_acc, mean_correct
        )
    mean_acc = get_mean_acc(class_acc)
    instance_acc = np.mean(mean_correct)
    return instance_acc, mean_acc


def run_val(args, logger, classifier):
    PCAC_VAL_DATASET = PCAC_dataset(args=args, split="validation")
    valDataLoader = torch.utils.data.DataLoader(
        PCAC_VAL_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=8
    )
    val_instance_acc, val_class_acc = track_accuracy(args, classifier, valDataLoader)
    logger.info(f"Val Overall Accuracy: {val_instance_acc:.4f}")
    logger.info(f"Val Mean Class Accuracy: {val_class_acc:.4f}")
    logger.info("End of validation test..")
    return None


def test_results(args, model, loader):
    mean_correct = []
    class_acc = np.zeros((args.num_class, 3))
    N_samples = len(loader.dataset)
    y_true = np.zeros(N_samples)
    y_pred = np.zeros(N_samples)
    ind = 0
    with torch.inference_mode():
        for data in tqdm(loader, total=len(loader)):
            class_acc, mean_correct, target, pred_choice = inference_loop(
                data, args, model, class_acc, mean_correct
            )

            current_batch_size = len(target)
            y_true[ind : ind + current_batch_size] = target.cpu().numpy()
            y_pred[ind : ind + current_batch_size] = pred_choice.cpu().numpy()
            ind = ind + current_batch_size
    mean_acc = get_mean_acc(class_acc)
    instance_acc = np.mean(mean_correct)
    return instance_acc, mean_acc, y_true, y_pred


def load_best_model(args, logger, pretrained=True):
    args.num_class = args.perturb_settings.n_classes
    if args.classifier == "FACT":
        args.input_dim, args = number_of_features(args)
        print(f"Input dim: {args.input_dim}")
        shutil.copy(
            hydra.utils.to_absolute_path(
                "classifiers/PointTransformers/models/{}/model.py".format(
                    args.model.name
                )
            ),
            ".",
        )

        if torch.cuda.is_available():
            classifier = getattr(
                importlib.import_module(
                    "classifiers.PointTransformers.models.{}.model".format(
                        args.model.name
                    )
                ),
                "PointTransformerCls",
            )(args).cuda()
        else:
            classifier = getattr(
                importlib.import_module(
                    "classifiers.PointTransformers.models.{}.model".format(
                        args.model.name
                    )
                ),
                "PointTransformerCls",
            )(args)
    elif args.classifier == "CorAl":
        shutil.copy(
            hydra.utils.to_absolute_path("classifiers/regression.py"),
            ".",
        )
        classifier = getattr(
            importlib.import_module("classifiers.regression"),
            "LogisticRegression",
        )(input_dim=2, output_dim=1)

    start_epoch = 0
    if pretrained and args.load_model_path:
        checkpoint = torch.load(args.load_model_path)
        start_epoch = checkpoint["epoch"]
        classifier.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Use pretrain model")

    return classifier, start_epoch, args


def run_cls(args, logger, pretrained=True):
    PCAC_TRAIN_DATASET = PCAC_dataset(args=args, split="train")
    PCAC_VAL_DATASET = PCAC_dataset(args=args, split="validation")
    trainDataLoader = torch.utils.data.DataLoader(
        PCAC_TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=8
    )
    valDataLoader = torch.utils.data.DataLoader(
        PCAC_VAL_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=8
    )
    del PCAC_TRAIN_DATASET, PCAC_VAL_DATASET

    # MODEL LOADING
    classifier, start_epoch, args = load_best_model(args, logger, pretrained=pretrained)
    criterion = torch.nn.CrossEntropyLoss()

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_gamma
    )
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0
    mean_correct = []

    train_accuracies = np.empty((args.epoch))
    val_accuracies = np.empty((args.epoch))

    # TRANING
    logger.info("Start training...")
    for epoch in range(start_epoch, args.epoch):
        logger.info("Epoch %d (%d/%s):" % (global_epoch + 1, epoch + 1, args.epoch))
        classifier.train()
        for _, data in tqdm(
            enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9
        ):
            points, target, scene_numbers = data
            if torch.cuda.is_available():
                points, target = points.cuda(), target[:, 0].cuda()
            else:
                target = target[:, 0]

            # points = append_spatial_features(args, points)
            points = normalize_data_on_condition(args, points)
            points = augment_data(args, points)

            optimizer.zero_grad()

            pred = classifier(points, inference=False)
            loss = get_loss(criterion, pred, target.long(), args.lambda_lf)

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

            loss.backward()
            optimizer.step()
            global_step += 1

        scheduler.step()

        train_instance_acc = np.mean(mean_correct)
        logger.info("Train Instance Accuracy (augmented data): %f" % train_instance_acc)

        with torch.inference_mode():
            if args.plot_train_acc:
                instance_acc_train, _ = track_accuracy(
                    args, classifier.eval(), trainDataLoader
                )
                logger.info(
                    "Train Instance Accuracy (regular data): %f" % instance_acc_train
                )
                train_accuracies[epoch] = instance_acc_train

            instance_acc, class_acc = track_accuracy(
                args, classifier.eval(), valDataLoader
            )
            val_accuracies[epoch] = instance_acc

            if instance_acc >= best_instance_acc:
                best_epoch = epoch + 1
                best_instance_acc = instance_acc
                best_class_acc = class_acc

                logger.info("Save model...")
                savepath = "best_model_" + args.model_identifier + ".pth"
                logger.info("Saving at %s" % savepath)
                state = {
                    "epoch": best_epoch,
                    "instance_acc": instance_acc,
                    "class_acc": class_acc,
                    "model_state_dict": classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(state, savepath)
            logger.info(
                "Vali Instance Accuracy: %f, Class Accuracy: %f"
                % (instance_acc, class_acc)
            )
            logger.info(
                "Best Instance Accuracy: %f, Class Accuracy: %f"
                % (best_instance_acc, best_class_acc)
            )
            global_epoch += 1

    # Load the best validation model
    best_model_path = "best_model_" + args.model_identifier + ".pth"
    checkpoint = torch.load(best_model_path)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    return train_accuracies, val_accuracies, classifier


def run_test(args, logger, classifier):
    PCAC_TEST_DATASET = PCAC_dataset(args=args, split="test")
    testDataLoader = torch.utils.data.DataLoader(
        PCAC_TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=8
    )
    del PCAC_TEST_DATASET

    # Test best validation settings on test data
    test_instance_acc, test_class_acc, y_true, y_pred = test_results(
        args, classifier.eval(), testDataLoader
    )
    logger.info(f"Test Overall Accuracy: {test_instance_acc:.4f}")
    logger.info(f"Test Mean Class Accuracy: {test_class_acc:.4f}")
    logger.info("End of training...")
    return test_instance_acc, test_class_acc, y_true, y_pred


@hydra.main(config_path="config", config_name="cls")
def main(args):
    if args.debug:
        start_debug()
    omegaconf.OmegaConf.set_struct(args, False)

    # HYPER PARAMETER
    logger = logging.getLogger(__name__)

    # Ludvig's code
    # Init Nusc object

    for i in range(args.running_iterations):
        # # # Setup data for new run
        if i == 0:
            args.classifier = "CorAl"
            args.feature_folder = "/home/luddi824/thesis/PCAC/data/PCAC_data/coral0_3"
            args.perturb_settings.r_bin = 0.03
            args.perturb_settings.t_bin = 0.3
            args.model_identifier = "coral0_3"
        elif i == 1:
            args.classifier = "FACT"
            args.feature_folder = (
                "/home/luddi824/thesis/PCAC/data/PCAC_data/FACT_binary0_3"
            )
            args.perturb_settings.r_bin = 0.03
            args.perturb_settings.t_bin = 0.3
            args.model_identifier = "FACT_binary0_3"
        elif i == 2:
            args.classifier = "CorAl"
            args.feature_folder = "/home/luddi824/thesis/PCAC/data/PCAC_data/coral0_1"
            args.perturb_settings.r_bin = 0.01
            args.perturb_settings.t_bin = 0.1
            args.model_identifier = "coral0_1"
        elif i == 3:
            args.classifier = "FACT"
            args.feature_folder = (
                "/home/luddi824/thesis/PCAC/data/PCAC_data/FACT_binary0_1"
            )
            args.perturb_settings.r_bin = 0.01
            args.perturb_settings.t_bin = 0.1
            args.model_identifier = "FACT_binary0_1"
        else:
            sys.exit(f"Not supposed to be more than {args.running_iterations} runs")

        logger.info(f"STARTING RUN {i + 1}. Settings:")
        logger.info(f"args.classifier {args.classifier}")
        logger.info(f"args.feature_folder {args.feature_folder}")
        logger.info(f"args.perturb_settings.r_bin {args.perturb_settings.r_bin}")
        logger.info(f"args.perturb_settings.t_bin {args.perturb_settings.t_bin}")
        logger.info(f"args.re_use_data {args.re_use_data}")
        logger.info(f"args.model_identifier: {args.model_identifier}")
        assert args.classifier in [
            "CorAl",
            "FACT",
        ], "ERROR: Did not get a valid classifier!"

        # NOTICE THAT THE BELOW OVERWRITES SOME SETTINGS
        if args.classifier == "CorAl":
            args = setup_coral_args(args, logger)
        elif args.classifier == "FACT":
            args = setup_fact_args(args, logger)

        # EXTRACT FEATURES
        if not args.re_use_data:
            logger.info("Load dataset ...")
            nusc = ns.nuscenes.NuScenes(
                version=args.dataset, dataroot=args.data_folder, verbose=False
            )
            logger.info("Start feature extraction")
            with torch.inference_mode():
                if args.classifier == "CorAl":
                    X_train, y_train, X_val, y_val, X_test, y_test = get_coral_features(
                        nusc, args, logger
                    )
                elif args.classifier == "FACT":
                    # Get features
                    extract_features_to_txt_files(nusc, args=args)
            torch.cuda.empty_cache()

        # PERFORM MODEL TRAINING
        if args.classifier == "CorAl":
            if args.rerun_only_test:
                classifier, _, args = load_best_model(args, logger)
                X_test = np.loadtxt(args.feature_folder + "/X_test.txt", delimiter=" ")
                y_test = np.loadtxt(args.feature_folder + "/y_test.txt", delimiter=" ")
            else:
                classifier = perform_coral_training(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    logger,
                    epochs=args.coral_settings.epochs,
                    learning_rate=args.coral_settings.learning_rate,
                )
        elif args.classifier == "FACT":
            # Get dataset (features in a data loader)
            args = process_features(args)
            # Run all
            if args.ablation.run_ablation:
                run_ablation_features(args, logger)
                sys.exit("Ablation finished!")
            else:
                if args.rerun_only_test:
                    classifier, _, args = load_best_model(args, logger)
                else:
                    train_accuracies, val_accuracies, classifier = run_cls(args, logger)

        # PERFORM MODEL INFERENCE
        if args.classifier == "CorAl":
            accuracy_test, y_pred = perform_coral_inference(X_test, y_test, classifier)
        elif args.classifier == "FACT":
            _, _, y_test, y_pred = run_test(args, logger, classifier)

        # VISUALIZE RESULTS
        if args.classifier == "CorAl":
            logger.info(f"Accuracy CorAl {accuracy_test:.4f}")
            model_plot(
                classifier,
                X_test,
                y_test,
                args=args,
                title=f"CorAl: {100*accuracy_test:.1f}% accuracy",
            )
            n_classes = 2
        elif args.classifier == "FACT":
            plot_accuracies(train_accuracies, val_accuracies, args=args)
            n_classes = args.perturb_settings.n_classes

        store_confusion_matrix(
            y_pred, y_test, N_classes=n_classes, logger=logger, args=args
        )

        # LOGGING
        logger.info(f"FINISHED RUN {i + 1}. Settings:")
        logger.info(f"args.classifier {args.classifier}")
        logger.info(f"args.feature_folder {args.feature_folder}")
        logger.info(f"args.perturb_settings.r_bin {args.perturb_settings.r_bin}")
        logger.info(f"args.perturb_settings.t_bin {args.perturb_settings.t_bin}")
        logger.info(f"args.re_use_data {args.re_use_data}")
        logger.info(f"args.model_identifier: {args.model_identifier}")


if __name__ == "__main__":
    main()
