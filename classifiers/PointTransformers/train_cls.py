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
from features.feature_utils import (process_features, run_ablation_features, number_of_features,
                                    augment_data, append_spatial_features,
                                    normalize_data_on_condition)
from utils.other import start_debug
from utils.pointclouds import PCAC_dataset
from visualization.classifications import plot_accuracies, store_confusion_matrix
from classifiers.loss_functions import get_loss
from classifiers.coral import coral


def track_accuracy(args, model, loader):
    mean_correct = []
    class_acc = np.zeros((args.num_class, 3))
    for data in tqdm(loader, total=len(loader)):
        points, target = data
        points, target = points.cuda(), target[:, 0].cuda()
        points = append_spatial_features(args, points)
        points = normalize_data_on_condition(args, points)

        classifier = model.eval()
        pred = classifier(points)  # [B, n_classes], here we have a score for each class
        pred_choice = pred.data.max(1)[1]  # highest score wins

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item()/float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def test_results(args, model, loader):
    mean_correct = []
    class_acc = np.zeros((args.num_class, 3))
    N_samples = len(loader.dataset)
    y_true = np.zeros(N_samples)
    y_pred = np.zeros(N_samples)
    ind = 0
    with torch.inference_mode():
        for data in tqdm(loader, total=len(loader)):
            points, target = data
            target = target[:, 0]

            current_batch_size = len(target)
            y_true[ind:ind + current_batch_size] = target.numpy()

            points, target = points.cuda(), target.cuda()
            points = append_spatial_features(args, points)
            points = normalize_data_on_condition(args, points)

            classifier = model.eval()
            pred = classifier(points)  # [B, n_classes], here we have a score for each class
            pred_choice = pred.data.max(1)[1]  # highest score wins

            y_pred[ind:ind + current_batch_size] = pred_choice.cpu().numpy()
            ind = ind + current_batch_size

            for cat in np.unique(target.cpu()):
                classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                class_acc[cat, 0] += classacc.item()/float(points[target == cat].size()[0])
                class_acc[cat, 1] += 1
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc, y_true, y_pred


def load_best_model(args, logger, pretrained=True):
    args.num_class = args.perturb_settings.n_classes
    args.input_dim, args = number_of_features(args)
    print(f"Input dim: {args.input_dim}")

    shutil.copy(hydra.utils.to_absolute_path(
        'classifiers/PointTransformers/models/{}/model.py'.format(args.model.name)),
                '.')
    classifier = getattr(importlib.import_module(
        'classifiers.PointTransformers.models.{}.model'.format(args.model.name)),
                         'PointTransformerCls')(args).cuda()

    start_epoch = 0
    if pretrained and args.load_model_path:
        checkpoint = torch.load(args.load_model_path)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')

    return classifier, start_epoch, args


# TODO: Maybe add functionality so that we can extract one feature at the time, and
# add this to previous extracted features. Note that we then have to save the
# pointcloud (the distorted versions) which should be unfeasible.
def run_cls(args, logger, pretrained=True):
    PCAC_TRAIN_DATASET = PCAC_dataset(args=args, split='train')
    PCAC_VAL_DATASET = PCAC_dataset(args=args, split='validation')
    trainDataLoader = torch.utils.data.DataLoader(PCAC_TRAIN_DATASET, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=8)
    valDataLoader = torch.utils.data.DataLoader(PCAC_VAL_DATASET, batch_size=args.batch_size,
                                                shuffle=False, num_workers=8)
    del PCAC_TRAIN_DATASET, PCAC_VAL_DATASET

    # MODEL LOADING
    classifier, start_epoch, args = load_best_model(args, logger, pretrained=pretrained)
    criterion = torch.nn.CrossEntropyLoss()

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0
    mean_correct = []

    train_accuracies = np.empty((args.epoch))
    val_accuracies = np.empty((args.epoch))

    # TRANING
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        classifier.train()
        for _, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points, target = points.cuda(), target[:, 0].cuda()

            # TODO: We actually should do not need to append the features and normalize the
            # data all the time, we could do that before instead. But currently, I am post-poning
            # this a bit.
            points = append_spatial_features(args, points)
            points = normalize_data_on_condition(args, points)
            points = augment_data(args, points)

            optimizer.zero_grad()

            pred = classifier(points)
            loss = get_loss(criterion, pred, target.long(), args.lambda_lf)

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

            loss.backward()
            optimizer.step()
            global_step += 1

        scheduler.step()

        train_instance_acc = np.mean(mean_correct)
        logger.info('Train Instance Accuracy (augmented data): %f' % train_instance_acc)

        with torch.inference_mode():
            if args.plot_train_acc:
                instance_acc_train, _ = track_accuracy(args, classifier.eval(), trainDataLoader)
                logger.info('Train Instance Accuracy (regular data): %f' % instance_acc_train)
                train_accuracies[epoch] = instance_acc_train

            instance_acc, class_acc = track_accuracy(args, classifier.eval(), valDataLoader)
            val_accuracies[epoch] = instance_acc

            if instance_acc >= best_instance_acc:
                best_epoch = epoch + 1
                best_instance_acc = instance_acc
                best_class_acc = class_acc

                logger.info('Save model...')
                savepath = 'best_model_' + args.model_identifier + '.pth'
                logger.info('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            logger.info('Vali Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc,
                                                                            class_acc))
            logger.info('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc,
                                                                            best_class_acc))
            global_epoch += 1

    # Load the best validation model
    best_model_path = 'best_model_' + args.model_identifier + '.pth'
    checkpoint = torch.load(best_model_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    return train_accuracies, val_accuracies, classifier


def run_test(args, logger, classifier):
    PCAC_TEST_DATASET = PCAC_dataset(args=args, split='test')
    testDataLoader = torch.utils.data.DataLoader(PCAC_TEST_DATASET, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=8)
    del PCAC_TEST_DATASET

    # Test best validation settings on test data
    test_instance_acc, test_class_acc, y_true, y_pred = test_results(
        args, classifier.eval(), testDataLoader)
    logger.info(f'Test Overall Accuracy: {test_instance_acc}')
    logger.info(f'Test Mean Class Accuracy: {test_class_acc}')
    logger.info('End of training...')
    return test_instance_acc, test_class_acc, y_true, y_pred


@hydra.main(config_path='config', config_name='cls')
def main(args):
    if args.debug:
        start_debug()
    omegaconf.OmegaConf.set_struct(args, False)

    # HYPER PARAMETER
    logger = logging.getLogger(__name__)

    # DATA LOADING
    logger.info('Load dataset ...')

    # Ludvig's code
    # Init Nusc object
    if not args.re_use_data:
        print("Start feature extraction", flush=True)
        nusc = ns.nuscenes.NuScenes(version=args.dataset, dataroot=args.data_folder, verbose=False)

    for i in range(args.running_iterations):
        # # Setup data for new run
        # sys.exit(f"Not supposed to be more than {args.running_iterations} runs")

        logger.info(f"STARTING RUN {i + 1}. Settings:")
        logger.info(f"args.features_to_create.use_csj {args.features_to_create.use_csj}")
        logger.info(f"args.features_to_use.use_csj {args.features_to_use.use_csj}")
        logger.info(f"args.model_identifier: {args.model_identifier}")
        ##

        if not args.re_use_data:
            if args.classifier == "CorAl":
                coral(nusc, args, logger)

            # Get features
            with torch.inference_mode():
                extract_features_to_txt_files(nusc, args=args)
            torch.cuda.empty_cache()
        args.n_samples = args.n_scenes*args.n_samples_per_scene
        # Get dataset (features in a data loader)
        args = process_features(args)
        # Run all
        if args.ablation:
            run_ablation_features(args, logger)
        else:
            if args.rerun_only_test is False:
                train_accuracies, val_accuracies, classifier = run_cls(args, logger)
                plot_accuracies(train_accuracies, val_accuracies, args.plot_train_acc,
                                args.model_identifier)
            else:
                classifier, _, args = load_best_model(args, logger)

            _, _, y_true, y_pred = run_test(args, logger, classifier)
            store_confusion_matrix(y_pred, y_true, N_classes=args.perturb_settings.n_classes,
                                   logger=logger, model_identifier=args.model_identifier)
        logger.info(f"FINISHED RUN {i + 1}. Settings:")
        logger.info(f"args.features_to_create.use_csj {args.features_to_create.use_csj}")
        logger.info(f"args.features_to_use.use_csj {args.features_to_use.use_csj}")
        logger.info(f"args.model_identifier: {args.model_identifier}")


if __name__ == '__main__':
    main()
