"""
Author: Benny
Date: Nov 2019
"""
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
from features.feature_utils import process_features, run_ablation_features, number_of_features
import classifiers.PointTransformers.provider as provider
from utils.other import start_debug
from utils.pointclouds import PCAC_dataset
from visualization.classifications import plot_accuracies, store_confusion_matrix


def track_accuracy(model, loader, num_class):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    for data in tqdm(loader, total=len(loader)):
        points, target = data
        target = target[:, 0]

        points, target = points.cuda(), target.cuda()
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


def test_results(model, loader, num_class):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    N_samples = len(loader.dataset)
    y_true = np.zeros(N_samples)
    y_pred = np.zeros(N_samples)
    ind = 0
    for data in tqdm(loader, total=len(loader)):
        points, target = data
        target = target[:, 0]

        current_batch_size = len(target)
        y_true[ind:ind + current_batch_size] = target.numpy()

        points, target = points.cuda(), target.cuda()
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


# TODO: Maybe add functionality so that we can extract one feature at the time, and
# add this to previous extracted features. Note that we then have to save the
# pointcloud (the distorted versions) which should be unfeasible.
def run_cls(n_samples, feature_filter, args, logger, pretrained=True):
    PCAC_TRAIN_DATASET = PCAC_dataset(n_samples=n_samples, root=args.feature_folder, split='train',
                                      feature_filter=feature_filter, train_ratio=args.train_ratio)
    PCAC_VAL_DATASET = PCAC_dataset(n_samples=n_samples, root=args.feature_folder, split='validation',
                                    feature_filter=feature_filter, train_ratio=args.train_ratio)
    PCAC_TEST_DATASET = PCAC_dataset(n_samples=n_samples, root=args.feature_folder, split='test',
                                     feature_filter=feature_filter, train_ratio=args.train_ratio)

    trainDataLoader = torch.utils.data.DataLoader(PCAC_TRAIN_DATASET, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=4)
    valDataLoader = torch.utils.data.DataLoader(PCAC_VAL_DATASET, batch_size=args.batch_size,
                                                shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(PCAC_TEST_DATASET, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=4)
    del PCAC_TRAIN_DATASET, PCAC_VAL_DATASET, PCAC_TEST_DATASET

    '''MODEL LOADING'''
    args.num_class = args.perturb_settings.n_classes
    args.input_dim = number_of_features(feature_filter)
    print(f"Input dim: {args.input_dim}")

    shutil.copy(hydra.utils.to_absolute_path(
        'classifiers/PointTransformers/models/{}/model.py'.format(args.model.name)),
                '.')
    classifier = getattr(importlib.import_module(
        'classifiers.PointTransformers.models.{}.model'.format(args.model.name)),
                         'PointTransformerCls')(args).cuda()
    criterion = torch.nn.CrossEntropyLoss()

    if pretrained:
        try:
            if args.load_model_path:
                checkpoint = torch.load(args.load_model_path)
            else:
                checkpoint = torch.load('best_model.pth')
            start_epoch = checkpoint['epoch']
            classifier.load_state_dict(checkpoint['model_state_dict'])
            logger.info('Use pretrain model')
        except FileNotFoundError as e:
            logger.info(f'Error loading model: {e}')
            logger.info('No existing model, starting training from scratch...')
            start_epoch = 0
    else:
        start_epoch = 0

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

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.85)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0
    mean_correct = []

    train_accuracies = np.empty((args.epoch))
    val_accuracies = np.empty((args.epoch))

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        classifier.train()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            # TODO: How to do with the augmentation?
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            target = target[:, 0]

            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            pred = classifier(points)
            loss = criterion(pred, target.long())
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

            # torch.cuda.empty_cache()
            loss.backward()
            optimizer.step()
            global_step += 1

        scheduler.step()

        train_instance_acc = np.mean(mean_correct)
        logger.info('Train Instance Accuracy (augmented data): %f' % train_instance_acc)

        with torch.no_grad():
            if args.plot_train_acc:
                instance_acc_train, class_acc_train = track_accuracy(classifier.eval(), trainDataLoader,
                                                                     num_class=args.num_class)
                logger.info('Train Instance Accuracy (regular data): %f' % instance_acc_train)
                train_accuracies[epoch] = instance_acc_train

            instance_acc, class_acc = track_accuracy(classifier.eval(), valDataLoader,
                                                     num_class=args.num_class)
            val_accuracies[epoch] = instance_acc

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            logger.info('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            logger.info('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc,
                                                                            best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = 'best_model.pth'
                logger.info('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    # Load the best validation model
    best_model_path = 'best_model.pth'
    checkpoint = torch.load(best_model_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])

    # Test best validation settings on test data
    test_instance_acc, test_class_acc, y_true, y_pred = test_results(classifier.eval(), testDataLoader,
                                                                     num_class=args.num_class)
    logger.info(f'Test Overall Accuracy: {test_instance_acc}')
    logger.info(f'Test Mean Class Accuracy: {test_class_acc}')
    logger.info('End of training...')
    return train_accuracies, val_accuracies, test_instance_acc, test_class_acc, y_true, y_pred


@hydra.main(config_path='config', config_name='cls')
def main(args):
    if args.debug:
        start_debug()
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    logger = logging.getLogger(__name__)

    '''DATA LOADING'''
    logger.info('Load dataset ...')

    # Ludvig's code
    # Init Nusc object
    if not args.re_use_data:
        print("Start feature extraction", flush=True)
        data_folder = '/home/luddi824/thesis/PCAC/data/nuscenes/'
        version = args.dataset
        nusc = ns.nuscenes.NuScenes(version=version, dataroot=data_folder, verbose=False)

        # Get features
        extract_features_to_txt_files(nusc, args=args)
        torch.cuda.empty_cache()
        del nusc
    n_samples = args.n_scenes*args.n_samples_per_scene
    # Get dataset (features in a data loader)
    feature_filter = process_features(args.features_to_use, args.features_to_create)

    # Run all
    if args.ablation:
        run_ablation_features(n_samples, feature_filter, args, logger)
    else:
        train_accuracies, val_accuracies, test_instance_acc, test_class_acc, y_true, y_pred =\
            run_cls(n_samples, feature_filter, args, logger)
        plot_accuracies(train_accuracies, val_accuracies, args.plot_train_acc)
        # TODO: Have test set to do confusion matrix, and final accuracy result on as well ...
        store_confusion_matrix(y_pred, y_true, N_classes=args.perturb_settings.n_classes,
                               logger=logger)


if __name__ == '__main__':
    main()
