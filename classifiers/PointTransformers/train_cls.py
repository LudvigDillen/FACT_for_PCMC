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
from features.feature_extractor import extract_features_to_txt_files, number_of_features
import classifiers.PointTransformers.provider as provider
from utils.other import start_debug
from utils.pointclouds import PCAC_dataset
from visualization.classifications import plot_accuracies


def test(model, loader, num_class=2):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
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


@hydra.main(config_path='config', config_name='cls')
def main(args):
    start_debug()
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    logger = logging.getLogger(__name__)

    '''DATA LOADING'''
    logger.info('Load dataset ...')

    # Ludvig's code
    # Init Nusc object
    extract_features = True
    if extract_features:
        data_folder = '/home/luddi824/thesis/PCAC/data/nuscenes/'
        version = 'v1.0-mini'
        nusc = ns.nuscenes.NuScenes(version=version, dataroot=data_folder, verbose=False)

        # Get features
        extract_features_to_txt_files(nusc, features=args.features, n_scenes=5,
                                      n_samples_per_scene=1, N_fps_points=args.num_point)
        args.input_dim = number_of_features(args.features)
        torch.cuda.empty_cache()
        del nusc
    else:
        args.input_dim = 8

    # Get dataset (features in a data loader)
    PCAC_TRAIN_DATASET = PCAC_dataset(split='train')
    PCAC_TEST_DATASET = PCAC_dataset(split='test')
    trainDataLoader = torch.utils.data.DataLoader(PCAC_TRAIN_DATASET, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(PCAC_TEST_DATASET, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=4)
    del PCAC_TRAIN_DATASET, PCAC_TEST_DATASET

    '''MODEL LOADING'''
    args.num_class = 2  # aligned or misaligned

    shutil.copy(hydra.utils.to_absolute_path(
        'classifiers/PointTransformers/models/{}/model.py'.format(args.model.name)),
                '.')
    classifier = getattr(importlib.import_module(
        'classifiers.PointTransformers.models.{}.model'.format(args.model.name)),
                         'PointTransformerCls')(args).cuda()
    criterion = torch.nn.CrossEntropyLoss()

    try:
        checkpoint = torch.load('best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except FileNotFoundError as e:
        logger.info(f'Error loading model: {e}')
        logger.info('No existing model, starting training from scratch...')
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

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)
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
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            target = target[:, 0]

            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            pred = classifier(points)
            loss = criterion(pred, target.long())
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

            torch.cuda.empty_cache()
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        train_accuracies[epoch] = train_instance_acc

        scheduler.step()

        train_instance_acc = np.mean(mean_correct)
        logger.info('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=args.num_class)
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

    logger.info('End of training...')
    plot_accuracies(train_accuracies, val_accuracies)


if __name__ == '__main__':
    main()
