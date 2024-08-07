# """
# Scripts for pairwise registration demo

# Author: Shengyu Huang
# Last modified: 22.02.2021
# """
# import os, torch, sys
# from pathlib import Path
# import numpy as np
# from easydict import EasyDict as edict
# from torch.utils.data import Dataset

# cwd = os.getcwd()
# sys.path.append(cwd)
# from OverlapPredator.datasets.indoor import IndoorDataset
# from OverlapPredator.datasets.dataloader import get_dataloader
# from OverlapPredator.models.architectures import KPFCNN
# from OverlapPredator.lib.utils import load_obj, setup_seed, load_config
# from OverlapPredator.lib.benchmark_utils import ransac_pose_estimation

# setup_seed(0)


# class ThreeDMatchDemo(Dataset):
#     """
#     Load subsampled coordinates, relative rotation and translation
#     Output(torch.Tensor):
#         src_pcd:        [N,3]
#         tgt_pcd:        [M,3]
#         rot:            [3,3]
#         trans:          [3,1]
#     """
#     def __init__(self,config, src_path, tgt_path):
#         super(ThreeDMatchDemo,self).__init__()
#         self.config = config
#         self.src_path = src_path
#         self.tgt_path = tgt_path

#     def __len__(self):
#         return 1

#     def __getitem__(self, item):
#         # get pointcloud
#         src_pcd = torch.load(self.src_path).astype(np.float32)
#         tgt_pcd = torch.load(self.tgt_path).astype(np.float32)   
#         src_feats=np.ones_like(src_pcd[:,:1]).astype(np.float32)
#         tgt_feats=np.ones_like(tgt_pcd[:,:1]).astype(np.float32)

#         # fake the ground truth information
#         rot = np.eye(3).astype(np.float32)
#         trans = np.ones((3,1)).astype(np.float32)
#         correspondences = torch.ones(1,2).long()

#         return (src_pcd, tgt_pcd, src_feats, tgt_feats, rot, trans, correspondences,
#                 src_pcd, tgt_pcd, torch.ones(1))


# def lighter(color, percent):
#     '''assumes color is rgb between (0, 0, 0) and (1,1,1)'''
#     color = np.array(color)
#     white = np.array([1, 1, 1])
#     vector = white-color
#     return color + vector * percent


# def main(config, demo_loader):
#     config.model.eval()
#     c_loader_iter = demo_loader.__iter__()
#     with torch.no_grad():
#         inputs = next(c_loader_iter)
#         ##################################
#         # load inputs to device.
#         for k, v in inputs.items():
#             if type(v) == list:
#                 inputs[k] = [item.to(config.device) for item in v]
#             else:
#                 inputs[k] = v.to(config.device)

#         ###############################################
#         # forward pass
#         feats, scores_overlap, scores_saliency = config.model(inputs)  #[N1, C1], [N2, C2]
#         pcd = inputs['points'][0]
#         len_src = inputs['stack_lengths'][0][0]

#         src_pcd, tgt_pcd = pcd[:len_src], pcd[len_src:]
#         src_feats, tgt_feats = feats[:len_src].detach().cpu(), feats[len_src:].detach().cpu()
#         src_overlap = scores_overlap[:len_src].detach().cpu()
#         src_saliency = scores_saliency[:len_src].detach().cpu()
#         tgt_overlap = scores_overlap[len_src:].detach().cpu()
#         tgt_saliency = scores_saliency[len_src:].detach().cpu()

#         ########################################
#         # do probabilistic sampling guided by the score
#         src_scores = src_overlap * src_saliency
#         tgt_scores = tgt_overlap * tgt_saliency

#         # Downsample pointcloud (TODO: possibly remove ...)
#         if src_pcd.size(0) > config.n_points:
#             idx = np.arange(src_pcd.size(0))
#             probs = (src_scores / src_scores.sum()).numpy().flatten()
#             idx = np.random.choice(idx, size= config.n_points, replace=False, p=probs)
#             src_pcd, src_feats = src_pcd[idx], src_feats[idx]
#         if tgt_pcd.size(0) > config.n_points:
#             idx = np.arange(tgt_pcd.size(0))
#             probs = (tgt_scores / tgt_scores.sum()).numpy().flatten()
#             idx = np.random.choice(idx, size= config.n_points, replace=False, p=probs)
#             tgt_pcd, tgt_feats = tgt_pcd[idx], tgt_feats[idx]

#         ########################################
#         # run ransac and draw registration
#         tsfm = ransac_pose_estimation(src_pcd, tgt_pcd, src_feats, tgt_feats, mutual=False)
#     return tsfm


# def setup_op_registration(config_path="OverlapPredator/configs/test/kitti.yaml"):
#     # load configs
#     # TODO: Perhaps it's better to have the KITTI configs (weights) since nuScenes is outdoors
#     path = str(Path(__file__).parent.parent) + "/" + config_path
#     config = load_config(path)
#     config = edict(config)
#     if config.gpu_mode:
#         config.device = torch.device('cuda')
#     else:
#         config.device = torch.device('cpu')

#     # model initialization
#     config.architecture = [
#         'simple',
#         'resnetb',
#     ]
#     for i in range(config.num_layers-1):
#         config.architecture.append('resnetb_strided')
#         config.architecture.append('resnetb')
#         config.architecture.append('resnetb')
#     for i in range(config.num_layers-2):
#         config.architecture.append('nearest_upsample')
#         config.architecture.append('unary')
#     config.architecture.append('nearest_upsample')
#     config.architecture.append('last_unary')
#     config.model = KPFCNN(config).to(config.device)

#     # load pretrained weights
#     assert config.pretrain != None
#     weight_path = str(Path(__file__).parent.parent) + "/OverlapPredator/" + config.pretrain
#     state = torch.load(weight_path)
#     config.model.load_state_dict(state['state_dict'])
#     return config


# def get_neighborhood_limits(config):
#     # Create dataset and dataloader
#     train_path = str(Path(__file__).parent.parent) + "/OverlapPredator/" + config.train_info
#     info_train = load_obj(train_path)
#     train_set = IndoorDataset(info_train,config,data_augmentation=True)
#     _, neighborhood_limits = get_dataloader(dataset=train_set,
#                                             batch_size=config.batch_size,
#                                             shuffle=True,
#                                             num_workers=config.num_workers)
#     return neighborhood_limits


# def get_pair_loader(config, neighborhood_limits, src_pth, tgt_pth):
#     demo_set = ThreeDMatchDemo(config, src_pth, tgt_pth)
#     demo_loader, _ = get_dataloader(dataset=demo_set, batch_size=config.batch_size, shuffle=False,
#                                     num_workers=1, neighborhood_limits=neighborhood_limits)
#     return demo_loader


# if __name__ == '__main__':
#     # Setup config file
#     config = setup_op_registration()
#     neighborhood_limits = get_neighborhood_limits(config)
#     demo_loader = get_pair_loader(config, neighborhood_limits)
#     # Do pose estimation
#     tsfm = main(config, demo_loader)
