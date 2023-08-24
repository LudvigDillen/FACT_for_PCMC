import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.pointnet_util import index_points, square_distance


class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        # Get index to nearest neighbors
        knn_idx = square_distance(xyz, xyz).argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)  # kN neighbors

        pre = features  # residual connection
        x = self.fc1(features)  # first fully connected

        # positional enc. + MLP
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f
        del xyz, knn_xyz

        # embed features
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
        del knn_idx, x

        v += pos_enc  # right part of sum

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)  # gamma of key-query+rel_enc

        normalizer = np.sqrt(k.size(-1))
        del k, q, pos_enc

        attn = F.softmax(attn / normalizer, dim=-2)  # b x n x k x f, rho
        # sum over all kNN and perform Hadamard product over attention and values
        res = torch.einsum('bmnf,bmnf->bmf', attn, v)
        del attn, v

        res = self.fc2(res) + pre  # second fully connected + residual connection
        return res
