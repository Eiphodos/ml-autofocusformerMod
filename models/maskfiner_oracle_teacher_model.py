# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from typing import Dict
import random
from einops import rearrange

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class OracleTeacherBackbone(nn.Module):
    def __init__(self, backbones, backbone_dims, out_dim, all_out_features, n_scales, num_classes):
        super().__init__()
        self.backbones = nn.ModuleList(backbones)
        self.out_dim = out_dim
        self.backbone_dims = backbone_dims
        self.all_out_features = all_out_features
        self.all_out_features_scales = {k: len(all_out_features) - i - 1 for i, k in enumerate(all_out_features)}
        self.n_scales = n_scales
        self.num_classes = num_classes

        feat_projs = []
        feat_norms = []
        for i in range(len(self.backbones)):
            scale_projs = []
            scale_norms = []
            for j in range(len(self.backbones[i]._out_features) - 1):
                f_proj = nn.Linear(backbone_dims[i], backbone_dims[j])
                f_norm = nn.LayerNorm(backbone_dims[j])
                scale_projs.append(f_proj)
                scale_norms.append(f_norm)
            scale_projs = nn.ModuleList(scale_projs)
            scale_norms = nn.ModuleList(scale_norms)
            feat_projs.append(scale_projs)
            feat_norms.append(scale_norms)
        self.feat_proj = nn.ModuleList(feat_projs)
        self.feat_norm = nn.ModuleList(feat_norms)

        #self.head = nn.Linear(out_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.head = MLP(sum(self.backbone_dims), sum(self.backbone_dims) // 2, num_classes, num_layers=3)

        '''
        out_projs = []
        for dim in backbone_dims:
            out_projs.append(nn.Sequential(
                        nn.Linear(dim, out_dim, bias=True),
                        nn.LayerNorm(out_dim)
                    ))
            nn.init.xavier_uniform_(out_projs[-1][0].weight, gain=1)
            nn.init.constant_(out_projs[-1][0].bias, 0)
        self.out_proj = nn.ModuleList(out_projs)
        '''
        '''
        upsamplers = []
        for i in range(len(self.backbones) - 1):
            upsample_out = MLP(backbone_dims[i], backbone_dims[i], 1, num_layers=3)
            upsamplers.append(upsample_out)
        self.upsamplers = nn.ModuleList(upsamplers)
        '''



        print("Successfully built OracleTeacherBackbone model!")


    def forward(self, im):
        upsampling_mask = None
        features = None
        features_pos = None
        outs = {}
        for scale in range(len(self.backbones)):
            output = self.backbones[scale](im, scale, features, features_pos, upsampling_mask)
            all_out_features = self.backbones[scale]._out_features
            all_feat = []
            all_scale = []
            all_pos = []
            all_ss = []
            for i, f in enumerate(all_out_features):
                feat = output[f]
                feat_pos = output[f + '_pos']
                feat_scale = output[f + '_scale']
                feat_ss = output[f + '_spatial_shape']
                curr_scale = self.all_out_features_scales[f]

                B, N, C = feat.shape

                #print("Output {} for scale {}: feat_shape: {}, pos_shape: {}, scale_shape: {}, spatial_shape: {}".format(f, scale, feat.shape, feat_pos.shape, feat_scale.shape, feat_ss))


                if f + '_pos' in outs:
                    pos_indices = self.find_pos_org_order(outs[f + '_pos'], feat_pos)
                    print("Shuffled pos: {}".format(feat_pos[0, 0:10, :]))
                    b_ = torch.arange(B).unsqueeze(-1).expand(-1, N)
                    feat = feat[b_, pos_indices]
                    feat_pos = feat_pos[b_, pos_indices]
                    feat_scale = feat_scale[b_, pos_indices]
                    print("Original pos: {}".format(outs[f + '_pos'][0,0:10,:]))
                    print("Ordered pos: {}".format(feat_pos[0, 0:10, :]))
                    assert (outs[f + '_pos'] == feat_pos).all()
                    orig_dtype = feat.dtype
                    outs[f] = outs[f] + self.feat_norm[scale][curr_scale](self.feat_proj[scale][curr_scale](feat).float()).to(orig_dtype)
                else:
                    outs[f] = feat
                    outs[f + '_pos'] = feat_pos
                    outs[f + '_scale'] = feat_scale
                    outs[f + '_spatial_shape'] = feat_ss

                all_feat.append(feat)
                all_pos.append(feat_pos)
                all_scale.append(feat_scale)
                all_ss.append(feat_ss)

            if scale < len(self.backbones) - 1:
                #B, N, C = all_feat[0].shape
                upsampling_mask = self.generate_random_upsampling_mask(B, N)
                #upsampling_mask = self.upsamplers[scale](all_feat[0]).squeeze(-1)

            #print("Upsampling mask for scale {}: pred: {}, oracle: {}".format(scale, upsampling_mask_pred.shape, upsampling_mask_oracle.shape))

            all_pos = torch.cat(all_pos, dim=1)
            all_scale = torch.cat(all_scale, dim=1)
            features_pos = torch.cat([all_scale.unsqueeze(2), all_pos], dim=2)
            features = torch.cat(all_feat, dim=1)
        outs['min_spatial_shape'] = output['min_spatial_shape']

        out_scale_vectors = []
        for i, f in enumerate(all_out_features[::-1]):
            feat = outs[f]
            pooled = feat.mean(1)
            out_scale_vectors.append(pooled)
        out_scale_vectors = torch.cat(out_scale_vectors, dim=1)
        out = self.head(out_scale_vectors)

        return out


    def generate_random_upsampling_mask(self, batch_size, n_tokens):
        upsampling_mask = torch.randn(batch_size, n_tokens).float().to('cuda')
        return upsampling_mask
    def find_pos_org_order(self, pos_org, pos_shuffled):
        dists = torch.cdist(pos_org.float(), pos_shuffled.float(), p=1)  # Manhattan distance
        pos_indices = torch.argmin(dists, dim=2)  # (B, N_)

        return pos_indices