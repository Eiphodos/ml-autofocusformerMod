# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from typing import Dict
import random
from einops import rearrange

import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

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
        final_upsampling_ratios = []
        for b in self.backbones:
            final_upsampling_ratios.append(b.upscale_ratio)
        self.final_upsampling_ratios = final_upsampling_ratios
        self.out_dim = out_dim
        self.backbone_dims = backbone_dims
        self.all_out_features = all_out_features
        self.all_out_features_scales = {k: len(all_out_features) - i - 1 for i, k in enumerate(all_out_features)}
        self.n_scales = n_scales
        self.num_classes = num_classes

        #self.head = nn.Linear(out_dim, num_classes) if num_classes > 0 else nn.Identity()
        '''
        head_projs = []
        d_out = []
        for i in range(self.n_scales):
            d = sum(self.backbone_dims[i:])
            do = d // 4
            head_proj = nn.Linear(d, do)
            head_projs.append(head_proj)
            d_out.append(do)
        self.head_projs = nn.ModuleList(head_projs)
        '''
        tot_out_dim = backbone_dims[-1] * self.n_scales
        self.head_norm = nn.LayerNorm(tot_out_dim)
        self.head = MLP(tot_out_dim, tot_out_dim, num_classes, num_layers=3)
        #self.head = nn.Linear(tot_out_dim, num_classes)

        self.apply(self._init_weights)

        print("Successfully built OracleTeacherBackbone model!")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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
                '''
                if f + '_pos' in outs:
                    pos_indices = self.find_pos_org_order(outs[f + '_pos'], feat_pos)
                    b_ = torch.arange(B).unsqueeze(-1).expand(-1, N)
                    feat = feat[b_, pos_indices]
                    feat_pos = feat_pos[b_, pos_indices]
                    feat_scale = feat_scale[b_, pos_indices]
                    assert (outs[f + '_pos'] == feat_pos).all()
                    outs[f] = torch.cat([outs[f], feat], dim=2)
                else:
                '''
                outs[f] = feat
                outs[f + '_pos'] = feat_pos
                outs[f + '_scale'] = feat_scale
                outs[f + '_spatial_shape'] = feat_ss

                all_feat.append(feat)
                all_pos.append(feat_pos)
                all_scale.append(feat_scale)
                all_ss.append(feat_ss)

            if scale < len(self.backbones) - 1:
                B, N, C = all_feat[0].shape
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
            #projed = self.head_projs[i](pooled)
            out_scale_vectors.append(pooled)
        out_scale_vectors = torch.cat(out_scale_vectors, dim=1)
        out_scale_vectors = self.head_norm(out_scale_vectors)
        #out_scale_vectors = nn.functional.gelu(out_scale_vectors)
        out = self.head(out_scale_vectors)

        return out


    def generate_random_upsampling_mask(self, batch_size, n_tokens):
        upsampling_mask = torch.randn(batch_size, n_tokens).float().to('cuda')
        return upsampling_mask
    def find_pos_org_order(self, pos_org, pos_shuffled):
        dists = torch.cdist(pos_org.float(), pos_shuffled.float(), p=1)  # Manhattan distance
        pos_indices = torch.argmin(dists, dim=2)  # (B, N_)

        return pos_indices

    def generate_max_norm_upsampling_mask(self, features):
        upsampling_mask = features.norm(dim=2)
        return upsampling_mask