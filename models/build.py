# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Adapted for AutoFocusFormer by Ziwen 2023

import torch
from .aff_transformer import AutoFocusFormer
from .maskfiner_oracle_teacher_model import OracleTeacherBackbone
from .maskfiner_up_down import UpDownBackbone
from .mixres_vit import MixResViT
from .mixres_neighbour import MixResNeighbour

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'aff':
        model = AutoFocusFormer(in_chans=config.DATA.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.AFF.EMBED_DIM,
                                cluster_size=config.MODEL.AFF.CLUSTER_SIZE,
                                nbhd_size=config.MODEL.AFF.NBHD_SIZE,
                                alpha=config.MODEL.AFF.ALPHA,
                                ds_rate=config.MODEL.AFF.DS_RATE,
                                reserve_on=config.MODEL.AFF.RESERVE,
                                depths=config.MODEL.AFF.DEPTHS,
                                num_heads=config.MODEL.AFF.NUM_HEADS,
                                mlp_ratio=config.MODEL.AFF.MLP_RATIO,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                patch_norm=config.MODEL.AFF.PATCH_NORM,
                                layer_scale=config.MODEL.AFF.LAYER_SCALE,
                                img_size=config.DATA.IMG_SIZE)
    elif model_type == 'maskfinerOT':
        backbones = []
        for layer_index, m in enumerate(config.MODEL.MR.NAME):
            if layer_index == 0:
                in_chans = 3
            else:
                in_chans = config.MODEL.MR.EMBED_DIM[layer_index - 1]
            if m == 'MixResViT':
                bb = MixResViT(patch_sizes=config.MODEL.MR.PATCH_SIZES[:layer_index + 1],
                                        n_layers=config.MODEL.MR.DEPTHS[layer_index],
                                        d_model=config.MODEL.MR.EMBED_DIM[layer_index],
                                        n_heads=config.MODEL.MR.NUM_HEADS[layer_index],
                                        mlp_ratio=config.MODEL.MR.MLP_RATIO[layer_index],
                                        dropout=config.MODEL.MR.DROP_RATE[layer_index],
                                        drop_path_rate=config.MODEL.MR.DROP_PATH_RATE[layer_index],
                                        split_ratio=config.MODEL.MR.SPLIT_RATIO[layer_index],
                                        channels=in_chans,
                                        n_scales=config.MODEL.MR.N_RESOLUTION_SCALES,
                                        min_patch_size=config.MODEL.MR.PATCH_SIZES[-1],
                                        upscale_ratio=config.MODEL.MR.UPSCALE_RATIO[layer_index],
                                        out_features= config.MODEL.MR.OUT_FEATURES[-(layer_index+1):])
            elif m == 'MixResNeighbour':
                bb = MixResNeighbour(patch_sizes=config.MODEL.MR.PATCH_SIZES[:layer_index + 1],
                                        n_layers=config.MODEL.MR.DEPTHS[layer_index],
                                        d_model=config.MODEL.MR.EMBED_DIM[layer_index],
                                        n_heads=config.MODEL.MR.NUM_HEADS[layer_index],
                                        mlp_ratio=config.MODEL.MR.MLP_RATIO[layer_index],
                                        dropout=config.MODEL.MR.DROP_RATE[layer_index],
                                        drop_path_rate=config.MODEL.MR.DROP_PATH_RATE[layer_index],
                                        attn_drop_rate=config.MODEL.MR.ATTN_DROP_RATE[layer_index],
                                        split_ratio=config.MODEL.MR.SPLIT_RATIO[layer_index],
                                        channels=in_chans,
                                        cluster_size=config.MODEL.MR.CLUSTER_SIZE[layer_index],
                                        nbhd_size=config.MODEL.MR.NBHD_SIZE[layer_index],
                                        n_scales=config.MODEL.MR.N_RESOLUTION_SCALES,
                                        keep_old_scale=config.MODEL.MR.KEEP_OLD_SCALE,
                                        scale=layer_index,
                                        add_image_data_to_all=config.MODEL.MR.ADD_IMAGE_DATA_TO_ALL,
                                        min_patch_size=config.MODEL.MR.PATCH_SIZES[-1],
                                        upscale_ratio=config.MODEL.MR.UPSCALE_RATIO[layer_index],
                                        out_features= config.MODEL.MR.OUT_FEATURES[-(layer_index+1):])
            else:
                raise NotImplementedError(f"Unkown model: {m}")
            backbones.append(bb)
        model = OracleTeacherBackbone(backbones=backbones,
                                      backbone_dims=config.MODEL.MR.EMBED_DIM,
                                      out_dim=config.MODEL.MR.OUT_DIM,
                                      all_out_features=config.MODEL.MR.OUT_FEATURES,
                                      n_scales=config.MODEL.MR.N_RESOLUTION_SCALES,
                                      num_classes=config.MODEL.NUM_CLASSES)
    elif model_type == 'maskfinerUD':
        bb_in_feats = [[None], ["res5"], ["res5", "res4"], ["res5", "res4", "res3"], ["res5", "res4", "res3"],
                       ["res5", "res4"], ["res5"], [None]]
        all_backbones = []
        n_scales = config.MODEL.MR.N_RESOLUTION_SCALES
        n_layers = len(config.MODEL.MR.NAME)
        min_patch_size = config.MODEL.MR.PATCH_SIZES[n_scales - 1]
        for layer_index, name in enumerate(config.MODEL.MR.NAME):
            if layer_index == 0:
                first_layer = True
                in_chans = 3
            else:
                first_layer = False
                in_chans = config.MODEL.MR.EMBED_DIM[layer_index - 1]
            if layer_index >= n_scales:
                scale = n_layers - layer_index - 1
                patch_sizes = config.MODEL.MR.PATCH_SIZES[layer_index:]
                out_features = config.MODEL.MR.OUT_FEATURES[-(n_layers - layer_index):]
                #in_chans = sum(config.MODEL.MR.EMBED_DIM[-(layer_index + 1):-(n_layers - layer_index)])
                in_chans = config.MODEL.MR.EMBED_DIM[layer_index - 1] + config.MODEL.MR.EMBED_DIM[n_layers - layer_index - 1]
            else:
                scale = layer_index
                patch_sizes = config.MODEL.MR.PATCH_SIZES[:layer_index + 1]
                out_features = config.MODEL.MR.OUT_FEATURES[-(layer_index+1):]
            drop_path_rate = config.MODEL.MR.DROP_PATH_RATE
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config.MODEL.MR.DEPTHS))]
            drop_path = dpr[sum(config.MODEL.MR.DEPTHS[:layer_index]):sum(config.MODEL.MR.DEPTHS[:layer_index + 1])]
            if name == 'MixResViT':
                bb = MixResViT(patch_sizes=patch_sizes,
                               n_layers=config.MODEL.MR.DEPTHS[layer_index],
                               d_model=config.MODEL.MR.EMBED_DIM[layer_index],
                               n_heads=config.MODEL.MR.NUM_HEADS[layer_index],
                               mlp_ratio=config.MODEL.MR.MLP_RATIO[layer_index],
                               dropout=config.MODEL.MR.DROP_RATE[layer_index],
                               drop_path_rate=drop_path,
                               split_ratio=config.MODEL.MR.SPLIT_RATIO[layer_index],
                               channels=in_chans,
                               n_scales=n_scales,
                               min_patch_size=min_patch_size,
                               upscale_ratio=config.MODEL.MR.UPSCALE_RATIO[layer_index],
                               out_features=out_features,
                               first_layer=first_layer,
                               layer_scale=config.MODEL.MR.LAYER_SCALE,
                               num_register_tokens=config.MODEL.MR.NUM_REGISTER_TOKENS)
            elif name == 'MixResNeighbour':
                bb = MixResNeighbour(patch_sizes=patch_sizes,
                                     n_layers=config.MODEL.MR.DEPTHS[layer_index],
                                     d_model=config.MODEL.MR.EMBED_DIM[layer_index],
                                     n_heads=config.MODEL.MR.NUM_HEADS[layer_index],
                                     mlp_ratio=config.MODEL.MR.MLP_RATIO[layer_index],
                                     dropout=config.MODEL.MR.DROP_RATE[layer_index],
                                     drop_path_rate=drop_path,
                                     attn_drop_rate=config.MODEL.MR.ATTN_DROP_RATE[layer_index],
                                     split_ratio=config.MODEL.MR.SPLIT_RATIO[layer_index],
                                     channels=in_chans,
                                     cluster_size=config.MODEL.MR.CLUSTER_SIZE[layer_index],
                                     nbhd_size=config.MODEL.MR.NBHD_SIZE[layer_index],
                                     n_scales=n_scales,
                                     keep_old_scale=config.MODEL.MR.KEEP_OLD_SCALE,
                                     scale=scale,
                                     add_image_data_to_all=config.MODEL.MR.ADD_IMAGE_DATA_TO_ALL,
                                     min_patch_size=min_patch_size,
                                     upscale_ratio=config.MODEL.MR.UPSCALE_RATIO[layer_index],
                                     layer_scale=config.MODEL.MR.LAYER_SCALE,
                                     out_features=out_features,
                                     first_layer=first_layer)
            else:
                raise NotImplementedError(f"Unkown model: {name}")
            all_backbones.append(bb)
        model = UpDownBackbone(backbones=all_backbones,
                                      backbone_dims=config.MODEL.MR.EMBED_DIM,
                                      out_dim=config.MODEL.MR.OUT_DIM,
                                      all_out_features=config.MODEL.MR.OUT_FEATURES,
                                      n_scales=config.MODEL.MR.N_RESOLUTION_SCALES,
                                      num_classes=config.MODEL.NUM_CLASSES,
                                      bb_in_feats=bb_in_feats,
                                      aux_loss=config.MODEL.MR.AUX_LOSS)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
