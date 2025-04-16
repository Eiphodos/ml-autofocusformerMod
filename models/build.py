# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Adapted for AutoFocusFormer by Ziwen 2023

from .aff_transformer import AutoFocusFormer
from .maskfiner_oracle_teacher_model import OracleTeacherBackbone
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
                raise NotImplementedError(f"Unkown model: {model_type}")
            backbones.append(bb)
        model = OracleTeacherBackbone(backbones=backbones,
                                      backbone_dims=config.MODEL.MR.EMBED_DIMS,
                                      out_dim=config.MODEL.MR.OUT_DIM,
                                      all_out_features=config.MODEL.MR.OUT_FEATURES,
                                      n_scales=config.MODEL.MR.N_RESOLUTION_SCALES,
                                      num_classes=config.MODEL.NUM_CLASSES)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
