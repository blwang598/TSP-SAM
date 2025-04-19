# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .modeling import (ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, Sam_0, Sam_1, TwoWayTransformer,
                       TinyViT, CombinedClassifier, Classifiy_TwoWayTransformer, FuseTSPG,
                       Classifier_ab0, Classifier_ab1, Classifier_ab2, CombinedClassifier_batch_fore,
                       CombinedClassifier_batch_aft_bl, CombinedClassifier_batch_aft_fh, CombinedClassifier_batch_aft_sd)  # 只需要更改后面两个分类器的结构就可以实现消融


def build_sam_vit_t_batched_mgda(checkpoint=None):  # 建立mobileSAM模型
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = Sam(
            image_encoder=TinyViT(
                img_size=1024,
                in_chans=3,
                num_classes=1000,  # image encoder用的是tinyVit
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8
            ),
            prompt_encoder=PromptEncoder(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
            ),
            fuse_tspg=FuseTSPG(
                endoder_transformer_dim=768,
                upsample_transformer_dim=256,
                sam_features_length=3,
            ),
            mask_decoder=MaskDecoder(
                num_multimask_outputs=3,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            ),
            classifier_fore=CombinedClassifier_batch_fore(
                transformer=Classifiy_TwoWayTransformer(
                    depth=1,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
            ),
            classifier_aft_bl=CombinedClassifier_batch_aft_bl(
                transformer_dim=prompt_embed_dim,
                mlp_dim=256,
                activation=torch.nn.GELU,
                num_class=2
            ),
            classifier_aft_fh=CombinedClassifier_batch_aft_fh(
                transformer_dim=prompt_embed_dim,
                mlp_dim=256,
                activation=torch.nn.GELU,
                num_class=3
            ),
            classifier_aft_sd=CombinedClassifier_batch_aft_sd(
                transformer_dim=prompt_embed_dim,
                mlp_dim=256,
                activation=torch.nn.GELU,
                num_class=3
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )

    mobile_sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        mobile_sam.load_state_dict(state_dict, strict=False)
    return mobile_sam




sam_model_registry = {
    "vit_t_batched_mgda": build_sam_vit_t_batched_mgda
}
