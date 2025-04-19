# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fontTools.misc.classifyTools import Classifier
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple, Union

from .tiny_vit_sam import TinyViT
from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .classifier import CombinedClassifier
from .fuse_tspg import FuseTSPG
from .classifier_ab1 import Classifier_ab1, Classifier_ab0
from .classifier_ab2 import Classifier_ab2
from .classifier_batch import CombinedClassifier_batch_fore, CombinedClassifier_batch_aft_bl, CombinedClassifier_batch_aft_sd, CombinedClassifier_batch_aft_fh


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        # image_encoder: ImageEncoderViT,  # 这里传入的已经是对象了，在build_sam文件中创建的
        # image_encoder: Union[ImageEncoderViT, TinyViT],
        image_encoder: TinyViT,
        prompt_encoder: PromptEncoder,
        fuse_tspg: FuseTSPG,
        mask_decoder: MaskDecoder,
        classifier_fore: Union[CombinedClassifier, Classifier_ab2, Classifier_ab1, CombinedClassifier_batch_fore],
        classifier_aft_bl: CombinedClassifier_batch_aft_bl,
        classifier_aft_fh: CombinedClassifier_batch_aft_fh,
        classifier_aft_sd: CombinedClassifier_batch_aft_sd,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.classifier_fore = classifier_fore
        self.classifier_aft_bl = classifier_aft_bl
        self.classifier_aft_fh = classifier_aft_fh
        self.classifier_aft_sd = classifier_aft_sd
        self.fuse_tspg = fuse_tspg
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device


    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


