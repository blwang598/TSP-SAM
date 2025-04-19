import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from sam_lora import LoRA_Sam, LoRA_TinySam


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.image_embeddings = None

    def get_checkpoint(self, model_type):
        if model_type == "vit_t_batched_mgda":
            checkpoint = os.path.join(self.cfg.model.checkpoint, 'mobile_sam.pt')
        else:
            raise ValueError("Model type error!")
        return checkpoint

    def setup(self):
        self.model_type = self.cfg.model.type
        checkpoint = self.get_checkpoint(self.cfg.model.type)
        self.model = sam_model_registry[self.cfg.model.type](checkpoint=checkpoint)

        self.model.train()
        if self.cfg.model.freeze.image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False 

    def finetune(self):
        LoRA_TinySam(self.model, 4)

    def set_norm_layer(self):
        for name, param in self.model.image_encoder.named_parameters():
            if "norm" in name:
                param.requires_grad = True

    def set_evp_adaptor_layer(self):
        for param in self.model.image_encoder.prompt_generator.parameters():
            param.requires_grad = True

    def set_prompt_layer(self):
        self.model.image_encoder.Prompt_Tokens.requires_grad = True

    def reset_parameters(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                if "linear_a" in name:
                    nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                if "linear_b" in name:
                    nn.init.zeros_(param)

    def forward(self, images, prompts):
        _, _, H, W = images.shape 
        image_embeddings, image_embeddings_blocks = self.encode(images) 
        task_prompts = self.fusetspg(image_embeddings_blocks) 
        pred_masks, ious, res_masks = self.decode((H, W), prompts)
        cls_preds_fore = self.classifier_fore(prompts, task_prompts, res_masks)
        if self.model_type == 'vit_t_batched_mgda':
            pred_bl = self.classifier_bl(cls_preds_fore[0])
            pred_fh = self.classifier_fh(cls_preds_fore[1])
            pred_sd = self.classifier_sd(cls_preds_fore[2])
            pred_masks = torch.stack(pred_masks)
            ious = torch.cat(ious, dim=0)
            return pred_masks, ious, [pred_bl, pred_fh, pred_sd]

    def forward_for_grad(self, images, prompts):
        _, _, H, W = images.shape 
        self.image_embeddings, self.image_embeddings_blocks = self.encode(images)
        task_prompts = self.fusetspg(self.image_embeddings_blocks)
        pred_masks, ious, res_masks = self.decode((H, W), prompts)
        cls_preds_fore = self.classifier_fore(prompts, task_prompts, res_masks)
        pred_masks = torch.stack(pred_masks)
        ious = torch.cat(ious, dim=0)
        return cls_preds_fore, pred_masks, ious 


    def encode(self, images):
        self.image_embeddings, self.image_embeddings_blocks = self.model.image_encoder(images) 
        return self.image_embeddings, self.image_embeddings_blocks

    def fusetspg(self, image_embeddings_blocks):
        task_prompts = self.model.fuse_tspg(image_embeddings_blocks)
        return task_prompts

    def decode(self, image_shape, prompts): 
        image_embeddings = self.image_embeddings
        if image_embeddings == None:
            raise "No image embeddings"

        pred_masks = []
        ious = []
        res_masks = []
        for prompt, embedding in zip(prompts, image_embeddings): 
 
            if isinstance(prompt, torch.Tensor): 
                prompt = prompt.to(device=embedding.device)
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=prompt,
                masks=None,
            )
            elif isinstance(prompt, tuple): 
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=prompt,
                boxes=None,
                masks=None,
            )

            low_res_masks, iou_predictions = self.model.mask_decoder( 
                image_embeddings=embedding.unsqueeze(0), 
                image_pe=self.model.prompt_encoder.get_dense_pe(), 
                sparse_prompt_embeddings=sparse_embeddings, 
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            masks = F.interpolate( 
                low_res_masks,
                image_shape,
                mode="bilinear",
                align_corners=False,
            )
            pred_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)
            res_masks.append(low_res_masks)
        return pred_masks, ious, res_masks


    def classifier_fore(self, prompts, task_prompts, mask_preds):
        image_embeddings = self.image_embeddings
        if image_embeddings == None:
            raise "No image embeddings"
        if self.model_type != "vit_t_batched" and self.model_type != "vit_t_batched_mgda":
            cls_preds = [] 
            for prompt, embedding, mask_pred, task_prompt in zip(prompts, image_embeddings, mask_preds, task_prompts): 
 
                if isinstance(prompt, torch.Tensor): 
                    prompt = prompt.to(device=embedding.device) 
                    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=None,
                    boxes=prompt,
                    masks=mask_pred,
                )
                elif isinstance(prompt, tuple): 
                    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=prompt,
                    boxes=None,
                    masks=mask_pred,
                ) 
                cls_pred = self.model.classifier(image_embeddings=embedding, 
                                                 image_pe=self.model.prompt_encoder.get_dense_pe(), 
                                                 task_prompt_embeddings=task_prompt.unsqueeze(0), 
                                                 sparse_prompt_embeddings=sparse_embeddings, 
                                                 dense_prompt_embeddings=dense_embeddings, 
                                                 ) 
                cls_preds.append(cls_pred)
            return cls_preds
        else: 
            sparse_emb_list = []
            dense_emb_list = [] 
            device = image_embeddings.device
            for prompt, mask_pred in zip(prompts, mask_preds): 
 
                if isinstance(prompt, torch.Tensor): 
                    prompt = prompt.to(device=device) 
                    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=None,
                    boxes=prompt,
                    masks=mask_pred,
                )
                elif isinstance(prompt, tuple): 
                    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=prompt,
                    boxes=None,
                    masks=mask_pred,
                )
                sparse_emb_list.append(sparse_embeddings) 
                dense_emb_list.append(dense_embeddings) 
 
            sparse_embeddings_all = torch.cat(sparse_emb_list, dim=0)
            dense_embeddings_all = torch.cat(dense_emb_list, dim=0)
            cls_pred_bl, cls_pred_fh, cls_pred_sd = self.model.classifier_fore(image_embeddings=image_embeddings,
                                              image_pe=self.model.prompt_encoder.get_dense_pe(),
                                              task_prompt_embeddings=task_prompts,
                                              sparse_prompt_embeddings=sparse_embeddings_all,
                                              dense_prompt_embeddings=dense_embeddings_all)

            return cls_pred_bl, cls_pred_fh, cls_pred_sd

    def classifier_bl(self, cls_pred_bl): 
        pred_bl = self.model.classifier_aft_bl(cls_pred_bl)
        return pred_bl

    def classifier_fh(self, cls_pred_fh): 
        pred_fh = self.model.classifier_aft_fh(cls_pred_fh)
        return pred_fh

    def classifier_sd(self, cls_pred_sd): 
        pred_sd = self.model.classifier_aft_sd(cls_pred_sd)
        return pred_sd


