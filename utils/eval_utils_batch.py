import os
import torch
import lightning as L
import segmentation_models_pytorch as smp
from box import Box
from torch.utils.data import DataLoader
from model import Model
from utils.sample_utils import get_point_prompts
from utils.tools import write_csv
import sklearn.metrics as sm
import numpy as np

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)
    batch_iou = batch_iou.unsqueeze(1)
    return batch_iou


def calc_dice(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7      
    batch_dice = 1 - 2*intersection / (union + epsilon)
    batch_dice = batch_dice.unsqueeze(1)
    return batch_dice


def get_prompts(cfg: Box, bboxes, gt_masks):
    if cfg.prompt == "box" or cfg.prompt == "coarse":
        prompts = bboxes
    elif cfg.prompt == "point":
        prompts = get_point_prompts(gt_masks, cfg.num_points)      
    else:
        raise ValueError("Prompt Type Error!")
    return prompts


def validate(fabric: L.Fabric, cfg: Box, model: Model, val_dataloader: DataLoader, name: str, step: int = 0, is_source=False):
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()
    with ((torch.no_grad())):
        first_test_iter = True
        acc_num_bl = torch.zeros(1).cuda()
        acc_num_fh = torch.zeros(1).cuda()
        acc_num_sd = torch.zeros(1).cuda()
        sample_num = torch.zeros(1).cuda()
        for iter, data in enumerate(val_dataloader):

            images, bboxes, gt_masks, cls_bl, cls_fh, cls_sd, W, H, name = data
            num_images = images.size(0)
            sample_num += num_images
            prompts = get_prompts(cfg, bboxes, gt_masks)
            pred_masks, _, cls_logits = model.forward(images, prompts)

            batch_stats = smp.metrics.get_stats(
                pred_masks,
                gt_masks.int(),
                mode='binary',
                threshold=0.5,
            )
            batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
            batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
            ious.update(batch_iou, num_images)
            f1_scores.update(batch_f1, num_images)
            cls_logit_bl, cls_logit_fh, cls_logit_sd = cls_logits[0], cls_logits[1], cls_logits[2]
            cls_pred_bl = torch.max(cls_logit_bl, dim=1)[1]
            cls_pred_fh = torch.max(cls_logit_fh, dim=1)[1]
            cls_pred_sd = torch.max(cls_logit_sd, dim=1)[1]
            acc_num_bl += torch.eq(cls_pred_bl, cls_bl).sum()
            acc_num_fh += torch.eq(cls_pred_fh, cls_fh).sum()
            acc_num_sd += torch.eq(cls_pred_sd, cls_sd).sum()
            if first_test_iter:
                all_probs_bl, all_probs_fh, all_probs_sd = cls_logit_bl, cls_logit_fh, cls_logit_sd
                all_labels_bl, all_labels_fh, all_labels_sd = cls_bl, cls_fh, cls_sd
                first_test_iter = False
            else:
                all_probs_bl = torch.cat((all_probs_bl, cls_logit_bl), dim=0)
                all_probs_fh = torch.cat((all_probs_fh, cls_logit_fh), dim=0)
                all_probs_sd = torch.cat((all_probs_sd, cls_logit_sd), dim=0)
                all_labels_bl = torch.cat((all_labels_bl, cls_bl), dim=0)
                all_labels_fh = torch.cat((all_labels_fh, cls_fh), dim=0)
                all_labels_sd = torch.cat((all_labels_sd, cls_sd), dim=0)      
      
      
            torch.cuda.empty_cache()
        _, predict_bl = torch.max(all_probs_bl, 1)
        _, predict_fh = torch.max(all_probs_fh, 1)
        _, predict_sd = torch.max(all_probs_sd, 1)
        predictions_bl = torch.squeeze(predict_bl).float()
        predictions_sd = torch.squeeze(predict_sd).float()
        predictions_fh = torch.squeeze(predict_fh).float()
        all_label_cpu_bl = [a.cpu() for a in all_labels_bl]
        all_label_cpu_fh = [a.cpu() for a in all_labels_fh]
        all_label_cpu_sd = [a.cpu() for a in all_labels_sd]

        avg_recall_bl = sm.balanced_accuracy_score(np.array(all_label_cpu_bl), predictions_bl.cpu().numpy())
        avg_recall_fh = sm.balanced_accuracy_score(np.array(all_label_cpu_fh), predictions_fh.cpu().numpy())
        avg_recall_sd = sm.balanced_accuracy_score(np.array(all_label_cpu_sd), predictions_sd.cpu().numpy())
        acc_bl = acc_num_bl / sample_num
        acc_fh = acc_num_fh / sample_num
        acc_sd = acc_num_sd / sample_num
        acc_bl = acc_bl.cpu().item()
        acc_fh = acc_fh.cpu().item()
        acc_sd = acc_sd.cpu().item()


    fabric.print(f'Validation [{step}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}] '
                 f'-- Accuracy bl: {acc_bl:.4f} --Recall_bl: {avg_recall_bl:.4f}'
                 f'-- Accuracy fh: {acc_fh:.4f} --Recall_fh: {avg_recall_fh:.4f}'
                 f'-- Accuracy sd: {acc_sd:.4f} --Recall_sd: {avg_recall_sd:.4f}'
                 )


    model.train()
    return ious.avg, f1_scores.avg, avg_recall_bl, avg_recall_fh, avg_recall_sd
