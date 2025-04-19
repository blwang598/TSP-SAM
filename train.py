import os
import time
import torch
import lightning as L
import torch.nn.functional as F
from box import Box
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torch.autograd import Variable
from configs.config import cfg
from losses import DiceLoss, FocalLoss, ContraLoss, MultiFocalLoss, BinaryFocalLoss

from model import Model
from utils.eval_utils_batch import AverageMeter, calc_iou, validate, get_prompts
from utils.tools import create_csv, reduce_instances
from losses import CrossEntropyLoss2d
from min_norm_solvers import MinNormSolver, gradient_normalizers
from datasets.WLI import load_datasets_soft

def train_sam(
        cfg: Box,
        fabric: L.Fabric,
        model: Model,
        optimizer: _FabricOptimizer,
        scheduler: _FabricOptimizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
):
    """The SAM training loop."""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    mask_gt_losses = AverageMeter()
    cls_blgt_losses = AverageMeter()
    cls_fhgt_losses = AverageMeter()
    cls_sdgt_losses = AverageMeter()
    pred_iou_losses = AverageMeter()
    total_losses = AverageMeter()
    end = time.time()

    # focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    pred_iou_loss = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')

    weight_bl = torch.tensor([0.6, 0.4]).cuda()
    weight_fh = torch.tensor([1.3, 0.7, 0]).cuda()
    weight_sd = torch.tensor([1.5, 0.5, 0]).cuda()

    wce_Loss_bl = CrossEntropyLoss2d(weight=weight_bl)
    wce_Loss_fh = CrossEntropyLoss2d(weight=weight_fh)
    wce_Loss_sd = CrossEntropyLoss2d(weight=weight_sd)

    scale_all_analysis = {}
    scale_all_analysis['bl'] = []
    scale_all_analysis['fh'] = []
    scale_all_analysis['sd'] = []
    for step in range(cfg.opt.train_steps):
        data = next(iter(train_dataloader))
        data_time.update(time.time() - end)
        images_weak, bboxes, gt_masks, cls_bingli, cls_fenhua, cls_shendu = data
        batch_size = images_weak.size(0)
        num_insts = sum(len(gt_mask) for gt_mask in gt_masks)
        if num_insts > cfg.max_nums:
            print(num_insts)
            bboxes, gt_masks = reduce_instances(bboxes, gt_masks, cfg.max_nums)

        prompts = get_prompts(cfg, bboxes, gt_masks)

        cls_pred_fore, pred_masks, iou_predictions = model.forward_for_grad(images_weak, prompts)
        rep_variable_bl = Variable(cls_pred_fore[0].data.clone(), requires_grad=True)
        rep_variable_fh = Variable(cls_pred_fore[1].data.clone(), requires_grad=True)
        rep_variable_sd = Variable(cls_pred_fore[2].data.clone(), requires_grad=True)

        grads = {}
        loss_data = {}
        scale = {}

        # pathological
        optimizer.zero_grad()
        bl_logits_grad = model.classifier_bl(rep_variable_bl)
        cls_bl_onehot = torch.tensor(F.one_hot(cls_bingli, num_classes=2), dtype=torch.float16)
        loss_bl_grad = wce_Loss_bl(bl_logits_grad[0], cls_bl_onehot)
        loss_data['bl'] = loss_bl_grad.data
        fabric.backward(loss_bl_grad)
        grads['bl'] = []
        grads['bl'].append(Variable(rep_variable_bl.grad.data.clone(), requires_grad=False))
        rep_variable_bl.grad.data.zero_()

        # differentiation
        optimizer.zero_grad()
        fh_logits_grad = model.classifier_fh(rep_variable_fh)
        cls_fh_onehot = torch.tensor(F.one_hot(cls_fenhua, num_classes=3), dtype=torch.float16)
        loss_fh_grad = wce_Loss_fh(fh_logits_grad[0], cls_fh_onehot)
        loss_data['fh'] = loss_fh_grad.data
        fabric.backward(loss_fh_grad)
        grads['fh'] = []
        grads['fh'].append(Variable(rep_variable_fh.grad.data.clone(), requires_grad=False))
        rep_variable_fh.grad.data.zero_()

        # infiltration
        optimizer.zero_grad()
        sd_logits_grad = model.classifier_sd(rep_variable_sd)
        cls_sd_onehot = torch.tensor(F.one_hot(cls_shendu, num_classes=3), dtype=torch.float16)
        loss_sd_grad = wce_Loss_fh(sd_logits_grad[0], cls_sd_onehot)
        grads['sd'] = []
        loss_data['sd'] = loss_sd_grad.data
        fabric.backward(loss_sd_grad)
        grads['sd'].append(Variable(rep_variable_sd.grad.data.clone(), requires_grad=False))
        rep_variable_sd.grad.data.zero_()

        # Normalize all gradients
        gn = gradient_normalizers(grads, loss_data, 'loss+')
        for gr_i in range(len(grads['bl'])):
            grads['bl'][gr_i] = grads['bl'][gr_i] / gn['bl']
        for gr_i in range(len(grads['fh'])):
            grads['fh'][gr_i] = grads['fh'][gr_i] / gn['fh']
        for gr_i in range(len(grads['sd'])):
            grads['sd'][gr_i] = grads['sd'][gr_i] / gn['sd']
        # Frank-Wolfe iteration to compute scales.
        sol, min_norm = MinNormSolver.find_min_norm_element([grads['bl'], grads['fh'], grads['sd']])
        scale['bl'] = float(sol[0])
        scale['fh'] = float(sol[1])
        scale['sd'] = float(sol[2])
        scale_all_analysis['bl'].append(float(sol[0]))
        scale_all_analysis['fh'].append(float(sol[1]))
        scale_all_analysis['sd'].append(float(sol[2]))

        optimizer.zero_grad()

        iou_label = torch.ones((batch_size), device=fabric.device).float()

        loss_pred_iou = pred_iou_loss(iou_predictions.squeeze(), iou_label)
        loss_mask_gt = dice_loss(pred_masks, gt_masks)

        pred_cls_pred_bl = model.classifier_bl(cls_pred_fore[0])
        pred_cls_pred_fh = model.classifier_fh(cls_pred_fore[1])
        pred_cls_pred_sd = model.classifier_sd(cls_pred_fore[2])
        loss_cls_blgt = wce_Loss_bl(pred_cls_pred_bl[0], cls_bl_onehot)
        loss_cls_fhgt = wce_Loss_fh(pred_cls_pred_fh[0], cls_fh_onehot)
        loss_cls_sdgt = wce_Loss_sd(pred_cls_pred_sd[0], cls_sd_onehot)

        loss_total = loss_mask_gt + scale['bl'] * loss_cls_blgt + scale['fh'] * loss_cls_fhgt + scale[
            'sd'] * loss_cls_sdgt + loss_pred_iou

        optimizer.zero_grad()
        fabric.backward(loss_total)

        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()

        batch_time.update(time.time() - end)
        end = time.time()

        mask_gt_losses.update(loss_mask_gt.item(), batch_size)
        cls_blgt_losses.update(loss_cls_blgt.item(), batch_size)
        cls_fhgt_losses.update(loss_cls_fhgt.item(), batch_size)
        cls_sdgt_losses.update(loss_cls_sdgt.item(), batch_size)
        pred_iou_losses.update(loss_pred_iou.item(), batch_size)
        total_losses.update(loss_total.item(), batch_size)

        fabric.print(f'Step: [{step + 1}/{len(train_dataloader)}]'
                     f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                     f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
                     f' | MaskGt Loss [{mask_gt_losses.val:.4f} ({mask_gt_losses.avg:.4f})]'
                     f' | Cls blLoss [{cls_blgt_losses.val:.4f} ({cls_blgt_losses.avg:.4f})]'
                     f' | Cls fhLoss [{cls_fhgt_losses.val:.4f} ({cls_fhgt_losses.avg:.4f})]'
                     f' | Cls sdLoss [{cls_sdgt_losses.val:.4f} ({cls_sdgt_losses.avg:.4f})]'
                     f' | Pred IouLoss [{pred_iou_losses.val:.4f} ({pred_iou_losses.avg:.4f})]'
                     f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')

        loss_logger = {"MaskGt Loss": mask_gt_losses.avg,
                       "Cls blgt": cls_blgt_losses.avg,
                       "Cls fhgt": cls_fhgt_losses.avg,
                       "Cls sdgt": cls_sdgt_losses.avg,
                       "Pred Iou": pred_iou_losses.avg,
                       "Total Loss": total_losses.avg
                       }
        fabric.log_dict(loss_logger, step)
        torch.cuda.empty_cache()

        if (step + 1) % cfg.eval_interval == 0:
            iou, f1_score, avg_recall_bl, avg_recall_fh, avg_recall_sd = validate(fabric, cfg, model, val_dataloader,
                                                                                  cfg.name, step, is_source=False)


def configure_opt(cfg: Box, model: Model):
    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor ** 2)

    parameter_groups = [{'params': model.model.image_encoder.parameters(), 'lr': cfg.opt.learning_rate},
                        {'params': model.model.mask_decoder.parameters(), 'lr': cfg.opt.learning_rate},
                        {'params': model.model.prompt_encoder.parameters(), 'lr': cfg.opt.learning_rate},
                        {'params': model.model.fuse_tspg.parameters(), 'lr': cfg.opt.learning_rate},
                        {'params': model.model.classifier_fore.parameters(), 'lr': cfg.opt.learning_rate},
                        {'params': model.model.classifier_aft_bl.parameters(), 'lr': cfg.opt.learning_rate},
                        {'params': model.model.classifier_aft_fh.parameters(), 'lr': 0.5 * cfg.opt.learning_rate},
                        {'params': model.model.classifier_aft_sd.parameters(), 'lr': 0.3 * cfg.opt.learning_rate},
                        ]

    optimizer = torch.optim.Adam(parameter_groups, lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def main(cfg: Box) -> None:
    fabric = L.Fabric(accelerator="auto",
                      devices=[1],
                      strategy="auto",
                      loggers=[TensorBoardLogger(cfg.out_dir)],
                      precision='16',
                      )
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    # with fabric.device:
    model = Model(cfg)
    model.setup()  # lora_vit

    train_data, val_data = load_datasets_soft(cfg, model.model.image_encoder.img_size)
    optimizer, scheduler = configure_opt(cfg, model)

    val_data = fabric._setup_dataloader(val_data)
    model, optimizer = fabric.setup(model, optimizer)

    if cfg.model.ckpt is not None:
        full_checkpoint = fabric.load(cfg.model.ckpt)
        model.load_state_dict(full_checkpoint["model"])

    model.train()

    train_sam(cfg, fabric, model, optimizer, scheduler, train_data, val_data)

    del model, train_data, val_data


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids

    main(cfg)
    torch.cuda.empty_cache()
