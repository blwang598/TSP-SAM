import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC

ALPHA = 0.25
GAMMA = 2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, preds, labels):
        eps = 1e-7
        # 开始计算
        loss_y1 = -1 * self.alpha * \
            torch.pow((1 - preds), self.gamma) * \
            torch.log(preds + eps) * labels
        loss_y0 = -1 * (1 - self.alpha) * torch.pow(preds, self.gamma) * torch.log(1 - preds + eps) * (1 - labels)
        loss = loss_y0 + loss_y1
        return torch.mean(loss)

class MultiFocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num=3, alpha=None, gamma=2, size_average=True):
        super(MultiFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = torch.Tensor(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = inputs
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss



class FocalLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE

        return focal_loss


class DiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class ContraLoss(nn.Module):

    def __init__(self, temperature = 0.3, weight=None, size_average=True):
        super().__init__()
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, embedd_x: torch.Tensor, embedd_y: torch.Tensor, mask_x: torch.Tensor, mask_y: torch.Tensor):
        x_embedding = self.norm_embed(embedd_x)
        y_embedding = self.norm_embed(embedd_y)
        # mask_y = mask_y.float()
        x_masks = F.interpolate(mask_x, size=x_embedding.shape[-2:], mode="bilinear", align_corners=False).detach()
        # x_masks = torch.clamp(x_masks, min=0, max=1)
        sum_x = x_masks.sum(dim=[-1, -2]).clone()

        y_masks = F.interpolate(mask_y, size=y_embedding.shape[-2:], mode="bilinear", align_corners=False).detach()
        # y_masks = torch.clamp(y_masks, min=0, max=1)
        sum_y = y_masks.sum(dim=[-1, -2]).clone()
        # [n, 1, H, W]
        multi_embedd_x = (x_embedding * x_masks).sum(dim=[-1, -2]) / sum_x
        multi_embedd_y = (y_embedding * y_masks).sum(dim=[-1, -2]) / sum_y

        flatten_x = multi_embedd_x.view(multi_embedd_x.size(0), -1)
        flatten_y = multi_embedd_y.view(multi_embedd_y.size(0), -1)
        similarity_matrix = F.cosine_similarity(flatten_x.unsqueeze(1), flatten_y.unsqueeze(0), dim=2)

        label_pos = torch.eye(x_masks.size(0)).bool().to(embedd_x.device)
        label_nag = ~label_pos

        similarity_matrix = similarity_matrix / self.temperature
        loss = -torch.log(
                similarity_matrix.masked_select(label_pos).exp().sum() / 
                similarity_matrix.exp().sum()
            )
        #         similarity_matrix.masked_select(label_pos).exp().sum()
        # loss = -torclass AutomaticWeightedLoss(nn.Module):
        #
        #         num: int，the number of loss
        #     Examples：
        #         loss2=2
        #         loss_sum = awl(loss1, loss2)
        #     def __init__(self, num=2):
        #         params = torch.ones(num, requires_grad=True)
        #
        #         loss_sum = 0
        #             loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        #         similarity_matrix.masked_select(label_pos).exp().sum() /
        #     )
        return loss

    def norm_embed(self, embedding: torch.Tensor):
        embedding = F.normalize(embedding, dim=0, p=2)
        return embedding

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss(weight, size_average)

    def forward(self, preds, targets):
        return self.nll_loss(preds, targets)
