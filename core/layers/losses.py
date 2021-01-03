import torch
from torch import nn

import functools

import torch.nn.functional as F


def reduce_loss(loss, reduction):
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


class SmoothL1Loss(nn.Module):

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


class EXTLoss(nn.Module):
    def __init__(self, ext_loss_type='smoothl1'):
        super(EXTLoss, self).__init__()
        self.ext_loss_type = ext_loss_type
        if ext_loss_type == 'smoothl1':
            self.loss_func = nn.SmoothL1Loss(reduction='none')

    def forward(self, pred, target, weight=None):
        losses = self.loss_func(pred, target).sum(dim=1)
        if weight is not None:
            return (losses * weight).sum()
        else:
            return losses.sum()


class DiceLoss(nn.Module):
    def __init__(self,
                 bce_weight=0,
                 ignore_value=255):
        super(DiceLoss, self).__init__()
        self.ignore_value = ignore_value
        if bce_weight != 0:
            self.bce_crit = nn.BCELoss()
        else:
            self.bce_crit = None
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        if len(target.size()) == 3:
            target = target.unsqueeze(1)
        assert pred.size() == target.size()

        target = target.float()

        if self.ignore_value:
            mask = torch.ne(target, self.ignore_value).float()
            pred *= mask
            target *= mask

        p2 = pred * pred
        g2 = target * target
        pg = pred * target

        p2 = torch.sum(p2, (3, 2, 1))
        g2 = torch.sum(g2, (3, 2, 1))
        pg = torch.sum(pg, (3, 2, 1))

        dice_coef = (2 * pg) / (p2 + g2 + 0.0001)

        dice_loss = (1.0 - dice_coef).sum()
        dice_loss /= target.size(0)

        if self.bce_crit is not None:
            bce_loss = self.bce_crit(pred, target)
            dice_loss += self.bce_weight * bce_loss

        return dice_loss


class IOULoss(nn.Module):
    """
    Codes from Adet (https://github.com/aim-uofa/AdelaiDet/blob/master/adet/layers/iou_loss.py)
    """
    def __init__(self, loc_loss_type='iou'):
        super(IOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        g_w_intersect = torch.max(pred_left, target_left) + \
                        torch.max(pred_right, target_right)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + \
                        torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect

        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        if self.loc_loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loc_loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loc_loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None:
            return (losses * weight).sum()
        else:
            return losses.sum()
