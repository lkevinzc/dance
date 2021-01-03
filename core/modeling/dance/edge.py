"""
 @ Created by: liuzichen@u.nus.edu
 @ Date: 2020-02-17
"""

from typing import Dict

import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
from core.layers import DiceLoss
from core.structures import ExtremePoints
from detectron2.layers import Conv2d, ShapeSpec
from detectron2.structures import Boxes, Instances
from detectron2.utils.registry import Registry
from torch import nn
from torch.nn import functional as F

from .head import RefineNet

__all__ = ["EDGE_HEADS_REGISTRY", "build_edge_det_head", "EdgeFPNHead"]

EDGE_HEADS_REGISTRY = Registry("EDGE_HEADS")


def build_edge_det_head(cfg, input_shape):
    name = cfg.MODEL.DANCE.EDGE.NAME
    return EDGE_HEADS_REGISTRY.get(name)(cfg, input_shape)


@EDGE_HEADS_REGISTRY.register()
class EdgeFPNHead(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.in_features = cfg.MODEL.DANCE.EDGE.IN_FEATURES
        feature_strides = {k: v.stride for k, v in input_shape.items()}
        feature_channels = {k: v.channels for k, v in input_shape.items()}
        self.ignore_value = cfg.MODEL.DANCE.EDGE.IGNORE_VALUE
        num_classes = cfg.MODEL.DANCE.EDGE.NUM_CLASSES
        conv_dims = cfg.MODEL.DANCE.EDGE.CONVS_DIM
        self.common_stride = cfg.MODEL.DANCE.EDGE.COMMON_STRIDE
        norm = cfg.MODEL.DANCE.EDGE.NORM
        self.loss_weight = cfg.MODEL.DANCE.EDGE.LOSS_WEIGHT
        # fmt: on

        self.loss = DiceLoss(
            cfg.MODEL.DANCE.EDGE.BCE_WEIGHT, cfg.MODEL.DANCE.EDGE.IGNORE_VALUE
        )

        self.gt_input = cfg.TEST.GT_IN.WHAT if cfg.TEST.GT_IN.ON else (None,)

        conv1 = Conv2d(
            1,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=nn.GroupNorm(4, 32),
            activation=F.relu,
        )
        conv2 = Conv2d(
            32,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=nn.GroupNorm(4, 32),
            activation=F.relu,
        )
        conv3 = Conv2d(
            32,
            1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            activation=torch.sigmoid,
        )
        weight_init.c2_msra_fill(conv1)
        weight_init.c2_msra_fill(conv2)
        nn.init.normal_(conv3.weight, 0, 0.01)
        nn.init.constant_(conv3.bias, 0)
        self.attender = nn.Sequential(conv1, conv2, conv3)

        self.scale_heads = []
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1,
                int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride)),
            )
            for k in range(head_length):
                norm_module = nn.GroupNorm(32, conv_dims) if norm == "GN" else None
                conv = Conv2d(
                    feature_channels[in_feature] if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=norm_module,
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if feature_strides[in_feature] != self.common_stride:
                    head_ops.append(
                        nn.Upsample(
                            scale_factor=2, mode="bilinear", align_corners=False
                        )
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])

        norm_module = nn.GroupNorm(32, conv_dims) if norm == "GN" else None
        conv = Conv2d(
            conv_dims,
            conv_dims,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not norm,
            norm=norm_module,
            activation=F.relu,
        )
        predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(conv)
        weight_init.c2_msra_fill(predictor)
        self.predictor = nn.Sequential(conv, predictor)

        self.refine_head = RefineNet(cfg)

    def forward(self, features, pred_instances=None, targets=None):

        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[f])
            else:
                x = x + self.scale_heads[i](features[f])

        pred_logits = self.predictor(x)
        pred_edge = pred_logits.sigmoid()

        att_map = self.attender(1 - pred_edge)  # regions that need evolution

        if self.training:
            edge_target = targets[0]
            snake_input = x
            pred_edge_full = F.interpolate(
                pred_edge,
                scale_factor=self.common_stride,
                mode="bilinear",
                align_corners=False,
            )
            snake_input = torch.cat([att_map, x], dim=1)

            # Quick fix for batches that do not have poly after filtering
            try:
                _, poly_loss = self.refine_head(snake_input, None, targets[1])
            except Exception:
                poly_loss = {}

            edge_loss = self.loss(pred_edge_full, edge_target) * self.loss_weight
            poly_loss.update(
                {
                    "loss_edge_det": edge_loss,
                }
            )
            return [], poly_loss, []
        else:

            snake_input = torch.cat([att_map, x], dim=1)

            if "instance" in self.gt_input:
                assert targets[1][0] is not None

                for im_i in range(len(targets[1][0])):
                    gt_instances_per_im = targets[1][0][im_i]
                    bboxes = gt_instances_per_im.gt_boxes.tensor
                    instances_per_im = Instances(pred_instances[im_i]._image_size)
                    instances_per_im.pred_boxes = Boxes(bboxes)
                    instances_per_im.pred_classes = gt_instances_per_im.gt_classes
                    instances_per_im.scores = torch.ones_like(
                        gt_instances_per_im.gt_classes, device=bboxes.device
                    )
                    if gt_instances_per_im.has("gt_masks"):
                        gt_masks = gt_instances_per_im.gt_masks
                        ext_pts_off = self.refine_head.get_simple_extreme_points(
                            gt_masks.polygons
                        ).to(bboxes.device)
                        ex_t = torch.stack(
                            [ext_pts_off[:, None, 0], bboxes[:, None, 1]], dim=2
                        )
                        ex_l = torch.stack(
                            [bboxes[:, None, 0], ext_pts_off[:, None, 1]], dim=2
                        )
                        ex_b = torch.stack(
                            [ext_pts_off[:, None, 2], bboxes[:, None, 3]], dim=2
                        )
                        ex_r = torch.stack(
                            [bboxes[:, None, 2], ext_pts_off[:, None, 3]], dim=2
                        )
                        instances_per_im.ext_points = ExtremePoints(
                            torch.cat([ex_t, ex_l, ex_b, ex_r], dim=1)
                        )

                    pred_instances[im_i] = instances_per_im

            new_instances, _ = self.refine_head(snake_input, pred_instances, None)

            pred_edge = att_map

            return pred_edge, {}, new_instances
