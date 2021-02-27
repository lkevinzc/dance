"""
Adapted from detectron2.modeling.meta_arch.samantic_seg.py
 @ Created by: liuzichen@u.nus.edu
 @ Date: 2020-02-17
"""

import numpy as np
import torch
from typing import Dict
import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec
from detectron2.utils.registry import Registry
from detectron2.structures import Instances, Boxes

from core.layers import DiceLoss
from core.structures import ExtremePoints
from .snake_head import RefineNet

from core.utils import timer

__all__ = ["EDGE_HEADS_REGISTRY", "build_edge_det_head", "EdgeSnakeFPNHead"]


EDGE_HEADS_REGISTRY = Registry("EDGE_HEADS")


def build_edge_det_head(cfg, input_shape):
    name = cfg.MODEL.EDGE_HEAD.NAME
    return EDGE_HEADS_REGISTRY.get(name)(cfg, input_shape)


@EDGE_HEADS_REGISTRY.register()
class EdgeSnakeFPNHead(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.in_features      = cfg.MODEL.EDGE_HEAD.IN_FEATURES
        feature_strides       = {k: v.stride for k, v in input_shape.items()}
        feature_channels      = {k: v.channels for k, v in input_shape.items()}
        self.ignore_value     = cfg.MODEL.EDGE_HEAD.IGNORE_VALUE
        num_classes           = cfg.MODEL.EDGE_HEAD.NUM_CLASSES
        conv_dims             = cfg.MODEL.EDGE_HEAD.CONVS_DIM
        self.common_stride    = cfg.MODEL.EDGE_HEAD.COMMON_STRIDE
        norm                  = cfg.MODEL.EDGE_HEAD.NORM
        self.edge_in          = cfg.MODEL.SNAKE_HEAD.EDGE_IN
        self.loss_weight      = cfg.MODEL.EDGE_HEAD.LOSS_WEIGHT
        self.coord_conv       = cfg.MODEL.SNAKE_HEAD.COORD_CONV
        self.edge_map_thre    = cfg.MODEL.SNAKE_HEAD.EDGE_IN_TH
        # fmt: on
        self.loss = DiceLoss(
            cfg.MODEL.EDGE_HEAD.BCE_WEIGHT, cfg.MODEL.EDGE_HEAD.IGNORE_VALUE
        )

        self.gt_input = cfg.TEST.GT_IN.WHAT if cfg.TEST.GT_IN.ON else (None,)

        self.strong_feat = cfg.MODEL.EDGE_HEAD.STRONG_FEAT

        self.attention = cfg.MODEL.SNAKE_HEAD.ATTENTION

        self.original = cfg.MODEL.SNAKE_HEAD.ORIGINAL

        if self.attention:
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

        self.pred_edge = cfg.MODEL.SNAKE_HEAD.PRED_EDGE

        self.edge_on = cfg.MODEL.EDGE_HEAD.TRAIN

        if self.edge_on:
            self.scale_heads = []
            for in_feature in self.in_features:
                head_ops = []
                head_length = max(
                    1,
                    int(
                        np.log2(feature_strides[in_feature])
                        - np.log2(self.common_stride)
                    ),
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

        if self.edge_on:
            if not self.strong_feat:
                self.predictor = Conv2d(
                    conv_dims, num_classes, kernel_size=1, stride=1, padding=0
                )
                weight_init.c2_msra_fill(self.predictor)
            else:
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
                predictor = Conv2d(
                    conv_dims, num_classes, kernel_size=1, stride=1, padding=0
                )
                weight_init.c2_msra_fill(conv)
                weight_init.c2_msra_fill(predictor)
                self.predictor = nn.Sequential(conv, predictor)

        self.refine_head = RefineNet(cfg)

        if self.edge_in:
            self.mean_filter = torch.nn.Conv2d(
                1, 1, kernel_size=3, stride=1, padding=1, bias=False
            )
            nn.init.constant_(self.mean_filter.weight.data, 1)
            self.mean_filter.weight.requires_grad = False

        self.selective_refine = cfg.MODEL.SNAKE_HEAD.SELECTIVE_REFINE
        if self.selective_refine:
            self.dilate_filter = torch.nn.Conv2d(
                1, 1, kernel_size=5, stride=1, padding=2, bias=False
            )
            nn.init.constant_(self.dilate_filter.weight.data, 1)
            self.dilate_filter.weight.requires_grad = False

        # refine_loss_type = cfg.MODEL.SNAKE_HEAD.LOSS_TYPE
        # refine_loss_weight = 100 if refine_loss_type == 'chamfer' else 10
        # point_weight = cfg.MODEL.SNAKE_HEAD.POINT_WEIGH
        # self.refine_loss_weight = 0.2 if point_weight else refine_loss_weight

    def forward(self, features, pred_instances=None, targets=None):
        if self.edge_on:
            with timer.env("pfpn_back"):
                for i, f in enumerate(self.in_features):
                    if i == 0:
                        x = self.scale_heads[i](features[f])
                    else:
                        x = x + self.scale_heads[i](features[f])

        if self.edge_on:
            with timer.env("edge"):
                pred_logits = self.predictor(x)
                pred_edge = pred_logits.sigmoid()
                if self.attention:
                    # print('pred edge', pred_edge)
                    att_map = self.attender(
                        1 - pred_edge
                    )  # regions that need evolution

        if self.training:
            edge_target = targets[0]
            if self.edge_in:
                edge_prior = targets[0].unsqueeze(1).float().clone()  # (B, 1, H, W)
                edge_prior[edge_prior == self.ignore_value] = 0  # remove ignore value

                edge_prior = self.mean_filter(edge_prior)
                edge_prior = F.interpolate(
                    edge_prior,
                    scale_factor=1 / self.common_stride,
                    mode="bilinear",
                    align_corners=False,
                )
                edge_prior[edge_prior > 0] = 1

                if self.strong_feat:
                    snake_input = torch.cat([edge_prior, x], dim=1)
                else:
                    snake_input = torch.cat([edge_prior, features["p2"]], dim=1)
            else:
                if self.strong_feat:
                    snake_input = x
                else:
                    snake_input = features["p2"]

            if self.edge_on:
                pred_edge_full = F.interpolate(
                    pred_edge,
                    scale_factor=self.common_stride,
                    mode="bilinear",
                    align_corners=False,
                )

            if self.selective_refine:
                edge_prior = targets[0].unsqueeze(1).float().clone()  # (B, 1, H, W)
                edge_prior[edge_prior == self.ignore_value] = 0  # remove ignore value
                edge_prior = self.dilate_filter(edge_prior)
                # edge_prior = self.dilate_filter(edge_prior)
                # edge_target = edge_prior.clone()
                edge_prior[edge_prior > 0] = 1
                edge_prior = F.interpolate(
                    edge_prior,
                    scale_factor=1 / self.common_stride,
                    mode="bilinear",
                    align_corners=False,
                )
                if self.strong_feat:
                    snake_input = torch.cat([edge_prior, x], dim=1)
                else:
                    if self.pred_edge:
                        snake_input = torch.cat(
                            [edge_prior, pred_logits, features["p2"]], dim=1
                        )
                    else:
                        snake_input = torch.cat([edge_prior, features["p2"]], dim=1)

            if self.attention:
                if self.strong_feat:
                    snake_input = torch.cat([att_map, x], dim=1)
                else:
                    # dont cater pred_edge option now
                    snake_input = torch.cat([att_map, features["p2"]], dim=1)

            ### Quick fix for batches that do not have poly after filtering
            _, poly_loss = self.refine_head(snake_input, None, targets[1])

            if self.edge_on:
                edge_loss = self.loss(pred_edge_full, edge_target) * self.loss_weight
                poly_loss.update(
                    {
                        "loss_edge_det": edge_loss,
                    }
                )

            return [], poly_loss, []
        else:
            if self.edge_in or self.selective_refine:
                if self.edge_map_thre > 0:
                    pred_edge = (pred_edge > self.edge_map_thre).float()

                if "edge" in self.gt_input:
                    assert targets[0] is not None
                    pred_edge = targets[0].unsqueeze(1).float().clone()
                    pred_edge[pred_edge == self.ignore_value] = 0  # remove ignore value

                    if self.selective_refine:
                        pred_edge = self.dilate_filter(pred_edge)
                        # pred_edge = self.dilate_filter(pred_edge)

                    pred_edge = F.interpolate(
                        pred_edge,
                        scale_factor=1 / self.common_stride,
                        mode="bilinear",
                        align_corners=False,
                    )

                    pred_edge[pred_edge > 0] = 1
                if self.strong_feat:
                    snake_input = torch.cat([pred_edge, x], dim=1)
                else:
                    snake_input = torch.cat([pred_edge, features["p2"]], dim=1)
            else:
                if self.strong_feat:
                    snake_input = x
                else:
                    snake_input = features["p2"]

            if self.attention:
                if self.strong_feat:
                    snake_input = torch.cat([att_map, x], dim=1)
                else:
                    # dont cater pred_edge option now
                    snake_input = torch.cat([att_map, features["p2"]], dim=1)

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

                        # TODO: NOTE: Test for theoretic limit. #####
                        # contours = self.refine_head.get_simple_contour(gt_masks)
                        # poly_sample_targets = []
                        # for i, cnt in enumerate(contours):
                        #     if cnt is None:
                        #         xmin, ymin = bboxes[:, 0], bboxes[:, 1]  # (n,)
                        #         xmax, ymax = bboxes[:, 2], bboxes[:, 3]  # (n,)
                        #         box = [
                        #             xmax, ymin, xmin, ymin, xmin, ymax, xmax, ymax
                        #         ]
                        #         box = torch.stack(box, dim=1).view(-1, 4, 2)
                        #         sampled_box = self.refine_head.uniform_upsample(box[None],
                        #                                                         self.refine_head.num_sampling)
                        #         poly_sample_targets.append(sampled_box[i])
                        #         # print(sampled_box.shape)
                        #         continue
                        #
                        #     # 1) uniform-sample
                        #     oct_sampled_targets = self.refine_head.uniform_sample(cnt,
                        #                                                           len(cnt) * self.refine_head.num_sampling)  # (big, 2)
                        #     tt_idx = np.random.randint(len(oct_sampled_targets))
                        #     oct_sampled_targets = np.roll(oct_sampled_targets, -tt_idx, axis=0)[::len(cnt)]
                        #     oct_sampled_targets = torch.tensor(oct_sampled_targets, device=bboxes.device)
                        #     poly_sample_targets.append(oct_sampled_targets)
                        #     # print(oct_sampled_targets.shape)
                        #
                        #     # 2) polar-sample
                        #     # ...
                        # poly_sample_targets = torch.stack(poly_sample_targets, dim=0)
                        # instances_per_im.pred_polys = PolygonPoints(poly_sample_targets)
                        # TODO: NOTE: Test for theoretic limit. #####

                    pred_instances[im_i] = instances_per_im

            new_instances, _ = self.refine_head(snake_input, pred_instances, None)
            # new_instances = pred_instances
            if not self.edge_on:
                pred_edge = torch.rand(1, 1, 5, 5, device=snake_input.device)

            if self.attention:
                pred_edge = att_map

            return pred_edge, {}, new_instances
