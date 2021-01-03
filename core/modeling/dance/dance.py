'''
 @ Created by: liuzichen@u.nus.edu
 @ Date: 2020-02-17
'''

import logging
import torch
from torch import nn

from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from .edge import build_edge_det_head
from core.modeling.postprocessing import (
    detector_postprocess,
    edge_map_postprocess
)


@META_ARCH_REGISTRY.register()
class Dance(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.refinement_head = build_edge_det_head(cfg, self.backbone.output_shape())

        self.visualize_path = cfg.MODEL.DANCE.VIS_PATH

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def single_test(self, batched_inputs):
        assert len(batched_inputs) == 1
        images = batched_inputs[0]["image"].to(self.device)
        images = self.normalizer(images)
        images = ImageList.from_tensors([images], self.backbone.size_divisibility)

        features = self.backbone(images.tensor)
        gt_instances = None
        gt_sem_seg = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        edge_map, head_losses, proposals = self.refinement_head(features,
                                                                proposals,
                                                                (gt_sem_seg,
                                                                 [gt_instances,
                                                                  images.image_sizes]))
        height = batched_inputs[0].get("height", images.image_sizes[0][0])
        width = batched_inputs[0].get("width", images.image_sizes[0][1])
        instance_r = detector_postprocess(proposals[0], height, width)
        processed_results = [{"instances": instance_r}]
        return processed_results

    def forward(self, batched_inputs):
        if not self.training and not self.visualize_path:
            return self.single_test(batched_inputs)

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        features = self.backbone(images.tensor)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        if "sem_seg" in batched_inputs[0]:
            gt_sem_seg = [x["sem_seg"].to(self.device) for x in batched_inputs]
            gt_sem_seg = ImageList.from_tensors(
                gt_sem_seg, self.backbone.size_divisibility, self.refinement_head.ignore_value
            ).tensor
        else:
            gt_sem_seg = None

        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        edge_map, head_losses, proposals = self.refinement_head(features,
                                                                proposals,
                                                                (gt_sem_seg,
                                                                 [gt_instances,
                                                                  images.image_sizes]))

        # In training, the proposals are not useful at all in RPN models; but not here
        # This makes RPN-only models about 5% slower.
        if self.training:
            proposal_losses.update(head_losses)
            return proposal_losses

        processed_results = []

        for per_edge_map, results_per_image, input_per_image, image_size in zip(
                edge_map, proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            edge_map_r = edge_map_postprocess(per_edge_map, image_size)
            instance_r = detector_postprocess(proposals[0], height, width)
            processed_results.append(
                {"instances": instance_r,
                 "edge_map": edge_map_r},
            )
        return processed_results
