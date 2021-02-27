import logging
import torch
from torch import nn

from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from .postprocessing import (
    detector_postprocess
)

from detectron2.structures import Instances, Boxes
from core.structures import ExtremePoints

from .dsnake_head import SnakeFPNHead


@META_ARCH_REGISTRY.register()
class FcosSnake(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.refinement_head = SnakeFPNHead(cfg, self.backbone.output_shape())
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

        self.gt_input = cfg.TEST.GT_IN.WHAT if cfg.TEST.GT_IN.ON else (None,)

    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0] :
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)

        if not self.training:
            if 'instance' in self.gt_input:
                assert gt_instances is not None

                for im_i in range(len(gt_instances)):
                    gt_instances_per_im = gt_instances[im_i]
                    bboxes = gt_instances_per_im.gt_boxes.tensor
                    instances_per_im = Instances(proposals[im_i]._image_size)
                    instances_per_im.pred_boxes = Boxes(bboxes)
                    instances_per_im.pred_classes = gt_instances_per_im.gt_classes
                    instances_per_im.scores = torch.ones_like(gt_instances_per_im.gt_classes).to(bboxes.device)

                    if gt_instances_per_im.has("gt_masks"):
                        gt_masks = gt_instances_per_im.gt_masks
                        ext_pts_off = self.refinement_head.refine_head.get_simple_extreme_points(
                            gt_masks.polygons).to(bboxes.device)
                        ex_t = torch.stack([ext_pts_off[:, None, 0], bboxes[:, None, 1]], dim=2)
                        ex_l = torch.stack([bboxes[:, None, 0], ext_pts_off[:, None, 1]], dim=2)
                        ex_b = torch.stack([ext_pts_off[:, None, 2], bboxes[:, None, 3]], dim=2)
                        ex_r = torch.stack([bboxes[:, None, 2], ext_pts_off[:, None, 3]], dim=2)
                        instances_per_im.ext_points = ExtremePoints(
                            torch.cat([ex_t, ex_l, ex_b, ex_r], dim=1))
                    else:
                        quad = self.refinement_head.refine_head.get_quadrangle(bboxes).view(-1, 4, 2)
                        instances_per_im.ext_points = ExtremePoints(quad)

                    proposals[im_i] = instances_per_im

        head_losses, proposals = self.refinement_head(features, proposals, gt_instances)

        # In training, the proposals are not useful at all in RPN models; but not here
        # This makes RPN-only models about 5% slower.
        if self.training:
            proposal_losses.update(head_losses)
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            instance_r = detector_postprocess(results_per_image,
                                              height,
                                              width)
            processed_results.append(
                {"instances": instance_r}
            )

        return processed_results
