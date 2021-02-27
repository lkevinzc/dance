import logging
import torch
from torch import nn

from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from .edge_det import build_edge_det_head
from core.modeling.postprocessing import detector_postprocess, edge_map_postprocess

from core.utils import timer


@META_ARCH_REGISTRY.register()
class Dance(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )

        self.refinement_head = build_edge_det_head(cfg, self.backbone.output_shape())

        self.mask_result_src = cfg.MODEL.DANCE.MASK_IN

        self.semantic_filter = cfg.MODEL.DANCE.SEMANTIC_FILTER
        self.semantic_filter_th = cfg.MODEL.DANCE.SEMANTIC_FILTER_TH

        self.need_concave_hull = (
            True if cfg.MODEL.SNAKE_HEAD.LOSS_TYPE == "chamfer" else False
        )

        self.roi_size = cfg.MODEL.DANCE.ROI_SIZE

        self.re_compute_box = cfg.MODEL.DANCE.RE_COMP_BOX

        self.visualize_path = cfg.MODEL.SNAKE_HEAD.VIS_PATH

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def single_test(self, batched_inputs):
        assert len(batched_inputs) == 1
        with timer.env("preprocess"):
            images = batched_inputs[0]["image"].to(self.device)
            images = self.normalizer(images)
            images = ImageList.from_tensors([images], self.backbone.size_divisibility)

        with timer.env("backbone"):
            features = self.backbone(images.tensor)

        gt_instances = None
        gt_sem_seg = None

        with timer.env("fcose"):
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

        if self.mask_result_src != "BOX":
            edge_map, head_losses, proposals = self.refinement_head(
                features, proposals, (gt_sem_seg, [gt_instances, images.image_sizes])
            )

        with timer.env("postprocess"):
            height = batched_inputs[0].get("height", images.image_sizes[0][0])
            width = batched_inputs[0].get("width", images.image_sizes[0][1])
            instance_r = detector_postprocess(
                self.semantic_filter,
                self.semantic_filter_th,
                self.mask_result_src,
                proposals[0],
                height,
                width,
                self.roi_size,
                self.need_concave_hull,
                self.re_compute_box,
            )
            processed_results = [{"instances": instance_r}]
            return processed_results

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        if not self.training and not self.visualize_path:
            return self.single_test(batched_inputs)

        with timer.env("preprocess"):
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [self.normalizer(x) for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        with timer.env("backbone"):
            features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10,
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        if "sem_seg" in batched_inputs[0]:
            gt_sem_seg = [x["sem_seg"].to(self.device) for x in batched_inputs]
            gt_sem_seg = ImageList.from_tensors(
                gt_sem_seg,
                self.backbone.size_divisibility,
                self.refinement_head.ignore_value,
            ).tensor
        else:
            gt_sem_seg = None

        with timer.env("fcose"):
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
        edge_map, head_losses, proposals = self.refinement_head(
            features, proposals, (gt_sem_seg, [gt_instances, images.image_sizes])
        )

        # In training, the proposals are not useful at all in RPN models; but not here
        # This makes RPN-only models about 5% slower.
        if self.training:
            timer.reset()
            proposal_losses.update(head_losses)
            return proposal_losses

        processed_results = []

        with timer.env("postprocess"):
            for per_edge_map, results_per_image, input_per_image, image_size in zip(
                edge_map, proposals, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                # TODO (OPT): NO need for interpolate then back for real speed test
                with timer.env("extra"):
                    edge_map_r = edge_map_postprocess(
                        per_edge_map, image_size, height, width
                    )
                instance_r = detector_postprocess(
                    self.semantic_filter,
                    self.semantic_filter_th,
                    self.mask_result_src,
                    results_per_image,
                    height,
                    width,
                    self.roi_size,
                    self.need_concave_hull,
                    self.re_compute_box,
                )
                processed_results.append(
                    {"instances": instance_r, "edge_map": edge_map_r},
                )
        return processed_results
