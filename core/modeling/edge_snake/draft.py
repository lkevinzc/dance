import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import measure
from shapely.geometry import Polygon
import pycocotools.mask as mask_util

from detectron2.layers import Conv2d, DeformConv, cat, ModulatedDeformConv
from core.structures import ExtremePoints, PolygonPoints

from core.layers import DFConv2d, SmoothL1Loss, ChamferLoss, extreme_utils

from core.modeling.fcose.utils import get_extreme_points

from .deform_head import DeformNet

from detectron2.utils import timer

def sample_octagons(self, pred_instances):
    poly_sample_locations = []
    image_index = []
    for im_i in range(len(pred_instances)):
        instance_per_im = pred_instances[im_i]
        ext_points = instance_per_im.ext_points
        octagons_per_im = ext_points.get_octagons().cpu().numpy().reshape(-1, 8, 2)
        for oct in octagons_per_im:
            # sampling from octagon
            oct_sampled_pts = self.uniform_sample(oct, self.num_sampling)

            oct_sampled_pts = oct_sampled_pts[::-1] if Polygon(
                oct_sampled_pts).exterior.is_ccw else oct_sampled_pts
            assert not Polygon(oct_sampled_pts).exterior.is_ccw, '1) contour must be clock-wise!'

            poly_sample_locations.append(torch.tensor(oct_sampled_pts, device=ext_points.device))
            image_index.append(im_i)

    if not poly_sample_locations:
        return poly_sample_locations, image_index

    poly_sample_locations = torch.stack(poly_sample_locations, dim=0)
    image_index = torch.tensor(image_index)
    return poly_sample_locations, image_index


def compute_loss_for_maskious(self, classes, targets, location_preds, scores):
    if isinstance(location_preds, list):
        # e.g. 4*sum{k}, 128, 2
        classes = classes.repeat(len(location_preds))
        targets = targets.repeat(len(location_preds), 1, 1)
        location_preds = torch.cat(location_preds, dim=0)
    elif len(location_preds) % len(classes) == 0:
        ratio = int(len(location_preds) / len(classes))
        classes = classes.repeat(ratio)
        targets = targets.repeat(ratio, 1, 1)
    else:
        raise ValueError('Number of pairs not match!')

    targets_np = targets.cpu().numpy().reshape(targets.size(0), -1)
    location_preds_np = location_preds.cpu().numpy().reshape(location_preds.size(0), -1)
    ious_w_valid = []
    for (t, l) in zip(targets_np, location_preds_np):
        ious_w_valid.append(_compute_iou_coco(t, l, self.ms_min_area))
    ious_w_valid = torch.tensor(ious_w_valid, device=targets.device)
    select = ious_w_valid[:, 0].bool()
    ious   = ious_w_valid[:, 1]

    maskiou_t = ious[select]
    classes = classes[select]
    scores = scores[select]

    if len(scores) == 0:
        return maskiou_t.sum() * 0

    maskiou_p = torch.gather(scores, dim=1, index=classes[:, None]).view(-1)
    return F.smooth_l1_loss(maskiou_p, maskiou_t, reduction='mean')


def single_segment_matching(num_sampling, dense_targets, sampled_pts, edge_idx):
    ext_idx = edge_idx[::3]  # try ext first, if work then consider finer segments
    aug_ext_idx = torch.cat([ext_idx, torch.tensor([num_sampling], device=ext_idx.device)], dim=0)
    ch_pts = sampled_pts[ext_idx]  # characteristic points
    diff = (ch_pts[:, None, :] - dense_targets[None, :, :]).pow(2).sum(2)
    min_idx = torch.argmin(diff, dim=1)
    # TODO: hard-code 3x.
    aug_min_idx = torch.cat([min_idx, torch.tensor([num_sampling * 3], device=min_idx.device)], dim=0)

    # estimate curvature
    shift_d_l = torch.cat([dense_targets[1:], dense_targets[:1]], dim=0)
    shift_d_r = torch.cat([dense_targets[-1:], dense_targets[:-1]], dim=0)
    cur = ((shift_d_l + shift_d_r) / 2 - dense_targets).pow(2).sum(1)

    cur[::3] += 1e-9    # regular pulses.

    segments = []
    for i in range(4):
        mask = torch.zeros_like(cur)
        mask[aug_min_idx[i]:aug_min_idx[i + 1]] = 1
        interest_idx = torch.argsort(mask * cur, descending=True)[:aug_ext_idx[i + 1] - aug_ext_idx[i]]
        segments.append(torch.sort(interest_idx)[0])
    segments = torch.cat(segments)
    return dense_targets[segments]

def single_uniform_segment_matching(self, dense_targets, sampled_pts, edge_idx):
    ext_idx = edge_idx[::3]  # try ext first, if work then consider finer segments
    aug_ext_idx = torch.cat([ext_idx, torch.tensor([self.num_sampling - 1], device=ext_idx.device)], dim=0)
    ch_pts = sampled_pts[ext_idx]  # characteristic points
    diff = (ch_pts[:, None, :] - dense_targets[None, :, :]).pow(2).sum(2)
    min_idx = torch.argmin(diff, dim=1)
    # TODO: hard-code 3x.
    aug_min_idx = torch.cat([min_idx, torch.tensor([self.num_sampling * 3 - 1], device=min_idx.device)], dim=0)

    before_i = 0
    after_i = 1

    segments = []
    for i in range(4):
        original_len = aug_min_idx[after_i] - aug_min_idx[before_i]
        assert original_len >= 0
        if original_len == 0:
            after_i += 1
            continue

        desired_num_seg = aug_ext_idx[after_i] - aug_ext_idx[before_i]
        assert desired_num_seg >= 0
        if desired_num_seg == 0:
            before_i += 1
            after_i += 1
            continue

        re_sampled_pts = self.uniform_sample_1d(
            dense_targets[aug_min_idx[before_i]: aug_min_idx[after_i]],
            desired_num_seg)

        segments.append(re_sampled_pts)

    segments = np.concatenate(segments, axis=0)
    assert len(segments) == self.num_sampling
    return segments


def segment_matching(dense_targets, sampled_pts, edge_idx):
    ext_idx = edge_idx[:, ::3]  # try ext first, if work then consider finer segments
    seq_idx = torch.arange(ext_idx.size(0)).repeat_interleave(ext_idx.size(1)).to(ext_idx.device)
    ch_pts = sampled_pts[seq_idx, ext_idx.view(-1)].reshape(ext_idx.size(0), ext_idx.size(1), 2)  # characteristic points
    diffs = (ch_pts[:, :, None, :] - dense_targets[:, None, :, :]).pow(2).sum(3)
    min_idx = torch.argmin(diffs, dim=2)


def uniform_sample_1d(pts, new_n):
    n = pts.shape[0]
    if n == new_n:
        return pts
    # len: n - 1
    segment_len = np.sqrt(np.sum((pts[1:] - pts[:-1]) ** 2, axis=1))

    # down-sample or up-sample
    # n
    start_node = np.cumsum(np.concatenate([np.array([0]), segment_len]))
    total_len = np.sum(segment_len)

    new_per_len = total_len / new_n

    mark_1d = ((np.arange(new_n-1) + 1) * new_per_len).reshape(-1, 1)
    locate = (start_node.reshape(1, -1) - mark_1d)
    iss, jss = np.where(locate > 0)
    cut_idx = np.cumsum(np.unique(iss, return_counts=True)[1])
    cut_idx = np.concatenate([np.array([0]), cut_idx[:-1]])

    after_idx = jss[cut_idx]
    before_idx = after_idx - 1

    after_idx[after_idx < 0] = 0

    before = locate[np.arange(new_n-1), before_idx]
    after = locate[np.arange(new_n-1), after_idx]

    w = (- before / (after - before)).reshape(-1, 1)

    sampled_pts = (1 - w) * pts[before_idx] + w * pts[after_idx]

    return np.concatenate([pts[:1], sampled_pts, pts[-1:]], axis=0)










