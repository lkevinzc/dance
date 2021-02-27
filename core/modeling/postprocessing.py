# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import numpy as np
from torch.nn import functional as F
import functools
import multiprocessing as mp
from detectron2.layers import ROIAlign
from detectron2.structures import Instances, polygons_to_bitmask, Boxes
import pycocotools.mask as mask_util
from core.structures import PolygonPoints
from core.utils import timer

A = 7


def inverse_liner_ramp(edge_map, p, q):
    return np.exp(A * (1 - edge_map[q[1], q[0]])) * \
           (np.abs(p[0] - q[0]) + np.abs([p[1] - q[1]])) / np.sqrt(2)


# def inverse_liner_ramp(edge_map, p, q):
#     return  (1 - edge_map[q[1], q[0]]) * \
#            (np.abs(p[0] - q[0]) + np.abs([p[1] - q[1]])) / np.sqrt(2)


def get_neighbour(seed, xmin, ymin, xmax, ymax):
    seed = list(seed)
    p = torch.tensor([seed]).repeat(8, 1)
    delta = torch.tensor([
        [0, 1],
        [0, -1],
        [1, 0],
        [-1, 0],
        [1, 1],
        [1, -1],
        [-1, 1],
        [-1, -1]
    ])
    p += delta
    p[:, 0].clamp_(min=xmin, max=xmax)
    p[:, 1].clamp_(min=ymin, max=ymax)
    p = np.unique(p, axis=0).tolist()
    if seed in p:
        p.remove(seed)
    return p


class LiveWire:
    def __init__(self, edge_map, cost_func, neighbour_func, ymax, xmax):
        self.cost_func = cost_func
        self.neighbour_func = neighbour_func
        self.edge_map = edge_map
        self.ymax = ymax
        self.xmax = xmax

    def solve(self, seed):
        self.L = []  # active pixels sorted by cost
        self.expanded = []  # list of pixels that are expanded
        self.total_cost = {}

        left_p = {}

        self.total_cost[seed] = 0
        self.L.append(seed)
        while self.L:
            idx = int(np.argmin([self.total_cost[key] for key in self.L]))
            q = self.L[idx]
            self.L.remove(q)
            self.expanded.append(q)
            for r in self.neighbour_func(q, 0, 0, seed[0], self.ymax):
                r = tuple(r)
                if r in self.expanded:
                    continue

                tmp_cost = self.total_cost[q] + float(self.cost_func(self.edge_map, q, r))

                if r in self.L and tmp_cost < self.total_cost[r]:
                    self.L.remove(r)  # remove higher cost neighbour
                if r not in self.L:
                    self.total_cost[r] = tmp_cost
                    left_p[r] = q
                    self.L.append(r)

        self.L = []  # active pixels sorted by cost
        self.expanded = []  # list of pixels that are expanded
        self.total_cost = {}

        right_p = {}

        self.total_cost[seed] = 0
        self.L.append(seed)
        while self.L:
            idx = int(np.argmin([self.total_cost[key] for key in self.L]))
            q = self.L[idx]
            self.L.remove(q)
            self.expanded.append(q)
            for r in self.neighbour_func(q, seed[0], 0, self.xmax, self.ymax):
                r = tuple(r)
                if r in self.expanded:
                    continue

                tmp_cost = self.total_cost[q] + float(self.cost_func(self.edge_map, q, r))

                if r in self.L and tmp_cost < self.total_cost[r]:
                    self.L.remove(r)  # remove higher cost neighbour
                if r not in self.L:
                    self.total_cost[r] = tmp_cost
                    right_p[r] = q
                    self.L.append(r)

        return left_p, right_p


def backtrack(path, p, s):
    contour = [p]
    while True:
        if p == s:
            return contour
        p = path[p]
        contour.append(p)


def flatten(l):
    return [item for sublist in l for item in sublist]


def walk(single_edge_map, single_ext_points, single_box, roi_size):
    assert single_edge_map.shape[0] == single_edge_map.shape[1]
    a = single_edge_map.shape[0]
    finder = LiveWire(single_edge_map,
                      inverse_liner_ramp,
                      get_neighbour,
                      a - 1,
                      a - 1)
    t = tuple(single_ext_points[0].tolist())
    l = tuple(single_ext_points[1].tolist())
    b = tuple(single_ext_points[2].tolist())
    r = tuple(single_ext_points[3].tolist())

    path_top_left_panel, path_top_right_panel = finder.solve(t)
    poly_tl = backtrack(path_top_left_panel, l, t)
    poly_tl.reverse()
    poly_rt = backtrack(path_top_right_panel, r, t)

    path_bottom_left_panel, path_bottom_right_panel = finder.solve(b)
    poly_lb = backtrack(path_bottom_left_panel, l, b)
    poly_br = backtrack(path_bottom_right_panel, r, b)
    poly_br.reverse()

    poly = poly_tl[:-1] + poly_lb[:-1] + poly_br[:-1] + poly_rt[:-1]

    p = torch.tensor(poly).to(single_box.device).float()

    w = single_box[2] - single_box[0]
    h = single_box[3] - single_box[1]

    p[:, 0] = (p[:, 0] / (roi_size - 1) * w + single_box[0]).floor()
    p[:, 1] = (p[:, 1] / (roi_size - 1) * h + single_box[1]).floor()
    return flatten(p.int().tolist())


# def get_masks(roi_edge_maps, roi_ext_pts, boxes, roi_size):
#     assert roi_edge_maps.shape[0] == roi_ext_pts.shape[0]  # D
#     polygons = []
#
#     for (single_edge_map, single_ext_points, single_box) in zip(
#         roi_edge_maps, roi_ext_pts.long(), boxes
#     ):
#         m = torch.zeros_like(single_edge_map)
#         m[single_ext_points[:, 1], single_ext_points[:, 0]] = 1
#         score = (m * single_edge_map).sum()
#         if score > 0.5:
#             p = walk(single_edge_map,
#                      single_ext_points,
#                      single_box,
#                      roi_size)
#         else:
#             x1, y1, x2, y2 = single_box
#             p = torch.stack([x1, y1, x1, y2, x2, y2, x2, y1]).tolist()
#
#         polygons.append(p)
#     return polygons


def get_masks(roi_edge_maps, roi_ext_pts, boxes, roi_size):
    assert roi_edge_maps.shape[0] == roi_ext_pts.shape[0]  # D
    polygons = []

    for (single_edge_map, single_ext_points, single_box) in zip(
        roi_edge_maps, roi_ext_pts.long(), boxes
    ):

        p = walk(single_edge_map,
                 single_ext_points,
                 single_box,
                 roi_size)

        polygons.append(p)
    return polygons


# TODO: Be CAREFUL to use this
def get_masks_mp(roi_edge_maps, roi_ext_pts, boxes, roi_size):
    assert roi_edge_maps.shape[0] == roi_ext_pts.shape[0]  # D

    pool = mp.Pool(processes=16)

    results = pool.starmap(
        functools.partial(walk, roi_size=roi_size),
        zip(roi_edge_maps, roi_ext_pts, boxes),
        chunksize=5,
    )

    pool.close()
    pool.join()

    return results


def get_bbox_mask(bboxes, image_shape):
    # input: N x 4
    x1, y1 = bboxes[:, 0], bboxes[:, 1]
    x2, y2 = bboxes[:, 0], bboxes[:, 3]
    x3, y3 = bboxes[:, 2], bboxes[:, 3]
    x4, y4 = bboxes[:, 2], bboxes[:, 1]
    rectangles = torch.stack([x1, y1, x2, y2, x3, y3, x4, y4], dim=1).cpu().numpy()
    masks = [polygons_to_bitmask([p], image_shape[0], image_shape[1]) for p in rectangles]
    if not masks:
        return bboxes.new_empty((0,) + image_shape, dtype=torch.uint8)
    return torch.stack([torch.from_numpy(x) for x in masks])

def get_bbox_rles(bboxes, image_shape):
    # input: N x 4
    x1, y1 = bboxes[:, 0], bboxes[:, 1]
    x2, y2 = bboxes[:, 0], bboxes[:, 3]
    x3, y3 = bboxes[:, 2], bboxes[:, 3]
    x4, y4 = bboxes[:, 2], bboxes[:, 1]
    rectangles = torch.stack([x1, y1, x2, y2, x3, y3, x4, y4], dim=1).cpu().numpy()
    h, w = image_shape
    rles = [
        mask_util.merge(mask_util.frPyObjects([o.tolist()], h, w))
        for o in rectangles
    ]
    return rles

def get_octagon_mask(octagons, image_shape):
    # input: N x 16
    octagons_np = octagons.cpu().numpy()
    masks = [polygons_to_bitmask([p], image_shape[0], image_shape[1]) for p in octagons_np]
    if not masks:
        return octagons.new_empty((0,) + image_shape, dtype=torch.uint8)
    return torch.stack([torch.from_numpy(x) for x in masks])


def get_octagon_rles(octagons, image_shape):
    # input: N x 16
    octagons = octagons.cpu().numpy()
    h, w = image_shape
    rles = [
        mask_util.merge(mask_util.frPyObjects([o.tolist()], h, w))
        for o in octagons
    ]
    return rles


def get_polygon_rles(polygons, image_shape):
    # input: N x (p*2)
    polygons = polygons.cpu().numpy()
    h, w = image_shape
    rles = [
        mask_util.merge(mask_util.frPyObjects([p.tolist()], h, w))
        for p in polygons
    ]
    return rles


# def get_polygon_rles(pois, image_shape):
#     # pois: list of list
#     h, w = image_shape
#     rles = [
#         mask_util.merge(mask_util.frPyObjects([p], h, w))
#         for p in pois
#     ]
#     return rles


def detector_postprocess(semantic_filter,
                         semantic_filter_th,
                         mask_result_src,
                         results,
                         output_height,
                         output_width,
                         roi_size,
                         need_concave_hull,
                         re_comp_box):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.

    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    # the results.image_size here is the one the model saw, typically (800, xxxx)

    # with timer.env('postprocess_sub1_get'):
    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
    results = Instances((output_height, output_width), **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes

    # with timer.env('postprocess_sub2_scale'):
    output_boxes.scale(scale_x, scale_y)
        # now the results.image_size is the one of raw input image
    # with timer.env('postprocess_sub3_clip'):
    output_boxes.clip(results.image_size)

    # with timer.env('postprocess_sub4_filter'):
    results = results[output_boxes.nonempty()]

    # with timer.env('postprocess_cp2'):
    if results.has("pred_polys"):
        if results.has("pred_path"):
            with timer.env('extra'):
                snake_path = results.pred_path
                for i in range(snake_path.size(1)):     # number of evolution
                    current_poly = PolygonPoints(snake_path[:, i, :, :])
                    current_poly.scale(scale_x, scale_y)
                    current_poly.clip(results.image_size)
                    snake_path[:, i, :, :] = current_poly.tensor

        # TODO: Note that we did not scale exts (no need if not for evaluation)
        if results.has("ext_points"):
            results.ext_points.scale(scale_x, scale_y)

        results.pred_polys.scale(scale_x, scale_y)

        if re_comp_box:
            results.pred_boxes = Boxes(results.pred_polys.get_box())

        # results.pred_polys.clip(results.image_size)
        # results.pred_masks = get_polygon_rles(results.pred_polys.flatten(),
        #                                       (output_height, output_width))

        return results


    # if semantic_filter and results.has("ext_points"):
    #     if len(results) > 0:
    #         output_ext_points = results.ext_points
    #         keep_on_edge = output_ext_points.onedge(edge_map,
    #                                                 (output_height, output_width),
    #                                                 threshold=semantic_filter_th)
    #         re_weight = keep_on_edge.float() * 0.1 + 0.9
    #         results.scores *= re_weight

    if mask_result_src == 'NO':
        return results

    if mask_result_src == 'BOX':
        results.pred_masks = get_bbox_rles(results.pred_boxes.tensor, (output_height, output_width))

    elif results.has("ext_points"):
        # directly from extreme points to get these results as masks
        results.ext_points.scale(scale_x, scale_y)
        results.ext_points.fit_to_box()

        if mask_result_src == 'OCT_BIT':
            results.pred_masks = get_octagon_mask(results.ext_points.get_octagons(),
                                                  (output_height, output_width))
        elif mask_result_src == 'OCT_RLE':
            results.pred_masks = get_octagon_rles(results.ext_points.get_octagons(),
                                                  (output_height, output_width))
        # elif mask_result_src == 'MASK':
        #     aligned_ext_pts = results.ext_points.align(roi_size).cpu()
        #     batch_inds = torch.tensor([[0.]], device=results.ext_points.device).expand(len(results), 1)
        #     rois = torch.cat([batch_inds, results.pred_boxes.tensor], dim=1)  # Nx5
        #     roi_edge_map = ROIAlign(
        #         (roi_size, roi_size), 1.0, 0, aligned=False
        #     ).forward(edge_map[None, None, :, :].clone(), rois).squeeze(1)
        #     roi_edge_map = roi_edge_map.cpu()   # (D, roi_size, roi_size)
        #
        #     pois = get_masks(roi_edge_map, aligned_ext_pts, results.pred_boxes.tensor.cpu(), roi_size)
        #
        #     results.pred_masks = get_polygon_rles(pois, (output_height, output_width))

    return results


def edge_map_postprocess(result, img_size, output_height, output_width):
    """
    Return semantic segmentation predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    # result = F.interpolate(
    #     result, size=(output_height, output_width), mode="bilinear", align_corners=False
    # )[0][0]
    return result[0][0]
