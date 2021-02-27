# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import numpy as np
from torch.nn import functional as F
import functools
import multiprocessing as mp
from detectron2.layers import ROIAlign
from detectron2.structures import Instances, polygons_to_bitmask
import pycocotools.mask as mask_util
from core.structures import PolygonPoints


def get_polygon_rles(polygons, image_shape):
    # input: N x (p*2)
    polygons = polygons.cpu().numpy()
    h, w = image_shape
    rles = [
        mask_util.merge(mask_util.frPyObjects([p.tolist()], h, w))
        for p in polygons
    ]
    return rles


def detector_postprocess(results,
                         output_height,
                         output_width):
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
    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
    results = Instances((output_height, output_width), **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes

    output_boxes.scale(scale_x, scale_y)
    # now the results.image_size is the one of raw input image
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    if results.has("pred_polys"):
        if results.has("pred_path"):
            snake_path = results.pred_path
            for i in range(snake_path.size(1)):     # number of evolution
                current_poly = PolygonPoints(snake_path[:, i, :, :])
                current_poly.scale(scale_x, scale_y)
                current_poly.clip(results.image_size)
                snake_path[:, i, :, :] = current_poly.tensor

        results.pred_polys.scale(scale_x, scale_y)
        results.pred_polys.clip(results.image_size)
        results.pred_masks = get_polygon_rles(results.pred_polys.flatten(),
                                              (output_height, output_width))
        return results

    else:
        raise ValueError('No pred_polys in instance prediction!')
