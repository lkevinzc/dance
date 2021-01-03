# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Adapted for edge map generation from panoptic segmentation data of COCO

import time
import functools
import json
import multiprocessing as mp
import numpy as np
import os
from PIL import Image
import cv2

from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

from panopticapi.utils import rgb2id

EDGE_THICKNESS = 1


def save_edge_map(instance_map, output_edge):
    canvas = np.zeros_like(instance_map)
    for i in range(np.max(instance_map)):
        instance_idx = i + 1
        contours, hierarchy = cv2.findContours(
            (instance_map == instance_idx).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, contours, -1, 1, EDGE_THICKNESS)
    cv2.imwrite(output_edge, canvas)


def _process_panoptic_to_instance(input_panoptic, output_edge, segments, stuff_ids):
    # assuming there is no more that 255 instances in one image;
    # if violated, consider do use rbg instead of gray-scale
    panoptic = np.asarray(Image.open(input_panoptic), dtype=np.uint32)
    panoptic = rgb2id(panoptic)  # map to stuff/thing object ids.
    instance_map = np.zeros_like(panoptic, dtype=np.uint8)
    instance_count = 1
    for seg in segments:
        cat_id = seg["category_id"]
        if cat_id in stuff_ids:
            continue
        else:
            assert instance_count <= 255, 'Too many instances (>256)'
            instance_map[panoptic == seg["id"]] = instance_count
            instance_count += 1
    save_edge_map(instance_map, output_edge)


def separate_coco_edge_map_from_panoptic(panoptic_json, panoptic_root, edge_root, categories):
    os.makedirs(edge_root, exist_ok=True)

    stuff_ids = [k["id"] for k in categories if k["isthing"] == 0]

    with open(panoptic_json) as f:
        obj = json.load(f)

    pool = mp.Pool(processes=max(mp.cpu_count() // 2, 4))

    def iter_annotations():
        for anno in obj["annotations"]:
            file_name = anno["file_name"]
            segments = anno["segments_info"]
            input = os.path.join(panoptic_root, file_name)
            output = os.path.join(edge_root, file_name)
            yield input, output, segments

    print("Start writing to {} ...".format(edge_root))
    start = time.time()
    pool.starmap(
        functools.partial(_process_panoptic_to_instance, stuff_ids=stuff_ids),
        iter_annotations(),
        chunksize=100,
    )
    print("Finished. time: {:.2f}s".format(time.time() - start))


if __name__ == "__main__":
    dataset_dir = os.path.join(os.path.dirname(__file__), "mycoco")
    for s in ["val2017", "train2017"]:
        separate_coco_edge_map_from_panoptic(
            os.path.join(dataset_dir, "annotations/panoptic_{}.json".format(s)),
            os.path.join(dataset_dir, "panoptic_{}".format(s)),
            os.path.join(dataset_dir, "edge_{}".format(s)),
            COCO_CATEGORIES,
        )
