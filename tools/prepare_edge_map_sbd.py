# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Modified by Zichen Liu for edge map purpose

import time
import functools
import multiprocessing as mp
import numpy as np
import os
import cv2

from pycocotools.coco import COCO

EDGE_THICKNESS = 1


def save_edge_map(edge_fn, mask, im_size):
    canvas = np.zeros(im_size)
    all_segs = list(map(lambda x: x['segmentation'], mask))
    counters = []
    for segs in all_segs:
        counters += [
            np.expand_dims(np.array(seg, dtype=np.int32).reshape(-1, 2), 0)
            for seg in segs
        ]
    cv2.drawContours(canvas, counters, -1, 1, EDGE_THICKNESS)
    cv2.imwrite(edge_fn, canvas)


def _process_json_to_mask(file_name, height, width, ann, edge_root):
    edge_fn = os.path.join(edge_root, os.path.basename(file_name))
    save_edge_map(edge_fn, ann, [height, width])


def generate_coco_edge_map_from_json(instance_json, edge_root):
    os.makedirs(edge_root, exist_ok=True)

    coco_api = COCO(instance_json)
    img_ids = sorted(coco_api.imgs.keys())

    pool = mp.Pool(processes=max(mp.cpu_count() // 2, 4))

    def iter_annotations():
        count = 0
        for img_id in img_ids:
            img_info = coco_api.loadImgs(img_id)[0]
            file_name = img_info['file_name']
            width = img_info['width']
            height = img_info['height']

            ann = coco_api.imgToAnns[img_id]
            count += 1
            yield file_name, height, width, ann
        print(count)

    print("Start writing to {} ...".format(edge_root))
    start = time.time()
    pool.starmap(
        functools.partial(_process_json_to_mask, edge_root=edge_root),
        iter_annotations(),
        chunksize=100,
    )
    print("Finished. time: {:.2f}s".format(time.time() - start))


if __name__ == "__main__":
    dataset_dir = '/ldap_home/zichen.liu/data/train_data/sbd/sbd'
    for s in ["sbd_train_instance", "sbd_trainval_instance"]:
        generate_coco_edge_map_from_json(
            os.path.join(dataset_dir, "annotations/{}.json".format(s)),
            os.path.join(dataset_dir, "edge_{}".format(s)),
        )
