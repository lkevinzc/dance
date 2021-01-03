import os

from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data import DatasetCatalog, MetadataCatalog
from .datasets import register_coco_edge_map, register_cityscapes_edge_map

'''
Register COCO dataset with edge map annotations
'''

SPLITS_COCO_W_EDGE = {
    "coco_2017_train_edge": (
        # original directory/annotations coco detection
        "coco/train2017",
        "coco/annotations/instances_train2017.json",
        # directory for edge map created by datasets/prepare_edge_map.py
        # takes ~ 12 mins on a machine with 64 Xeon(R) Gold 6130 CPUs
        "coco/edge_train2017"
    ),
    "coco_2017_val_edge": (
        "coco/val2017",
        "coco/annotations/instances_val2017.json",
        "coco/edge_val2017"
    ),
}


def register_all_coco_edge(root="datasets"):
    for name, (image_root, json_file, edge_root) in SPLITS_COCO_W_EDGE.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_edge_map(
            name,
            _get_builtin_metadata("coco"),
            os.path.join(root, image_root),
            os.path.join(root, edge_root),
            os.path.join(root, json_file) if "://" not in json_file else json_file
        )


register_all_coco_edge()

'''
Register CITYSCAPES dataset with edge map annotations
'''

SPLITS_CITY_W_EDGE = {
    "cityscapes_train_edge": (
        # original directory/annotations coco detection
        "cityscape-coco/coco_img/train",
        "cityscape-coco/coco_ann/instance_train.json",
        "cityscape-coco/edge_train"
    ),
    "cityscapes_val_edge": (
        "cityscape-coco/coco_img/val",
        "cityscape-coco/coco_ann/instance_val.json",
        "cityscape-coco/edge_val"
    ),
}


def register_all_cityscapes_edge(root="datasets"):
    for name, (image_root, json_file, edge_root) in SPLITS_CITY_W_EDGE.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_cityscapes_edge_map(
            name,
            {},
            os.path.join(root, image_root),
            os.path.join(root, edge_root),
            os.path.join(root, json_file) if "://" not in json_file else json_file
        )


register_all_cityscapes_edge()


def register_cityscapes(root="datasets"):
    # Assume pre-defined datasets live in `./datasets`.
    DatasetCatalog.register('cityscapes_coco_fine_instance_seg_train',
                            lambda: load_coco_json(
                                os.path.join(root, 'cityscape-coco/coco_ann/instance_train.json'),
                                os.path.join(root, 'cityscape-coco/coco_img/train'),
                                'cityscapes_coco_fine_instance_seg_train'))

    DatasetCatalog.register('cityscapes_coco_fine_instance_seg_val',
                            lambda: load_coco_json(
                                os.path.join(root, 'cityscape-coco/coco_ann/instance_val.json'),
                                os.path.join(root, 'cityscape-coco/coco_img/val'),
                                'cityscapes_coco_fine_instance_seg_val'))
    MetadataCatalog.get('cityscapes_coco_fine_instance_seg_train').set(
        evaluator_type="coco",
    )
    MetadataCatalog.get('cityscapes_coco_fine_instance_seg_val').set(
        evaluator_type="coco",
    )


register_cityscapes()

'''
Register SBD dataset
'''

_PREDEFINED_SPLITS_SBD = {
    "sbd_train": ("sbd/images", "sbd/annotations/sbd_train_instance.json"),
    "sbd_val": ("sbd/images", "sbd/annotations/sbd_val_instance.json"),
}

SBD_CATEGORIES = [
    {"color": [220, 20, 60], 'id': 1, 'name': 'aeroplane'},
    {"color": [119, 11, 32], 'id': 2, 'name': 'bicycle'},
    {"color": [0, 0, 142], 'id': 3, 'name': 'bird'},
    {"color": [0, 0, 230], 'id': 4, 'name': 'boat'},
    {"color": [106, 0, 228], 'id': 5, 'name': 'bottle'},
    {"color": [0, 60, 100], 'id': 6, 'name': 'bus'},
    {"color": [0, 80, 100], 'id': 7, 'name': 'car'},
    {"color": [0, 0, 70], 'id': 8, 'name': 'cat'},
    {"color": [0, 0, 192], 'id': 9, 'name': 'chair'},
    {"color": [250, 170, 30], 'id': 10, 'name': 'cow'},
    {"color": [100, 170, 30], 'id': 11, 'name': 'diningtable'},
    {"color": [220, 220, 0], 'id': 12, 'name': 'dog'},
    {"color": [175, 116, 175], 'id': 13, 'name': 'horse'},
    {"color": [0, 82, 0], 'id': 14, 'name': 'motorbike'},
    {"color": [0, 82, 100], 'id': 15, 'name': 'person'},
    {"color": [82, 82, 100], 'id': 16, 'name': 'pottedplant'},
    {"color": [182, 8, 100], 'id': 17, 'name': 'sheep'},
    {"color": [182, 8, 0], 'id': 18, 'name': 'sofa'},
    {"color": [182, 18, 0], 'id': 19, 'name': 'train'},
    {"color": [12, 18, 192], 'id': 20, 'name': 'tvmonitor'}
]

thing_ids = [k["id"] for k in SBD_CATEGORIES]
thing_colors = [k["color"] for k in SBD_CATEGORIES]
assert len(thing_ids) == 20, len(thing_ids)
# Mapping from the incontiguous COCO category id to an id in [0, 19]
thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
thing_classes = [k["name"] for k in SBD_CATEGORIES]
metadata = {
    "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
    "thing_classes": thing_classes,
    "thing_colors": thing_colors,
}


def register_all_coco(root="datasets"):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_SBD.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            metadata,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


register_all_coco()
