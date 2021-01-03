import copy

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json, load_sem_seg


def register_coco_edge_map(
        name, metadata, image_root, edge_root, instances_json
):
    ds_name = name
    DatasetCatalog.register(
        ds_name,
        lambda: merge_to_panoptic(
            load_coco_json(instances_json, image_root, ds_name),
            load_sem_seg(edge_root, image_root),
        ),
    )
    MetadataCatalog.get(ds_name).set(
        image_root=image_root,
        edge_root=edge_root,
        json_file=instances_json,
        evaluator_type="coco+edge_map",
        **metadata
    )

    semantic_name = name + "_edgeonly"
    DatasetCatalog.register(semantic_name, lambda: load_sem_seg(edge_root, image_root))
    MetadataCatalog.get(semantic_name).set(
        sem_seg_root=edge_root, image_root=image_root, evaluator_type="sem_seg", **metadata
    )


def register_cityscapes_edge_map(
        name, metadata, image_root, edge_root, instances_json
):
    ds_name = name
    DatasetCatalog.register(
        ds_name,
        lambda: merge_to_panoptic(
            load_coco_json(instances_json, image_root, ds_name),
            load_sem_seg(edge_root, image_root, image_ext='png'),
        ),
    )
    MetadataCatalog.get(ds_name).set(
        image_root=image_root,
        edge_root=edge_root,
        json_file=instances_json,
        evaluator_type="coco+edge_map",
        **metadata
    )

    semantic_name = name + "_edgeonly"
    DatasetCatalog.register(semantic_name, lambda: load_sem_seg(edge_root, image_root))
    MetadataCatalog.get(semantic_name).set(
        sem_seg_root=edge_root, image_root=image_root, evaluator_type="sem_seg", **metadata
    )


def merge_to_panoptic(detection_dicts, sem_seg_dicts):
    """
    Create dataset dicts for panoptic segmentation, by
    merging two dicts using "file_name" field to match their entries.

    Args:
        detection_dicts (list[dict]): lists of dicts for object detection or instance segmentation.
        sem_seg_dicts (list[dict]): lists of dicts for semantic segmentation.

    Returns:
        list[dict] (one per input image): Each dict contains all (key, value) pairs from dicts in
            both detection_dicts and sem_seg_dicts that correspond to the same image.
            The function assumes that the same key in different dicts has the same value.
    """
    results = []
    sem_seg_file_to_entry = {x["file_name"]: x for x in sem_seg_dicts}
    assert len(sem_seg_file_to_entry) > 0

    for det_dict in detection_dicts:
        dic = copy.copy(det_dict)
        dic.update(sem_seg_file_to_entry[dic["file_name"]])
        results.append(dic)
    return results
