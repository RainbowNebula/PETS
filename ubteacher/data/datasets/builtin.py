 # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import contextlib
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from fvcore.common.timer import Timer
from fvcore.common.file_io import PathManager
import io
import logging

logger = logging.getLogger(__name__)

JSON_ANNOTATIONS_DIR = ""
_SPLITS_COCO_FORMAT = {"coco": {
    "coco_2017_unlabel": (
        "coco/unlabeled2017",
        "coco/annotations/image_info_unlabeled2017.json",
    ),
    "coco_2017_for_voc20": (
        "coco",
        "coco/annotations/google/instances_unlabeledtrainval20class.json",
    ),
    "bdd100k_daytime_car_unlabel": (
        "bdd100k/images/100k/train",
        "bdd100k/cocoAnnotations/caronly_labels/bdd100k_labels_images_daytime_caronly_train_coco.json"
    ),
    "bdd100k_dawn_unlabel": (
        "bdd100k/images/100k/train",
        "bdd100k/cocoAnnotations/time_labels/bdd100k_labels_images_dawn_train_coco.json"
    )
}}

_SPLITS_COCO_FORMAT_LABEL = {"coco": {
    "bdd100k_daytime_car_train": (
        "bdd100k/images/100k/train",
        "bdd100k/cocoAnnotations/caronly_labels/bdd100k_labels_images_daytime_caronly_train_coco.json"
    ),
    "bdd100k_daytime_car_val": (
        "bdd100k/images/100k/val",
        "bdd100k/cocoAnnotations/caronly_labels/bdd100k_labels_images_daytime_caronly_val_coco.json"
    ),
    "cityscapes_car_train": (
        "Cityscapes/leftImg8bit/train",
        "Cityscapes/cocoAnnotations/cityscapes_train_caronly_cocostyle.json"
    ),
    "cityscapes_car_val": (
        "Cityscapes/leftImg8bit/val",
        "Cityscapes/cocoAnnotations/cityscapes_val_caronly_cocostyle.json"
    ),
    "cityscapes_car_train_0.02": (
        "Cityscapes/leftImg8bit/train",
        "Cityscapes/cocoAnnotations/cityscapes_train_caronly_cocostyle.json"
    ),
    "cityscapes_car_val_0.02": (
        "Cityscapes/leftImg8bit/val",
        "Cityscapes/cocoAnnotations/cityscapes_val_caronly_cocostyle.json"
    ),
    "KITTI_car_train": (
        "KITTI/JPEGImages",
        "KITTI/coco_Annotations/train_caronly.json"
    ),
    "KITTI_car_val": (
        "KITTI/JPEGImages",
        "KITTI/coco_Annotations/train_caronly.json"
    ),
    "bdd100k_daytime_train": (
        "bdd100k/images/100k/train",
        "bdd100k/cocoAnnotations/time_labels/bdd100k_labels_images_daytime_train_coco_da.json"
    ),
    "bdd100k_daytime_val": (
        "bdd100k/images/100k/val",
        "bdd100k/cocoAnnotations/time_labels/bdd100k_labels_images_daytime_val_coco_da.json"
    ),
    "bdd100k_daytime_train_notrain": (
        "bdd100k/images/100k/train",
        "bdd100k/cocoAnnotations/time_labels/bdd100k_labels_images_daytime_train_coco_da_notrain.json"
    ),
    "bdd100k_daytime_val_notrain": (
        "bdd100k/images/100k/val",
        "bdd100k/cocoAnnotations/time_labels/bdd100k_labels_images_daytime_val_coco_da_notrain.json"
    ),
    "bdd100k_dawn_train": (
        "bdd100k/images/100k/train",
        "bdd100k/cocoAnnotations/time_labels/bdd100k_labels_images_dawn_train_coco.json"
    ),
    "bdd100k_dawn_val": (
        "bdd100k/images/100k/val",
        "bdd100k/cocoAnnotations/time_labels/bdd100k_labels_images_dawn_val_coco.json"
    ),
    "bdd100k_night_train": (
        "bdd100k/images/100k/train",
        "bdd100k/cocoAnnotations/time_labels/bdd100k_labels_images_night_train_coco.json"
    ),
    "bdd100k_night_val": (
        "bdd100k/images/100k/val",
        "bdd100k/cocoAnnotations/time_labels/bdd100k_labels_images_night_val_coco.json"
    ),
    "cityscapes_train_1": (
        "Cityscapes/leftImg8bit",
        "Cityscapes/cocoAnnotations_2/cityscapes_coco_train.json"
    ),
    "cityscapes_val_1": (
        "Cityscapes/leftImg8bit",
        "Cityscapes/cocoAnnotations_2/cityscapes_coco_val.json"
    ),
    "cityscapes_foggy_train_1": (
        "Cityscapes/leftImg8bit_foggy",
        "Cityscapes/cocoAnnotations_2/foggy_cityscapes_coco_train.json"
    ),
    "cityscapes_foggy_val_1": (
        "Cityscapes/leftImg8bit_foggy",
        "Cityscapes/cocoAnnotations_2/foggy_cityscapes_coco_val.json"
    ),
    "cityscapes_foggy_train_0.02_1": (
        "Cityscapes/leftImg8bit_foggy",
        "Cityscapes/cocoAnnotations_2/foggy_cityscapes_coco_train_0.02.json"
    ),
    "cityscapes_foggy_val_0.02_1": (
        "Cityscapes/leftImg8bit_foggy",
        "Cityscapes/cocoAnnotations_2/foggy_cityscapes_coco_val_0.02.json"
    ),
    "sim10k_car_train": (
        "Sim10k/JPEGImages",
        "Sim10k/coco/trainval10k_caronly.json"
    ),
    "sim10k_car_val": (
        "Sim10k/JPEGImages",
        "Sim10k/coco/trainval10k_caronly.json"
    ),
}}


def register_coco_unlabel(root):
    for _, splits_per_dataset in _SPLITS_COCO_FORMAT.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            meta = {}
            register_coco_unlabel_instances(
                key, meta, os.path.join(root, json_file), os.path.join(root, image_root)
            )


def register_coco_label(root):
    for _, splits_per_dataset in _SPLITS_COCO_FORMAT_LABEL.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            meta = {}
            register_coco_instances(
                key, meta, os.path.join(root, json_file), os.path.join(root, image_root)
            )



def register_coco_unlabel_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root

    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_coco_unlabel_json(json_file, image_root, name)
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def load_coco_unlabel_json(
        json_file, image_root, dataset_name=None, extra_annotation_keys=None
):
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds())
        )

    id_map = None
    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())

    imgs = coco_api.loadImgs(img_ids)

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs), json_file))

    dataset_dicts = []

    for img_dict in imgs:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        dataset_dicts.append(record)

    return dataset_dicts




_root = os.path.abspath("/mnt/csip-107/dataset_for_UDA_OD/")
register_coco_unlabel(_root)
register_coco_label(_root)

