######
# @title 3. Import Libraries and other Utilities
######
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()
setup_logger(name="oneformer")

# Import libraries
import numpy as np
import cv2
import torch

# from google.colab.patches import cv2_imshow
import imutils

# Import detectron2 utilities
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog

import os
import sys
import PIL

sys.path.append(os.path.join(os.path.dirname(__file__), "OneFormer_colab_segtools"))
from demo.defaults import DefaultPredictor
from demo.visualizer import Visualizer, ColorMode


# import OneFormer Project
from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)

from .download_weights import *

cpu_device = torch.device("cpu")
config_dir = os.path.join(os.path.dirname(__file__), "OneFormer_colab_segtools/configs")
SWIN_CFG_DICT = {
    "cityscapes": "cityscapes/oneformer_swin_large_IN21k_384_bs16_90k.yaml",
    "coco": "coco/oneformer_swin_large_IN21k_384_bs16_100ep.yaml",
    "ade20k": "ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml",
}

DINAT_CFG_DICT = {
    "cityscapes": "cityscapes/oneformer_dinat_large_bs16_90k.yaml",
    "coco": "coco/oneformer_dinat_large_bs16_100ep.yaml",
    "ade20k": "ade20k/oneformer_dinat_large_IN21k_384_bs16_160k.yaml",
}

SWIN_CFG_DICT = {k: os.path.join(config_dir, v) for k, v in SWIN_CFG_DICT.items()}
DINAT_CFG_DICT = {k: os.path.join(config_dir, v) for k, v in DINAT_CFG_DICT.items()}


def setup_cfg(dataset, model_path, use_swin):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    if use_swin:
        cfg_path = SWIN_CFG_DICT[dataset]
    else:
        cfg_path = DINAT_CFG_DICT[dataset]
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.MODEL.WEIGHTS = model_path
    cfg.freeze()
    return cfg


def setup_modules(dataset, model_path, use_swin):
    cfg = setup_cfg(dataset, model_path, use_swin)
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
    )
    if "cityscapes_fine_sem_seg_val" in cfg.DATASETS.TEST_PANOPTIC[0]:
        from cityscapesscripts.helpers.labels import labels

        stuff_colors = [k.color for k in labels if k.trainId != 255]
        metadata = metadata.set(stuff_colors=stuff_colors)

    return predictor, metadata


def panoptic_run(img, predictor, metadata):
    visualizer = Visualizer(
        img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE
    )
    predictions = predictor(img, "panoptic")
    panoptic_seg, segments_info = predictions["panoptic_seg"]
    out, classes = visualizer.draw_panoptic_seg_predictions(
        panoptic_seg.to(cpu_device), segments_info, alpha=0.5
    )

    # クラス名も追加
    for idx, segment_info in enumerate(segments_info):
        segments_info[idx].update({"class": classes[segment_info["category_id"]]})
    return out, panoptic_seg, segments_info


def instance_run(img, predictor, metadata):
    visualizer = Visualizer(
        img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE
    )
    predictions = predictor(img, "instance")
    instances = predictions["instances"].to(cpu_device)
    out = visualizer.draw_instance_predictions(predictions=instances, alpha=0.5)
    return out


def semantic_run(img, predictor, metadata):
    visualizer = Visualizer(
        img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE
    )
    predictions = predictor(img, "semantic")
    out = visualizer.draw_sem_seg(
        predictions["sem_seg"].argmax(dim=0).to(cpu_device), alpha=0.5
    )
    return out


TASK_INFER = {
    "panoptic": panoptic_run,
    "instance": instance_run,
    "semantic": semantic_run,
}

ade20k_predictor_swin, ade20k_metadata_swin = setup_modules(
    "ade20k", "weights/250_16_swin_l_oneformer_ade20k_160k.pth", True
)

cityscapes_predictor_swin, cityscapes_metadata_swin = setup_modules(
    "cityscapes", "weights/250_16_swin_l_oneformer_cityscapes_90k.pth", True
)

coco_predictor_swin, coco_metadata_swin = setup_modules(
    "coco", "weights/150_16_swin_l_oneformer_coco_100ep.pth", True
)

ade20k_predictor_dinat, ade20k_metadata_dinat = setup_modules(
    "ade20k", "weights/250_16_dinat_l_oneformer_ade20k_160k.pth", False
)

cityscapes_predictor_dinat, cityscapes_metadata_dinat = setup_modules(
    "cityscapes", "weights/250_16_dinat_l_oneformer_cityscapes_90k.pth", False
)

coco_predictor_dinat, coco_metadata_dinat = setup_modules(
    "coco", "weights/150_16_dinat_l_oneformer_coco_100ep.pth", False
)


def process_panoptic(image, predictor, metadata):
    # task = "panoptic"  # @param
    # out = TASK_INFER[task](image, predictor, metadata).get_image()
    out, panoptic_seg, segments_info = panoptic_run(image, predictor, metadata)
    output_image = out.get_image()[:, :, ::-1]
    return output_image


def one_ade20k(image, use_swin=False):
    """ade20kのセグメンテーションを行う

    Args:
        image: PILでもnumpyでもパスでも可

    Returns:
        result_image: セグメンテーション結果
    """

    if use_swin:
        predictor, metadata = ade20k_predictor_swin, ade20k_metadata_swin
    else:
        predictor, metadata = ade20k_predictor_dinat, ade20k_metadata_dinat

    if isinstance(image, str):
        image = cv2.imread(image)
    elif isinstance(image, PIL.Image.Image):
        image = np.array(image)
    elif isinstance(image, np.ndarray):
        image = image.copy()
    else:
        raise ValueError("image type is not supported")

    return process_panoptic(image, predictor, metadata)


def one_cityscapes(image, use_swin=False):
    """cityscapeのセグメンテーションを行う

    Args:
        image: PILでもnumpyでもパスでも可

    Returns:
        result_image: セグメンテーション結果
    """

    if use_swin:
        predictor, metadata = cityscapes_predictor_swin, cityscapes_metadata_swin
    else:
        predictor, metadata = cityscapes_predictor_dinat, cityscapes_metadata_dinat

    if isinstance(image, str):
        image = cv2.imread(image)
    elif isinstance(image, PIL.Image.Image):
        image = np.array(image)
    elif isinstance(image, np.ndarray):
        image = image.copy()
    else:
        raise ValueError("image type is not supported")

    return process_panoptic(image, predictor, metadata)


def one_coco(image, use_swin=False):
    """cocoのセグメンテーションを行う

    Args:
        image: PILでもnumpyでもパスでも可

    Returns:
        result_image: セグメンテーション結果
    """

    if use_swin:
        predictor, metadata = coco_predictor_swin, coco_metadata_swin
    else:
        predictor, metadata = coco_predictor_dinat, coco_metadata_dinat

    if isinstance(image, str):
        image = cv2.imread(image)
    elif isinstance(image, PIL.Image.Image):
        image = np.array(image)
    elif isinstance(image, np.ndarray):
        image = image.copy()
    else:
        raise ValueError("image type is not supported")

    return process_panoptic(image, predictor, metadata)
