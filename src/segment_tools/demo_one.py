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
    out = visualizer.draw_panoptic_seg_predictions(
        panoptic_seg.to(cpu_device), segments_info, alpha=0.5
    )
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

use_swin = False

# download model checkpoint
import os
import subprocess

if not use_swin:
    if not os.path.exists("250_16_dinat_l_oneformer_cityscapes_90k.pth"):
        subprocess.run(
            "wget https://shi-labs.com/projects/oneformer/cityscapes/250_16_dinat_l_oneformer_cityscapes_90k.pth",
            shell=True,
        )
    predictor, metadata = setup_modules(
        "cityscapes", "250_16_dinat_l_oneformer_cityscapes_90k.pth", use_swin
    )
else:
    if not os.path.exists("250_16_swin_l_oneformer_cityscapes_90k.pth"):
        subprocess.run(
            "wget https://shi-labs.com/projects/oneformer/cityscapes/250_16_swin_l_oneformer_cityscapes_90k.pth",
            shell=True,
        )
    predictor, metadata = setup_modules(
        "cityscapes", "250_16_swin_l_oneformer_cityscapes_90k.pth", use_swin
    )


def process_panoptic(image):
    # task = "panoptic"  # @param
    # out = TASK_INFER[task](image, predictor, metadata).get_image()
    out, panoptic_seg, segments_info = panoptic_run(image, predictor, metadata)
    print(type(out))
    print(type(panoptic_seg))
    print(type(segments_info), len(segments_info))
    print(segments_info)
    cv2.imwrite("result.png", out[:, :, ::-1])
    return out[:, :, ::-1]


def cityscape_test(image):
    """cityscapeのセグメンテーションを行う

    Args:
        image: PILでもnumpyでもパスでも可

    Returns:
        result_image: セグメンテーション結果
    """
    return process_panoptic(image)

image = cv2.imread("cityscapes.png")
out = process_panoptic(image)