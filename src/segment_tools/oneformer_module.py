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

sys.path.append(os.path.join(os.path.dirname(__file__), "OneFormer_segtools"))
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

from .utils import mask_class_objects, draw_multi_mask, check_image_type, mask_class_objects_multi

from itertools import cycle

cpu_device = torch.device("cpu")
config_dir = os.path.join(os.path.dirname(__file__), "OneFormer_segtools/configs")
SWIN_CFG_DICT = {
    "ade20k":       "ade20k/swin/oneformer_swin_large_bs16_160k.yaml",
    "cityscapes":   "cityscapes/swin/oneformer_swin_large_bs16_90k.yaml",
    "coco":         "coco/swin/oneformer_swin_large_bs16_100ep.yaml",
    "vistas":       "mapillary_vistas/swin/oneformer_swin_large_bs16_300k.yaml", 
}

CONVNEXT_CFG_DICT = {
    "ade20k":       "ade20k/convnext/oneformer_convnext_large_bs16_160k.yaml",
    "cityscapes":   "cityscapes/convnext/oneformer_convnext_large_bs16_90k.yaml",
    "coco":         "",
    "vistas":       "mapillary_vistas/convnext/oneformer_convnext_large_bs16_300k.yaml",
}

DINAT_CFG_DICT = {
    "ade20k":       "ade20k/dinat/coco_pretrain_oneformer_dinat_large_bs16_160k_1280x1280.yaml",
    "cityscapes":   "cityscapes/dinat/oneformer_dinat_large_bs16_90k.yaml",
    "coco":         "coco/dinat/oneformer_dinat_large_bs16_100ep.yaml",
    "vistas":       "mapillary_vistas/dinat/oneformer_dinat_large_bs16_300k.yaml",
}

SWIN_CFG_DICT = {k: os.path.join(config_dir, v) for k, v in SWIN_CFG_DICT.items()}
CONVNEXT_CFG_DICT = {k: os.path.join(config_dir, v) for k, v in CONVNEXT_CFG_DICT.items()}
DINAT_CFG_DICT = {k: os.path.join(config_dir, v) for k, v in DINAT_CFG_DICT.items()}


def setup_cfg(dataset, model_path, use_swin, use_convnext):
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
    elif use_convnext:
        cfg_path = CONVNEXT_CFG_DICT[dataset]
    else:
        cfg_path = DINAT_CFG_DICT[dataset]
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.MODEL.WEIGHTS = model_path
    cfg.freeze()
    return cfg


def setup_modules(dataset, model_path, use_swin, use_convnext):
    cfg = setup_cfg(dataset, model_path, use_swin, use_convnext)
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
    classes = metadata.stuff_classes

    # クラス名も追加
    for idx, segment_info in enumerate(segments_info):
        segments_info[idx].update({"class": classes[segment_info["category_id"]]})
        
    panoptic_seg = panoptic_seg.cpu().numpy()
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

class OneFormer:
    def __init__(self, dataset="ade20k", use_swin=False, use_convnext=False):
        if dataset == "ade20k":
            if use_swin:
                weight_path = "weights/250_16_swin_l_oneformer_ade20k_160k.pth"
            elif use_convnext:
                weight_path = "weights/250_16_convnext_l_oneformer_ade20k_160k.pth"
            else:
                weight_path = "weights/250_16_dinat_l_oneformer_ade20k_160k.pth"
            if not os.path.exists(weight_path):
                download_weights_ade20k(weight_path, use_swin, use_convnext)
            self.predictor, self.metadata = setup_modules(dataset, weight_path, use_swin, use_convnext)
            
        elif dataset == "cityscapes":
            if use_swin:
                weight_path = "weights/250_16_swin_l_oneformer_cityscapes_90k.pth"
            elif use_convnext:
                weight_path = "weights/250_16_convnext_l_oneformer_cityscapes_90k.pth"
            else:
                weight_path = "weights/250_16_dinat_l_oneformer_cityscapes_90k.pth"
            if not os.path.exists(weight_path):
                download_weights_cityscapes(weight_path, use_swin, use_convnext)
            self.predictor, self.metadata = setup_modules(dataset, weight_path, use_swin, use_convnext)
            
        elif dataset == "coco":
            assert not use_convnext, "convnext is not supported in coco dataset"
            if use_swin:
                weight_path = "weights/150_16_swin_l_oneformer_coco_100ep.pth"
            else:
                weight_path = "weights/150_16_dinat_l_oneformer_coco_100ep.pth"
            if not os.path.exists(weight_path):
                download_weights_coco(weight_path, use_swin)
            self.predictor, self.metadata = setup_modules(dataset, weight_path, use_swin, use_convnext)
            
        elif dataset == "vistas":
            if use_swin:
                weight_path = "weights/250_16_swin_l_oneformer_mapillary_300k.pth"
            elif use_convnext:
                weight_path = "weights/250_16_convnext_l_oneformer_mapillary_300k.pth"
            else:
                weight_path = "weights/250_16_dinat_l_oneformer_mapillary_300k.pth"
            if not os.path.exists(weight_path):
                download_weights_vistas(weight_path, use_swin, use_convnext)
            self.predictor, self.metadata = setup_modules(dataset, weight_path, use_swin, use_convnext)
                
        else:
            raise ValueError("dataset is not supported")
                
    def run(self, image, prompt=None, color="random", task="panoptic", panoptic_mask=False):
        image = check_image_type(image)
        prompt = [prompt] if isinstance(prompt, str) else prompt
        color = [color] if isinstance(color, str) else color
        
        out, panoptic_seg, segments_info = TASK_INFER[task](image, self.predictor, self.metadata)

        try:
            out = out[0] if len(out) == 2 else out
        except:
            pass

        # promptがNoneでない場合のみ、draw_multi_maskを実行(多分重いので)
        # panoptic_maskがTrueの場合、colorはrandomしかできない(色指定する方も面倒かも)
        if prompt is not None:
            prompt_color_map = {prompt_: color_ for prompt_, color_ in zip(prompt, cycle(color))}
            panoptic_seg, output_image = mask_class_objects_multi(seg=panoptic_seg, ann=segments_info, stuff_classes=self.metadata.stuff_classes, image=image, panoptic_mask=panoptic_mask, prompt_color_map=prompt_color_map)
        else:
            output_image = out.get_image()[:, :, ::-1]
        
        return {"image": output_image, "mask": panoptic_seg, "info": segments_info}
                
    def get_labels(self):
        return self.metadata.stuff_classes
