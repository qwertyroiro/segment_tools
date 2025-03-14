import math
import itertools
from functools import partial

import torch
import torch.nn.functional as F
import sys
import os
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "dinov2"))
from dinov2.eval.depth.models import build_depther
import matplotlib
from torchvision import transforms
import urllib
import mmcv
from mmcv.runner import load_checkpoint
import cv2
import numpy as np


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(
            itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1])
        )
        output = F.pad(x, pads)
        return output


def create_depther(cfg, backbone_model, backbone_size, head_type):
    train_cfg = cfg.get("train_cfg")
    test_cfg = cfg.get("test_cfg")
    depther = build_depther(cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)

    depther.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
        return_class_token=cfg.model.backbone.output_cls_token,
        norm=cfg.model.backbone.final_norm,
    )

    if hasattr(backbone_model, "patch_size"):
        depther.backbone.register_forward_pre_hook(
            lambda _, x: CenterPadding(backbone_model.patch_size)(x[0])
        )

    return depther


def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()


def make_depth_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            lambda x: 255.0 * x[:3],  # Discard alpha component and scale by 255
            transforms.Normalize(
                mean=(123.675, 116.28, 103.53),
                std=(58.395, 57.12, 57.375),
            ),
        ]
    )


def render_depth(values, colormap_name="magma_r") -> Image:
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)

    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True)  # ((1)xhxwx4)
    colors = colors[:, :, :3]  # Discard alpha component
    return cv2.cvtColor(np.array(colors), cv2.COLOR_BGR2RGB)


def check_image_type(image):
    if isinstance(image, str):
        from PIL import Image

        image_source = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        from PIL import Image

        image_source = Image.fromarray(image).convert("RGB")
    else:
        image_source = image.copy()
    return image_source


class DINOv2_depth:
    def __init__(self, BACKBONE_SIZE="base"):
        BACKBONE_SIZE = BACKBONE_SIZE  # in ("small", "base", "large" or "giant")
        backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        backbone_arch = backbone_archs[BACKBONE_SIZE]
        backbone_name = f"dinov2_{backbone_arch}"

        backbone_model = torch.hub.load(
            repo_or_dir="facebookresearch/dinov2", model=backbone_name
        )
        backbone_model.eval()
        backbone_model.cuda()

        HEAD_DATASET = "nyu"  # in ("nyu", "kitti")
        HEAD_TYPE = "dpt"  # in ("linear", "linear4", "dpt")

        DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
        head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
        head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

        cfg_str = load_config_from_url(head_config_url)
        cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

        self.model = create_depther(
            cfg,
            backbone_model=backbone_model,
            backbone_size=BACKBONE_SIZE,
            head_type=HEAD_TYPE,
        )

        load_checkpoint(self.model, head_checkpoint_url, map_location="cpu")
        self.model.eval()
        self.model.cuda()
        
    def __call__(self, image):
        return self.run(image)

    def run(self, image):
        """画像から深度マップを生成する

        Args:
            image: PILでもnumpyでもパスでも可

        Returns:
            image: 深度マップを描画した画像
            depth: 深度マップ
        """
        image = check_image_type(image)
        transform = make_depth_transform()
        scale_factor = 1
        rescaled_image = image.resize(
            (scale_factor * image.width, scale_factor * image.height)
        )
        transformed_image = transform(rescaled_image)
        batch = transformed_image.unsqueeze(0).cuda()  # Make a batch of one image

        with torch.inference_mode():
            result = self.model.whole_inference(batch, img_meta=None, rescale=True)

        depth = result.squeeze().cpu()
        depth_image = render_depth(depth)
        return {"image": depth_image, "depth": depth.numpy()}
