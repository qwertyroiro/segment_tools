import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "Depth-Anything_segtools"))
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

import cv2
import torch
from torchvision.transforms import Compose
import matplotlib
from PIL import Image
import numpy as np
from .utils import check_image_type

def render_depth(values, colormap_name="magma_r"):
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)

    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True) # ((1)xhxwx4)
    colors = colors[:, :, :3] # Discard alpha component
    return np.array(colors).convert('RGB')
    # return Image.fromarray(colors)

def Depth_Anything(image, encoder='vitl'):
    image = check_image_type(image)
    image_height, image_width = image.shape[:2]
    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder)).eval()

    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0)

    # depth shape: 1xHxW
    depth = depth_anything(image)
    depth = depth.detach().cpu().numpy()
    # resize depth to original image resolution
    depth = cv2.resize(depth[0], (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    depth_img = render_depth(depth)
    return {'image': depth_img, 'depth': depth}