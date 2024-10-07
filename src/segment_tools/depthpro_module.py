from .utils import check_image_type
import depth_pro
import cv2
import matplotlib
import numpy as np


class Depth_Pro:
    def __init__(self):
        self.model, self.transform = depth_pro.create_model_and_transforms()
        self.model.eval()

    def run(self, image):
        image = check_image_type(image)
        image = self.transform(image)
        prediction = self.model.infer(image, f_px=None)
        depth = prediction["depth"]
        depth_image = self.__render_depth(depth)
        return {"image": depth_image, "depth": depth}

    def __render_depth(self, values, colormap_name="magma_r"):
        min_value, max_value = values.min(), values.max()
        normalized_values = (values - min_value) / (max_value - min_value)

        colormap = matplotlib.colormaps[colormap_name]
        colors = colormap(normalized_values, bytes=True)  # ((1)xhxwx4)
        colors = colors[:, :, :3]  # Discard alpha component
        return cv2.cvtColor(np.array(colors), cv2.COLOR_BGR2RGB)
