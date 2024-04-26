from PIL import Image
import numpy as np
import segment_tools as st
import cv2

image_path = "cityscapes.png"
image_pil = Image.open(image_path)  # Open image with Pillow
image_np = np.array(image_pil)      # Convert to numpy array

import logging
logging.getLogger("fvcore").setLevel(logging.ERROR)
logging.getLogger("detectron2").setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("dinov2").setLevel(logging.ERROR)

oneformer_city = st.OneFormer(dataset="cityscapes")

prompts = ["sidewalk", "road", "car"] # red, green, blue

oneformer_city = st.OneFormer(dataset="cityscapes")

# Using prompt
result = oneformer_city.run(image_np, prompts, color=["Red", "Green", "Blue"])
if result is not None:
    image, ann = result["image"], result["mask"]


cv2.imwrite("test_multi.png", image)
