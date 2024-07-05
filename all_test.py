from PIL import Image
import numpy as np
import segment_tools as st
import cv2
image_path = "cityscapes.jpg"
image_pil = Image.open(image_path)  # Open image with Pillow
image_np = np.array(image_pil)      # Convert to numpy array

import logging
logging.getLogger("fvcore").setLevel(logging.ERROR)
logging.getLogger("detectron2").setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("dinov2").setLevel(logging.ERROR)

prompt = "car"  # Define your prompt


fastsam = st.FastSAM()
result = fastsam.run(image_np)
if result is not None:
    image, ann = result["image"], result["mask"]
image.save("fastsam.jpg")

fastsam = st.FastSAM()
result = fastsam.run(image_np, prompt)
if result is not None:
    image, ann = result["image"], result["mask"]
image.save("fastsam_prompt.jpg")


clipseg = st.CLIPSeg()
result = clipseg.run(image_np, prompt)
if result is not None:
    image, ann = result["image"], result["mask"]
image.save("clipseg.jpg")


dino = st.DINO()
result = dino.run(image_np, prompt)
if result is not None:
    image, bbox = result["image"], result["bbox"]
image.save("dino.jpg")


dinoseg = st.DINOSeg(sam_checkpoint="vit_h")
result = dinoseg.run(image_np, prompt)
if result is not None:
    image, ann = result["image"], result["mask"]
image.save("dinoseg.jpg")

# ADE20K
oneformer_ade20k = st.OneFormer(dataset="ade20k")
result = oneformer_ade20k.run(image_np)
if result is not None:
    image, ann, info = result["image"], result["mask"], result["info"]
image.save("oneformer_ade20k.jpg")

oneformer_ade20k = st.OneFormer(dataset="ade20k")
result = oneformer_ade20k.run(image_np, prompt)
if result is not None:
    image, ann = result["image"], result["mask"]
image.save("oneformer_ade20k_prompt.jpg")

oneformer_ade20k_swin = st.OneFormer(dataset="ade20k", use_swin=True)
result = oneformer_ade20k_swin.run(image_np)
if result is not None:
    image, ann, info = result["image"], result["mask"], result["info"]
image.save("oneformer_ade20k_swin.jpg")

oneformer_ade20k_swin = st.OneFormer(dataset="ade20k", use_swin=True)
result = oneformer_ade20k_swin.run(image_np, prompt)
if result is not None:
    image, ann = result["image"], result["mask"]
image.save("oneformer_ade20k_swin_prompt.jpg")

# Cityscapes
oneformer_city = st.OneFormer(dataset="cityscapes")
result = oneformer_city.run(image_np)
if result is not None:
image, ann, info = result["image"], result["mask"], result["info"]