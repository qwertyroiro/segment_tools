from PIL import Image
import numpy as np
import segment_tools as st

image_path = "cityscapes.png"
image_pil = Image.open(image_path)  # Open image with Pillow
image_np = np.array(image_pil)      # Convert to numpy array

import logging
logging.getLogger("fvcore").setLevel(logging.ERROR)
logging.getLogger("detectron2").setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("dinov2").setLevel(logging.ERROR)

prompt = "car"  # Define your prompt

# fastsam
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
del fastsam

# clipseg
clipseg = st.CLIPSeg()
result = clipseg.run(image_np, prompt)
if result is not None:
    image, ann = result["image"], result["mask"]
image.save("clipseg.jpg")
del clipseg

# dino
dino = st.DINO()
result = dino.run(image_np, prompt)
if result is not None:
    image, bbox = result["image"], result["bbox"]
image.save("dino.jpg")
del dino

# dinoseg
dinoseg = st.DINOSeg(sam_checkpoint="vit_h")
result = dinoseg.run(image_np, prompt)
if result is not None:
    image, ann = result["image"], result["mask"]
image.save("dinoseg.jpg")
del dinoseg

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
del oneformer_ade20k

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
del oneformer_ade20k_swin

# Cityscapes
oneformer_city = st.OneFormer(dataset="cityscapes")
result = oneformer_city.run(image_np)
if result is not None:
    image, ann, info = result["image"], result["mask"], result["info"]
image.save("oneformer_city.jpg")

oneformer_city = st.OneFormer(dataset="cityscapes")
result = oneformer_city.run(image_np, prompt)
if result is not None:
    image, ann = result["image"], result["mask"]
image.save("oneformer_city_prompt.jpg")
del oneformer_city

oneformer_city_swin = st.OneFormer(dataset="cityscapes", use_swin=True)
result = oneformer_city_swin.run(image_np)
if result is not None:
    image, ann, info = result["image"], result["mask"], result["info"]
image.save("oneformer_city_swin.jpg")

oneformer_city_swin = st.OneFormer(dataset="cityscapes", use_swin=True)
result = oneformer_city_swin.run(image_np, prompt)
if result is not None:
    image, ann = result["image"], result["mask"]
image.save("oneformer_city_swin_prompt.jpg")
del oneformer_city_swin

# COCO
oneformer_coco = st.OneFormer(dataset="coco")
result = oneformer_coco.run(image_np)
if result is not None:
    image, ann, info = result["image"], result["mask"], result["info"]
image.save("oneformer_coco.jpg")

oneformer_coco = st.OneFormer(dataset="coco")
result = oneformer_coco.run(image_np, prompt)
if result is not None:
    image, ann = result["image"], result["mask"]
image.save("oneformer_coco_prompt.jpg")
del oneformer_coco

oneformer_coco_swin = st.OneFormer(dataset="coco", use_swin=True)
result = oneformer_coco_swin.run(image_np)
if result is not None:
    image, ann, info = result["image"], result["mask"], result["info"]
image.save("oneformer_coco_swin.jpg")

oneformer_coco_swin = st.OneFormer(dataset="coco", use_swin=True)
result = oneformer_coco_swin.run(image_np, prompt)
if result is not None:
    image, ann = result["image"], result["mask"]
image.save("oneformer_coco_swin_prompt.jpg")
del oneformer_coco_swin

# depth
depth_model = st.Depth_Anything(encoder="vitl") # vits or vitb or vitl
result = depth_model.run(image)
if result is not None:
    image, depth = result["image"], result["depth"]
image.save("depth.jpg")
del depth_model

# DINOv2_depth
depth_model = st.DINOv2_depth(BACKBONE_SIZE="base") # small, base, large, giant
result = depth_model.run(image)
if result is not None:
    depth_img, depth = result["image"], result["depth"]
depth_img.save("depth_dinov2.jpg")
del depth_model