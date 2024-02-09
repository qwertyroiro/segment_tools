from PIL import Image
import numpy as np
import cv2
import os
import sys
import contextlib
import logging
import segment_tools as st

image_path = "cityscapes.png"
image_pil = Image.open(image_path)
image_np = np.array(image_pil)
if image_np.shape[2] >= 4:
    image_np = image_np[:, :, :3]
prompt = "car"

image_dir = "image_dir"

os.makedirs(f"{image_dir}", exist_ok=True)

import logging
logging.getLogger("fvcore").setLevel(logging.ERROR)
logging.getLogger("detectron2").setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

print("\nOneFormer_ade20k(dinat)のテスト")
oneformer_ade20k = st.OneFormer(dataset="ade20k")
result = oneformer_ade20k.run(image_np)
if result is None:
    print("no result")
else:
    image, ann = result["image"], result["mask"]
    print(image.shape, ann.shape)
    print(f"最大値: {np.max(ann)}, 最小値: {np.min(ann)}")
    cv2.imwrite(f"{image_dir}/OneFormer_ade20k(dinat).png", image)

print("\nOneFormer_ade20k(swin)のテスト")
oneformer_ade20k = st.OneFormer(dataset="ade20k", use_swin=True)
result = oneformer_ade20k.run(image_np)
if result is None:
    print("no result")
else:
    image, ann = result["image"], result["mask"]
    print(image.shape, ann.shape)
    print(f"最大値: {np.max(ann)}, 最小値: {np.min(ann)}")
    cv2.imwrite(f"{image_dir}/OneFormer_ade20k(swin).png", image)

print("\nOneFormer_cityscapes(dinat)のテスト")
oneformer_city = st.OneFormer(dataset="cityscapes")
result = oneformer_city.run(image_np)
if result is None:
    print("no result")
else:
    image, ann = result["image"], result["mask"]
    print(image.shape, ann.shape)
    print(f"最大値: {np.max(ann)}, 最小値: {np.min(ann)}")
    cv2.imwrite(f"{image_dir}/OneFormer_cityscapes(dinat).png", image)

print("\nOneFormer_cityscapes(swin)のテスト")
oneformer_city = st.OneFormer(dataset="cityscapes", use_swin=True)
result = oneformer_city.run(image_np)
if result is None:
    print("no result")
else:
    image, ann = result["image"], result["mask"]
    print(image.shape, ann.shape)
    print(f"最大値: {np.max(ann)}, 最小値: {np.min(ann)}")
    cv2.imwrite(f"{image_dir}/OneFormer_cityscapes(swin).png", image)

print("\nOneFormer_coco(dinat)のテスト")
oneformer_coco = st.OneFormer(dataset="coco")
result = oneformer_coco.run(image_np)
if result is None:
    print("no result")
else:
    image, ann = result["image"], result["mask"]
    print(image.shape, ann.shape)
    print(f"最大値: {np.max(ann)}, 最小値: {np.min(ann)}")
    cv2.imwrite(f"{image_dir}/OneFormer_coco(dinat).png", image)

print("\nOneFormer_coco(swin)のテスト")
oneformer_coco = st.OneFormer(dataset="coco", use_swin=True)
result = oneformer_coco.run(image_np)
if result is None:
    print("no result")
else:
    image, ann = result["image"], result["mask"]
    print(image.shape, ann.shape)
    print(f"最大値: {np.max(ann)}, 最小値: {np.min(ann)}")
    cv2.imwrite(f"{image_dir}/OneFormer_coco(swin).png", image)
    
print("\nFastSAM(プロンプトなし)のテスト")
result = st.FastSAM(image_np)
if result is None:
    print("no result")
else:
    image, ann = result["image"], result["mask"]
    print(image.shape, ann.shape)
    print(f"最大値: {np.max(ann)}, 最小値: {np.min(ann)}")
    cv2.imwrite(f"{image_dir}/FastSAM.png", image)


if prompt is not None:
    print("\nOneFormer_ade20k(dinat)(prompt)のテスト")
    oneformer_ade20k = st.OneFormer(dataset="ade20k")
    result = oneformer_ade20k.run(image_np, prompt)
    if result is None:
        print("no result")
    else:
        image, ann = result["image"], result["mask"]
        print(image.shape, ann.shape)
        print(f"最大値: {np.max(ann)}, 最小値: {np.min(ann)}")
        cv2.imwrite(f"{image_dir}/OneFormer_ade20k(dinat)(prompt).png", image)

    print("\nOneFormer_ade20k(swin)(prompt)のテスト")
    oneformer_ade20k = st.OneFormer(dataset="ade20k", use_swin=True)
    result = oneformer_ade20k.run(image_np, prompt)
    if result is None:
        print("no result")
    else:
        image, ann = result["image"], result["mask"]
        print(image.shape, ann.shape)
        print(f"最大値: {np.max(ann)}, 最小値: {np.min(ann)}")
        cv2.imwrite(f"{image_dir}/OneFormer_ade20k(swin)(prompt).png", image)

    print("\nOneFormer_cityscapes(dinat)(prompt)のテスト")
    oneformer_city = st.OneFormer(dataset="cityscapes")
    result = oneformer_city.run(image_np, prompt)
    if result is None:
        print("no result")
    else:
        image, ann = result["image"], result["mask"]
        print(image.shape, ann.shape)
        print(f"最大値: {np.max(ann)}, 最小値: {np.min(ann)}")
        cv2.imwrite(f"{image_dir}/OneFormer_cityscapes(dinat)(prompt).png", image)
    
    print("\nOneFormer_cityscapes(swin)(prompt)のテスト")
    oneformer_city = st.OneFormer(dataset="cityscapes", use_swin=True)
    result = oneformer_city.run(image_np, prompt)
    if result is None:
        print("no result")
    else:
        image, ann = result["image"], result["mask"]
        print(image.shape, ann.shape)
        print(f"最大値: {np.max(ann)}, 最小値: {np.min(ann)}")
        cv2.imwrite(f"{image_dir}/OneFormer_cityscapes(swin)(prompt).png", image)

    print("\nOneFormer_coco(dinat)(prompt)のテスト")
    oneformer_coco = st.OneFormer(dataset="coco")
    result = oneformer_coco.run(image_np, prompt)
    if result is None:
        print("no result")
    else:
        image, ann = result["image"], result["mask"]
        print(image.shape, ann.shape)
        print(f"最大値: {np.max(ann)}, 最小値: {np.min(ann)}")
        cv2.imwrite(f"{image_dir}/OneFormer_coco(dinat)(prompt).png", image)

    print("\nOneFormer_coco(swin)(prompt)のテスト")
    oneformer_coco = st.OneFormer(dataset="coco", use_swin=True)
    result = oneformer_coco.run(image_np, prompt)
    if result is None:
        print("no result")
    else:
        image, ann = result["image"], result["mask"]
        print(image.shape, ann.shape)
        print(f"最大値: {np.max(ann)}, 最小値: {np.min(ann)}")
        cv2.imwrite(f"{image_dir}/OneFormer_coco(swin)(prompt).png", image)

    print("\nDINOのテスト")
    result = st.DINO(image_path, prompt)
    if result is None:
        print("no result")
    else:
        image, bbox = result["image"], result["bbox"]
        print(image.shape, bbox.shape)
        print(f"最大値: {np.max(bbox)}, 最小値: {np.min(bbox)}")
        cv2.imwrite(f"{image_dir}/dino.png", image)

    print("\nDINOSegのテスト")
    result = st.DINOSeg(image_path, prompt)
    if result is None:
        print("no result")
    else:
        image, ann = result["image"], result["mask"]
        print(image.shape, ann.shape)
        print(f"最大値: {np.max(ann)}, 最小値: {np.min(ann)}")
        cv2.imwrite(f"{image_dir}/DINOSeg.png", image)
        
    print("\nFastSAM(プロンプトあり)のテスト")
    result = st.FastSAM(image_np, prompt)
    if result is None:
        print("no result")
    else:
        image, ann = result["image"], result["mask"]
        print(image.shape, ann.shape)
        print(f"最大値: {np.max(ann)}, 最小値: {np.min(ann)}")
        cv2.imwrite(f"{image_dir}/FastSAM_prompt.png", image)

    print("\nCLIPSegのテスト")
    result = st.CLIPSeg(image_np, prompt)
    if result is None:
        print("no result")
    else:
        image, ann = result["image"], result["mask"]
        print(image.shape, ann.shape)
        print(f"最大値: {np.max(ann)}, 最小値: {np.min(ann)}")
        cv2.imwrite(f"{image_dir}/CLIPSeg.png", image)

    print("\nOneFormer_cityscapes(dinat)(prompt)(combine)のテスト")
    oneformer_city = st.OneFormer(dataset="cityscapes")
    result = oneformer_city.run(image_np, prompt)
    if result is None:
        print("no result")
    else:
        image, ann = result["image"], result["mask"]
        print(image.shape, ann.shape)
        print(f"最大値: {np.max(ann)}, 最小値: {np.min(ann)}")
        ann = st.combine_masks(ann)
        ann = ann * 255
        cv2.imwrite(f"{image_dir}/OneFormer_cityscapes(dinat)(prompt)(combine).png", ann)
            
print("\nテスト完了")
