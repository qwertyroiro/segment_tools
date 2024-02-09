from PIL import Image
import numpy as np
import cv2
import os
import sys
import contextlib
import segment_tools as st

image_path = "cityscapes.png"
image_pil = Image.open(image_path)
image_np = np.array(image_pil)
if image_np.shape[2] >= 4:
    image_np = image_np[:, :, :3]

image_dir = "image_dir"

os.makedirs(f"{image_dir}", exist_ok=True)

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

print("\nテスト完了")
