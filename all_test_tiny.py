from PIL import Image
import numpy as np
import cv2
import os
import sys
import contextlib
import logging

image_path = "cityscapes.png"
image_pil = Image.open(image_path)
image_np = np.array(image_pil)
if image_np.shape[2] >= 4:
    image_np = image_np[:, :, :3]
prompt = "person"

image_dir = "image_dir"


@contextlib.contextmanager
def suppress_output(suppress_stdout=False, suppress_logging=True):
    # 既存の標準出力と標準エラー出力を保存
    old_stdout, old_stderr = sys.stdout, sys.stderr
    # 既存のロギングハンドラーを保存
    old_logging_level = logging.root.manager.disable

    try:
        if suppress_stdout:
            # 標準出力と標準エラー出力を無効化するためのダミーのファイル記述子
            with open(os.devnull, "w") as devnull:
                sys.stdout, sys.stderr = devnull, devnull
                if suppress_logging:
                    # ロギングをCRITICAL以上に設定して、ほとんどのログを無効化
                    logging.disable(logging.CRITICAL)
                yield
        else:
            if suppress_logging:
                # ロギングをCRITICAL以上に設定して、ほとんどのログを無効化
                logging.disable(logging.CRITICAL)
            yield
    finally:
        # 標準出力と標準エラー出力を元に戻す
        sys.stdout, sys.stderr = old_stdout, old_stderr
        # ロギングの設定を元に戻す
        logging.disable(old_logging_level)


with suppress_output():
    import segment_tools as st

os.makedirs(f"{image_dir}", exist_ok=True)

with suppress_output():
    print("\nOneFormer_ade20k(dinat)のテスト")
    oneformer_ade20k = st.OneFormer_ade20k()
    result = oneformer_ade20k.run(image_np)
    if result is None:
        print("no result")
    else:
        image, ann = result["image"], result["mask"]
        cv2.imwrite(f"{image_dir}/OneFormer_ade20k(dinat).png", image)

    print("\nOneFormer_ade20k(swin)のテスト")
    oneformer_ade20k = st.OneFormer_ade20k(use_swin=True)
    result = oneformer_ade20k.run(image_np)
    if result is None:
        print("no result")
    else:
        image, ann = result["image"], result["mask"]
        cv2.imwrite(f"{image_dir}/OneFormer_ade20k(swin).png", image)

    print("\nOneFormer_cityscapes(dinat)のテスト")
    oneformer_city = st.OneFormer_cityscapes()
    result = oneformer_city.run(image_np)
    if result is None:
        print("no result")
    else:
        image, ann = result["image"], result["mask"]
        cv2.imwrite(f"{image_dir}/OneFormer_cityscapes(dinat).png", image)

    print("\nOneFormer_cityscapes(swin)のテスト")
    oneformer_city = st.OneFormer_cityscapes(use_swin=True)
    result = oneformer_city.run(image_np)
    if result is None:
        print("no result")
    else:
        image, ann = result["image"], result["mask"]
        cv2.imwrite(f"{image_dir}/OneFormer_cityscapes(swin).png", image)

    print("\nOneFormer_coco(dinat)のテスト")
    oneformer_coco = st.OneFormer_coco()
    result = oneformer_coco.run(image_np)
    if result is None:
        print("no result")
    else:
        image, ann = result["image"], result["mask"]
        cv2.imwrite(f"{image_dir}/OneFormer_coco(dinat).png", image)

    print("\nOneFormer_coco(swin)のテスト")
    oneformer_coco = st.OneFormer_coco(use_swin=True)
    result = oneformer_coco.run(image_np)
    if result is None:
        print("no result")
    else:
        image, ann = result["image"], result["mask"]
        cv2.imwrite(f"{image_dir}/OneFormer_coco(swin).png", image)

    if prompt is not None:
        print("\nOneFormer_ade20k(dinat)(prompt)のテスト")
        oneformer_ade20k = st.OneFormer_ade20k()
        result = oneformer_ade20k.run(image_np, prompt)
        if result is None:
            print("no result")
        else:
            image, ann = result["image"], result["mask"]
            cv2.imwrite(f"{image_dir}/OneFormer_ade20k(dinat)(prompt).png", image)

        print("\nOneFormer_ade20k(swin)(prompt)のテスト")
        oneformer_ade20k = st.OneFormer_ade20k(use_swin=True)
        result = oneformer_ade20k.run(image_np, prompt)
        if result is None:
            print("no result")
        else:
            image, ann = result["image"], result["mask"]
            cv2.imwrite(f"{image_dir}/OneFormer_ade20k(swin)(prompt).png", image)

        print("\nOneFormer_cityscapes(dinat)(prompt)のテスト")
        oneformer_city = st.OneFormer_cityscapes()
        result = oneformer_city.run(image_np, prompt)
        if result is None:
            print("no result")
        else:
            image, ann = result["image"], result["mask"]
            cv2.imwrite(f"{image_dir}/OneFormer_cityscapes(dinat)(prompt).png", image)

        print("\nOneFormer_cityscapes(swin)(prompt)のテスト")
        oneformer_city = st.OneFormer_cityscapes(use_swin=True)
        result = oneformer_city.run(image_np, prompt)
        if result is None:
            print("no result")
        else:
            image, ann = result["image"], result["mask"]
            cv2.imwrite(f"{image_dir}/OneFormer_cityscapes(swin)(prompt).png", image)

        print("\nOneFormer_coco(dinat)(prompt)のテスト")
        oneformer_coco = st.OneFormer_coco()
        result = oneformer_coco.run(image_np, prompt)
        if result is None:
            print("no result")
        else:
            image, ann = result["image"], result["mask"]
            cv2.imwrite(f"{image_dir}/OneFormer_coco(dinat)(prompt).png", image)

        print("\nOneFormer_coco(swin)(prompt)のテスト")
        oneformer_coco = st.OneFormer_coco(use_swin=True)
        result = oneformer_coco.run(image_np, prompt)
        if result is None:
            print("no result")
        else:
            image, ann = result["image"], result["mask"]
            cv2.imwrite(f"{image_dir}/OneFormer_coco(swin)(prompt).png", image)

        print("\ndinoのテスト")
        result = st.dino(image_path, prompt)
        if result is None:
            print("no result")
        else:
            image, bbox = result["image"], result["bbox"]
            cv2.imwrite(f"{image_dir}/dino.png", image)

        print("\ndinosegのテスト")
        result = st.dinoseg(image_path, prompt)
        if result is None:
            print("no result")
        else:
            image, ann = result["image"], result["mask"]
            cv2.imwrite(f"{image_dir}/dinoseg.png", image)

    print("\nfastsam(プロンプトなし)のテスト")
    result = st.fastsam(image_np)
    if result is None:
        print("no result")
    else:
        image, ann = result["image"], result["mask"]
        cv2.imwrite(f"{image_dir}/fastsam.png", image)

    if prompt is not None:
        print("\nfastsam(プロンプトあり)のテスト")
        result = st.fastsam(image_np, prompt)
        if result is None:
            print("no result")
        else:
            image, ann = result["image"], result["mask"]
            cv2.imwrite(f"{image_dir}/fastsam_prompt.png", image)

        print("\nclipsegのテスト")
        result = st.clipseg(image_np, prompt)
        if result is None:
            print("no result")
        else:
            image, ann = result["image"], result["mask"]
            cv2.imwrite(f"{image_dir}/clipseg.png", image)

        print("\nOneFormer_cityscapes(dinat)(prompt)(combine)のテスト")
        oneformer_city = st.OneFormer_cityscapes()
        result = oneformer_city.run(image_np, prompt)
        if result is None:
            print("no result")
        else:
            image, ann = result["image"], result["mask"]
            ann = st.combine_masks(ann)
            ann = ann * 255
            cv2.imwrite(
                f"{image_dir}/OneFormer_cityscapes(dinat)(prompt)(combine).png", ann
            )

print("\nテスト完了")