from .FastSAM.fastsam import FastSAM, FastSAMPrompt

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch

import numpy as np
import warnings
from PIL import Image
import cv2
warnings.filterwarnings('ignore')

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# プロセッサとモデルの準備
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clip_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
fastsam_model = FastSAM('weights/FastSAM.pt')

def fastsam(image_path, text=None, require_image=True, points=None, point_labels=None, bboxes=None, bbox_labels=None):
    """ FastSAMを用いた画像のセグメンテーション
    Args:
        image_path: PILでもnumpyでもパスでも可
        text: テキストプロンプト. Defaults to None.
        points: ポイントプロンプト. points default [[0,0]] or [[x1,y1],[x2,y2]] など. Defaults to None.
        point_labels: 背景か前景か. point_label default [0] or [1,0] or [1] 0:背景, 1:前景. Defaults to None.
        bboxes: ボックスプロンプト.bbox default shape [0,0,0,0] -> [x1,y1,x2,y2].  Defaults to None.
        bbox_labels (_type_, optional): _description_. Defaults to None.

    Returns:
        image: 結果の画像
        ann: アノテーション
    """
    IMAGE_PATH = image_path
    everything_results = fastsam_model(IMAGE_PATH, device=device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
    prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=device)

    # everything prompt
    ann = prompt_process.everything_prompt()
    ann = ann.cpu().numpy()

    if bboxes is not None:
        # # bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
        ann = prompt_process.box_prompt(bboxes=bboxes, boxlabel=bbox_labels)

    if text is not None:
        # text prompt
        ann = prompt_process.text_prompt(text=text)

    if points is not None:
        # point prompt
        # points default [[0,0]] [[x1,y1],[x2,y2]]
        # point_label default [0] [1,0] 0:background, 1:foreground
        ann = prompt_process.point_prompt(points=points, pointlabel=point_labels)
    
    output_image = prompt_process.plot_to_result(annotations=ann)
    
    if require_image:
        return output_image, ann
    else:
        return ann
    
def clipseg(image, text, threshold=100, require_image=True):
    """clipsegを用いた画像のセグメンテーション

    Args:
        image: PILでもnumpyでもパスでも可
        text: テキストプロンプト

    Returns:
        output_np: セグメンテーション結果
    """

    # image_pathがパス or ndarrayの場合は、PIL.Imageへ変換
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    else:
        image = image.copy()
        
    image_shape = (image.width, image.height)
    # 画像のリサイズとRGB変換
    image_ = image.resize((352, 352))
    image_ = image.convert('RGB')
    # プロンプトの準備
    prompts = [text]
    # 推論の準備
    inputs = processor(
        text=prompts,
        images=[image_] * len(prompts),
        padding="max_length",
        return_tensors="pt")
    inputs = inputs.to(device)
    # 推論
    with torch.no_grad():
        outputs = clip_model(**inputs)
    # 確率を取得(352x352)
    preds = outputs.logits
    # 確率をsigmoidにかける(正規化?)
    output = torch.sigmoid(preds)
    # torchからnumpyへ
    output_np = output.cpu().numpy()
    # 0-255へ
    output_np = Image.fromarray((output_np * 255).astype(np.uint8))
    # 元の画像サイズへ
    output_np = output_np.resize(image_shape)
    # PILからnumpyへ
    output_np = np.array(output_np)
    
    output_image = cv2.threshold(output_np, threshold, 255, cv2.THRESH_BINARY)[1]
    ann = output_np.copy()

    if require_image:
        return output_image, ann
    else:
        return ann