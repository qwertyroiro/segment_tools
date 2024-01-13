from .FastSAM.fastsam import FastSAM, FastSAMPrompt

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch

import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# プロセッサとモデルの準備
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clip_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
fastsam_model = FastSAM('weights/FastSAM.pt')

def fastsam(image_path, text=None, points=None, point_labels=None, bboxes=None, bbox_labels=None,):
    """ FastSAMを用いた画像のセグメンテーション
    Args:
        image_path: PILでもnumpyでもパスでも可
        text: テキストプロンプト. Defaults to None.
        points: ポイントプロンプト. points default [[0,0]] or [[x1,y1],[x2,y2]] など. Defaults to None.
        point_labels: 背景か前景か. point_label default [0] or [1,0] or [1] 0:背景, 1:前景. Defaults to None.
        bboxes: ボックスプロンプト.bbox default shape [0,0,0,0] -> [x1,y1,x2,y2].  Defaults to None.
        bbox_labels (_type_, optional): _description_. Defaults to None.

    Returns:
        result_image: 結果の画像
        ann: アノテーション
    """
    IMAGE_PATH = image_path
    everything_results = fastsam_model(IMAGE_PATH, device=device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
    prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=device)

    # everything prompt
    ann = prompt_process.everything_prompt()

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
    
    result_image = prompt_process.plot_to_result(annotations=ann)

    return result_image, ann
    
def clipseg(image, text):
    """clipsegを用いた画像のセグメンテーション

    Args:
        image: PILでもnumpyでもパスでも可
        text: テキストプロンプト

    Returns:
        output_np: セグメンテーション結果
    """

    # image_pathがパス or ndarrayの場合は、PIL.Imageへ変換
    if isinstance(image, str):
        from PIL import Image
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        from PIL import Image
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
    # 推論
    with torch.no_grad():
        outputs = clip_model(**inputs)
    # 確率を取得(352x352)
    preds = outputs.logits
    # 確率をsigmoidにかける(正規化?)
    output = torch.sigmoid(preds)
    # torchからnumpyへ
    output_np = output.cpu().numpy()
    # # numpyからPIL.Imageへ
    # output_pil = Image.fromarray((output_np * 255).astype(np.uint8))
    # 元のサイズへリサイズ
    output_np = np.asarray(Image.fromarray(output_np[0][0]).resize(image_shape))
    # # 元のサイズへリサイズ
    # output_pil = output.resize(image_shape)

    return output_np