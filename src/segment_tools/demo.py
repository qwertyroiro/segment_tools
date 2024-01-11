from .FastSAM.fastsam import FastSAM, FastSAMPrompt

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch

import numpy as np

# プロセッサとモデルの準備
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

def pre_check():
    # もしFastSAM/weights/FastSAM.ptが存在しない場合はdownload_weights.pyを実行
    import os
    if not os.path.exists('weights/FastSAM.pt'):
        print("weights/FastSAM.pt not found. Downloading...")
        from .download_weights import download_weights
        download_weights('weights/FastSAM.pt')

def fastsam(image_path, text=None, points=None, point_labels=None, bboxes=None, bbox_labels=None,):
    pre_check()
    model = FastSAM('weights/FastSAM.pt')
    IMAGE_PATH = image_path
    DEVICE = 'cpu'
    everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
    prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

    # everything prompt
    ann = prompt_process.everything_prompt()

    # # bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
    # ann = prompt_process.box_prompt(bboxes=[[200, 200, 300, 300]])

    # # text prompt
    # ann = prompt_process.text_prompt(text='a photo of a dog')

    # # point prompt
    # # points default [[0,0]] [[x1,y1],[x2,y2]]
    # # point_label default [0] [1,0] 0:background, 1:foreground
    # ann = prompt_process.point_prompt(points=[[620, 360]], pointlabel=[1])

    return prompt_process.plot_to_result(annotations=ann)
    
def clipseg(image, text):

    # image_pathがパスの場合は、PIL.Imageへ変換
    if isinstance(image, str):
        from PIL import Image
        image = Image.open(image)
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
        outputs = model(**inputs)
    # 確率を取得(352x352)
    preds = outputs.logits
    # 確率をsigmoidにかける(正規化?)
    output = torch.sigmoid(preds)
    # torchからnumpyへ
    output = output.cpu().numpy()
    # numpyからPIL.Imageへ
    output = Image.fromarray((output * 255).astype(np.uint8))
    # 元のサイズへリサイズ
    output = output.resize(image_shape)

    return output