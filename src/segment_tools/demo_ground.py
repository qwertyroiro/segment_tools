from PIL import Image
# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, predict
# segment anything
from segment_anything import build_sam, SamPredictor 
import numpy as np
# diffusers
import torch
from huggingface_hub import hf_hub_download
import warnings
warnings.filterwarnings('ignore')
from segment_tools.utils import draw_multi_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
sam_checkpoint = 'weights/sam_vit_h_4b8939.pth'
sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
   
# groundingdinoのモデルを返す
def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    args.device = device
    model = build_model(args)
    model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model   

groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device)

# groundingdino用の画像の前処理
def __load_image(image):
    # image_pathがパス or ndarrayの場合は、PIL.Imageへ変換
    if isinstance(image, str):
        from PIL import Image
        image_source = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        from PIL import Image
        image_source = Image.fromarray(image).convert("RGB")
    else:
        image_source = image.copy()
    
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


# detect object using grounding DINO
def __detect(image, image_source, text_prompt, model, box_threshold = 0.3, text_threshold = 0.25):
  boxes, logits, phrases = predict(
      model=model, 
      image=image, 
      caption=text_prompt,
      box_threshold=box_threshold,
      text_threshold=text_threshold,
      device=device,
      )
  annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
  annotated_frame = annotated_frame[...,::-1] # BGR to RGB 
  return annotated_frame, boxes 

# segment object using segment anything
def __segment(image, sam_model, boxes):
  sam_model.set_image(image)
  H, W, _ = image.shape
  boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

  transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(device), image.shape[:2])
  masks, _, _ = sam_model.predict_torch(
      point_coords = None,
      point_labels = None,
      boxes = transformed_boxes,
      multimask_output = False,
      )
  return masks.cpu()

# draw mask from segment anything
def __draw_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

# draw mask from segment anything
def __draw_multi_mask(masks, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = masks[0].shape[-2:]
    mask_image = masks[0].reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    temp_mask = np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))
    
    for mask in masks[1:]:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")
        temp_mask = np.array(Image.alpha_composite(Image.fromarray(temp_mask), mask_image_pil))
        
    return temp_mask

    

def DINO(image, text):
    """DINOを用いた画像のゼロショット物体検出

    Args:
        image: PILでもnumpyでもパスでも可
        text: テキストプロンプト

    Returns:
        annotated_frame: 物体検出結果
        detected_boxes: 物体検出結果のバウンディングボックス
    """
    image_source, image = __load_image(image)
    annotated_frame, detected_boxes = __detect(image, image_source, text_prompt=text, model=groundingdino_model)
    # detected_boxesをtensorからndarrayに変換
    detected_boxes = detected_boxes.cpu().numpy()
    
    # RGB to BGR
    annotated_frame = annotated_frame[:, :, ::-1]
    
    return {"image": annotated_frame, "bbox": detected_boxes}
    
def DINOSeg(image, text):
    """dinoを用いた画像のセグメンテーション

    Args:
        image: PILでもnumpyでもパスでも可
        text: テキストプロンプト

    Returns:
        segmented_frame_masks: セグメンテーション結果
        annotated_frame_with_mask: セグメンテーション結果を重ねた画像
        detected_boxes: 物体検出結果のバウンディングボックス
    """
    image_source, image = __load_image(image)
    annotated_frame, detected_boxes = __detect(image, image_source, text_prompt=text, model=groundingdino_model)
    if len(detected_boxes) == 0:
        print("警告: 物体が検出されませんでした。")
        return None
    segmented_frame_masks = __segment(image_source, sam_predictor, boxes=detected_boxes)
    segmented_frame_masks = segmented_frame_masks.cpu().numpy()
    segmented_frame_masks = segmented_frame_masks[:, 0, :, :]
    annotated_frame_with_mask = draw_multi_mask(segmented_frame_masks, annotated_frame)
    annotated_frame_with_mask = annotated_frame_with_mask[:, :, :3]
    
    
    detected_boxes = detected_boxes.cpu().numpy()
    segmented_frame_masks = segmented_frame_masks.astype(int)
    
    # RGB to BGR
    annotated_frame_with_mask = annotated_frame_with_mask[:, :, ::-1]
    
    return {"image": annotated_frame_with_mask, "mask": segmented_frame_masks, "bbox": detected_boxes}