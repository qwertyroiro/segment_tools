import numpy as np
from PIL import Image
from scipy.ndimage import label

# draw mask from segment anything
def draw_multi_mask(masks, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = masks[0].shape[-2:]
    mask_image = masks[0].reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    # mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA") # tensor
    mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")

    temp_mask = np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))
    
    for mask in masks[1:]:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        # mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA") # tensor
        mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")
        temp_mask = np.array(Image.alpha_composite(Image.fromarray(temp_mask), mask_image_pil))
        
    return temp_mask

def mask_class_objects(seg: np.ndarray, ann: list, class_name: str) -> np.ndarray:
    # 指定された'class'に対応する'id'を取得
    target_ids = [item['id'] for item in ann if item['class'] == class_name]
    assert len(target_ids) > 0, "class_name is not found in ann"
    separate_masks = []
    # target_idsに含まれるidの位置を1に設定
    for target_id in target_ids:
        mask = np.zeros_like(seg)
        mask[seg == target_id] = 1
        separate_masks.append(mask)
    
    separate_masks = np.array(separate_masks)
    return separate_masks

# def separate_class_masks(seg: np.ndarray, ann: list, class_name: str) -> list:
#     # 指定された'class'に対応する'id'を取得
#     target_ids = [item['id'] for item in ann if item['class'] == class_name]
    
#     # target_idsに含まれるidの位置を1に設定し、それ以外を0に設定するマスクを作成
#     mask = np.isin(seg, target_ids).astype(int)
    
#     # 連結成分のラベリングを使用して、個別のマスクを取得
#     labeled_mask, num_features = label(mask)
    
#     # 個別のマスクをリストに格納
#     separate_masks = []
#     for i in range(1, num_features + 1):
#         separate_masks.append((labeled_mask == i).astype(int))
    
#     separate_masks = np.array(separate_masks)
#     return separate_masks

def separate_masks(seg: np.ndarray) -> list:
    # 連結成分のラベリングを使用して、個別のマスクを取得
    labeled_mask, num_features = label(seg)
    
    # 個別のマスクをリストに格納
    separate_masks = []
    for i in range(1, num_features + 1):
        separate_masks.append((labeled_mask == i).astype(int))
        
    separate_masks = np.array(separate_masks)
    
    return separate_masks