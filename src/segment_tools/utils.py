import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label

# draw mask from segment anything
def draw_multi_mask(masks, image, label=None, random_color=True, alpha=0.5):
    """マスクと画像を与えることで、すべてのマスクを画像に重ね合わせた画像を返す関数。

    Args:
        masks: 複数のマスクが含まれるNumPy配列。形状は(x, H, W)で、xはマスクの数、Hは縦のサイズ、Wは横のサイズ。
        image: 画像のNumPy配列。形状は(H, W, C)で、Hは縦のサイズ、Wは横のサイズ、Cはチャンネル数。
        label: マスクに付けるラベル。デフォルトはNone。
        random_color: ランダムな色を使用するかどうか。デフォルトはTrue。
        alpha: マスクの透明度。デフォルトは0.5。
    """
    def get_color(random_color):
        if random_color:
            return np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
        else:
            return np.array([30/255, 144/255, 255/255, alpha])
    
    def draw_label(draw, position, label, font):
        text_size = draw.textsize(label, font=font)
        text_position = (position[0] - text_size[0] // 2, position[1] - text_size[1] // 2)
        # 黒背景を追加
        draw.rectangle([text_position[0] - 2, text_position[1] - 2, text_position[0] + text_size[0] + 2, text_position[1] + text_size[1] + 2], fill=(0, 0, 0, 255))
        draw.text(text_position, label, fill=(255, 255, 255, 255), font=font)
    
    h, w = masks.shape[1], masks.shape[2]
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    
    # フォントの設定（フォントサイズを大きくし、フォントの太さを調整）
    font = ImageFont.truetype("/usr/share/fonts/OTF/ipag.ttf", 24)  # フォントサイズを24に設定
    
    for mask in masks:
        color = get_color(random_color)
        mask_image = np.where(mask.reshape(h, w, 1) > 0, color.reshape(1, 1, -1), 0)
        mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")
        
        if label is not None:
            y, x = np.where(mask > 0)
            if len(x) > 0 and len(y) > 0:
                center_x, center_y = np.mean(x), np.mean(y)
                draw = ImageDraw.Draw(mask_image_pil)
                draw_label(draw, (center_x, center_y), label, font)
        
        annotated_frame_pil = Image.alpha_composite(annotated_frame_pil, mask_image_pil)
    
    return np.array(annotated_frame_pil)

def mask_class_objects(seg: np.ndarray, ann: list, class_name: str, stuff_classes) -> np.ndarray:
    """
    指定されたクラス(int)のオブジェクトをセグメンテーションマスクから分離し、そのマスクを返す関数。
    複数のマスクが出力されず、単一のndarrayにすべてのクラス情報が入っているようなOneFormerなどで使用

    Args:
        seg (np.ndarray): セグメンテーションマスクの配列
        ann (list): 検出結果のアノテーションリスト
        class_name (str): 分離するオブジェクトのクラス名
        stuff_classes: セグメンテーションマスクに含まれるクラスのリスト

    Returns:
        np.ndarray: 分離されたオブジェクトのマスク配列
    """
    
    # ラベルがmetadata['stuff_classes']に含まれていない場合は警告を出す
    if class_name not in stuff_classes:
        print(f"警告: {class_name} はラベルに含まれていません。")
        return seg, True, False
    
    # 指定された'class'に対応する'id'を取得
    target_ids = [item['id'] for item in ann if item['class'] == class_name]
    if len(target_ids) == 0:
        print(f"警告: {class_name} は検出結果に含まれていません。")
        return seg, False, True
    
    separate_masks = []
    # target_idsに含まれるidの位置を1に設定
    for target_id in target_ids:
        mask = np.zeros_like(seg)
        mask[seg == target_id] = 1
        separate_masks.append(mask)
    
    separate_masks = np.array(separate_masks)
    return separate_masks, False, False

def separate_masks(seg: np.ndarray) -> list:
    """
    連結成分のラベリングを使用して、個別のマスクを取得します。
    clipsegのような単一のndarrayにクラス情報のないセグメンテーションマスクを分離するために使用します。

    Parameters:
        seg (np.ndarray): ラベリングされたセグメンテーションマスク

    Returns:
        list: 個別のマスクのリスト
    """
    
    labeled_mask, num_features = label(seg)
    
    separate_masks = []
    for i in range(1, num_features + 1):
        separate_masks.append((labeled_mask == i).astype(int))
        
    separate_masks = np.array(separate_masks)
    
    return separate_masks

def combine_masks(masks: np.ndarray) -> np.ndarray:
    """
    複数のマスクを結合する関数。
    
    :param masks: 形状が(x, H, W)のNumPy配列。xはマスクの数、Hは縦のサイズ、Wは横のサイズ。
    :return: 結合されたマスク(H, W)を返す。
    """
    # 論理和を使ってマスクを結合する
    combined_mask = np.logical_or.reduce(masks, axis=0)
    
    # 結果をboolからintに変換する（必要に応じて）
    combined_mask = combined_mask.astype(int)
    
    return combined_mask