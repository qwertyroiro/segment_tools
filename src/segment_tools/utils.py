import PIL
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label
import cv2


def get_color(color):
    colors = {
        "Red": [255, 0, 0],
        "Green": [0, 128, 0],
        "Blue": [0, 0, 255],
        "White": [255, 255, 255],
        "Black": [0, 0, 0],
        "Yellow": [255, 255, 0],
        "Cyan": [0, 255, 255],
        "Magenta": [255, 0, 255],
        "Silver": [192, 192, 192],
        "Gray": [128, 128, 128],
        "Maroon": [128, 0, 0],
        "Olive": [128, 128, 0],
        "Purple": [128, 0, 128],
        "Teal": [0, 128, 128],
        "Navy": [0, 0, 128],
        "DodgerBlue": [30, 144, 255],
        "Orange": [255, 165, 0],
        "Pink": [255, 192, 203],
        "Brown": [165, 42, 42],
        "Gold": [255, 215, 0]
    }
    if color == "random":
        return np.random.randint(0, 256, size=3).tolist()
    elif color in colors:
        return colors[color]
    elif isinstance(color, list) and len(color) == 3:
        return color
    else:
        return colors["DodgerBlue"] # 既定の色を返す

# draw mask from segment anything
def draw_multi_mask(
    masks,
    image,
    label=None,
    color="random",
    alpha=0.5,
    font_scale=0.7,
    thickness=1,
    padding=1,
    panoptic_mask=False
):
    """マスクと画像を与えることで、すべてのマスクを画像に重ね合わせた画像を返す関数。

    Args:
        masks: 複数のマスクが含まれるNumPy配列。形状は(x, H, W)で、xはマスクの数、Hは縦のサイズ、Wは横のサイズ。
        image: 画像のNumPy配列。形状は(H, W, C)で、Hは縦のサイズ、Wは横のサイズ、Cはチャンネル数。
        label: マスクに付けるラベル。デフォルトはNone。
        color: どんな色を使用するかどうか。デフォルトはrandom。
        alpha: マスクの透明度。デフォルトは0.5。
        font_scale: テキストのフォントスケール。デフォルトは1。
        thickness: テキストの太さ。デフォルトは2。
        padding: テキストの背景のパディング。デフォルトは3。
    """
    annotated_frame = image.copy()

    color = get_color(color)

    for mask in masks:
        if panoptic_mask:
            color = get_color("random")
        # マスクを描画
        for c in range(3):
            annotated_frame[:, :, c] = np.where(
                mask > 0,
                annotated_frame[:, :, c] * (1 - alpha) + alpha * color[c],
                annotated_frame[:, :, c],
            )

        ## ラベルを描画
        if label is not None:
            y, x= np.where(mask > 0)
            if len(x) > 0 and len(y) > 0:
                center_x, center_y = int(np.mean(x)), int(np.mean(y))
                # テキストのサイズを取得
                text_size = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness
                )[0]
                text_x = center_x - text_size[0] // 2
                text_y = center_y + text_size[1] // 2
                # 黒背景を描画
                vertical_padding = padding - 30  # 縦方向のパディングを少し減らす

                cv2.rectangle(
                    annotated_frame,
                    (text_x - padding, center_y - text_size[1] - vertical_padding),
                    (
                        text_x + text_size[0] + padding,
                        center_y + text_size[1] + vertical_padding,
                    ),
                    (0, 0, 0),
                    -1,
                )
                # cv2.putTextを使用してテキストを描画
                cv2.putText(
                    annotated_frame,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                )

    return annotated_frame


def mask_class_objects(
    seg: np.ndarray, ann: list, class_name: str, stuff_classes, panoptic_mask=False
) -> np.ndarray:
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
        return seg, False

    # 指定された'class'に対応する'id'を取得
    target_ids = [item["id"] for item in ann if item["class"] == class_name]
    if len(target_ids) == 0:
        print(f"警告: {class_name} は検出結果に含まれていません。")
        return seg, False

    separate_masks = []
    # target_idsに含まれるidの位置を1に設定
    for target_id in target_ids:
        mask = np.zeros_like(seg)
        if panoptic_mask:
            mask[seg == target_id] = 1
        else:
            mask[seg == target_id] = 1
        separate_masks.append(mask)

    separate_masks = np.array(separate_masks)

    return separate_masks, True

def mask_class_objects_multi(
    seg: np.ndarray, ann: list, stuff_classes, image: np.ndarray, panoptic_mask=False, prompt_color_map=None
) -> np.ndarray:
    """
    mask_class_objectsをプロンプトごとに複数回実行し、複数のマスクを返す関数。
    """
    
    masks = []
    annotated_frame = image.copy()
    
    for prompt, color in prompt_color_map.items():
        separated_masks, execute_flag= mask_class_objects(seg, ann, prompt, stuff_classes, panoptic_mask=panoptic_mask)
        if execute_flag:
            masks.append(separated_masks)
            annotated_frame = draw_multi_mask(separated_masks, annotated_frame, label=prompt, color=color, panoptic_mask=panoptic_mask)
            

    return masks, annotated_frame[:, :, :3]

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

def check_image_type(image, type="numpy"):
    """path, PIL, numpyのいずれかの形式で与えられた画像をtype(numpy or pil)形式に変換する関数。

    Args:
        image (_type_): path, PIL, numpyのいずれかの形式で与えられた画像
        type (str, optional): numpy or pil. Defaults to "numpy".

    Raises:
        ValueError: image type is not supported

    Returns:
        _type_: type(numpy or pil)形式に変換された画像
    """
    if type == "numpy":
        if isinstance(image, str):
            image = cv2.imread(image)
        elif isinstance(image, PIL.Image.Image):
            image = np.array(image)
        elif isinstance(image, np.ndarray):
            image = image.copy()
        else:
            raise ValueError("image type is not supported")
    elif type == "pil":
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, PIL.Image.Image):
            image = image.copy()
        else:
            raise ValueError("image type is not supported")
    else:
        raise ValueError("type is not supported")
    
    return image

def calc_bboxes(masks):
    """
    セグメンテーションマスクからバウンディングボックスを計算する関数
    
    :param masks: shape [x, H, W] の numpy array
    :return: shape [x, 4] の numpy array (中心x, 中心y, 幅, 高さ) (すべて画像のH,Wに対する相対値)
    """
    
    if type(masks) == list: # prompt指定したとき
        bounding_boxes_list = []
        for mask in masks: # prompt回分
            x, H, W = mask.shape
            bounding_boxes = np.zeros((x, 4), dtype=np.float32)
            
            for i in range(x):
                mask = mask[i]
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                
                # バウンディングボックスの中心座標（相対値）
                center_x = (x_min + x_max) / (2 * W)
                center_y = (y_min + y_max) / (2 * H)
                
                # バウンディングボックスの幅と高さ（相対値）
                width = (x_max - x_min + 1) / W
                height = (y_max - y_min + 1) / H
                
                bounding_boxes[i] = [center_x, center_y, width, height]
                
        bounding_boxes_list.append(bounding_boxes)
        return bounding_boxes_list
    else:
        x, H, W = masks.shape
        bounding_boxes = np.zeros((x, 4), dtype=np.float32)
        
        for i in range(x):
            mask = masks[i]
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # バウンディングボックスの中心座標（相対値）
            center_x = (x_min + x_max) / (2 * W)
            center_y = (y_min + y_max) / (2 * H)
            
            # バウンディングボックスの幅と高さ（相対値）
            width = (x_max - x_min + 1) / W
            height = (y_max - y_min + 1) / H
            
            bounding_boxes[i] = [center_x, center_y, width, height]
        
        return bounding_boxes

def draw_bboxes(image, bboxes, color=(0, 255, 0), thickness=2, point_radius=5):
    """
    画像に複数のバウンディングボックスを描画する関数
    
    :param image: numpy array形式の画像 (BGR順)
    :param bboxes: shape [x, 4] の numpy array (中心x, 中心y, 幅, 高さ)
    :param color: バウンディングボックスの色 (BGR順)
    :param thickness: バウンディングボックスの線の太さ
    :param point_radius: 中心点の半径
    :return: バウンディングボックスが描画された画像 (PIL.Image形式)
    """
    image = check_image_type(image, type="numpy")
    height, width, _ = image.shape

    for bbox in bboxes:
        x_center, y_center, box_width, box_height = bbox

        # 中心座標を画像のピクセル座標に変換
        x_center = int(x_center * width)
        y_center = int(y_center * height)

        # 中心点を描画
        cv2.circle(image, (x_center, y_center), point_radius, color, -1)

        # 幅と高さを画像のピクセル座標に変換
        box_width = int(box_width * width)
        box_height = int(box_height * height)

        # バウンディングボックスの左上と右下の座標を計算
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # バウンディングボックスを描画
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # OpenCV形式(BGR)からPIL形式(RGB)に変換
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)

def calc_polygons(masks):
    """
    セグメンテーションマスクからポリゴンを計算する関数
    
    :param masks: shape [x, H, W] の numpy array
    :return: リスト of リスト of (x, y) 座標のタプル。各マスクに対して1つのポリゴン。
    """
    
    if type(masks) == list: # prompt指定したとき
        polygon_list = []
        for mask in masks: # prompt回分
            x, H, W = masks.shape
            polygons = []
            
            for i in range(x):
                mask = masks[i].astype(np.uint8)  # OpenCVは uint8 を期待する
                
                # 輪郭を見つける
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # 最大の輪郭を選択（通常はこれがオブジェクトの輪郭）
                    contour = max(contours, key=cv2.contourArea)
                    
                    # 輪郭を単純化（オプション）
                    epsilon = 0.005 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # 相対座標に変換
                    polygon = [(float(point[0][0])/W, float(point[0][1])/H) for point in approx]
                    
                    polygons.append(polygon)
                else:
                    # 輪郭が見つからない場合は空のリストを追加
                    polygons.append([])            
        polygon_list.append(polygons)
        return polygon_list
    else:
        x, H, W = masks.shape
        polygons = []
        
        for i in range(x):
            mask = masks[i].astype(np.uint8)
            
            # 輪郭を見つける
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 最大の輪郭を選択（通常はこれがオブジェクトの輪郭）
                contour = max(contours, key=cv2.contourArea)
                
                # 輪郭を単純化（オプション）
                epsilon = 0.005 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # 相対座標に変換
                polygon = [(float(point[0][0])/W, float(point[0][1])/H) for point in approx]
                
                polygons.append(polygon)
        return polygons

def draw_polygons(image, polygons, fill=False, alpha=0.5, color=(0, 255, 0)):
    """
    画像にポリゴンを描画する関数
    
    :param image: 描画対象の画像 (numpy array, shape: [H, W, 3])
    :param polygons: ポリゴンのリスト。各ポリゴンは (x, y) 座標のリスト
    :param fill: ポリゴンを塗りつぶすかどうか (デフォルトは False)
    :param alpha: 塗りつぶす場合の透明度 (0.0 ~ 1.0, デフォルトは 0.5)
    :param color: ポリゴンの色 (B, G, R)
    :return: ポリゴンが描画された画像
    """
    H, W = image.shape[:2]
    result = check_image_type(image, type="numpy")
    overlay = result.copy()
    
    for polygon in polygons:
        # 相対座標を画像の実際の座標に変換
        points = np.array([(int(x * W), int(y * H)) for x, y in polygon], np.int32)
        points = points.reshape((-1, 1, 2))
        
        if fill:
            # ポリゴンを塗りつぶす
            cv2.fillPoly(overlay, [points], color)
        else:
            # ポリゴンの輪郭のみを描画
            cv2.polylines(result, [points], True, color, 2)
    
    if fill:
        # 透明度を適用して元の画像と合成
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
    
    return result