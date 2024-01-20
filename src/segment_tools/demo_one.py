from .OneFormer_segtools.demo.demo import process_image

def cityscape_test(image):
    """cityscapeのセグメンテーションを行う

    Args:
        image: PILでもnumpyでもパスでも可

    Returns:
        result_image: セグメンテーション結果
    """
    return process_image(image)