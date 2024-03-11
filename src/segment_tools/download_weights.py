import os
import subprocess

def make_dir(path):
    os.makedirs(path, exist_ok=True)

def download_weights_FastSAM(weight_path): # onpath
    url = "https://huggingface.co/spaces/An-619/FastSAM/resolve/main/weights/FastSAM.pt"

    make_dir("weights")

    # wgetコマンドを使用してファイルをダウンロード
    subprocess.run(["wget", url, "-O", weight_path])
    
def download_weights_SAM(weight_path):
    weight_path_model = os.path.basename(weight_path)
    url = f"https://dl.fbaipublicfiles.com/segment_anything/{weight_path_model}"
    
    make_dir("weights")
    
    # wgetコマンドを使用してファイルをダウンロード
    subprocess.run(["wget", url, "-O", weight_path])

def download_weights_ade20k(weight_path, use_swin):
    if use_swin:
        url = "https://shi-labs.com/projects/oneformer/ade20k/250_16_swin_l_oneformer_ade20k_160k.pth"
    else:
        url = "https://shi-labs.com/projects/oneformer/ade20k/250_16_dinat_l_oneformer_ade20k_160k.pth"
    
    make_dir("weights")
    
    # wgetコマンドを使用してファイルをダウンロード
    subprocess.run(["wget", url, "-O", weight_path])
    
def download_weights_cityscapes(weight_path, use_swin):
    if use_swin:
        url = "https://shi-labs.com/projects/oneformer/cityscapes/250_16_swin_l_oneformer_cityscapes_90k.pth"
    else:
        url = "https://shi-labs.com/projects/oneformer/cityscapes/250_16_dinat_l_oneformer_cityscapes_90k.pth"
    
    make_dir("weights")
    
    # wgetコマンドを使用してファイルをダウンロード
    subprocess.run(["wget", url, "-O", weight_path])
    
def download_weights_coco(weight_path, use_swin):
    if use_swin:
        url = "https://shi-labs.com/projects/oneformer/coco/150_16_swin_l_oneformer_coco_100ep.pth"
    else:
        url = "https://shi-labs.com/projects/oneformer/coco/150_16_dinat_l_oneformer_coco_100ep.pth"
    
    make_dir("weights")
    
    # wgetコマンドを使用してファイルをダウンロード
    subprocess.run(["wget", url, "-O", weight_path])