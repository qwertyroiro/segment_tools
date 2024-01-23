import os
import subprocess

def download_weights_FastSAM(weight_path):
    url = "https://huggingface.co/spaces/An-619/FastSAM/resolve/main/weights/FastSAM.pt"
    output_directory = os.path.dirname(weight_path)
    output_path = weight_path

    # ディレクトリが存在しない場合は作成
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # wgetコマンドを使用してファイルをダウンロード
    subprocess.run(["wget", url, "-O", output_path])
    
def download_weights_DINO(weight_path):
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    output_directory = os.path.dirname(weight_path)
    output_path = weight_path
    
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    # wgetコマンドを使用してファイルをダウンロード
    subprocess.run(["wget", url, "-O", output_path])

def download_weights_ade20k(weight_path, use_swin):
    if use_swin:
        url = "https://shi-labs.com/projects/oneformer/ade20k/250_16_swin_l_oneformer_ade20k_160k.pth"
    else:
        url = "https://shi-labs.com/projects/oneformer/ade20k/250_16_dinat_l_oneformer_ade20k_160k.pth"
    
    output_directory = os.path.dirname(weight_path)
    output_path = weight_path
    
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    # wgetコマンドを使用してファイルをダウンロード
    subprocess.run(["wget", url, "-O", output_path])
    
def download_weights_cityscapes(weight_path, use_swin):
    if use_swin:
        url = "https://shi-labs.com/projects/oneformer/cityscapes/250_16_swin_l_oneformer_cityscapes_90k.pth"
    else:
        url = "https://shi-labs.com/projects/oneformer/cityscapes/250_16_dinat_l_oneformer_cityscapes_90k.pth"
    
    output_directory = os.path.dirname(weight_path)
    output_path = weight_path
    
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    # wgetコマンドを使用してファイルをダウンロード
    subprocess.run(["wget", url, "-O", output_path])
    
def download_weights_coco(weight_path, use_swin):
    if use_swin:
        url = "https://shi-labs.com/projects/oneformer/coco/150_16_swin_l_oneformer_coco_100ep.pth"
    else:
        url = "https://shi-labs.com/projects/oneformer/coco/150_16_dinat_l_oneformer_coco_100ep.pth"
    
    output_directory = os.path.dirname(weight_path)
    output_path = weight_path
    
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    # wgetコマンドを使用してファイルをダウンロード
    subprocess.run(["wget", url, "-O", output_path])