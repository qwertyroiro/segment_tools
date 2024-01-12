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