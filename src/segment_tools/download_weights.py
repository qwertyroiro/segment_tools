import os
import requests
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TextColumn,
)

def make_dir(weight_path):
    print(f"{weight_path} not found. Downloading...")
    dir_name = os.path.dirname(weight_path)
    os.makedirs(dir_name, exist_ok=True)

def download_file(url, weight_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 1024

    with open(weight_path, 'wb') as file, Progress(
        SpinnerColumn(),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[progress.description]{task.description}", justify="right"),
    ) as progress:
        task = progress.add_task(f"{weight_path} is Downloading...", total=total_size)
        for data in response.iter_content(chunk_size=chunk_size):
            file.write(data)
            progress.update(task, advance=len(data))

def download_weights_FastSAM(weight_path):
    url = "https://huggingface.co/spaces/An-619/FastSAM/resolve/main/weights/FastSAM.pt"
    make_dir(weight_path)
    download_file(url, weight_path)

def download_weights_SAM(weight_path, vit_size):
    url_list = {
        "vit_h": "sam_vit_h_4b8939.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_b": "sam_vit_b_01ec64.pth",
    }
    url = f"https://dl.fbaipublicfiles.com/segment_anything/{url_list[vit_size]}"
    make_dir(weight_path)
    download_file(url, weight_path)

def download_weights_ade20k(weight_path, use_swin, use_convnext):
    if use_swin:
        url = "https://shi-labs.com/projects/oneformer/ade20k/250_16_swin_l_oneformer_ade20k_160k.pth"
    elif use_convnext:
        url = "https://shi-labs.com/projects/oneformer/ade20k/250_16_convnext_l_oneformer_ade20k_160k.pth"
    else:
        url = "https://shi-labs.com/projects/oneformer/ade20k/250_16_dinat_l_oneformer_ade20k_160k.pth"
    make_dir(weight_path)
    download_file(url, weight_path)

def download_weights_cityscapes(weight_path, use_swin, use_convnext):
    if use_swin:
        url = "https://shi-labs.com/projects/oneformer/cityscapes/250_16_swin_l_oneformer_cityscapes_90k.pth"
    elif use_convnext:
        url = "https://shi-labs.com/projects/oneformer/mapillary/mapillary_pretrain_250_16_convnext_l_oneformer_mapillary_300k.pth"
    else:
        url = "https://shi-labs.com/projects/oneformer/cityscapes/250_16_dinat_l_oneformer_cityscapes_90k.pth"
    make_dir(weight_path)
    download_file(url, weight_path)

def download_weights_coco(weight_path, use_swin):
    if use_swin:
        url = "https://shi-labs.com/projects/oneformer/coco/150_16_swin_l_oneformer_coco_100ep.pth"
    else:
        url = "https://shi-labs.com/projects/oneformer/coco/150_16_dinat_l_oneformer_coco_100ep.pth"
    make_dir(weight_path)
    download_file(url, weight_path)
    
def download_weights_vistas(weight_path, use_swin, use_convnext):
    if use_swin:
        url = "https://shi-labs.com/projects/oneformer/mapillary/250_16_swin_l_oneformer_mapillary_300k.pth"
    elif use_convnext:
        url = "https://shi-labs.com/projects/oneformer/mapillary/250_16_convnext_l_oneformer_mapillary_300k.pth"
    else:
        url = "https://shi-labs.com/projects/oneformer/mapillary/250_16_dinat_l_oneformer_mapillary_300k.pth"
    make_dir(weight_path)
    download_file(url, weight_path)

def download_weights_xmem(weight_path, use_BL30K):
    if use_BL30K:
        url = "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"
    else:
        url = "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth"
    make_dir(weight_path)
    download_file(url, weight_path)

def download_weights_grit(weight_path):
    url = "https://datarelease.blob.core.windows.net/grit/models/grit_b_densecap_objectdet.pth"
    make_dir(weight_path)
    download_file(url, weight_path)