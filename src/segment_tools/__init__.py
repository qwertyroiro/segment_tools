# もしFastSAM/weights/FastSAM.ptが存在しない場合はdownload_weights.pyを実行
import os
if not os.path.exists('weights/FastSAM.pt'):
    print("weights/FastSAM.pt not found. Downloading...")
    from .download_weights import download_weights_FastSAM
    download_weights_FastSAM('weights/FastSAM.pt')
    
if not os.path.exists("weights/sam_vit_h_4b8939.pth"):
    print("weights/sam_vit_h_4b8939.pth not found. Downloading...")
    from .download_weights import download_weights_DINO
    download_weights_DINO("weights/sam_vit_h_4b8939.pth")
    
if not os.path.exists("weights/250_16_swin_l_oneformer_ade20k_160k.pth"):
    print("weights/250_16_swin_l_oneformer_ade20k_160k.pth not found. Downloading...")
    from .download_weights import download_weights_ade20k
    download_weights_ade20k("weights/250_16_swin_l_oneformer_ade20k_160k.pth", use_swin=True)
    
if not os.path.exists("weights/250_16_dinat_l_oneformer_ade20k_160k.pth"):
    print("weights/250_16_dinat_l_oneformer_ade20k_160k.pth not found. Downloading...")
    from .download_weights import download_weights_ade20k
    download_weights_ade20k("weights/250_16_dinat_l_oneformer_ade20k_160k.pth", use_swin=False)

if not os.path.exists("weights/250_16_swin_l_oneformer_cityscapes_90k.pth"):
    print("weights/250_16_swin_l_oneformer_cityscapes_90k.pth not found. Downloading...")
    from .download_weights import download_weights_cityscapes
    download_weights_cityscapes("weights/250_16_swin_l_oneformer_cityscapes_90k.pth", use_swin=True)

if not os.path.exists("weights/250_16_dinat_l_oneformer_cityscapes_90k.pth"):
    print("weights/250_16_dinat_l_oneformer_cityscapes_90k.pth not found. Downloading...")
    from .download_weights import download_weights_cityscapes
    download_weights_cityscapes("weights/250_16_dinat_l_oneformer_cityscapes_90k.pth", use_swin=False)

if not os.path.exists("weights/150_16_swin_l_oneformer_coco_100ep.pth"):
    print("weights/150_16_swin_l_oneformer_coco_100ep.pth not found. Downloading...")
    from .download_weights import download_weights_coco
    download_weights_coco("weights/150_16_swin_l_oneformer_coco_100ep.pth", use_swin=True)

if not os.path.exists("weights/150_16_dinat_l_oneformer_coco_100ep.pth"):
    print("weights/150_16_dinat_l_oneformer_coco_100ep.pth not found. Downloading...")
    from .download_weights import download_weights_coco
    download_weights_coco("weights/150_16_dinat_l_oneformer_coco_100ep.pth", use_swin=False)

from .demo_seg import *
from .demo_ground import *
from .demo_one import *