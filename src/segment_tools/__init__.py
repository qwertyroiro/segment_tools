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

from .demo_seg import *
from .demo_ground import *
from .demo_one import *
from .demo_depth import *
from .demo_dinov2 import *
from .utils import *