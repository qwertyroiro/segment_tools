import torch
import numpy as np
from PIL import Image
import os
import sys

# sys.path.append(os.path.join(os.path.dirname(__file__), "XMem"))
from .XMem.model.network import XMem
from .XMem.inference.inference_core import InferenceCore
import cv2
from .XMem.inference.interact.interactive_utils import (
    image_to_torch,
    index_numpy_to_one_hot_torch,
    torch_prob_to_numpy_mask,
    overlay_davis,
)

class XMem:
    def __init__(
        self,
        top_k=30,
        mem_every=5,
        deep_update_every=-1,
        enable_long_term=True,
        enable_long_term_count_usage=True,
        num_prototypes=128,
        min_mid_term_frames=5,
        max_mid_term_frames=10,
        max_long_term_elements=10000,
    ):
        torch.set_grad_enabled(False)
        # default configuration
        self.config = {
            "top_k": top_k,
            "mem_every": mem_every,
            "deep_update_every": deep_update_every,
            "enable_long_term": enable_long_term,
            "enable_long_term_count_usage": enable_long_term_count_usage,
            "num_prototypes": num_prototypes,
            "min_mid_term_frames": min_mid_term_frames,
            "max_mid_term_frames": max_mid_term_frames,
            "max_long_term_elements": max_long_term_elements,
        }
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.network = XMem(self.config, './saves/XMem.pth').eval().to(self.device)
        
        self.processor = InferenceCore(self.network, config=self.config)
        
        # ...
        
        
    def run(self, mask, video):
        num_objects = len(np.unique(mask)) - 1
        torch.cuda.empty_cache()
        
        self.processor.set_all_labels(range(1, num_objects+1))
        
        # ...
        
        # return {"video": output_video, "mask": mask_name}
