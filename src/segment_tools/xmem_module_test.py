import torch
import numpy as np
from PIL import Image
from XMem.model.network import XMem
from XMem.inference.inference_core import InferenceCore
import cv2
from XMem.inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis

torch.set_grad_enabled(False)

# default configuration
config = {
    'top_k': 30,
    'mem_every': 5,
    'deep_update_every': -1,
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    'num_prototypes': 128,
    'min_mid_term_frames': 5,
    'max_mid_term_frames': 10,
    'max_long_term_elements': 10000,
}

if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'

network = XMem(config, './saves/XMem.pth').eval().to(device)
processor = InferenceCore(network, config=config)
video_name = 'video.mp4'
mask_name = 'test.png'

mask = np.array(Image.open(mask_name))
num_objects = len(np.unique(mask)) - 1

torch.cuda.empty_cache()


processor.set_all_labels(range(1, num_objects+1)) # consecutive labels
cap = cv2.VideoCapture(video_name)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
output_video_name = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_name, fourcc, fps, (frame_width, frame_height))

# You can change these two numbers
frames_to_propagate = 200
visualize_every = 20

current_frame_index = 0

with torch.cuda.amp.autocast(enabled=True):
  while (cap.isOpened()):
    # load frame-by-frame
    _, frame = cap.read()
    if frame is None or current_frame_index > frames_to_propagate:
      break

    # convert numpy array to pytorch tensor format
    frame_torch, _ = image_to_torch(frame, device=device)
    if current_frame_index == 0:
      # initialize with the mask
      mask_torch = index_numpy_to_one_hot_torch(mask, num_objects+1).to(device)
      # the background mask is not fed into the model
      prediction = processor.step(frame_torch, mask_torch[1:])
    else:
      # propagate only
      prediction = processor.step(frame_torch)

    # argmax, convert to numpy
    prediction = torch_prob_to_numpy_mask(prediction)

    # if current_frame_index % visualize_every == 0:
    visualization = overlay_davis(frame, prediction)
    out.write(visualization)
    current_frame_index += 1


cap.release()
out.release()
print(f"Output video saved as {output_video_name}")