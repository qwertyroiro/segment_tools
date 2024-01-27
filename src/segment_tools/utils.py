import numpy as np
from PIL import Image

# draw mask from segment anything
def __draw_multi_mask(masks, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = masks[0].shape[-2:]
    mask_image = masks[0].reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    temp_mask = np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))
    
    for mask in masks[1:]:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")
        temp_mask = np.array(Image.alpha_composite(Image.fromarray(temp_mask), mask_image_pil))
        
    return temp_mask