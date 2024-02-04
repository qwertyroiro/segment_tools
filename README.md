## Prerequisites
Python: 3.10
CUDA: 11.8
torch: 2.1.0  
torchvison:  0.16.0  
https://pytorch.org/get-started/previous-versions/#v210
```bash
pip install git+https://github.com/qwertyroiro/segment_tools.git
pip install natten==0.14.4 -f https://shi-labs.com/natten/wheels/cu113/torch1.10.1/index.html
``````
## Usage

### Image Preparation
```python
from PIL import Image
import numpy as np

image_path = "dogs.jpg"
image_pil = Image.open(image_path)  # Open image with Pillow
image_np = np.array(image_pil)      # Convert to numpy array
```

### Define Prompt
```python
prompt = "dog"  # Define your prompt
```

### Segment Tools Usage

#### FastSAM
```python
import segment_tools as st

# Segment without prompt
result = st.fastsam(image_np)
if result is not None:
    image, ann = result["image"], result["mask"]

# Segment with prompt
result = st.fastsam(image_np, prompt)
if result is not None:
    image, ann = result["image"], result["mask"]
```

#### CLIPSeg
```python
result = st.clipseg(image_np, prompt)
if result is not None:
    image, ann = result["image"], result["mask"]
```

#### DINO
```python
result = st.dino(image_path, prompt)
if result is not None:
    image, bbox = result["image"], result["bbox"]
```

#### DINOSeg
```python
result = st.dinoseg(image_path, prompt)
if result is not None:
    image, maskimage, bbox = result["image"], result["mask"], result["bbox"]
```

### OneFormer Variants

#### OneFormer (ADE20K Dataset)
```python
oneformer_ade20k = st.OneFormer_ade20k()
result = oneformer_ade20k.run(image_np)
if result is not None:
    image, ann = result["image"], result["mask"]

# With SWIN Transformer
oneformer_ade20k_swin = st.OneFormer_ade20k(use_swin=True)
result = oneformer_ade20k_swin.run(image_np)
if result is not None:
    image, ann = result["image"], result["mask"]

# Using prompt
result = oneformer_ade20k.run(image_np, prompt)
if result is not None:
    image, ann = result["image"], result["mask"]
result = oneformer_ade20k_swin.run(image_np, prompt)
if result is not None:
    image, ann = result["image"], result["mask"]
```

#### OneFormer (Cityscapes Dataset)
```python
oneformer_city = st.OneFormer_cityscapes()
result = oneformer_city.run(image_np)
if result is not None:
    image, ann = result["image"], result["mask"]

# With SWIN Transformer
oneformer_city_swin = st.OneFormer_cityscapes(use_swin=True)
result = oneformer_city_swin.run(image_np)
if result is not None:
    image, ann = result["image"], result["mask"]

# Using prompt
result = oneformer_city.run(image_np, prompt)
if result is not None:
    image, ann = result["image"], result["mask"]
result = oneformer_city_swin.run(image_np, prompt)
if result is not None:
    image, ann = result["image"], result["mask"]
```

#### OneFormer (COCO Dataset)
```python
oneformer_coco = st.OneFormer_coco()
result = oneformer_coco.run(image_np)
if result is not None:
    image, ann = result["image"], result["mask"]

# With SWIN Transformer
oneformer_coco_swin = st.OneFormer_coco(use_swin=True)
result = oneformer_coco_swin.run(image_np)
if result is not None:
    image, ann = result["image"], result["mask"]

# Using prompt
result = oneformer_coco.run(image_np, prompt)
if result is not None:
    image, ann = result["image"], result["mask"]
result = oneformer_coco_swin.run(image_np, prompt)
if result is not None:
    image, ann = result["image"], result["mask"]
```

### Additional Notes
- The `run` method can be called with or without a prompt for all OneFormer variants.
- The `use_swin=True` parameter enables the use of the SWIN Transformer as the backbone for the OneFormer models.
- The `image` and `ann` (annotations) are obtained from the `result` dictionary, which is the output from the segmentation models.
- For DINO and DINOSeg, the outputs are `image`, `bbox` (bounding boxes), and `maskimage` (segmentation masks), respectively, also obtained from the `result` dictionary.
- If `result` is `None`, it indicates that the segmentation process was not successful. This could be due to various reasons such as incorrect input data or model limitations. It is important to handle this case in your code to avoid errors.
```