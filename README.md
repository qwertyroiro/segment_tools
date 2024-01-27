## Prerequisites
torch: 2.1.0  
torchvison:  0.16.0  
https://pytorch.org/get-started/previous-versions/#v210
```bash
pip install git+https://github.com/qwertyroiro/segment_tools.git
pip install natten==0.14.4 -f https://shi-labs.com/natten/wheels/cu113/torch1.10.1/index.html
``````
## Usage
#### Read Image
```python
from PIL import Image
import numpy as np
import segment_tools as st

image_path = "dogs.jpg"
image_pil = Image.open(image_path) # Pillow
image_np = np.array(image_pil) # ndarray
```

#### define prompt
```python
prompt = "dog"
```

#### FastSAM
```python
# segment anything
image, ann = st.fastsam(image_path)
image, ann = st.fastsam(image_pil)
image, ann = st.fastsam(image_np)

# Use prompt
image, ann = st.fastsam(image_path, prompt)
image, ann = st.fastsam(image_pil, prompt)
image, ann = st.fastsam(image_np, prompt)
```

#### CLIPSeg
```python
image, ann = st.clipseg(image_path, prompt)
image, ann = st.clipseg(image_pil, prompt)
image, ann = st.clipseg(image_np, prompt)
```

#### DINO
```python
image, bbox = st.dino(image_path, prompt)
image, bbox = st.dino(image_pil, prompt)
image, bbox = st.dino(image_np, prompt)
```

#### DINOSeg
```python
image, maskimage, bbox = st.dinoseg(image_path, prompt)
image, maskimage, bbox = st.dinoseg(image_pil, prompt)
image, maskimage, bbox = st.dinoseg(image_np, prompt)
```

#### OneFormer (ade20k)
```python
oneformer_ade20k = st.OneFormer_ade20k()
image, ann, ann_info = oneformer_ade20k.run(image_path)
image, ann, ann_info = oneformer_ade20k.run(image_pil)
image, ann, ann_info = oneformer_ade20k.run(image_np)
```

#### OneFormer (cityscapes)
```python
oneformer_city = st.OneFormer_cityscapes()
image, ann, ann_info = oneformer_city.run(image_path)
image, ann, ann_info = oneformer_city.run(image_pil)
image, ann, ann_info = oneformer_city.run(image_np)
```

#### OneFormer (coco)
```python
oneformer_coco = st.OneFormer_coco()
image, ann, ann_info = oneformer_coco.run(image_path)
image, ann, ann_info = oneformer_coco.run(image_pil)
image, ann, ann_info = oneformer_coco.run(image_np)
```