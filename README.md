## Usage
pip install git+https://github.com/qwertyroiro/segment_tools.git

#### read image
```python
from PIL import Image
import numpy as np
import segment_tools as st

image_path = "dogs.jpg"
image_pil = Image.open(image_path) # Pillow
image_np = np.array(image_pil) # ndarray
```

#### fastsam
```python
image, ann = st.fastsam(image_path)
image, ann = st.fastsam(image_pil)
image, ann = st.fastsam(image_np)

image, ann = st.fastsam(image_path, "dog")
image, ann = st.fastsam(image_pil, "dog")
image, ann = st.fastsam(image_np, "dog")
```

#### clipseg
```python
output = st.clipseg(image_path, "dog")
output = st.clipseg(image_pil, "dog")
output = st.clipseg(image_np, "dog")
```

#### dino
```python
image, bbox = st.dino(image_path, "dog")
image, bbox = st.dino(image_pil, "dog")
image, bbox = st.dino(image_np, "dog")
```

#### dinoseg
```python
image, maskimage, bbox = st.dinoseg(image_path, "dog")
image, maskimage, bbox = st.dinoseg(image_pil, "dog")
image, maskimage, bbox = st.dinoseg(image_np, "dog")
```
