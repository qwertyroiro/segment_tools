import os
from setuptools import setup, find_packages

# 現在のファイルのディレクトリを取得
PKG_DIR = os.path.dirname(os.path.abspath(__file__))

# 絶対パスを指定
groundingdino_whl = os.path.join(
    PKG_DIR, "groundingdino-0.1.0-cp310-cp310-linux_x86_64.whl"
)
clip_whl = os.path.join(PKG_DIR, "clip-1.0-py3-none-any.whl")
detectron2_whl = os.path.join(PKG_DIR, "detectron2-0.6-cp310-cp310-linux_x86_64.whl")
natten_whl = os.path.join(
    PKG_DIR, "natten-0.15.1+torch210cu121-cp310-cp310-linux_x86_64.whl"
)

setup(
    name="segment_tools",
    version="1.0.0",
    author="roiro",
    description="Segmentation tools",
    license="MIT License",
    python_requires="==3.10.*",
    install_requires=[
        "matplotlib==3.8.2",
        "opencv-python==4.9.0.80",
        "Pillow==9.3.0",
        "PyYAML==6.0.1",
        "scipy==1.12.0",
        "torch>=1.7.0",
        "torchvision>=0.8.1",
        "tqdm==4.66.1",
        "pandas==2.2.0",
        "seaborn==0.13.2",
        "ultralytics==8.0.120",
        f"GroundingDINO @ file://{groundingdino_whl}",
        f"clip @ file://{clip_whl}",
        "segment_anything @ git+https://github.com/facebookresearch/segment-anything.git@6fdee8f2727f4506cfbbe553e23b895e27956588",
        "transformers==4.37.1",
        "addict==2.4.0",
        "diffusers==0.25.1",
        "huggingface_hub==0.20.3",
        "numpy==1.23.5",
        "onnxruntime==1.16.3",
        "pycocotools==2.0.7",
        "requests==2.31.0",
        "setuptools==69.0.3",
        "supervision==0.18.0",
        "termcolor==2.4.0",
        "timm==0.9.12",
        "yapf==0.40.2",
        "nltk==3.8.1",
        "fairscale==0.4.13",
        "litellm==1.20.1",
        "accelerate==0.26.1",
        "panopticapi @ git+https://github.com/cocodataset/panopticapi.git@7bb4655548f98f3fedc07bf37e9040a992b054b0",
        "cityscapesscripts @ git+https://github.com/mcordts/cityscapesScripts.git@a7ac7b4062d1a80ed5e22d2ea2179c886801c77d",
        "cython==3.0.8",
        "shapely==2.0.2",
        "h5py==3.7.0",
        "submitit==1.4.2",
        "scikit-image",
        "einops==0.4.1",
        "icecream==2.1.2",
        "inflect==5.6.0",
        "diffdist==0.1",
        "mmcv==1.6.2",
        f"detectron2 @ file://{detectron2_whl}",
        "imutils==0.5.4",
        "rich==13.7.1",
        f"natten @ file://{natten_whl}",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
