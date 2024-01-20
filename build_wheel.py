import subprocess
import sys

def build_wheel():
    # setup.pyがあるディレクトリに移動
    subprocess.check_call(['cd', './src/segment_tools/OneFormer_segtools/oneformer/modeling/pixel_decoder/ops'], shell=True)
    # setup.pyを実行してビルド
    # subprocess.check_call([sys.executable, 'setup.py', 'build_ext', '--inplace'], shell=True)
    subprocess.check_call([sys.executable, 'setup.py', 'build', 'install'], shell=True)

if __name__ == "__main__":
    build_wheel()