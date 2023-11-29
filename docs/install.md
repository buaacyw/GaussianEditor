1. Clone our repo and create Virtual environment
```
git clone https://github.com/buaacyw/GaussianEditor.git && cd GaussianEditor
conda create -n GaussianEditor python==3.8 && conda activate GaussianEditor
```

2. Install
```
# CUDA version 11.7
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
# CUDA version 11.8
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e gaussiansplatting/submodules/diff-gaussian-rasterization
pip install -e gaussiansplatting/submodules/simple-knn
```

4. (Optional) Install our forked viser [Required by WebUI)
```
mkdir extern && cd extern
git clone https://github.com/heheyas/viser 
pip install -e viser
cd ..
```

7. (Optional) Download Wonder3D checkpoints [Required by <b>Add</b>]
```bash
sh download_wonder3d.sh
```
