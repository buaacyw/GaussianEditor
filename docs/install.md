1. Clone our repo
```
git clone https://github.com/buaacyw/GaussianEditor.git && cd GaussianEditor`
```
2. Create Virtual environment and install pytorch
```
conda create -n GaussianEditor python==3.8 && conda activate GaussianEditor
# CUDA version 11.7
`pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117`
# CUDA version 11.8
`pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118`
```
3. Install requirements
```
pip install -r requirements.txt
```
4. Install our forked viser (Optional, need by WebUI)
```
mkdir extern && cd extern
git clone https://github.com/heheyas/viser 
pip install -e viser
cd ..
```

6. Build Gaussian Splatting Renderer and submodules
```
pip install -e gaussiansplatting/submodules/diff-gaussian-rasterization
pip install -e gaussiansplatting/submodules/simple-knn
```

7. (Optional) Download Wonder3D checkpoints [Required by 3D inpainting] using our script:
```bash
sh download_wonder3d.sh
```
