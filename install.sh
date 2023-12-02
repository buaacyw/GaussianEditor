pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ninja
pip install -r requirements_2.txt
pip install Pillow==9.5.0
cd gaussiansplatting/submodules
pip install ./diff-gaussian-rasterization
pip install ./simple-knn
pip install easydict
pip install webdataset
pip install albumentations==0.5.2
pip install kornia==0.7.0
pip install diffusers[torch]==0.19.3
pip install rembg
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git
pip install viser
pip install torch_efficient_distloss
pip install mediapy
pip install plyfile
pip uninstall torch torchvision
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
cd ../..
mkdir extern && cd extern
git clone https://github.com/heheyas/viser
pip install -e viser
cd ..
