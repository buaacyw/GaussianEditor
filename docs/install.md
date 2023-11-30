1. Clone our repo and create conda environment
```
git clone https://github.com/buaacyw/GaussianEditor.git && cd GaussianEditor
conda env create -f environment.yaml 
```

2. (Optional) Install our forked viser [Required by WebUI)
```
mkdir extern && cd extern
git clone https://github.com/heheyas/viser 
pip install -e viser
cd ..
```

3. (Optional) Download Wonder3D checkpoints [Required by <b>Add</b>]
```bash
sh download_wonder3d.sh
```
