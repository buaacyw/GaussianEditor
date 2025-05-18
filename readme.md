# EV 2025 Spring final project

This project is a clone and extension from the original work [buaacyw/GaussianEditor](https://github.com/buaacyw/GaussianEditor).

## Install environment
```bash
conda env create -f environment.yaml
conda activate GaussianEditor

pip install -r base_requirements.txt
pip install --no-deps -r git_requirements.txt

# Download DPT models, ref: https://github.com/EPFL-VILAB/omnidata
mkdir dpt
cd dpt
wget 'https://zenodo.org/records/10447888/files/omnidata_dpt_depth_v2.ckpt?download=1'
cd ..
```

To use Gaussian Editor's web UI, install their modified version of viser:
```bash
mkdir extern && cd extern
git clone https://github.com/heheyas/viser 
pip install -e viser
cd ..
```
For the "added" function, download the Wonder3D dataset:
```bash
sh download_wonder3d.sh
```

To load the model before launching the server: 
```bash
$ python
>>> import torch
>>> from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    DDIMScheduler,
)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
```

To launch the web UI server, 
```bash
python webui.py --gs_source <your-ply-file> --colmap_dir <dataset-dir>
```
where `--gs_source` refers to the pre-trained .ply file (something like ../../point_cloud.ply), and `--colmap_dir` refers to where the Colmap output resides (the colmap output sparse folder should be the subfolder of `--colmap_dir`).
See the [original repo](https://github.com/buaacyw/GaussianEditor/blob/master/docs/webui.md) for more information.
