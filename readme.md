
<p align="center">
  <img src="docs/figs/logo.png" align="center" width="50%">
  
  <h3 align="center"><strong>[CVPR 2024] GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting</strong></h3>

  <p align="center">
    <a href="https://buaacyw.github.io/">Yiwen Chen</a><sup>*1,2</sup>,</span>
    <a href="https://scholar.google.com/citations?user=2pbka1gAAAAJ&hl=en">Zilong Chen</a><sup>*3</sup>,</span>
    <a href="https://icoz69.github.io/">Chi Zhang</a><sup>2</sup>,
    <a href="https://scholar.google.com/citations?user=bKG4Un8AAAAJ&hl=en">Feng Wang</a><sup>3</sup>,
    <a href="https://www.researchgate.net/scientific-contributions/Xiaofeng-Yang-2185243877">Xiaofeng Yang</a><sup>2</sup>,    <br>
    <a href="https://yikaiw.github.io/">Yikai Wang</a><sup>3</sup>,
    <a href="https://caizhongang.github.io/">Zhongang Cai</a><sup>4</sup>
    <a href="https://scholar.google.com.hk/citations?user=jZH2IPYAAAAJ&hl=en">Lei Yang</a><sup>4</sup>
    <a href="https://sites.google.com/site/thuliuhuaping">Huaping Liu</a><sup>3</sup>
    <a href="https://guosheng.github.io/">Guosheng Lin</a><sup>**1,2</sup>
    <br>
    <sup>*</sup>Equal contribution.
    <sup>**</sup>Corresponding author.
    <br>
    <sup>1</sup>S-Lab, Nanyang Technological University,
    <br>
<sup>2</sup>School of Computer Science and Engineering, Nanyang Technological University,
    <br>
    <sup>3</sup>Department of Computer Science and Technology, Tsinghua University,
    <sup>4</sup>SenseTime Research,

</p>

<div align="center">

<a href='https://arxiv.org/abs/2311.14521'><img src='https://img.shields.io/badge/arXiv-2311.14521-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://buaacyw.github.io/gaussian-editor/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://www.youtube.com/watch?v=TdZIICSFqsU&ab_channel=YiwenChen'><img src='https://img.shields.io/badge/Youtube-Video-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://github.com/buaacyw/GaussianEditor/blob/master/LICENSE.txt'><img src='https://img.shields.io/badge/License-SLab-blue'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

</div>


## Demo Videos
<details open>
  <summary>Swift and controllable 3D editing with only 2-7 minutes.</summary>

https://github.com/buaacyw/GaussianEditor/assets/52091468/10740174-3208-4408-b519-23f58604339e

https://github.com/buaacyw/GaussianEditor/assets/52091468/44797174-0242-4c82-a383-2d7b3d4fd693


https://github.com/buaacyw/GaussianEditor/assets/52091468/18dd3ef2-4066-428a-918d-c4fe673d0af8
</details>

## Release
- [12/5] Docker support. Great thanks to [jhuangBU](https://github.com/jhuangBU). For windows, you can try [this guide](https://github.com/buaacyw/GaussianEditor/issues/9) and [this guide](https://github.com/buaacyw/GaussianEditor/issues/14).
- [11/29] Release segmentation confidence score scaler. You can now scale the threshold of semantic tracing masks. 
- [11/27] 🔥 We released **GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting** and beta version of GaussianEditing WebUI.

## Contents
- [Demo Videos](#demo-videos)
- [Release](#release)
- [Contents](#contents)
- [Installation](#installation)
- [WebUI Guide](#webui-guide)
- [How to achieve better result](#how-to-achieve-better-result)
- [Command Line](#command-line)
- [TODO](#todo)
- [FAQ](#faq)

## Installation
Our environment has been tested on Ubuntu 22, CUDA 11.8 with 3090, A5000 and A6000.
1. Clone our repo and create conda environment
```
git clone https://github.com/buaacyw/GaussianEditor.git && cd GaussianEditor

# (Option one) Install by conda
conda env create -f environment.yaml

# (Option two) You can also install by pip
# CUDA version 11.7
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
# CUDA version 11.8
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# (Option three) If the below two options fail, please try this:
# For CUDA 11.8
bash install.sh
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

## WebUI Guide
Please be aware that our WebUI is currently in a beta version. Powered by [Viser](https://github.com/nerfstudio-project/viser/tree/main), you can use our WebUI even if you are limited to remote server. For details, please follow [WebUI Guide](https://github.com/buaacyw/GaussianEditor/blob/master/docs/webui.md).

## How to achieve better result

The demand for 3D editing is very diverse. For instance, if you only want to change textures and materials or significantly modify geometry, it's clear that a one-size-fits-all hyperparameter won't work. Therefore, we cannot provide a default hyperparameter setting that works effectively in all scenarios. Therefore, if your results do not meet expectations, please refer to our [hyperparameter tuning](https://github.com/buaacyw/GaussianEditor/blob/master/docs/hyperparameter.md) document. In it, we detail the function of each hyperparameter and advise on which parameters to adjust when you encounter specific issues. 

## Command Line
We also provide a command line version of GaussianEditor. Like WebUI, you need to specify your path to the pretrained Gaussians and COLMAP outputs as mentioned in [here](https://github.com/buaacyw/GaussianEditor/blob/1fa96851c132258e0547ba73372f37cff83c92c3/docs/webui.md?plain=1#L20).
Please check scripts in `sciprt` folder. Simply change `data.source` to your COLMAP output directory and 
`system.gs_source` to your pretrained Gaussians and run our demo scripts.


## TODO

The repo is still being under construction, thanks for your patience. 
- [x] Tutorial for hyperparameter tuning.
- [x] Step-by-step tutorial for WebUI .
- [x] Realised WebUI beta version and GaussianEditor.

## FAQ

- Bad result for <b>Edit</b>. We are using [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix) to generate edited 2D images as editing guidance.
Unfortunately, InstructPix2Pix only works on limited prompts, please first try [here](https://huggingface.co/spaces/timbrooks/instruct-pix2pix) if you are not sure whether your text prompts work.
- Bad result for <b>Add</b>. We use [ControlNet-Inpainting](https://github.com/lllyasviel/ControlNet) to first generate 2D inpainting and then transfer it into 3D. Also it doesn't work for bad prompts. Please try to enlarge your inpainting mask and try more seeds.
- Bad result for <b>Segmentation</b>. Try scale the segmentation threshold, which changes the confidence score for segmentation.
- Missing weights for DPT. Please read this [issue](https://github.com/buaacyw/GaussianEditor/issues/10)

## Acknowledgement

Our code is based on these wonderful repos:

* [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
* [Wonder3D](https://github.com/xxlong0/Wonder3D)
* [Threestudio](https://github.com/threestudio-project/threestudio)
* [Viser](https://github.com/nerfstudio-project/viser)
* [InstructNerf2Nerf](https://github.com/ayaanzhaque/instruct-nerf2nerf)
* [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix)
* [Controlnet](https://github.com/lllyasviel/ControlNet)



