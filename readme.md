
<p align="center">
  <img src="docs/figs/logo.png" align="center" width="50%">
  
  <h3 align="center"><strong>GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting</strong></h3>

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

[//]: # (<a href='https://arxiv.org/abs/2310.15169'><img src='https://img.shields.io/badge/arXiv-2310.15169-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;)
 <a href='https://buaacyw.github.io/gaussian-editor/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://www.youtube.com/watch?v=TdZIICSFqsU&ab_channel=YiwenChen'><img src='https://img.shields.io/badge/Youtube-Video-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://github.com/buaacyw/GaussianEditor/blob/master/LICENSE.txt'><img src='https://img.shields.io/badge/License-MIT-blue'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

</div>
<div align="center">
  <b>Note: This repo is still under construction. We aim to build this repo as a swift, controllable, and interactive 3D editing tool. We welcome any collaboration. If you're interested in connecting or partnering with us, please don't hesitate to reach out via email (YIWEN002@e.ntu.edu.sg).</b>
</div>

## Demo Videos
<details open>
  <summary>Swift and controllable 3D editing with only 2-7 minutes.</summary>

https://github.com/buaacyw/GaussianEditor/assets/52091468/10740174-3208-4408-b519-23f58604339e

https://github.com/buaacyw/GaussianEditor/assets/52091468/44797174-0242-4c82-a383-2d7b3d4fd693


https://github.com/buaacyw/GaussianEditor/assets/52091468/18dd3ef2-4066-428a-918d-c4fe673d0af8
</details>

## Release
- [11/27] 🔥 We released **GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting** and beta version of GaussianEditing WebUI.

## Contents
- [Install](#install)
- [WebUI Guide](#webui-guide)
- [Command Line](#command-line)
- [TODO](#todo)
- [FAQ](#faq)

## Install
Our environment has been tested on Ubuntu 22 with CUDA 11.8. Please follow [Installation](https://github.com/buaacyw/GaussianEditor/blob/master/docs/install.md).

## WebUI Guide
Please be aware that our WebUI is currently in a beta version. Powered by [Viser](https://github.com/nerfstudio-project/viser/tree/main), you can use our WebUI even if you are limited to remote server. For details, please follow [WebUI Guide](https://github.com/buaacyw/GaussianEditor/blob/master/docs/webui.md).

## Command Line
We also provide a command line version of GaussianEditor. Like WebUI, you need to specify your path to the pretrained Gaussians and COLMAP outputs as mentioned in [here](https://github.com/buaacyw/GaussianEditor/blob/1fa96851c132258e0547ba73372f37cff83c92c3/docs/webui.md?plain=1#L20).
Please check scripts in `sciprt` folder. Simply change `data.source` to your COLMAP output directory and 
`system.gs_source` to your pretrained Gaussians and run our demo scripts.


## TODO

The repo is still being under construction, thanks for your patience. 
- [ ] Step-by-step tutorial for WebUI .
- [ ] Tutorial for hyperparameter tuning.
- [ ] Colab.
- [ ] Windows support.
- [ ] Docker support.
- [x] Realised WebUI beta version and GaussianEditor.

## FAQ


