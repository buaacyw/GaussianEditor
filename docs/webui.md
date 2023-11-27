### WebUI Instructions

#### 1. Prepare dataset and pre-trained models
Download MipNeRF-360 dataset with our script `download.sh`:
```
sh download.sh
```
Or use other dataset, following the instruction of [3DGS](https://github.com/graphdeco-inria/gaussian-splatting#processing-your-own-scenes)


Download pre-trained `.ply` model from [3DGS](https://github.com/graphdeco-inria/gaussian-splatting#evaluation) (already done if you use `download.sh`):
```
mkdir dataset
cd dataset
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip
unzip models.zip
```
Or train from scratch following instructions of [3DGS](https://github.com/graphdeco-inria/gaussian-splatting#running).

#### 2. Start our webUI
If you are ready with dataset, you can start our webUI by
```bash
python webui.py --gs_source <your-ply-file> --colmap_dir <dataset-dir>
```
where `--gs_source` refers to the pre-trained `.ply` file, and `--colmap_dir` refers to where the posed multiview images resides (with `cameras.json` inside).


For example, if you are using `download.sh` (which means adopting pre-trained GS from 3DGS and download the corresponding `.ply` files into ./dataset/<scene-name>), you can start with
```bash
python webui.py \
    --colmap_dir ./dataset/<scene-name> \
    --gs_source ./dataset/<scene-name>/point_cloud/iteration_7000/point_cloud.ply
```
#### 3. Step-by-Step Guide for WebUI
We suggest that you first take a look at our paper and our [demo video](https://www.youtube.com/watch?v=TdZIICSFqsU&ab_channel=YiwenChen).
The WebUI of GaussianEditor currently features five functionalities: <b>semantic tracing (3D segmentation) by text, tracing by click, edit, delete, and add.</b>

Our WebUI requires cameras output from COLMAP to provide the training perspectives for <b>segmentation, edit, and delete</b>. Currently, we only accept `PINHOLE` cameras. We first introduce the basic usages of our WebUI and then explain the above five functionalities in order.

##### (1) Basic Usage
<img width="235" alt="1701045938591" src="https://github.com/buaacyw/GaussianEditor/assets/52091468/bcb8ef14-651b-47d8-b816-064ed72cab8c">

- `Resolution`. Resolution of the renderings that sent to your screen. Note that changing this won't affect the training resolution for <b>edit, and delete</b>, which is fixed to 512. However, it would affect the 2D inpainting used in `add` since we ask users for providing a 2D mask. If you find your WebUI very slow, please lower `Resolution`.
- `Resolution`. Resolution of the renderings 

