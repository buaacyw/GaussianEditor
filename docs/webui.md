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