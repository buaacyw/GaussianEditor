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

Our WebUI requires cameras output from COLMAP to provide the training perspectives for <b>segmentation, edit, and delete</b>. Currently, we only accept `PINHOLE` cameras. Notice that the current views of your WebUI won't affect the training process since we will load COLMAP cameras for training. We first introduce the basic usages of our WebUI and then explain the above five functionalities in order.

##### (1) Basic Usage
<img width="235" alt="1701045938591" src="https://github.com/buaacyw/GaussianEditor/assets/52091468/bcb8ef14-651b-47d8-b816-064ed72cab8c">

- `Resolution`. Resolution of renderings sent to your screen. Note that changing this won't affect the training resolution for <b>edit, and delete</b>, which is fixed to 512. However, it would affect the 2D inpainting used in `add` since we ask users for providing a 2D mask. If you find your WebUI very slow, please lower `Resolution`.
- `FoV Scaler`. Scale the FoV of renderings sent to your screen. Same as `Resolution`, it will only affect `add`. Typically you don't need to change this.
- `Renderer Output`. Render type sent to your screen. Depth will be updated in the future.
- `Save Gaussian`. After your editing. Click it to save your Gaussians to `ui_result`. You can then relaunch WebUI and load the saved Gaussian to apply sencond time editing. There is still some bugs for directly editing Gaussians multiple time with single launch of our WebUI.
- `Show Frame`. Due to the view navigation setting of [Viser](https://github.com/nerfstudio-project/viser/tree/main/examples), it's a bit hard to control cameras because the frams of Gaussian scenes are typically not default to be standard. So you may find yourself hard to control your views. In this case, click `Show Frame` and jump to one of the preload camera views from COLMAP.

##### (1) Semantic Tracing by Text

![image](https://github.com/buaacyw/GaussianEditor/assets/52091468/1e66ce57-aa79-4144-9b4c-9918712ce0fb)

Steps:
1. Specify `Text Seg Prompt` to the desired target.
2. Click `Tracing Begins!`.
3. After 30-60 seconds, `Semantic Group` will add a new group, named by your text in `Text Seg Prompt`.
4. Click `Show Semantic Mask` to view the segmentation results.

- `Text Seg Prompt`. Prompt for conduct prompt SAM.
- `Semantic Group`. All your segmentation results. Switch between them!
- `Seg Camera Nums`. How many views are we used for SAM. Less views, faster segmentation. Typically 12 views will generate a good enough result.
- `Show Semantic Mask`. Show your mask. Notice that this won't affect training.

##### (2) Tracing by Click

Steps:
1. Open `Enable SAM` and `Add SAM Points`.
2. Click on the part that you want to segment. Typically you don't need to move your views since adding point from single view can already provides nice results, but if you want, you need to first close `Add SAM Points`, then move your views and then open it.
3. After you add your point, close `Add SAM Points` and then specify the name of your desired part in `SAM Group Name`. Notice that `SAM Group Name` is only used as the name in `Semantic Group`, it won't be used as text prompt for segnmentation. 
4. Clik `Show Semantic Mask` to view the segmentation results.

##### (3) Edit
![image](https://github.com/buaacyw/GaussianEditor/assets/52091468/7b0a13b6-dec3-4135-b892-3bf5e4a7315d)

Simply input your prompt and begin editing. You can specify the part that we want to change by switch `Semantic Group`. The default `ALL` means that the whole Gaussians will be updated. After the training begins, you can view the current edit 2D frames by opening `Show Edit Frame`.

##### (4) Delete

![image](https://github.com/buaacyw/GaussianEditor/assets/52091468/09c9da14-7ac0-4903-9688-39f095428a39)

Same step as <b>Edit</b>, but you must specify masks in `Semantic Group`. The masked part will be delete and than we use 2D inpainting methods to fix the artifacts cased by this action. Note that the `Text` is used to fix the artifacts on background. Therefore, you should not input the object category into `Text` but rather the background descriptions.  

##### (5) Add

