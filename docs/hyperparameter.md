# Functions
- [Semantic Tracing](#semantic-tracing)
- [Edit](#edit)
- [Delete](#delete)
- [Add](#add)

## Semantic Tracing
- `Seg Camera Nums`: The number of camera views that we perform semantic unprojection. It controls the number of times that we query [SAM](https://github.com/facebookresearch/segment-anything). Reducing this number speeds up semantic tracing. However, if you notice inaccuracies in your mask, consider increasing this value for better results.

- `Seg Threshold`: The confidence score of segmentation. The larger this value, the more areas will be considered as the targeted category.
  
## Edit
- `Camera Num`: The camera numbers used for training. If you find that the overall editing results are satisfactory, but some views appear weird (blurry, incomplete), this may be due to sparse training cameras at certain angles. 
Increasing the number of views for training cameras might resolve this issue. However, it's important to note that we sample training cameras from the Colmap cameras you provide. Therefore, if the Colmap cameras are sparse in that view, increasing this value won't improve the problem. 
It's also worth noting that raising this value will slightly increase time of initial stage of training, as we first generate edited 2D images for each view, and then randomly edit an image at each Edit Interval. 48 views typically work for most cases.

- `Total Step`: Total training step. Please note that simply increasing this will not significantly enhance the training time or the results. 
The rendering and backpropagation speeds of Gaussian Splatting are extremely fast, so the majority of our training time is spent updating the edited 2D images using diffusion models. 
We stop updating the edited 2D images after `Edit Until Step`, meaning that from the `Edit Until Step` to the `Total Step`, Gaussians only fit on a fixed dataset. 
Generally, we set the `Total Step` to be slightly higher than the `Edit Until Step` to ensure proper fitting of the Gaussians.

- `Densify Until Step`: When do we end Gaussians densification. It's typically set a little larger than `Total step` to ensure better fitting.

- `Densify Interval`&`Max Densify Percent`: How often and how much do we densify the Guassians. 
As mentioned in our paper, different from vanilla [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), we only densify those Gaussians whose gradients rank in the top `Max Densify Percent` to adapt to the unstable generative loss.
These two values need to be carefully and coordinately set. 
The smaller the interval and the larger the percent, the stronger the densification.
If you notice a lot of noise in the image, it's very likely due to over-densification. 
Since only the traced areas (as defined by the `Semantic Group`) will be densified, setting the `Semantic Group` to `ALL` means millions of points in the entire scene will be densified.
In such cases, you need to reduce the degree of densification.
Our default setting of 100 & 0.01 works well in most cases, as it is intended for updating only specific areas of the scene. 
In some instances, if you wish to make minor texture modifications or significant edits, consider reducing or increasing densification.
During each densification, low-opacity Gaussians are also pruned, so if you aim to remove certain parts, you should set `Max Densify Percent` to 0 but increase the `Densify Interval`.

- `Edit Interval`: Interval for updating the edited images. Lowering this value will result in more frequent updates of the edited 2D images, thereby enhancing the effect but also increasing the training time, and vice versa.
The default setting of 10 works well in most situations.

- `Edit Begin Step`&`Edit Until Step`: Period for updating edited images. same as `Edit Interval`, increase this period will result in better results but also increase the training time, and vice versa.

- `Learning Rate Scaler`: Learning rate scaler for Gaussians. This value will be multiplied to the learning rate of corresponding properties of Gaussians.
The default setting is a moderate value. If you aim to only update textures, consider lowering other learning rates.

- `Lambda L1`&`Lambda Perceptual`: Loss weight for L1 loss and perceptual loss. Typically you don't have to modify it.

- `Anchor Init G0`&`Anchor Init`&`Anchor Multiplier`: Anchor loss setting. 
As mentioned in our paper, the proposed hierarchical gaussian splatting (HGS) will apply anchor loss to elder generations.
`Anchor Init G0` defines the initial anchor loss weight we apply to Generation 0 (i.e. the origin Gaussians).
`Anchor Init` defines the initial anchor loss weight we apply to i-th generation when (i+1)-th generation is generated.
After each densification, the anchor loss weights of all Gaussians are multiplied by `Anchor Multiplier`.
So lowering these values will decrease the anchor loss weight, which increases the fluidity of the Gaussians.
Typically, you don't need to change these values.
However, if you reduce the `Densify Interval`, you should lower the `Anchor Multiplier` to achieve balance, and vice versa.

- `Lambda Anchor`: Anchor loss weight for different properties. Anchor loss weight defined above will be multiplied by `Lambda Anchor` for different properties.
Typically you don't need to modify this.

## Delete
- `Camera Num`: Same as in <b>Edit</b>. 
After deleting the Gaussians in the traced area, holes or artifacts are created. 
To repair these areas, we rely on [Controlnet-inpainting](https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint) to generate repaired images from various viewpoints. 
This parameter defines the number of viewpoints from which images are generated. 
The higher it is, the more time it takes to generate, but the better the results.

- `Total Step`: Same as in <b>Edit</b>.


- `Edit Interval`&`Edit Begin Step`&`Edit Until Step`: Disabled in <b>Delete</b>. This is because we generate inpainted 2D images for each view only once, which is done to increase speed.

- `Densification`: Same as in <b>Edit</b>. But notice that the traced area will be changed to the local area of the deleted object, which makes the number of traced Gaussians relatively less.
So if you find it hard to fix the holes caused by deleting, increase densification.

- `Inpaint Scale`: As mentioned above, we need to trace the local area of the deleted object. This parameter defines how local we will trace.
The higher this value, the more areas are defined as local, resulting in a greater number of Gaussians being traced. 

- `Mask Dilate`&`Fix Holes`: When we query [Controlnet-inpainting](https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint), we need to provide it a 2D mask to tell it the area that we want to restore.
Typically we first detect the local areas according  to `Inpaint Scale` and then project these local areas to get 2D masks.
These 2D masks will then dilate by `Mask Dilate` pixel. If `Fix Holes` is enabled, the holes in the dilated 2D masks will be filled in.
This process aids Controlnet in generating better inpainted 2D images, particularly in scenes similar to those in our demo videos.

- `Learning Rate Scaler`&`Anchor Loss`: Same as in <b>Edit</b>.

## Add
Currently we don't have any hyperparameters for tuning in <b>Add</b>.
