# Functions
- [Semantic Tracing](#semantic-tracing)
- [Edit](#edit)
- [Delete](#delete)
- [Add](#add)

## Semantic Tracing
- `Seg Camera Nums`: The number of camera views that we perform semantic unprojection. It controls the number of times that we query [SAM](https://github.com/facebookresearch/segment-anything). Reducing this number speeds up semantic tracing. However, if you notice inaccuracies in your mask, consider increasing this value for better results.

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
Our default setting of 100 & 0.01 is designed for updating only certain areas of the scene.



## Delete

## Add
