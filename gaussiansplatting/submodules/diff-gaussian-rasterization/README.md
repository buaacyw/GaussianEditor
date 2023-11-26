# Differential Gaussian Rasterization

This is a forked repository of the rasterization pipeline from the paper "3D Gaussian Splatting for Real-Time Rendering of Radiance Fields". I have made some extensions to it: 

- main branch incorporates only the forward pass of depth, which is used for depth visualization. 
- 4th-degree: add the 4th degree of SH
- depth: add both the forward and backward pass of depth, which is used for some tasks with depth supervision.
- latest: is the dev branch that contains acc and depth visualization, together with depth backward pass. 

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
</section>