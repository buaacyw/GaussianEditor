#!/usr/bin/env bash

set -x

# download model & dataset
bash download_wonder3d.sh &
git clone https://huggingface.co/datasets/jhuangbu/gaussian-editor-garden-scene dataset &
wait

# run webui
/root/miniforge3/bin/conda run \
  --live-stream -n GaussianEditor \
  python webui.py --gs_source dataset/garden/point_cloud/iteration_30000/point_cloud.ply --colmap_dir dataset/garden