# not using official CUDA image as they routinely wipe them out
FROM kchawlapi/cuda11.8.0-ubuntu22.04:latest

# install system dependencies
RUN apt-get update \
    && apt-get install curl wget git git-lfs -y \
    && apt-get install build-essential -y \
    && apt-get install libgl1 libglib2.0-0 -y

# install miniforge
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
    && bash Miniforge3-$(uname)-$(uname -m).sh -b \
    && /root/miniforge3/bin/conda init bash

# setup conda environment, without pip deps
COPY environment.yaml environment.yaml
COPY environment-disable-pip.patch environment-disable-pip.patch
RUN patch environment.yaml environment-disable-pip.patch
RUN /root/miniforge3/bin/mamba env create -f environment.yaml && /root/miniforge3/bin/mamba clean -afy

# install pip dependencies
COPY requirements.lock.txt requirements.lock.txt
# GPU architecture spec for native extensions
# 8.6 for A5000/A6000 GPUS, supported by pytorch 2.1.0
ENV TCNN_CUDA_ARCHITECTURES=86
ENV TORCH_CUDA_ARCH_LIST=8.6
# hack: link libcuda.so so the tiny-cuda-nn compiles at build
RUN ln -s /root/miniforge3/envs/GaussianEditor/lib/stubs/libcuda.so /root/miniforge3/envs/GaussianEditor/lib
COPY gaussiansplatting gaussiansplatting
RUN /root/miniforge3/bin/conda run --live-stream -n GaussianEditor pip install -r requirements.lock.txt \
    && /root/miniforge3/bin/conda run --live-stream -n GaussianEditor pip cache purge

# additional - install viser & build webclient
RUN mkdir extern && cd extern \
    && git clone https://github.com/heheyas/viser \
    && /root/miniforge3/bin/mamba run -n GaussianEditor --live-stream pip install -e viser \
    && cd .. \
    && /root/miniforge3/bin/conda run --live-stream -n GaussianEditor python -c "import viser; viser.ViserServer()" \
    && rm -rf /extern/viser/src/viser/client/node_modules

# other local files
COPY . GaussianEditor

# startup commands
WORKDIR /GaussianEditor
RUN chmod +x Dockerfile.bootstrap.sh
CMD ./Dockerfile.bootstrap.sh