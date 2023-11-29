mkdir threestudio/utils/wonder3D/ckpts
mkdir threestudio/utils/wonder3D/ckpts/unet
wget https://huggingface.co/camenduru/Wonder3D/resolve/main/random_states_0.pkl -P threestudio/utils/wonder3D/ckpts
wget https://huggingface.co/camenduru/Wonder3D/resolve/main/scaler.pt -P threestudio/utils/wonder3D/ckpts
wget https://huggingface.co/camenduru/Wonder3D/resolve/main/scheduler.bin -P threestudio/utils/wonder3D/ckpts
wget https://huggingface.co/camenduru/Wonder3D/resolve/main/unet/diffusion_pytorch_model.bin -P threestudio/utils/wonder3D/ckpts/unet
wget https://huggingface.co/camenduru/Wonder3D/resolve/main/unet/config.json -P threestudio/utils/wonder3D/ckpts/unet
