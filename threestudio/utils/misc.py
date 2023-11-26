import gc
import os
import re

import tinycudann as tcnn
import torch
from packaging import version

from threestudio.utils.config import config_to_primitive
from threestudio.utils.typing import *
import torch.nn.functional as F
import cv2
import numpy as np
from scipy import ndimage
@torch.no_grad()
def dilate_mask(mask, pixels):
    kernel = torch.ones(1, 1, pixels*2+1, pixels*2+1, dtype=mask.dtype, device=mask.device)
    return F.max_pool2d(mask.unsqueeze(0), kernel_size=kernel.shape[2:], stride=1, padding=pixels).squeeze(0)

@torch.no_grad()
def erode_mask(mask, pixels):
    kernel = torch.ones(1, 1, pixels*2+1, pixels*2+1, dtype=mask.dtype, device=mask.device)
    padded_mask = F.pad(mask, (pixels, pixels, pixels, pixels), mode='constant', value=1)
    return F.conv2d(padded_mask.unsqueeze(0), kernel, stride=1).squeeze(0) == kernel.numel()
def fill_closed_areas(mask):
    mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)

    mask_np=ndimage.binary_fill_holes(mask_np).astype(int)

    filled_mask_tensor = torch.tensor(mask_np, device=mask.device,dtype=torch.bool)

    return filled_mask_tensor[None]


def parse_version(ver: str):
    return version.parse(ver)


def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def get_device():
    return torch.device(f"cuda:{get_rank()}")


def load_module_weights(
    path, module_name=None, ignore_modules=None, map_location=None
) -> Tuple[dict, int, int]:
    if module_name is not None and ignore_modules is not None:
        raise ValueError("module_name and ignore_modules cannot be both set")
    if map_location is None:
        map_location = get_device()

    ckpt = torch.load(path, map_location=map_location)
    state_dict = ckpt["state_dict"]
    state_dict_to_load = state_dict

    if ignore_modules is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            ignore = any(
                [k.startswith(ignore_module + ".") for ignore_module in ignore_modules]
            )
            if ignore:
                continue
            state_dict_to_load[k] = v

    if module_name is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            m = re.match(rf"^{module_name}\.(.*)$", k)
            if m is None:
                continue
            state_dict_to_load[m.group(1)] = v

    return state_dict_to_load, ckpt["epoch"], ckpt["global_step"]


def C(value: Any, epoch: int, global_step: int) -> float:
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = config_to_primitive(value)
        if not isinstance(value, list):
            raise TypeError("Scalar specification only supports list, got", type(value))
        if len(value) == 3:
            value = [0] + value
        assert len(value) == 4
        start_step, start_value, end_value, end_step = value
        if isinstance(end_step, int):
            current_step = global_step
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
        elif isinstance(end_step, float):
            current_step = epoch
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
    return value


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    tcnn.free_temporary_memory()


def finish_with_cleanup(func: Callable):
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)
        cleanup()
        return out

    return wrapper


def _distributed_available():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def barrier():
    if not _distributed_available():
        return
    else:
        torch.distributed.barrier()


def broadcast(tensor, src=0):
    if not _distributed_available():
        return tensor
    else:
        torch.distributed.broadcast(tensor, src=src)
        return tensor


def enable_gradient(model, enabled: bool = True) -> None:
    for param in model.parameters():
        param.requires_grad_(enabled)


def step_check(step: int, step_size: int, run_at_zero: bool = False) -> bool:
    """Returns true based on current step and step interval. credit: nerfstudio"""
    if step_size == 0:
        return False
    return (run_at_zero or step != 0) and step % step_size == 0
