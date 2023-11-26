import argparse
import os
import nvitop
import time
import sys
import datetime
from nvitop import select_devices
from omegaconf import OmegaConf
from pathlib import Path
from itertools import product
from copy import deepcopy

import threestudio
from threestudio.utils.config import config_to_primitive
from threestudio.utils.typing import *


def find_available_gpus(minial_free_memory=7.0, devices=None):
    """find availables gpus given conditions

    Args:
        minial_free_memory (float, optional): mininal free VRAM of gpus, in GiB. Defaults to 7.0.
    """
    if devices is not None:
        devices = [nvitop.Device(i) for i in devices]
    return select_devices(
        devices=devices, format="index", min_free_memory=f"{minial_free_memory}GiB"
    )


def find_gpus_without_process_of_current_user():
    n_devices = nvitop.Device.count()
    gpus = [nvitop.Device(i) for i in range(n_devices)]
    available_gpus = []
    user_name = os.environ["USER"]
    for gpu in gpus:
        no_current_user = True
        for process in gpu.processes().values():
            if process.username() == user_name:
                no_current_user = False
                break
        if no_current_user:
            available_gpus.append(gpu.index)

    return available_gpus


def set_cfg_field(cfg, field, value):
    fields = field.split(".")
    for f in fields[:-1]:
        cfg = cfg[f]
    cfg[fields[-1]] = value
    # return cfg


def generate_sweep_configs(sweep_config: Union[str, Dict, DictConfig, Path]):
    # sweep_cfg = OmegaConf.load(f"./conf/sweep/{sweep_config}.yaml")
    if isinstance(sweep_config, (str, Path)):
        sweep_cfg = OmegaConf.load(sweep_config)
    elif isinstance(sweep_config, dict):
        sweep_cfg = OmegaConf.create(sweep_config)

    joint_fields = config_to_primitive(sweep_cfg.joint_fields)
    if not isinstance(joint_fields, list):
        joint_fields = [joint_fields]
    if not isinstance(joint_fields[0], list):
        joint_fields = [joint_fields]

    for j_field in joint_fields:
        field_length = len(sweep_cfg[j_field[0]])
        for jf in j_field:
            assert field_length == len(
                sweep_cfg[jf]
            ), f"joint fields {jf} have different length"

    all_fields = sweep_cfg.keys()
    cross_fields = []
    cross_fields = list(filter(lambda x: x not in sum(joint_fields, []), all_fields))
    cross_fields.remove("joint_fields")

    num_total_cfgs = 1
    for field in cross_fields:
        num_total_cfgs *= len(sweep_cfg[field])
    for fields in joint_fields:
        num_total_cfgs *= len(sweep_cfg[fields[0]])

    threestudio.info(
        f"Sweep: Total number of configs: {num_total_cfgs}, joint fields: {joint_fields}, cross fields: {cross_fields}"
    )

    cnt = 0
    cross_field_strings = []
    for cross_items in product(*[sweep_cfg[f] for f in cross_fields]):
        cross_field_string = []
        for c_idx, cross_item in enumerate(cross_items):
            cross_field_string.append(f"{cross_fields[c_idx]}={cross_item}")
        cross_field_strings.append(cross_field_string)

    joint_field_strings = []
    for ids in product(
        *[range(len(sweep_cfg[j_field[0]])) for j_field in joint_fields]
    ):
        joint_field_string = []
        for k_idx, f_id in enumerate(ids):
            for j_field in joint_fields[k_idx]:
                joint_field_string.append(f"{j_field}={sweep_cfg[j_field][f_id]}")
        joint_field_strings.append(joint_field_string)

    cnt = len(joint_field_strings) * len(cross_field_strings)

    assert (
        cnt == num_total_cfgs
    ), "number of configs does not match, this script is buggy"

    config_strings = []
    for jfs, cfs in product(joint_field_strings, cross_field_strings):
        config_strings.append(jfs + cfs)

    return config_strings


if __name__ == "__main__":
    print(find_gpus_without_process_of_current_user())
    print(find_available_gpus(1.0, find_gpus_without_process_of_current_user()))
    parser = argparse.ArgumentParser()
    parser.add_argument("base_name", type=str)
    parser.add_argument("--sweep_config", type=str, default=None)
    parser.add_argument("--sweep_group_name", type=str, default=None)
    parser.add_argument("--offset", type=int, default=0)

    opt = parser.parse_args()

    if opt.sweep_config is not None:
        generate_sweep_configs(
            opt.base_name, opt.sweep_config, opt.sweep_group_name, opt.offset
        )
