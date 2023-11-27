import bisect
import random
from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured

from threestudio.utils.typing import *
import numpy as np


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


def convert_camera_to_world_transform(transform):
    converted_transform = transform.clone()

    converted_transform[:, 2] *= -1

    converted_transform[[0, 2], :] = converted_transform[[2, 0], :]

    return converted_transform


def circle_poses(
    device, radius=torch.tensor([3.2]), theta=torch.tensor([60]), phi=torch.tensor([0])
):
    theta = theta / 180 * np.pi
    phi = phi / 180 * np.pi

    centers = torch.stack(
        [
            radius * torch.sin(theta) * torch.sin(phi),
            radius * torch.cos(theta),
            radius * torch.sin(theta) * torch.cos(phi),
        ],
        dim=-1,
    )  # [B, 3]

    # lookat
    forward_vector = safe_normalize(centers)
    up_vector = (
        torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(len(centers), 1)
    )
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = (
        torch.eye(4, dtype=torch.float, device=device)
        .unsqueeze(0)
        .repeat(len(centers), 1, 1)
    )
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


trans_t = lambda t: torch.Tensor(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]
).float()

rot_phi = lambda phi: torch.Tensor(
    [
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ]
).float()

rot_theta = lambda th: torch.Tensor(
    [
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1],
    ]
).float()


def rodrigues_mat_to_rot(R):
    eps = 1e-16
    trc = np.trace(R)
    trc2 = (trc - 1.0) / 2.0
    s = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if (1 - trc2 * trc2) >= eps:
        tHeta = np.arccos(trc2)
        tHetaf = tHeta / (2 * (np.sin(tHeta)))
    else:
        tHeta = np.real(np.arccos(trc2))
        tHetaf = 0.5 / (1 - tHeta / 6)
    omega = tHetaf * s
    return omega


def rodrigues_rot_to_mat(r):
    wx, wy, wz = r
    theta = np.sqrt(wx * wx + wy * wy + wz * wz)
    a = np.cos(theta)
    b = (1 - np.cos(theta)) / (theta * theta)
    c = np.sin(theta) / theta
    R = np.zeros([3, 3])
    R[0, 0] = a + b * (wx * wx)
    R[0, 1] = b * wx * wy - c * wz
    R[0, 2] = b * wx * wz + c * wy
    R[1, 0] = b * wx * wy + c * wz
    R[1, 1] = a + b * (wy * wy)
    R[1, 2] = b * wy * wz - c * wx
    R[2, 0] = b * wx * wz - c * wy
    R[2, 1] = b * wz * wy + c * wx
    R[2, 2] = a + b * (wz * wz)
    return R


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        torch.Tensor(
            np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        )
        @ c2w
    )
    return c2w


def convert_camera_pose(camera_pose):
    # Clone the tensor to avoid in-place operations
    colmap_pose = camera_pose.clone()

    # Extract rotation and translation components
    rotation = colmap_pose[:, :3, :3]
    translation = colmap_pose[:, :3, 3]

    # Change rotation orientation
    rotation[:, 0, :] *= -1
    rotation[:, 1, :] *= -1

    # Change translation position
    translation[:, 0] *= -1
    translation[:, 1] *= -1

    return colmap_pose


def convert_camera_pose(camera_pose):
    # Clone the tensor to avoid in-place operations
    colmap_pose = camera_pose.clone()

    # Extract rotation and translation components
    rotation = colmap_pose[:, :3, :3]
    translation = colmap_pose[:, :3, 3]

    # Change rotation orientation
    rotation[:, 0, :] *= -1
    rotation[:, 1, :] *= -1

    # Change translation position
    translation[:, 0] *= -1
    translation[:, 1] *= -1

    return colmap_pose


@dataclass
class GSLoadDataModuleConfig:
    # height, width, and batch_size should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    source: str = None
    height: Any = 512
    width: Any = 512
    batch_size: Any = 1
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    eval_height: int = -1
    eval_width: int = -1
    eval_batch_size: int = 1
    max_view_num: int = 48
    n_val_views: int = 8
    n_test_views: int = 120
    elevation_range: Tuple[float, float] = (-10, 45)
    elevation_view_num: int = 2
    azimuth_range: Tuple[float, float] = (-180, 180)
    azimuth_view_num: int = 8
    camera_distance_range: Tuple[float, float] = (4.0, 6.0)
    fovy_range: Tuple[float, float] = (
        40,
        70,
    )  # in degrees, in vertical direction (along height)
    camera_perturb: float = 0.0
    center_perturb: float = 0.0
    up_perturb: float = 0.0
    light_position_perturb: float = 1.0
    light_distance_range: Tuple[float, float] = (0.8, 1.5)
    eval_elevation_deg: float = 15.0
    eval_camera_distance: float = 6.0
    eval_fovy_deg: float = 70.0
    light_sample_strategy: str = "dreamfusion"
    batch_uniform_azimuth: bool = True
    progressive_until: int = 0  # progressive ranges for elevation, azimuth, r, fovy


class GSLoadIterableDataset(IterableDataset, Updateable):
    def __init__(self, cfg, scene) -> None:
        super().__init__()
        self.cfg: GSLoadDataModuleConfig = cfg
        self.scene = scene
        self.total_view_num = len(self.scene.cameras)
        random.seed(0)  # make sure same views
        self.n2n_view_index = random.sample(
            range(0, self.total_view_num),
            min(self.total_view_num, self.cfg.max_view_num),
        )
        self.view_index_stack = self.n2n_view_index.copy()
        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        self.batch_sizes: List[int] = (
            [self.cfg.batch_size]
            if isinstance(self.cfg.batch_size, int)
            else self.cfg.batch_size
        )
        assert len(self.heights) == len(self.widths) == len(self.batch_sizes)
        self.resolution_milestones: List[int]
        if (
            len(self.heights) == 1
            and len(self.widths) == 1
            and len(self.batch_sizes) == 1
        ):
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.batch_size: int = self.batch_sizes[0]

    def collate(self, batch) -> Dict[str, Any]:
        # sample elevation angles
        cam_list = []
        index_list = []
        for _ in range(self.batch_size):
            if not self.view_index_stack:
                self.view_index_stack = self.n2n_view_index.copy()
            view_index = random.choice(self.view_index_stack)
            self.view_index_stack.remove(view_index)
            cam_list.append(self.scene.cameras[view_index])
            index_list.append(view_index)

        return {
            "index": index_list,
            "camera": cam_list,
            "height": self.height,
            "width": self.width,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        self.width = self.widths[size_ind]
        self.batch_size = self.batch_sizes[size_ind]
        # self.directions_unit_focal = self.directions_unit_focals[size_ind]
        threestudio.debug(
            f"Training height: {self.height}, width: {self.width}, batch_size: {self.batch_size}"
        )
        # progressive view
        self.progressive_view(global_step)

    def __iter__(self):
        while True:
            yield {}

    def progressive_view(self, global_step):
        pass


class GSLoadDataset(Dataset):
    def __init__(self, cfg, split, scene, train_view_list=None) -> None:
        super().__init__()
        self.cfg: GSLoadDataModuleConfig = cfg
        self.split = split
        self.scene = scene
        self.total_view_num = len(self.scene.cameras)

        if split == "val":
            self.n_views = self.cfg.n_val_views
            self.h = self.cfg.height
            self.w = self.cfg.width
        else:
            self.n_views = self.total_view_num
            self.h = self.cfg.eval_height if self.cfg.eval_height > 0 else self.scene.cameras[0].image_height
            self.w = self.cfg.eval_width if self.cfg.eval_width > 0 else self.scene.cameras[0].image_width

        if train_view_list is None:
            self.selected_views = torch.linspace(
                0, self.total_view_num - 1, self.n_views, dtype=torch.int
            )
        else:
            train_view_list = sorted(train_view_list)
            self.selected_views = torch.linspace(
                0, len(train_view_list) - 1, self.n_views, dtype=torch.int
            )
            self.selected_views = [train_view_list[idx] for idx in self.selected_views]

    def __len__(self):
        return self.n_views

    def __getitem__(self, index):
        return {
            "index": self.selected_views[index] if self.split == "val" else index,
            "height": self.h,
            "width": self.w,
        }

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.h, "width": self.w})
        return batch


@register("gs-load")
class GS_load(pl.LightningDataModule):
    cfg: GSLoadDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        from gaussiansplatting.scene.camera_scene import CamScene

        super().__init__()
        self.cfg = parse_structured(GSLoadDataModuleConfig, cfg)
        self.train_scene = CamScene(
            self.cfg.source, h=self.cfg.height, w=self.cfg.width
        )
        self.eval_scene = CamScene(
            self.cfg.source, h=self.cfg.eval_height, w=self.cfg.eval_width
        )

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = GSLoadIterableDataset(self.cfg, self.train_scene)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = GSLoadDataset(
                self.cfg, "val", self.eval_scene, self.train_dataset.n2n_view_index
            )
        if stage in [None, "test", "predict"]:
            self.test_dataset = GSLoadDataset(self.cfg, "test", self.eval_scene)

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset,
            batch_size=self.cfg.n_val_views,
            collate_fn=self.val_dataset.collate,
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )
