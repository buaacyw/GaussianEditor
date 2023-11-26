import numpy as np
import torch
from kornia.geometry.quaternion import Quaternion


@torch.no_grad()
def scale_gaussians(gaussian, scale):
    gaussian._xyz.data = gaussian._xyz.data * scale
    g_scale = gaussian.get_scaling * scale
    gaussian._scaling.data = torch.log(g_scale + 1e-7)


@torch.no_grad()
def rotate_gaussians(gaussian, rotmat):
    rot_q = Quaternion.from_matrix(rotmat[None, ...])
    g_qvec = Quaternion(gaussian.get_rotation)
    gaussian._rotation.data = (rot_q * g_qvec).data

    gaussian._xyz.data = torch.einsum("ij,bj->bi", rotmat, gaussian._xyz.data)


@torch.no_grad()
def translate_gaussians(gaussian, tvec):
    gaussian._xyz.data = gaussian._xyz.data + tvec[None, ...]


from scipy.spatial.transform import Rotation as R

default_model_mtx = (
    torch.from_numpy(R.from_rotvec(-np.pi / 2 * np.array([1.0, 0.0, 0.0])).as_matrix())
    .float()
    .cuda()
)
