import os
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.special import expit
import struct
import argparse
C0 = 0.28209479177387814

def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

def to_splat(ply_file):
    ply_file = Path(ply_file)

    plydata = PlyData.read(ply_file)

    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")
    ]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
    # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
    max_sh_degree = int(((len(extra_f_names) + 3) / 3) ** 0.5 - 1)
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape(
        (features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1)
    )

    scale_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")
    ]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
    ]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    pos = xyz
    rgb = (SH2RGB(features_dc.squeeze()) * 255.0).astype(np.uint8).clip(0, 255)
    # shs = np.concatenate((features_dc, features_extra), axis=2)
    # print(shs.shape)
    # dirs = np.array([[1.0, 0.0, 0.0]] * shs.shape[0])
    # rgb = eval_sh(max_sh_degree, shs, dirs)

    # rgb = (rgb * 255.0).astype(np.uint8).clip(0, 255)

    # breakpoint()

    opacity = (expit(opacities) * 255.0).astype(np.uint8).clip(0, 255)
    svec = np.exp(scales)
    qvec = rots
    qvec = qvec / np.linalg.norm(qvec, axis=1, keepdims=True)
    qvec = qvec * 128 + 128
    qvec = qvec.astype(np.uint8).clip(0, 255)

    n = pos.shape[0]

    volume = np.prod(svec, axis=1) * opacity[..., 0]
    index = list(range(n))
    index = sorted(index, key=lambda i: volume[i], reverse=True)

    filename = ply_file.stem + ".splat"
    filename = ply_file.parent / filename
    with open(filename, "wb") as f:
        for i in index:
            f.write(struct.pack("fff", pos[i, 0], pos[i, 1], pos[i, 2]))
            f.write(struct.pack("fff", svec[i, 0], svec[i, 1], svec[i, 2]))
            f.write(struct.pack("BBBB", rgb[i, 0], rgb[i, 1], rgb[i, 2], opacity[i, 0]))
            f.write(struct.pack("BBBB", qvec[i, 0], qvec[i, 1], qvec[i, 2], qvec[i, 3]))


if __name__ == "__main__":
    ply_dir = "D:/lib/demo/to_trans"
    for f in os.listdir(ply_dir):
        cur_ply = os.path.join(ply_dir,f)
        to_splat(cur_ply)
