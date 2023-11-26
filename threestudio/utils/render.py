import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import trimesh
import pyrender
from mediapy import write_image, write_video

from gaussiansplatting.scene.cameras import Camera, Simple_Camera, C2W_Camera

os.environ["PYOPENGL_PLATFORM"] = "egl"


def get_scene_radius(c2ws, scale: float = 1.1):
    camera_centers = c2ws[..., :3, 3]
    camera_centers = np.linalg.norm(
        camera_centers - np.mean(camera_centers, axis=0, keepdims=True), axis=-1
    )
    return np.max(camera_centers) * scale


def get_c2w_from_up_and_look_at(
    up,
    look_at,
    pos,
    return_pt=False,
):
    up = up / np.linalg.norm(up)
    z = look_at - pos
    z = z / np.linalg.norm(z)
    y = -up
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    c2w = np.zeros([4, 4], dtype=np.float32)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = pos
    c2w[3, 3] = 1.0

    if return_pt:
        c2w = torch.from_numpy(c2w).to(torch.float32)

    return c2w

def get_horizontal_poses(num_images, camera_dist, coord="opengl"):
    points = []
    azi =[]
    ele = []
    camera_dist_list=[]
    for _ in range(num_images):
        # front view xz=0 y=-1 theta = 1.5pi
        camera_dist_list.append(camera_dist*np.random.uniform(1.00, 1.00))
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.arccos(np.random.uniform(-0.10, 0.4))
        # theta = 1.0 * np.pi
        # phi = np.arccos(0.5)

        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        points.append(np.array([x, y, z]))
        azi.append((theta/np.pi -1.5) *180) # -270 to 90
        ele.append(90-phi/np.pi*180) # change to angle

    look_at = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])

    poses = []

    for idx, p in enumerate(points):
        pose = get_c2w_from_up_and_look_at(
            up, look_at, p * camera_dist_list[idx], return_pt=False
        )
        if coord == "opengl":
            pose[..., 1:3] *= -1
        poses.append(pose)

    poses = np.stack(poses, axis=0)

    return poses, azi, ele


def get_uniform_poses(num_images, camera_dist, coord="opengl"):
    points = []
    for idx in range(num_images):
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.arccos(1 - 2 * np.random.uniform(0, 1))
        # theta = idx % 4 * 0.5 * np.pi + 1e-7
        # phi = np.arccos(0)

        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        points.append(np.array([x, y, z]))

    look_at = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])

    poses = []

    for p in points:
        pose = get_c2w_from_up_and_look_at(
            up, look_at, p * camera_dist, return_pt=False
        )
        if coord == "opengl":
            pose[..., 1:3] *= -1
        poses.append(pose)

    poses = np.stack(poses, axis=0)

    return poses


def render_multiview_images_from_mesh(
    mesh_file, horizontal=False,n_images=36, camera_dist=1.0, fov=np.pi / 3.0, reso=512, save_path=None
):
    fuze_trimesh = trimesh.load(mesh_file)
    mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0])
    mesh_node = pyrender.Node(mesh=mesh, matrix=np.eye(4))
    scene.add_node(mesh_node)

    if horizontal:
        poses, azi, ele = get_horizontal_poses(int(n_images*1.5), camera_dist, coord="opengl")
    else:
        poses = get_uniform_poses(n_images, camera_dist*1.0, coord="opengl")
    cam_nodes = []
    for p in poses:
        cam = pyrender.PerspectiveCamera(yfov=fov, aspectRatio=1.0)
        cam_node = pyrender.Node(camera=cam, matrix=p)
        scene.add_node(cam_node)
        cam_nodes.append(cam_node)

    r = pyrender.OffscreenRenderer(reso, reso)
    frames = []
    for c in cam_nodes:
        scene.main_camera_node = c
        frame, _ = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT)

        frames.append(frame)

    frames = np.stack(frames, axis=0)
    if save_path is not None:
        write_video(save_path, frames, fps=2)

    scene_radius = get_scene_radius(poses)

    poses = torch.from_numpy(poses).cuda()
    # opengl to opencv
    poses[..., 1:3] *= -1
    poses = torch.inverse(poses)
    poses[..., :3, :3] = poses[..., :3, :3].transpose(-1, -2)

    cams = []
    for idx, pose in enumerate(poses):
        if horizontal:
            cams.append(C2W_Camera(pose, fov, reso, reso, azimuth=azi[idx],elevation=ele[idx], dist=camera_dist))
        else:
            cams.append(C2W_Camera(pose, fov, reso, reso))

    frames = frames.astype(np.float32) / 255.0

    return cams, frames, scene_radius


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mesh_file", required=True, help="path to mesh file", type=str
    )

    opt = parser.parse_args()

    render_multiview_images_from_mesh(opt.mesh_file)
