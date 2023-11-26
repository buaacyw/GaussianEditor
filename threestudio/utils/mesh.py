import numpy as np
import torch
import trimesh


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.

    reference: https://github.com/mikedh/trimesh/issues/507#issuecomment-514973337
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()
                )
            )
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh


def load_mesh_as_pcd_trimesh(mesh_file, num_points):
    mesh = as_mesh(trimesh.load_mesh(mesh_file))
    n = num_points
    points = []
    while n > 0:
        p, _ = trimesh.sample.sample_surface_even(mesh, n)
        n -= p.shape[0]
        if n >= 0:
            points.append(p)
        else:
            points.append(p[:n])
    if len(points) > 1:
        points = np.concatenate(points, axis=0)
    else:
        points = points[0]
    points = torch.from_numpy(points.astype(np.float32))

    return points, torch.rand_like(points)


def get_random_poses(num_images, camera_dist, coord="opengl"):
    pass


def generate_dataset_given_mesh(mesh_file, num_images):
    mesh = trimesh.load_mesh(str(mesh_file))
    mesh = trimesh.Scene(mesh)
