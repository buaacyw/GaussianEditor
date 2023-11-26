import torch
pytorch3d_capable = True
try:
    import pytorch3d
    from pytorch3d.ops import estimate_pointcloud_normals
    from pytorch3d.ops import sample_farthest_points
    from pytorch3d.ops import knn_points
except ImportError:
    pytorch3d_capable = False


@torch.no_grad()
def K_nearest_neighbors(
    mean: torch.Tensor,
    K: int,
    query: None,
    return_dist: bool = False,
):
    # REMINDER: [2023-10-20] I have changed this function to return nn_dist in additional to index and the point. Do not panic if some bugs occur.
    if not pytorch3d_capable:
        raise ImportError("pytorch3d is not installed, which is required for KNN")
    # TODO: finish this
    if query is None:
        query = mean
    dist, idx, nn = knn_points(
        query[None, ...], mean[None, ...], K=K, return_nn=True
    )

    if not return_dist:
        return nn[0, :, 1:, :], idx[0, :, 1:]
    else:
        return (
            nn[0],
            idx[0],
            torch.sqrt(dist[0] + torch.finfo(dist.dtype).eps),
        )
