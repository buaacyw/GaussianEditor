import torch
from scipy.spatial import KDTree


@torch.no_grad()
def K_nearest_neighbors(
    mean: torch.Tensor, K: int, query: None, return_dist: bool = False
):
    mean_np = mean.detach().cpu().numpy()
    query_np = query.detach().cpu().numpy()

    kdtree = KDTree(mean_np)

    nn_dist, nn_idx = kdtree.query(query_np, k=K)

    nn_dist = torch.from_numpy(nn_dist).to(mean)
    nn_idx = torch.from_numpy(nn_idx).to(mean.device).to(torch.long)

    if not return_dist:
        return mean[nn_idx], nn_idx
    else:
        return mean[nn_idx], nn_idx, nn_dist


if __name__ == "__main__":
    mean = torch.randn(100000, 3, device="cuda")
    query = torch.randn(100000, 3, device="cuda")

    nn_dist, nn_idx = K_nearest_neighbors(mean, 10, query)

    breakpoint()
