"""v0.5 feature tests: torch bridge (tensors, differentiable sampling) and VolumeBatch."""
import numpy as np
import pytest

import xuvdb
from xuvdb.gpu import GpuVolume, VolumeBatch

torch = pytest.importorskip("torch")

from xuvdb.torch_bridge import grid_to_tensors, sample, sample_t, tensors_to_grid  # noqa: E402


def make_grid(offset=0.0):
    grid = xuvdb.VdbGrid(background=0.15, voxel_size=0.05, leaf_log2=4,
                         name=f"g{offset}", grid_class="level set")
    grid.stamp_sphere((0.3 + offset, 0.2, 0.1), radius=0.25, band=3.0)
    return grid


PTS = np.random.default_rng(12).uniform(-0.05, 0.85, size=(256, 3))


def host(grid, pts):
    return np.array([grid.sample_linear(p) for p in pts])


def test_tensor_roundtrip_cpu_and_cuda_if_available():
    grid = make_grid()
    for device in (["cpu"] + (["cuda"] if torch.cuda.is_available() else [])):
        t = grid_to_tensors(grid, device=device)
        assert t["keys"].device.type == device and t["values"].shape[1] == grid.leaf_dim**3
        back = tensors_to_grid(t, grid.voxel_size, grid.origin_world, grid.leaf_log2,
                               grid.background, grid.name, grid.grid_class)
        assert np.allclose(host(grid, PTS), host(back, PTS), atol=1e-6)


def test_differentiable_forward_matches_host():
    grid = make_grid()
    got = sample(grid, PTS).numpy()
    assert np.allclose(got, host(grid, PTS), atol=1e-5)


def test_gradient_wrt_values_matches_finite_differences():
    grid = make_grid()
    t = grid_to_tensors(grid)
    t["values"].requires_grad_(True)
    pts = torch.as_tensor(PTS[:64], dtype=torch.float32)
    out = sample_t(t, pts, grid.voxel_size, grid.origin_world, grid.leaf_log2, grid.background)
    out.sum().backward()
    g = t["values"].grad.numpy()  # (n_leaves, dim^3)

    # finite differences on a few leaf cells with nonzero analytic gradient
    rng = np.random.default_rng(3)
    cells = [(r, c) for r, c in zip(*np.nonzero(np.abs(g) > 1e-3))][:12] or \
        [(0, int(rng.integers(0, g.shape[1])))]
    vals0 = t["values"].detach().numpy().copy()
    eps = 1e-2
    for r, c in cells:
        pert = vals0.copy()
        pert[r, c] += eps
        t2 = {"keys": t["keys"], "values": torch.as_tensor(pert), "active": t["active"]}
        f1 = sample_t(t2, pts, grid.voxel_size, grid.origin_world, grid.leaf_log2,
                      grid.background).sum().item()
        f0 = float(out.sum().item())
        assert abs((f1 - f0) / eps - g[r, c]) < 5e-3 * max(1.0, abs(g[r, c])), (r, c)


def test_gradient_wrt_points_matches_finite_differences():
    grid = make_grid()
    t = grid_to_tensors(grid)
    pts = torch.tensor(PTS[:32], dtype=torch.float32, requires_grad=True)
    out = sample_t(t, pts, grid.voxel_size, grid.origin_world, grid.leaf_log2, grid.background)
    out.sum().backward()
    g = pts.grad.detach().numpy()
    eps = 1e-3
    with torch.no_grad():
        base = sample_t(t, pts, grid.voxel_size, grid.origin_world, grid.leaf_log2,
                        grid.background).sum().item()
    for i in (0, 7, 31):
        for a in range(3):
            p2 = pts.detach().clone()
            p2[i, a] += eps
            f2 = sample_t(t, p2, grid.voxel_size, grid.origin_world, grid.leaf_log2,
                          grid.background).sum().item()
            assert abs((f2 - base) / eps - g[i, a]) < 5e-3 * max(1.0, abs(g[i, a])), (i, a)


def test_volume_batch_sample_parity():
    g0 = make_grid(0.0)
    g1 = make_grid(0.15)  # different field position; same voxel size
    g2 = make_grid(0.0)
    g2.voxel_size = np.array([0.025, 0.05, 0.05])  # different transform, same data shape
    batch = VolumeBatch([g0, g1, g2])
    assert len(batch) == 3
    vols = [GpuVolume(g) for g in (g0, g1, g2)]
    rng = np.random.default_rng(5)
    ids = rng.integers(0, 3, size=300)
    pts = np.empty((300, 3), np.float32)
    for i, vid in enumerate(ids):
        c = np.array([0.3 + (0.15 if vid == 1 else 0.0), 0.2, 0.1])
        pts[i] = c + rng.normal(scale=0.12, size=3)
    got = batch.sample(pts, ids)
    want = np.empty(300, np.float32)
    for i, vid in enumerate(ids):
        want[i] = vols[vid].sample(pts[i:i + 1], linear=True)[0]
    assert np.allclose(got, want, atol=1e-5), np.abs(got - want).max()
    with pytest.raises(IndexError):
        batch.sample(pts[:2], [0, 3])
