"""v0.3 feature tests: triquadratic sampler, gradient/stencils, per-leaf stats, GPU reduce."""
import numpy as np
import pytest

import xuvdb
from xuvdb.gpu import GpuVolume
from xuvdb.tree import _quad_weight


def make_grid():
    grid = xuvdb.VdbGrid(background=0.15, voxel_size=0.05, leaf_log2=4,
                         name="shield", grid_class="level set")
    grid.stamp_sphere((0.3, 0.2, 0.1), radius=0.25, band=3.0)
    second = grid.copy()
    second.stamp_sphere((0.5, 0.2, 0.1), radius=0.10)
    grid.csg(second, "union")
    return grid


def dense_triquadratic_reference(grid, dense, ijk_min, pts):
    """Independent quadratic B-spline triquadratic over the dense array (test-side reference)."""
    h = grid.voxel_size
    coords = (pts - grid.origin_world) / h
    base = np.floor(coords).astype(np.int64)
    u = coords - base
    shape = np.asarray(dense.shape[:3])
    out = np.zeros(len(pts))
    for j in range(3):
        for k in range(3):
            for l in range(3):
                w = (_quad_weight(u[:, 0], j) * _quad_weight(u[:, 1], k) * _quad_weight(u[:, 2], l))
                idx = base + np.array([j - 1, k - 1, l - 1]) - ijk_min
                inside = np.all((idx >= 0) & (idx < shape), axis=1)
                vals = np.full(len(pts), grid.background)
                vv = dense[idx[inside, 0], idx[inside, 1], idx[inside, 2]]
                vals[inside] = np.asarray(vv, dtype=np.float64).reshape(-1)
                out += w * vals
    return out


def test_quadratic_matches_dense_reference():
    grid = make_grid()
    dense, ijk_min = grid.to_dense(pad=3)
    pts = np.random.default_rng(3).uniform(-0.05, 0.85, size=(400, 3))
    ref = dense_triquadratic_reference(grid, np.asarray(dense), ijk_min, pts)
    got = _vec(grid.sample_quadratic, pts)
    assert np.allclose(got, ref, atol=1e-5), np.abs(got - ref).max()


def _vec(fn, pts):
    return np.array([fn(p) for p in pts], dtype=np.float64)


def test_quadratic_constant_field_and_normalization():
    grid = xuvdb.VdbGrid(background=0.15, voxel_size=0.05, name="fog", grid_class="fog volume")
    grid.fill_box((0, 0, 0), (30, 30, 30), 2.5)
    pts = np.random.default_rng(4).uniform(0.1, 1.3, size=(200, 3))
    vals = _vec(grid.sample_quadratic, pts)
    inside = np.all((pts / 0.05 >= 1) & (pts / 0.05 <= 29), axis=1)
    assert np.allclose(vals[inside], 2.5, atol=1e-6)  # weights sum to 1: constant reproduced


def test_kernel_quadratic_matches_host():
    grid = make_grid()
    vol = GpuVolume(grid)
    pts = np.random.default_rng(5).uniform(-0.05, 0.85, size=(512, 3))
    host = np.array([grid.sample_quadratic(p) for p in pts])
    kern = vol.sample(pts.astype(np.float32), order=2)
    assert np.allclose(kern, host, atol=2e-5), np.abs(kern - host).max()
    # order kwarg backward compatibility
    assert np.allclose(vol.sample(pts[:8].astype(np.float32), linear=True),
                       vol.sample(pts[:8].astype(np.float32), order=1))
    assert np.allclose(vol.sample(pts[:8].astype(np.float32)),
                       vol.sample(pts[:8].astype(np.float32), order=0))
    with pytest.raises(ValueError):
        vol.sample(pts[:2].astype(np.float32), order=3)


@pytest.mark.parametrize("order", [1, 2])
def test_gradient_is_radial_on_sphere(order):
    grid = xuvdb.VdbGrid(background=0.15, voxel_size=0.05, leaf_log2=4,
                         name="s", grid_class="level set")
    c, r = np.array([0.4, 0.4, 0.4]), 0.3
    grid.stamp_sphere(c, radius=r, band=3.0)
    dirs = np.random.default_rng(6).normal(size=(64, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pts = c + dirs * r  # exactly on the surface: gradient must be the radial unit vector
    grad = grid.sample_gradient(pts, order=order)
    norm = np.linalg.norm(grad, axis=1)
    assert np.allclose(norm, 1.0, atol=5e-2)
    assert np.allclose((grad * dirs).sum(axis=1), 1.0, atol=5e-2)


def test_stencils_match_direct_probes():
    grid = make_grid()
    idxs = grid.active_indices()[:200]
    s7 = grid.stencil7_batch(idxs)
    s19 = grid.stencil19_batch(idxs)
    off7 = np.asarray(grid._STENCIL7)
    off19 = np.asarray(grid._STENCIL7 + grid._STENCIL19_EDGES)
    for n in (0, 17, 199):
        for a, off in enumerate(off7):
            assert s7[n, a] == grid.get_value(idxs[n] + off)
        for a, off in enumerate(off19):
            assert s19[n, a] == grid.get_value(idxs[n] + off)


def test_stencil_central_difference_exact_on_linear_field():
    grid = xuvdb.VdbGrid(voxel_size=0.1, name="lin")
    grid.fill_box((0, 0, 0), (40, 40, 40), 0.0)
    a = np.array([0.7, -1.3, 2.1])
    ijk = np.stack(np.meshgrid(range(41), range(41), range(41), indexing="ij"), axis=-1).reshape(-1, 3)
    lin = ijk @ a + 3.0
    for row, v in zip(ijk, lin):
        grid.set_value((int(row[0]), int(row[1]), int(row[2])), float(v))
    probes = np.array([[10, 20, 16], [30, 8, 32]], dtype=np.int64)
    vals = grid.stencil7_batch(probes)
    for n in range(len(probes)):
        c = vals[n, 0]
        gx = (vals[n, 1] - vals[n, 2]) / 0.2  # (+x -x) / 2h
        gy = (vals[n, 3] - vals[n, 4]) / 0.2
        gz = (vals[n, 5] - vals[n, 6]) / 0.2
        assert np.allclose([gx, gy, gz], a / 0.1, atol=1e-3)  # world-space gradient: a / h
        assert c == grid.get_value(tuple(probes[n]))


def test_value_range_cache_and_invalidation():
    grid = make_grid()
    act = np.concatenate([l.values[l.active] for l in grid.leaves()])
    lo, hi = grid.value_range()
    assert abs(lo - act.min()) < 1e-6 and abs(hi - act.max()) < 1e-6
    grid.fill_box((2, 2, 2), (60, 60, 60), 9.5)  # must invalidate the stats cache
    act2 = np.concatenate([l.values[l.active] for l in grid.leaves()])
    lo2, hi2 = grid.value_range()
    assert hi2 == pytest.approx(9.5) and lo2 == pytest.approx(act2.min())
    with pytest.raises(TypeError):
        xuvdb.VdbGrid(dtype=np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32)])
                      ).value_range()


def test_reduce_matches_brute_force():
    grid = make_grid()
    vol = GpuVolume(grid)
    r = vol.reduce()
    act = np.concatenate([l.values[l.active] for l in grid.leaves()])
    assert r["count"] == len(act)
    assert r["sum"] == pytest.approx(float(act.sum()), abs=1e-3)
    assert r["min"] == pytest.approx(float(act.min()), abs=1e-6)
    assert r["max"] == pytest.approx(float(act.max()), abs=1e-6)
    # structural edit + repack: reduce follows the new mask
    grid.fill_box((0, 0, 0), (30, 30, 30), 1.0)
    vol2 = GpuVolume(grid)
    r2 = vol2.reduce()
    act2 = np.concatenate([l.values[l.active] for l in grid.leaves()])
    assert r2["count"] == len(act2)
    assert r2["sum"] == pytest.approx(float(act2.sum()), abs=1e-3)
    # a grid whose packed leaves are all-inactive reduces to zeros, not +-1e30
    empty = xuvdb.VdbGrid(voxel_size=0.05, name="e")
    empty.fill_box((0, 0, 0), (3, 3, 3), 1.0, active=False)
    r3 = GpuVolume(empty).reduce()
    assert r3["count"] == 0 and r3["min"] is None and r3["max"] is None
