"""SDF offset tests: eikonal dilation/erosion, the LevelSetFilter::offset equivalent."""
import time

import numpy as np
import pytest

import xuvdb
from xuvdb.offset import offset


def make_sphere(radius=0.25, voxel=0.02, background=0.2, center=(0.0, 0.0, 0.0)):
    """Narrow-band level-set sphere with a wide-enough band for +/-0.1 offsets."""
    grid = xuvdb.VdbGrid(background=background, voxel_size=voxel, leaf_log2=4,
                         name="ls", grid_class="level set")
    grid.stamp_sphere(center, radius=radius, band=3.0)
    return grid


def radial_errors(grid, r_new, n=200, t_lo=0.02, t_hi=0.08, seed=0):
    """|sample_linear - true distance to the r_new sphere| at random outside points."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    t = rng.uniform(t_lo, t_hi, size=n)
    pts = dirs * (r_new + t)[:, None]
    got = np.array([float(grid.sample_linear(p)) for p in pts])
    return np.abs(got - t)


def test_offset_dilation_matches_true_distance():
    """+0.1 on a r=0.25 sphere: samples land on the r=0.35 SDF within 1.5 voxels."""
    grid = make_sphere()
    assert offset(grid, 0.1) is grid  # in place, returns the grid
    err = radial_errors(grid, r_new=0.35)
    print(f"dilation max error: {err.max():.4f} m ({err.max() / 0.02:.2f} voxels)")
    assert err.max() < 0.03


def test_offset_erosion_matches_true_distance():
    """-0.1: samples land on the r=0.15 SDF within 1.5 voxels."""
    grid = make_sphere()
    offset(grid, -0.1)
    err = radial_errors(grid, r_new=0.15)
    print(f"erosion max error: {err.max():.4f} m ({err.max() / 0.02:.2f} voxels)")
    assert err.max() < 0.03


def test_offset_monotonic_volume():
    """Positive offset grows the active region, negative shrinks it (equal bands)."""
    base = make_sphere().active_voxel_count
    grown = make_sphere()
    offset(grown, 0.1, band=3.0)
    shrunk = make_sphere()
    offset(shrunk, -0.1, band=3.0)
    assert grown.active_voxel_count > base
    assert shrunk.active_voxel_count < base


def test_offset_invariants():
    """background / grid_class unchanged; active values stay inside the band."""
    grid = make_sphere()
    bg, cls = grid.background, grid.grid_class
    band_vox = grid.background / float(np.mean(grid.voxel_size))
    band_world = band_vox * float(np.mean(grid.voxel_size))
    voxel = float(np.mean(grid.voxel_size))
    offset(grid, 0.1)
    assert grid.background == bg
    assert grid.grid_class == cls
    vals = np.abs(grid.active_values())
    assert vals.max() <= band_world + 2.0 * voxel


def test_offset_zero_distance_is_noop():
    grid = make_sphere()
    before = grid.copy()
    assert offset(grid, 0.0) is grid
    assert grid.active_voxel_count == before.active_voxel_count
    for (c0, v0), (c1, v1) in zip(grid.iter_voxels(), before.iter_voxels()):
        assert c0 == c1 and v0 == pytest.approx(v1, abs=1e-6)


def test_offset_bridges_separated_spheres():
    """Union of two close spheres dilated by +0.05: the gap bridges (midpoint inside)."""
    grid = make_sphere()
    grid.stamp_sphere((0.56, 0.0, 0.0), radius=0.25, band=3.0)
    mid = (0.28, 0.0, 0.0)
    assert grid.sample_linear(mid) > 0.0  # outside before the offset
    offset(grid, 0.05)
    assert grid.sample_linear(mid) < 0.0  # bridged after


def test_offset_rejects_vec_grids():
    grid = xuvdb.VdbGrid(dtype=np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32)]))
    with pytest.raises(TypeError):
        offset(grid, 0.1)


def test_offset_performance():
    """dim=16 leaves, ~20k active voxels: one offset must finish well under 2 s."""
    grid = make_sphere(radius=0.35)
    n_active = grid.active_voxel_count
    t0 = time.perf_counter()
    offset(grid, 0.1)
    elapsed = time.perf_counter() - t0
    print(f"offset(+0.1) on {n_active} active voxels, {grid.n_leaves} leaves: {elapsed:.2f}s")
    assert elapsed < 2.0
    assert grid.active_voxel_count > n_active
