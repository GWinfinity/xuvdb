"""Level-set smoothing tests: the LevelSetFilter::smooth equivalent (Jacobi relaxation)."""
import numpy as np
import pytest

import xuvdb
from xuvdb.smooth import smooth


def make_sphere(radius=0.25, voxel=0.02, background=0.2):
    grid = xuvdb.VdbGrid(background=background, voxel_size=voxel, leaf_log2=4,
                         name="ls", grid_class="level set")
    grid.stamp_sphere((0.0, 0.0, 0.0), radius=radius, band=3.0)
    return grid


def radial_errors(grid, r_true, n=200, t_lo=0.02, t_hi=0.05, seed=0):
    """|sample_linear - true distance| at random points outside the r_true sphere."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    t = rng.uniform(t_lo, t_hi, size=n)
    pts = dirs * (r_true + t)[:, None]
    got = np.array([float(grid.sample_linear(p)) for p in pts])
    return np.abs(got - t)


def band_laplacian_magnitude(grid):
    """max |discrete Laplacian| over the stored (allocated) values."""
    worst = 0.0
    vs = float(np.mean(grid.voxel_size))
    for leaf in grid.leaves():
        p = np.pad(leaf.values.astype(np.float64), 1, mode="edge")
        lap = (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1]
               + p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1]
               + p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2]
               - 6.0 * p[1:-1, 1:-1, 1:-1]) / (vs * vs)
        active = np.pad(leaf.active, 1, mode="constant", constant_values=False)
        m = active[1:-1, 1:-1, 1:-1] & np.pad(leaf.active, 1, mode="edge")[1:-1, 1:-1, 1:-1]
        if m.any():
            worst = max(worst, float(np.abs(lap[m]).max()))
    return worst


def test_smooth_sphere_stays_accurate():
    """A handful of sweeps must not wreck an already-smooth SDF (mask unchanged)."""
    grid = make_sphere()
    n_active_before = grid.active_voxel_count
    assert smooth(grid, iterations=5, alpha=0.5) is grid
    assert grid.active_voxel_count == n_active_before  # active mask preserved
    err = radial_errors(grid, r_true=0.25)
    assert err.max() < 0.03  # well within a couple of voxels


def test_smooth_reduces_noise():
    """Corrupted band values: smoothing must shrink the Laplacian magnitude hard."""
    grid = make_sphere()
    rng = np.random.default_rng(7)
    for leaf in grid.leaves():
        noise = rng.normal(0.0, 0.01, size=leaf.values.shape)  # half a voxel
        leaf.values[...] = (leaf.values + noise).astype(np.float32)
        leaf.invalidate()
    before = band_laplacian_magnitude(grid)
    smooth(grid, iterations=10, alpha=1.0)
    after = band_laplacian_magnitude(grid)
    print(f"max |laplacian|: {before:.1f} -> {after:.1f}")
    assert after < 0.5 * before
    # the underlying sphere is still intact
    err = radial_errors(grid, r_true=0.25)
    assert err.max() < 0.04  # within the (un-re-stamped) narrow band


def test_smooth_rounds_box_creases():
    """Crease (large |laplacian|) of a box level set softens under smoothing."""
    grid = xuvdb.VdbGrid(background=0.2, voxel_size=0.02, leaf_log2=4,
                         name="box", grid_class="level set")
    c, h = (0.1, 0.1, 0.1), 0.05
    pts = np.array([[c[0] + sx * (h + 0.06), c[1] + sy * (h + 0.06),
                     c[2] + sz * (h + 0.06)]
                    for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    grid.union_spheres(pts, radius=0.06)  # blobby union; creases at the joints
    before = band_laplacian_magnitude(grid)
    smooth(grid, iterations=10, alpha=1.0)
    after = band_laplacian_magnitude(grid)
    assert after < 0.6 * before


def test_smooth_equivalence_single_vs_multi_sweep():
    """iterations=N in one call == N chained calls of iterations=1 (buffer swap)."""
    g1 = make_sphere()
    g2 = make_sphere()
    smooth(g1, iterations=4, alpha=0.8)
    for _ in range(4):
        smooth(g2, iterations=1, alpha=0.8)
    pts = np.random.default_rng(1).normal(size=(64, 3)) * 0.3
    v1 = np.array([g1.sample_linear(p) for p in pts])
    v2 = np.array([g2.sample_linear(p) for p in pts])
    # chained calls quantize to float32 between sweeps; one call does not
    assert np.allclose(v1, v2, atol=2e-4)


def test_smooth_validation_and_noop():
    grid = make_sphere()
    assert smooth(grid, iterations=0) is grid  # no-op
    with pytest.raises(ValueError):
        smooth(grid, alpha=0.0)
    with pytest.raises(ValueError):
        smooth(grid, alpha=1.5)
    vec_grid = xuvdb.VdbGrid(voxel_size=0.02,
                             dtype=np.dtype([("x", np.float32), ("y", np.float32),
                                             ("z", np.float32)]))
    with pytest.raises(TypeError):
        smooth(vec_grid)
    # rotation must be given at construction (`is_axis_aligned` is fixed then)
    rotated = xuvdb.VdbGrid(background=0.2, voxel_size=0.02, leaf_log2=4,
                            grid_class="level set",
                            rotation=np.array([[0.0, -1.0, 0.0],
                                               [1.0, 0.0, 0.0],
                                               [0.0, 0.0, 1.0]]))
    with pytest.raises(ValueError):
        smooth(rotated)


def test_smooth_far_field_untouched():
    """Voxels outside the active bbox keep the background value."""
    grid = make_sphere()
    smooth(grid, iterations=4)
    idx = np.array([[100, 100, 100], [-80, 40, 12]])
    vals, act = grid.probe_batch(idx)
    assert np.all(vals == float(grid.background))
    assert not act.any()
