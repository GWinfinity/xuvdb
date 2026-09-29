"""v0.2 feature tests: leaf active-bbox cache, bbox-culling DDA, kernel ray, batch queries."""
import pytest

import numpy as np

import xuvdb
from xuvdb.gpu import GpuVolume


def make_grid():
    """Union of two spheres via csg (the README quick-start shape), 16^3 leaves."""
    grid = xuvdb.VdbGrid(background=0.15, voxel_size=0.05, leaf_log2=4,
                         name="shield", grid_class="level set")
    grid.stamp_sphere((0.3, 0.2, 0.1), radius=0.25, band=3.0)
    second = grid.copy()
    second.stamp_sphere((0.5, 0.2, 0.1), radius=0.10)
    grid.csg(second, "union")
    return grid


def brute_leaf_bbox(leaf):
    xs, ys, zs = np.nonzero(leaf.active)
    if xs.size == 0:
        return (None, None)  # an all-inactive leaf (pre-prune) caches an empty bbox
    return (np.array([xs.min(), ys.min(), zs.min()]), np.array([xs.max(), ys.max(), zs.max()]))


def assert_bbox_equal(got, ref):
    assert (got[0] is None) == (ref[0] is None)
    if ref[0] is not None:
        assert np.array_equal(got[0], ref[0]) and np.array_equal(got[1], ref[1])


@pytest.mark.parametrize("mutate", ["fill", "set", "csg", "clear"])
def test_leaf_bbox_cache_tracks_mutations(mutate):
    grid = make_grid()
    for leaf in grid.leaves():
        assert_bbox_equal(leaf.active_bbox(), brute_leaf_bbox(leaf))
    if mutate == "fill":
        grid.fill_box((2, 2, 2), (5, 5, 5), 0.0)
    elif mutate == "set":
        grid.set_value((3, 3, 3), -0.1)
    elif mutate == "csg":
        other = grid.copy()
        other.stamp_sphere((0.4, 0.25, 0.15), radius=0.05)
        grid.csg(other, "union")
    else:
        grid.fill_box((0, 0, 0), (60, 60, 60), 0.0, active=False)
    for leaf in grid.leaves():
        lo, hi = leaf.active_bbox()
        blo, bhi = brute_leaf_bbox(leaf)
        assert np.array_equal(lo, blo) and np.array_equal(hi, bhi)


def test_bbox_matches_io_roundtrip(tmp_path):
    grid = make_grid()
    path = str(tmp_path / "g.xuvdb")
    xuvdb.save(path, [grid])
    back = xuvdb.load(path)[0]
    for leaf, ref in zip(back.leaves(), [brute_leaf_bbox(l) for l in grid.leaves()]):
        assert_bbox_equal(leaf.active_bbox(), ref)


RAY_SETS = [
    ((0.3, 0.2, 2.0), (0, 0, -1)),      # front hit, through both leaves
    ((0.3, 0.45, 2.0), (0, 0, -1)),     # top-of-sphere grazing hit
    ((0.1, 0.1, -1.0), (0, 0, 1)),      # from below
    ((2.0, 2.0, 2.0), (-1, -1, -1)),    # diagonal that misses
    ((0.3, 0.2, 1.0), (0.5, 0.2, -1)),  # off-center diagonal
]


@pytest.mark.parametrize("origin,direction", RAY_SETS)
def test_bbox_dda_matches_reference_walk(origin, direction):
    grid = make_grid()
    fast = xuvdb.ray_surface_hit(grid, origin, direction, use_bbox=True)
    ref = xuvdb.ray_surface_hit(grid, origin, direction, use_bbox=False)
    if ref is None:
        assert fast is None
    else:
        assert fast is not None
        assert abs(fast[0] - ref[0]) < 1e-9
        assert np.allclose(fast[1], ref[1], atol=1e-9)


def test_batch_query_apis():
    grid = make_grid()
    idxs = grid.active_indices()
    vals = grid.active_values()
    assert len(idxs) == len(vals) == grid.active_voxel_count
    # every reported voxel probes active with the matching value; iter_voxels agrees
    vals_b, act_b = grid.probe_batch(idxs)
    assert act_b.all()
    assert np.allclose(np.asarray(vals).astype(np.float64).reshape(len(vals), -1),
                       np.asarray(vals_b).astype(np.float64).reshape(len(vals), -1))
    ref = {coord: float(v) for coord, v in grid.iter_voxels()}
    assert len(ref) == len(idxs)
    for (x, y, z), v in zip(idxs, np.asarray(vals).reshape(-1)):
        assert abs(ref[(int(x), int(y), int(z))] - float(v)) < 1e-6
    # unallocated indices read as (background, False)
    far = np.array([[999, 999, 999], [idxs[0][0], idxs[0][1], idxs[0][2]]])
    vals_f, act_f = grid.probe_batch(far)
    assert act_f.tolist() == [False, True]
    assert float(np.asarray(vals_f[0]).reshape(-1)[0]) == pytest.approx(grid.background)


def test_kernel_ray_matches_host():
    grid = make_grid()
    vol = GpuVolume(grid)
    for origin, direction in RAY_SETS:
        h = xuvdb.ray_surface_hit(grid, origin, direction)
        k = vol.ray_surface_hit(origin, direction)
        if h is None:
            assert k is None
        else:
            assert k is not None
            assert abs(k[0] - h[0]) < 1e-5, (origin, direction, k[0], h[0])
            assert np.allclose(k[1], h[1], atol=1e-4)
            assert abs(float(k[2]) - float(h[2])) < 1e-5
    # tmax truncation behaves the same on both sides
    h = xuvdb.ray_surface_hit(grid, (0.3, 0.2, 2.0), (0, 0, -1), tmax=1.0)
    k = vol.ray_surface_hit((0.3, 0.2, 2.0), (0, 0, -1), tmax=1.0)
    assert h is None and k is None
    # non-zero isovalue
    h = xuvdb.ray_surface_hit(grid, (0.3, 0.2, 2.0), (0, 0, -1), isovalue=0.1)
    k = vol.ray_surface_hit((0.3, 0.2, 2.0), (0, 0, -1), isovalue=0.1)
    assert (h is None) == (k is None)
    if h is not None:
        assert abs(k[0] - h[0]) < 1e-4
