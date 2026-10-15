"""Chunked-slab tests: forced-small chunks must match the monolithic operators."""
import numpy as np
import pytest

import xuvdb
from xuvdb.offset import offset
from xuvdb.smooth import smooth
from xuvdb.slab import tile_ranges


def make_sphere(radius=0.25, voxel=0.02):
    grid = xuvdb.VdbGrid(background=0.2, voxel_size=voxel, leaf_log2=4,
                         name="ls", grid_class="level set")
    grid.stamp_sphere((0.0, 0.0, 0.0), radius=radius, band=3.0)
    return grid


def samples(grid, n=200, seed=1):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.45, 0.45, size=(n, 3))
    return pts, np.array([grid.sample_linear(p) for p in pts])


def test_tile_ranges_budget_and_min_width():
    lo, hi = np.array([0, 0, 0]), np.array([999, 99, 49])
    chunks = tile_ranges(lo, hi, 60_000, min_width=1)
    total = 0
    for a, b in chunks:
        vol = int(np.prod(b - a + 1))
        total += vol
        assert vol <= 60_000
    # contiguous, covers the whole range
    assert len(chunks) > 1
    assert np.array_equal(chunks[0][0], lo) and np.array_equal(chunks[-1][1], hi)
    # min_width forces fewer, wider chunks
    wide = tile_ranges(lo, hi, 60_000, min_width=200)
    assert len(wide) < len(chunks)
    assert all(int(b[0] - a[0] + 1) >= 199 for a, b in wide)


def test_tile_ranges_single_chunk_when_it_fits():
    out = tile_ranges(np.zeros(3, np.int64), np.full(3, 31, np.int64), 64_000_000)
    assert len(out) == 1


def test_smooth_chunked_matches_monolithic():
    g_ref = make_sphere()
    g_chk = make_sphere()
    smooth(g_ref, iterations=6, alpha=0.7)
    smooth(g_chk, iterations=6, alpha=0.7, max_slab_voxels=1_000_000)
    _, v_ref = samples(g_ref)
    _, v_chk = samples(g_chk)
    # lockstep Jacobi: every voxel reads previous-iteration neighbours
    assert np.allclose(v_ref, v_chk, atol=1e-6)


def test_offset_chunked_matches_monolithic():
    g_ref = make_sphere()
    g_chk = make_sphere()
    offset(g_ref, 0.1)
    offset(g_chk, 0.1, max_slab_voxels=1_000_000)
    pts, v_ref = samples(g_ref)
    _, v_chk = samples(g_chk, seed=1)
    assert np.allclose(v_ref, v_chk, atol=2e-3)


def test_offset_chunked_erosion_too():
    g_ref = make_sphere()
    g_chk = make_sphere()
    offset(g_ref, -0.1)
    offset(g_chk, -0.1, max_slab_voxels=1_000_000)
    _, v_ref = samples(g_ref)
    _, v_chk = samples(g_chk)
    assert np.allclose(v_ref, v_chk, atol=2e-3)


def test_max_slab_voxels_validation():
    g = make_sphere()
    with pytest.raises(ValueError):
        offset(g, 0.05, max_slab_voxels=100)
    with pytest.raises(ValueError):
        smooth(g, max_slab_voxels=100)
