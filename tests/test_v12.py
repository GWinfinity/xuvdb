"""v1.2 feature tests: constant-leaf encoding, compress(), streaming primitives."""
import numpy as np
import pytest

import xuvdb
from xuvdb.io import iter_leaves, iter_slabs


def make_dusty_grid():
    """A grid whose far-field leaves carry inactive junk (exact distances from stamping)."""
    grid = xuvdb.VdbGrid(background=0.15, voxel_size=0.05, leaf_log2=4,
                         name="g", grid_class="level set")
    grid.stamp_sphere((0.3, 0.2, 0.1), radius=0.25, band=3.0)
    grid.stamp_sphere((0.5, 0.2, 0.1), radius=0.10, band=3.0)
    return grid


def test_constant_leaf_encoding_shrinks_files(tmp_path):
    """Uniform leaves collapse to one value on disk: the tile-equivalent win."""
    grid = xuvdb.VdbGrid(background=0.15, voxel_size=0.05, name="fog", grid_class="fog volume")
    grid.fill_box((0, 0, 0), (31, 15, 15), 2.5)  # fills whole 16^3 leaves uniformly
    dense_path = str(tmp_path / "dense.xuvdb")
    for leaf in grid.leaves():
        if leaf.active.all() and np.all(leaf.values == leaf.values.flat[0]):
            leaf.values[0, 0, 0] = leaf.values.flat[0] + 1e-3  # break constantness
    xuvdb.save(dense_path, [grid])
    dense2 = xuvdb.VdbGrid(background=0.15, voxel_size=0.05, name="fog", grid_class="fog volume")
    dense2.fill_box((0, 0, 0), (31, 15, 15), 2.5)  # stays constant
    const_path = str(tmp_path / "const.xuvdb")
    xuvdb.save(const_path, [dense2])
    sz_dense = len(open(dense_path, "rb").read())
    sz_const = len(open(const_path, "rb").read())
    assert sz_const < sz_dense / 10  # whole blocks -> one value each
    back = xuvdb.load(const_path)[0]
    assert back.get_value((5, 5, 5)) == pytest.approx(2.5)
    assert back.probe((5, 5, 5))[1] is True or back.probe((5, 5, 5))[1] == np.True_


def test_compress_drops_far_field_junk_but_keeps_surface():
    grid = make_dusty_grid()
    before_leaves = grid.n_leaves
    surf = np.array([0.3, 0.45, 0.1])  # on sphere-1, an ACTIVE voxel
    before = grid.sample_linear(surf)
    dropped, reset = grid.compress()
    assert dropped > 0 and reset > 0
    assert grid.n_leaves < before_leaves
    # ACTIVE-voxel values survive compress exactly; far-field inactive junk is the trade-off
    after = grid.sample_linear(surf)
    assert abs(after - before) < 1e-6
    assert abs(float(after)) < 2e-2


def test_compress_roundtrips_through_formats(tmp_path):
    grid = make_dusty_grid()
    grid.compress()
    for name in ("a.xuvdb",):
        path = str(tmp_path / name)
        xuvdb.save(path, [grid], compress=True)
        back = xuvdb.load(path)[0]
        pts = np.random.default_rng(1).uniform(0.1, 0.55, size=(128, 3))
        a = np.array([grid.sample_linear(p) for p in pts])
        b = np.array([back.sample_linear(p) for p in pts])
        assert np.allclose(a, b, atol=1e-6)


def test_iter_leaves_and_slabs_match_tree(tmp_path):
    grid = make_dusty_grid()
    grid.fill_box((-1, -1, -1), (25, 25, 25), 0.05)  # add a uniform region too
    path = str(tmp_path / "s.xuvdb")
    xuvdb.save(path, [grid], compress=False)  # streaming walks the raw payload
    leaves = list(iter_leaves(path))
    assert len(leaves) == grid.n_leaves
    back = xuvdb.VdbGrid(background=grid.background, voxel_size=grid.voxel_size,
                         leaf_log2=grid.leaf_log2, name="g", grid_class=grid.grid_class)
    for origin, values, active in leaves:
        leaf = back._get_or_create_leaf(tuple(origin >> grid.leaf_log2))
        leaf.values[...] = values.reshape([grid.leaf_dim] * 3)
        leaf.active[...] = active.reshape([grid.leaf_dim] * 3)
    pts = np.random.default_rng(2).uniform(0.0, 1.0, size=(256, 3))
    a = np.array([grid.sample_linear(p) for p in pts])
    b = np.array([back.sample_linear(p) for p in pts])
    assert np.allclose(a, b, atol=1e-6)
    # slabs reassemble into the full field; sampled everywhere == the grid
    chunks = [(ijk_min.copy(), slab, slab_act)
              for ijk_min, slab, slab_act in iter_slabs(path, axis=0, slab_leaves=2)]
    lo = np.minimum.reduce([c[0] for c in chunks])
    hi = np.maximum.reduce([c[0] + np.array(c[1].shape) for c in chunks])
    full = np.full(tuple(hi - lo) + tuple(grid.value_shape), grid.background, dtype=np.float32)
    full_act = np.zeros(tuple(hi - lo), dtype=bool)
    for ijk_min, slab, slab_act in chunks:
        dst = tuple(slice(int(ijk_min[a] - lo[a]), int(ijk_min[a] - lo[a]) + slab.shape[a])
                    for a in range(3))
        full[dst] = slab
        full_act[dst] = slab_act
    pts2 = np.random.default_rng(3).uniform(0, 1, size=(256, 3))
    idx = np.clip(np.rint(pts2 / 0.05).astype(int), 0, np.array(full.shape[:3]) - 1)
    for n in range(256):
        i, j, k = idx[n]
        gv = float(back.get_value((int(i + lo[0]), int(j + lo[1]), int(k + lo[2]))))
        assert abs(float(full[i, j, k]) - gv) < 1e-6
        _, act = back.probe((int(i + lo[0]), int(j + lo[1]), int(k + lo[2])))
        assert bool(full_act[i, j, k]) == bool(act)
