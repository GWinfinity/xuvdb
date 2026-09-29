"""v1.0 stability tests: stamp min-union semantics, frozen public API, frozen format version."""
import numpy as np
import pytest

import xuvdb
import xuvdb.io as xio


def leaves_snapshot(grid):
    return [(l.origin.copy(), l.values.copy(), l.active.copy()) for l in grid.leaves()]


def assert_same_grid(a, b, msg):
    la, lb = leaves_snapshot(a), leaves_snapshot(b)
    assert len(la) == len(lb), msg
    for (oa, va, aa), (ob, vb, ab) in zip(la, lb):
        assert np.array_equal(oa, ob), msg
        assert np.array_equal(va, vb), f"{msg}: values differ in leaf {oa}"
        assert np.array_equal(aa, ab), f"{msg}: active differs in leaf {oa}"


def test_double_sdf_stamp_equals_csg_union():
    """1.0 semantics: stamping two SDF spheres on one grid == copy + stamp + csg('union')."""
    def by_two_stamps():
        g = xuvdb.VdbGrid(background=0.15, voxel_size=0.05, leaf_log2=4, name="s",
                          grid_class="level set")
        g.stamp_sphere((0.3, 0.2, 0.1), radius=0.25, band=3.0)
        g.stamp_sphere((0.5, 0.2, 0.1), radius=0.10, band=3.0)
        return g

    def by_csg():
        g = xuvdb.VdbGrid(background=0.15, voxel_size=0.05, leaf_log2=4, name="s",
                          grid_class="level set")
        g.stamp_sphere((0.3, 0.2, 0.1), radius=0.25, band=3.0)
        second = g.copy()
        second.stamp_sphere((0.5, 0.2, 0.1), radius=0.10, band=3.0)
        g.csg(second, "union")
        return g

    assert_same_grid(by_two_stamps(), by_csg(), "two stamps must equal csg union")

    # and the union is a real SDF: surface ~0 on the OUTER surface of each sphere, interior
    # negative on the bridge (sphere 2 pokes out of sphere 1 toward +x)
    g = by_two_stamps()
    assert abs(g.sample_linear((0.3, 0.2, 0.35))) < 2e-2   # sphere-1 far top
    assert abs(g.sample_linear((0.6, 0.2, 0.1))) < 2e-2    # sphere-2 far side, outside sphere 1
    assert g.sample_linear((0.4, 0.2, 0.1)) < 0            # between the centers: union interior
    assert g.sample_linear((0.5, 0.2, 0.2)) < 0            # sphere-2 top sits inside sphere 1


def test_fresh_stamp_keeps_background_outside_band():
    g = xuvdb.VdbGrid(background=0.15, voxel_size=0.05, name="s", grid_class="level set")
    g.stamp_sphere((0.4, 0.4, 0.4), radius=0.1, band=3.0)
    far = (0.0, 0.0, 0.0)
    assert g.sample_linear(far) == pytest.approx(0.15, abs=1e-6)
    assert not g.probe(g.world_to_index(far).astype(int))[1]


def test_fog_stamp_still_overwrites():
    g = xuvdb.VdbGrid(voxel_size=0.05, name="fog", grid_class="fog volume")
    g.stamp_sphere((0.3, 0.3, 0.3), radius=0.2, value=1.0)
    g.stamp_sphere((0.3, 0.3, 0.3), radius=0.1, value=2.0)
    assert g.get_value((6, 6, 6)) == pytest.approx(2.0)  # smaller sphere overwrote


PUBLIC_API = [
    "VdbGrid", "Leaf", "GpuVolume", "VolumeBatch", "save", "load", "write_vdb", "read_vdb",
    "to_openvdb", "from_openvdb", "ray_surface_hit", "init_runtime", "torch_bridge",
]
GRID_METHODS = [
    "set_value", "get_value", "probe", "fill_box", "stamp_sphere", "scatter_particles",
    "union_spheres", "csg", "prune", "to_dense", "from_dense", "bbox", "value_range",
    "active_indices", "active_values", "probe_batch", "stencil7_batch", "stencil19_batch",
    "sample_nearest", "sample_linear", "sample_quadratic", "sample_gradient",
    "index_to_world", "world_to_index", "copy", "iter_voxels", "leaves",
]


def test_public_api_frozen():
    major = int(xuvdb.__version__.split(".")[0])
    assert major == 1  # semver: breaking changes bump the major
    assert xio.VERSION == 2  # .xuvdb format v2 is frozen; readers keep v1 forever
    for name in PUBLIC_API:
        assert hasattr(xuvdb, name), name
    for name in GRID_METHODS:
        assert hasattr(xuvdb.VdbGrid, name), name
    for name in ("sample", "sample_t", "grid_to_tensors", "tensors_to_grid"):
        assert hasattr(xuvdb.torch_bridge, name), name
    for name in ("sample", "sdf_normal", "write_voxels", "sync_to_host", "ray_surface_hit", "reduce"):
        assert hasattr(xuvdb.GpuVolume, name), name
    assert hasattr(xuvdb.VolumeBatch, "sample")
