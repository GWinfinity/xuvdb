"""Smoke tests for the packaged `xuvdb` (standalone import)."""
import numpy as np

import xuvdb

C1, R1 = np.array([0.3, 0.2, 0.1]), 0.25
C2, R2 = np.array([0.5, 0.2, 0.1]), 0.10


def make_union_grid():
    """Narrow-band level set of two spheres unioned via csg (equivalent to two stamps since 1.0.0)."""
    grid = xuvdb.VdbGrid(background=3 * 0.05, voxel_size=0.05, leaf_log2=4,
                         name="shield", grid_class="level set")
    grid.stamp_sphere(C1, radius=R1, band=3.0)
    second = grid.copy()
    second.stamp_sphere(C2, radius=R2)
    grid.csg(second, "union")
    grid.prune()
    return grid


sdf_union = lambda p: min(np.linalg.norm(p - C1) - R1, np.linalg.norm(p - C2) - R2)


def test_native_roundtrip_lossless(tmp_path):
    grid = make_union_grid()
    path = str(tmp_path / "scene.xuvdb")
    xuvdb.save(path, [grid])
    back = xuvdb.load(path)[0]
    pts = np.random.default_rng(0).uniform(-0.1, 0.9, size=(256, 3))
    v0 = np.array([grid.sample_linear(p) for p in pts])
    v1 = np.array([back.sample_linear(p) for p in pts])
    assert np.allclose(v0, v1, atol=1e-6)


def test_vdb_roundtrip_band_fidelity(tmp_path):
    grid = make_union_grid()
    path = str(tmp_path / "scene.vdb")
    xuvdb.write_vdb(path, [grid])
    with open(path, "rb") as f:
        assert f.read(8) == b"\x20\x42\x44\x56\x00\x00\x00\x00"
    vb = xuvdb.read_vdb(path, grid_name="shield")[0]
    pts = np.random.default_rng(1).uniform(-0.1, 0.9, size=(512, 3))
    band = pts[np.abs([sdf_union(p) for p in pts]) <= 0.10][:32]
    v0 = np.array([grid.sample_linear(p) for p in band])
    v1 = np.array([vb.sample_linear(p) for p in band])
    assert np.allclose(v0, v1, atol=1e-5)


def test_kernel_sample_matches_host_and_write_roundtrips():
    grid = make_union_grid()
    vol = xuvdb.GpuVolume(grid)
    pts = np.array([[0.3, 0.45, 0.1], [0.3, 0.2, 0.1]], np.float32)
    assert abs(vol.sample(pts[:1], linear=True)[0]) < 2e-2          # on surface
    assert abs(vol.sample(pts[1:], linear=True)[0] + 0.25) < 2e-2   # deep inside
    n = vol.sdf_normal(pts[:1])[0]
    assert np.allclose(n, [0, 1, 0], atol=5e-2)                     # sphere top -> +y
    # kernel write must reach the host tree through sync_to_host
    vol.write_voxels(pts[1:], np.array([-0.05], np.float32)).sync_to_host()
    assert abs(grid.sample_linear(pts[1]) + 0.05) < 1e-6


def test_scatter_particles_mass_conserved():
    drops = np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    fog = xuvdb.VdbGrid(voxel_size=0.05, name="liquid", grid_class="fog volume")
    fog.scatter_particles(drops, h=4 * 0.05, weights=1.0)
    mass = sum(leaf.values.sum() for leaf in fog.leaves()) * 0.05**3
    assert abs(mass - 2.0) < 1e-4  # unit-integral kernel, discretization error only


def test_ray_surface_hit():
    grid = make_union_grid()
    t, point, value = xuvdb.ray_surface_hit(grid, (0.3, 0.2, 2.0), (0, 0, -1))
    assert t is not None
    assert abs(point[2] - (0.1 + 0.25)) < 3e-2  # sphere-1 front surface
    assert abs(float(value)) < 5e-2
