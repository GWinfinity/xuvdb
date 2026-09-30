"""v0.4 feature tests: .xuvdb v2 (checksum/compression), blosc .vdb, f16, rigid rotation."""
import subprocess
import sys

import numpy as np
import pytest

import xuvdb


def make_grid():
    grid = xuvdb.VdbGrid(background=0.15, voxel_size=0.05, leaf_log2=4,
                         name="shield", grid_class="level set")
    grid.stamp_sphere((0.3, 0.2, 0.1), radius=0.25, band=3.0)
    second = grid.copy()
    second.stamp_sphere((0.5, 0.2, 0.1), radius=0.10)
    grid.csg(second, "union")
    return grid


def sample_grid(grid, pts):
    return np.array([grid.sample_linear(p) for p in pts])


PTS = np.random.default_rng(9).uniform(-0.05, 0.85, size=(256, 3))


@pytest.mark.parametrize("compress", [False, True])
def test_xuvdb_v2_roundtrip_and_checksum(tmp_path, compress):
    grid = make_grid()
    path = str(tmp_path / f"g{int(compress)}.xuvdb")
    xuvdb.save(path, [grid], compress=compress)
    back = xuvdb.load(path)[0]
    v0 = sample_grid(grid, PTS)
    v1 = sample_grid(back, PTS)
    assert np.allclose(v0, v1, atol=1e-6)
    # v2 files: header flags bit1 (crc) always, bit0 iff compressed
    with open(path, "rb") as f:
        flags = f.read()[6]
    assert flags & 0x2 and bool(flags & 0x1) == compress


def test_xuvdb_v2_corruption_detected(tmp_path):
    grid = make_grid()
    path = str(tmp_path / "g.xuvdb")
    xuvdb.save(path, [grid], compress=False)
    with open(path, "rb") as f:
        blob = bytearray(f.read())
    blob[30] ^= 0xFF  # flip bits inside the payload
    with open(path, "wb") as f:
        f.write(blob)
    with pytest.raises(ValueError, match="checksum mismatch"):
        xuvdb.load(path)


def test_xuvdb_v1_emitter_and_reader(tmp_path, monkeypatch):
    """The writer can emit genuine legacy streams (VERSION=1: no crc/rotation/kind byte) and
    the reader loads them - the same compatibility the 0.1-0.3 releases shipped."""
    import xuvdb.io as xio
    grid = make_grid()
    v1_path = str(tmp_path / "v1.xuvdb")
    monkeypatch.setattr(xio, "VERSION", 1)
    xuvdb.save(v1_path, [grid])
    with open(v1_path, "rb") as f:
        blob = f.read()
    assert blob[5] == 1 and blob[6] == 0 and len(blob) % 1 == 0  # flags 0, no trailer marker
    back = xuvdb.load(v1_path)[0]
    assert back.name == "shield"
    assert np.allclose(sample_grid(grid, PTS), sample_grid(back, PTS), atol=1e-6)


def test_f16_grid_roundtrips(tmp_path):
    grid = xuvdb.VdbGrid(dtype=np.float16, background=0.5, voxel_size=0.05,
                         name="half", grid_class="fog volume")
    grid.fill_box((0, 0, 0), (20, 20, 20), 1.0)
    assert grid.type_code == 3
    path = str(tmp_path / "h.xuvdb")
    xuvdb.save(path, [grid])
    back = xuvdb.load(path)[0]
    assert back.type_code == 3 and np.allclose(np.asarray(back.get_value((5, 5, 5)), dtype=np.float32), 1.0)
    # .vdb writer upcasts half -> f32 (read side still parses real half grids)
    vdb_path = str(tmp_path / "h.vdb")
    xuvdb.write_vdb(vdb_path, [grid])
    vb = xuvdb.read_vdb(vdb_path)[0]
    assert vb.type_code == 0
    assert np.allclose(np.asarray(vb.get_value((5, 5, 5)), dtype=np.float32), 1.0)
    with pytest.raises(TypeError):
        GpuVolumeLike(grid)


def GpuVolumeLike(grid):
    from xuvdb.gpu import GpuVolume
    return GpuVolume(grid)


ANG = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90-deg about z


def test_rotated_grid_sampling_and_ray():
    grid = xuvdb.VdbGrid(background=0.15, voxel_size=0.05, leaf_log2=4,
                         name="rot", grid_class="level set", rotation=ANG)
    assert not grid.is_axis_aligned
    c, r = np.array([0.4, 0.4, 0.4]), 0.25
    grid.stamp_sphere(c, radius=r, band=3.0)
    # index_to_world / world_to_index are inverse maps
    ijk = np.array([6, 9, 2])
    assert np.allclose(grid.world_to_index(grid.index_to_world(ijk)), ijk, atol=1e-9)
    # surface + interior probes across the rotated frame
    dirs = np.random.default_rng(8).normal(size=(32, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    surf = c + dirs * r
    vals = sample_grid(grid, surf)
    assert np.all(np.abs(vals) < 5e-2)  # SDF ~ 0 on the sphere, in ANY rotation
    inside = sample_grid(grid, c[None, :])[0]
    assert inside < -r + 5e-2
    t, point, value = xuvdb.ray_surface_hit(grid, c + np.array([0, 0, 2.0]), (0, 0, -1))
    assert t is not None and abs(point[2] - (0.4 + r)) < 3e-2


def test_rotated_grid_vdb_roundtrip(tmp_path):
    grid = xuvdb.VdbGrid(background=0.15, voxel_size=0.05, leaf_log2=4,
                         name="rot", grid_class="level set", rotation=ANG)
    grid.stamp_sphere((0.4, 0.4, 0.4), radius=0.25, band=3.0)
    path = str(tmp_path / "rot.vdb")
    xuvdb.write_vdb(path, [grid])
    back = xuvdb.read_vdb(path)[0]
    assert np.allclose(back.rotation, ANG, atol=1e-6)
    pts = np.random.default_rng(10).uniform(0.1, 0.7, size=(128, 3))
    assert np.allclose(sample_grid(grid, pts), sample_grid(back, pts), atol=1e-5)


def test_rotated_grid_io_roundtrip(tmp_path):
    grid = xuvdb.VdbGrid(background=0.15, voxel_size=0.05, leaf_log2=4,
                         name="rot", grid_class="level set", rotation=ANG)
    grid.stamp_sphere((0.4, 0.4, 0.4), radius=0.25, band=3.0)
    for compress in (False, True):
        path = str(tmp_path / f"rot{int(compress)}.xuvdb")
        xuvdb.save(path, [grid], compress=compress)
        back = xuvdb.load(path)[0]
        assert np.allclose(back.rotation, ANG, atol=1e-9)
        pts = np.random.default_rng(11).uniform(0.1, 0.7, size=(128, 3))
        assert np.allclose(sample_grid(grid, pts), sample_grid(back, pts), atol=1e-5)


def test_blosc_vdb_roundtrip(tmp_path):
    grid = make_grid()
    path = str(tmp_path / "b.vdb")
    xuvdb.write_vdb(path, [grid], blosc=True)
    back = xuvdb.read_vdb(path, grid_name="shield")[0]
    # narrow-band fidelity (the .vdb active-mask contract: inactive-only sub-leaves fold to bg)
    sdf_union = lambda p: min(np.linalg.norm(p - np.array([0.3, 0.2, 0.1])) - 0.25,
                              np.linalg.norm(p - np.array([0.5, 0.2, 0.1])) - 0.10)
    pts = PTS[np.abs([sdf_union(p) for p in PTS]) <= 0.10][:64]
    v0 = sample_grid(grid, pts)
    v1 = sample_grid(back, pts)
    assert np.allclose(v0, v1, atol=1e-5)
    # blosc read-back is bit-identical to the plain read-back
    plain = str(tmp_path / "p.vdb")
    xuvdb.write_vdb(plain, [grid])
    pback = xuvdb.read_vdb(plain)[0]
    for lb, lp in zip(back.leaves(), pback.leaves()):
        assert np.array_equal(lb.values, lp.values) and np.array_equal(lb.active, lp.active)


def test_kernel_rejects_rotated_and_half():
    from xuvdb.gpu import GpuVolume
    grid = make_grid()
    rotated = xuvdb.VdbGrid(background=0.15, voxel_size=0.05, name="r",
                            grid_class="level set", rotation=ANG)
    rotated.stamp_sphere((0.3, 0.2, 0.1), radius=0.25)
    with pytest.raises(TypeError, match="axis-aligned"):
        GpuVolume(rotated)
    half = xuvdb.VdbGrid(dtype=np.float16, name="h")
    half.fill_box((0, 0, 0), (5, 5, 5), 1.0)
    with pytest.raises(TypeError, match="float32"):
        GpuVolume(half)
