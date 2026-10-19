"""Regression tests for the security-audit findings (H-1..H-3, M-1..M-3).

Every malformed input is crafted byte-level per the published wire formats - the same
construction method as the audit's PoC scripts. Acceptance follows the audit's re-test
table: rejection happens at parse time, before materialization.
"""

import struct
import zlib

import numpy as np
import pytest
import xuvdb
from xuvdb import io as xio
from xuvdb import openvdb_file as ovf


# ------------------------------------------------------------------ .xuvdb crafting helpers


def _pack_str(s):
    b = s.encode("utf-8")
    return struct.pack("<I", len(b)) + b


def _wrap_xuvdb(payload, compress=False):
    flags = 0x02 | (0x01 if compress else 0)
    buf = b"XUVDB" + struct.pack("<BBI", 3, flags, 1) + payload
    if compress:
        buf = buf[:11] + zlib.compress(payload)
    return buf + struct.pack("<I", zlib.crc32(payload) & 0xFFFFFFFF)


def _grid_payload(name="g", type_code=0, leaf_log2=4, class_code=1,
                  voxel=(0.05, 0.05, 0.05), origin=(0.0, 0.0, 0.0), bg=1.0, n_leaves=0,
                  leaf_bytes=b""):
    p = _pack_str(name)
    p += struct.pack("<BBBB", type_code, leaf_log2, class_code, 0)
    p += struct.pack("<3d", *voxel) + struct.pack("<3d", *origin)
    p += struct.pack("<f", bg) if type_code == 0 else struct.pack("<d", bg)
    p += struct.pack("<I", n_leaves)
    return p + leaf_bytes


def _constant_leaf_records(n, leaf_log2, value=1.0):
    """n minimal v3 constant-leaf records (origin + kind=1 + one f32)."""
    return b"".join(
        struct.pack("<3iB", 0, 0, i << leaf_log2, 1) + struct.pack("<f", value) for i in range(n)
    )


# ------------------------------------------------------------------ .vdb crafting helpers


def _vdb_str(s):
    b = s.encode("utf-8")
    return struct.pack("<I", len(b)) + b


def _vdb_meta_string(key, value):
    payload = value.encode("utf-8")
    return _vdb_str(key) + _vdb_str("string") + struct.pack("<I", len(payload)) + payload


def _read_vdb_bytes(data):
    import os
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".vdb")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return ovf.read_vdb(path)
    finally:
        os.unlink(path)


def _craft_tile_vdb(meta_name="g", n_tiles=0, n_int5=0, value_bits=0, int5_origin=(0, 0, 0)):
    """A minimal but spec-shaped .vdb: one grid, UniformScaleMap, no leaves, optional
    int5 nodes whose value-mask bits declare `value_bits` active 128^3 root tiles."""
    b = bytearray()
    b += struct.pack("<q", ovf.OPENVDB_MAGIC)
    b += struct.pack("<I", 224) + struct.pack("<II", 11, 0) + b"\x01" + b"U" * 36
    b += struct.pack("<I", 0)              # file-level metadata
    b += struct.pack("<i", 1)              # grid count
    b += _vdb_str("g") + _vdb_str("Tree_float_5_4_3") + _vdb_str("")
    off = len(b)
    b += b"\x00" * 24                      # grid/block/end offsets, patched below
    grid_pos = len(b)
    b += struct.pack("<I", 0)              # compression = none
    b += struct.pack("<I", 1) + _vdb_meta_string("name", meta_name)
    b += _vdb_str("UniformScaleMap")
    b += struct.pack("<3d", 0.05, 0.05, 0.05)
    for _ in range(4):
        b += struct.pack("<3d", 0.0, 0.0, 0.0)
    b += struct.pack("<i", 1)              # buffer count
    b += struct.pack("<f", 1.0)            # background
    b += struct.pack("<I", n_tiles) + struct.pack("<I", n_int5)
    for _ in range(n_int5):
        b += struct.pack("<3i", *int5_origin)
        b += b"\x00" * (512 * 8)           # child mask (32768 bits)
        vm = bytearray(512 * 8)
        for i in range(value_bits):
            vm[i // 8] |= 1 << (i % 8)
        b += bytes(vm)                     # value mask
        b += b"\x06"                       # _NO_MASK_AND_ALL_VALS
        b += struct.pack("<32768f", *([1.0] * 32768))
    end_pos = len(b)
    struct.pack_into("<qqq", b, off, grid_pos, grid_pos, end_pos)
    return bytes(b)


# ------------------------------------------------------------------ H-1 object amplification


def test_forged_n_leaves_rejected(tmp_path):
    f = tmp_path / "forged.xuvdb"
    f.write_bytes(_wrap_xuvdb(_grid_payload(n_leaves=10_000_000)))
    with pytest.raises(ValueError, match="n_leaves"):
        xuvdb.load(str(f))
    with pytest.raises(ValueError, match="n_leaves"):
        list(xio.iter_leaves(str(f)))


def test_leaf_materialization_budget(tmp_path):
    # byte-count check passes (the records are physically present), but 100 x 64^3 =
    # 26.2M voxels exceeds the 2^24 materialization budget
    f = tmp_path / "big.xuvdb"
    f.write_bytes(_wrap_xuvdb(_grid_payload(leaf_log2=6, n_leaves=100,
                                            leaf_bytes=_constant_leaf_records(100, 6))))
    with pytest.raises(ValueError, match="budget"):
        xuvdb.load(str(f))


def test_forged_n_tiles_rejected():
    with pytest.raises(ValueError, match="n_tiles"):
        _read_vdb_bytes(_craft_tile_vdb(n_tiles=10_000_000))


def test_tile_materialization_budget():
    # the budget fills exactly 8 tiles of 2^21 voxels (8 x 2^21 = 2^24); the 9th trips the guard
    with pytest.warns(UserWarning, match="budget"):
        grids = _read_vdb_bytes(_craft_tile_vdb(n_int5=1, value_bits=64))
    assert grids[0].n_leaves == 8 * 4096


def test_far_tile_origin_rejected():
    with pytest.raises(ValueError, match="kernel-key"):
        _read_vdb_bytes(_craft_tile_vdb(n_int5=1, value_bits=1, int5_origin=(0, 0, 1 << 28)))


# ------------------------------------------------------------------ H-2 decompression bombs


def test_zlib_bomb_xuvdb_rejected(tmp_path):
    # 320 MiB of zeros compresses to ~300 KB with a fully VALID CRC; the bounded inflater
    # must refuse before the expansion is allocated (the CRC no longer comes first).
    bomb = b"\x00" * (320 << 20)
    f = tmp_path / "bomb.xuvdb"
    f.write_bytes(_wrap_xuvdb(bomb, compress=True))
    with pytest.raises(ValueError, match="budget"):
        xuvdb.load(str(f))


def test_zip_block_length_checked():
    payload = zlib.compress(b"\x00" * 4096)
    s = ovf._ReadStream(struct.pack("<q", len(payload)) + payload)
    with pytest.raises(ValueError):
        ovf._read_data_blob(s, 2048, ovf.COMPRESS_ZIP)
    s = ovf._ReadStream(struct.pack("<q", len(payload)) + payload)
    assert len(ovf._read_data_blob(s, 4096, ovf.COMPRESS_ZIP)) == 4096


# ------------------------------------------------------------------ H-3 key space


def test_far_leaf_origin_rejected_by_load(tmp_path):
    g = xuvdb.VdbGrid(background=1.0)
    g.set_value((0, 0, 1 << 24), 42.0)  # leaf key (0,0,2^20) - beyond the +-2^19 domain
    f = tmp_path / "far.xuvdb"
    xuvdb.save(str(f), [g])
    with pytest.raises(ValueError, match="kernel-key"):
        xuvdb.load(str(f))


def test_pack_key_range_check():
    g = xuvdb.VdbGrid(background=1.0)
    g.set_value((0, 0, 1 << 24), 42.0)
    with pytest.raises(ValueError, match="packed-key"):
        xuvdb.GpuVolume(g)


def test_kernel_host_consistency_far_coordinates():
    g = xuvdb.VdbGrid(background=0.15, voxel_size=0.05, grid_class="level set")
    g.stamp_sphere((0.0, 0.0, 0.0), radius=0.2)
    vol = xuvdb.GpuVolume(g)
    pts = np.array([[0.0, 0.0, 0.25],             # inside the sphere
                    [0.0, 12.8, 0.0],             # in-range key, no leaf there
                    [0.0, 0.0, 2.0e7]], np.float32)  # far outside the key domain
    kernel = vol.sample(pts, linear=False)
    host = np.array([g.sample_nearest(p) for p in pts])
    np.testing.assert_allclose(kernel, host, atol=1e-6)


# ------------------------------------------------------------------ M-1 non-finite parameters


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_voxel_size_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match="finite"):
        xuvdb.VdbGrid(voxel_size=(bad, 0.05, 0.05))
    f = tmp_path / "nan.xuvdb"
    f.write_bytes(_wrap_xuvdb(_grid_payload(voxel=(bad, 0.05, 0.05))))
    with pytest.raises(ValueError, match="finite"):
        xuvdb.load(str(f))


# ------------------------------------------------------------------ M-2 wire enums whitelist


def test_unknown_type_code_rejected(tmp_path):
    f = tmp_path / "badtype.xuvdb"
    f.write_bytes(_wrap_xuvdb(_grid_payload(type_code=5)))
    with pytest.raises(ValueError, match="type code"):
        xuvdb.load(str(f))
    with pytest.raises(ValueError, match="type code"):
        list(xio.iter_leaves(str(f)))


def test_unknown_class_code_rejected(tmp_path):
    f = tmp_path / "badclass.xuvdb"
    f.write_bytes(_wrap_xuvdb(_grid_payload(class_code=9)))
    with pytest.raises(ValueError, match="class code"):
        xuvdb.load(str(f))
    with pytest.raises(ValueError, match="class code"):
        list(xio.iter_leaves(str(f)))


# ------------------------------------------------------------------ M-3 grid name confusion


def test_descriptor_metadata_name_mismatch_rejected():
    with pytest.raises(ValueError, match="name mismatch"):
        _read_vdb_bytes(_craft_tile_vdb(meta_name="B"))


def test_write_vdb_rejects_separator_in_name(tmp_path):
    g = xuvdb.VdbGrid(name="A\x1eB")
    with pytest.raises(ValueError, match="U\\+001E"):
        xuvdb.write_vdb(str(tmp_path / "x.vdb"), [g])
