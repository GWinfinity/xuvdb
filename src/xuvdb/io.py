"""Serialization of XUVDB grids to the native `.xuvdb` binary format.

The extension is deliberately not `.vdb`: Houdini, Blender, Cycles and Arnold all parse `.vdb` as
OpenVDB, and foreign bytes under that suffix fail silently or crash - `save()` refuses such paths
outright. Interop with OpenVDB is an explicit, separate export (`openvdb_file.write_vdb`), which
emits a real OpenVDB stream.

Format (little-endian throughout; `u` = unsigned, `i` = signed, `f` = IEEE float):

    magic    5 bytes "XUVDB"
    version  u8       (currently 1)
    flags    u8       (0)
    n_grids  u16

    per grid:
      name           u32 length + utf-8 bytes
      type_code      u8     (0 = float32, 1 = float64, 2 = vec3 float32)
      leaf_log2      u8
      class_code     u8     (0 unknown, 1 level set, 2 fog volume, 3 staggered)
      reserved       u8
      voxel_size     f64[3]
      origin_world   f64[3]
      background     f32 / f64 / f32[3]  (per type_code)
      n_leaves       u32
      per leaf, sorted by leaf origin (i, then j, then k):
        origin      i32[3]                 (in voxels, of the leaf's (0,0,0) corner)
        mask        u64[dim^3 / 64]        (active bits, z-fastest bit order)
        values      value_type[dim^3]      (full dense leaf, z-fastest order)

The z-fastest voxel order (linear index `n = x*dim^2 + y*dim + z`) matches the OpenVDB leaf layout,
so round trips through `openvdb_file.py` are transpose-free. Like OpenVDB leaf nodes, a leaf's full
dense buffer is stored - inactive values (e.g. a level set's -background interior) survive a save /
load cycle.
"""

import struct

import numpy as np

from .tree import GRID_CLASSES, VdbGrid

MAGIC = b"XUVDB"
VERSION = 1

_TYPE_BY_CODE = {0: np.float32, 1: np.float64}
_CODE_BY_TYPE = {np.dtype(np.float32): 0, np.dtype(np.float64): 1}


def _pack_str(s):
    b = s.encode("utf-8")
    return struct.pack("<I", len(b)) + b


def _mask_words(active):
    """Pack a bool (dim,dim,dim) mask into u64 words, bit n = voxel n (z-fastest C order)."""
    bits = np.packbits(active.reshape(-1), bitorder="little")
    return np.pad(bits, (0, (-len(bits)) % 8)).view(np.uint64)


def _mask_from_words(words, count):
    bits = words.view(np.uint8)
    active = np.unpackbits(bits[: (count + 7) // 8], count=count, bitorder="little")
    return active.astype(bool)


def save(path, grids):
    """Write one or more grids to a `.xuvdb` file."""
    path = str(path)
    if path.lower().endswith(".vdb"):
        raise ValueError(
            f"refusing to write the XUVDB native format to {path!r}: every major volume tool "
            "(Houdini, Blender, Cycles, Arnold) parses '.vdb' as OpenVDB and fails opaquely on "
            "foreign bytes. Use the '.xuvdb' extension, or export a real OpenVDB stream with "
            "write_vdb()."
        )
    grids = list(grids)
    buf = bytearray()
    buf += MAGIC
    buf += struct.pack("<BBI", VERSION, 0, len(grids))
    for grid in grids:
        buf += _pack_str(grid.name)
        bg = np.asarray(grid.background, dtype=np.float64).reshape(-1)
        buf += struct.pack("<BBBB", grid.type_code, grid.leaf_log2, GRID_CLASSES.index(grid.grid_class), 0)
        buf += grid.voxel_size.astype("<f8").tobytes()
        buf += grid.origin_world.astype("<f8").tobytes()
        if grid.type_code == 0:
            buf += struct.pack("<f", float(bg[0]))
        elif grid.type_code == 1:
            buf += struct.pack("<d", float(bg[0]))
        else:
            buf += bg.astype("<f4").tobytes()
        leaves = grid.leaves()
        buf += struct.pack("<I", len(leaves))
        for leaf in leaves:
            buf += leaf.origin.astype("<i4").tobytes()
            mask = _mask_words(leaf.active)
            buf += mask.astype("<u8").tobytes()
            vals = leaf.values.reshape(-1)  # z-fastest C order
            if grid.type_code == 1:
                buf += vals.astype("<f8").tobytes()
            else:
                buf += vals.astype("<f4").tobytes()
    with open(path, "wb") as f:
        f.write(buf)
    return len(buf)


class _Reader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def read(self, n):
        if self.pos + n > len(self.data):
            raise ValueError("truncated .xuvdb stream")
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out


def load(path):
    """Read all grids from a `.xuvdb` file."""
    with open(path, "rb") as f:
        data = f.read()
    r = _Reader(data)
    if r.read(len(MAGIC)) != MAGIC:
        raise ValueError(f"{path} is not an XUVDB file")
    version, flags, n_grids = struct.unpack("<BBI", r.read(6))
    if version != VERSION:
        raise ValueError(f"unsupported XUVDB version {version}")
    if flags != 0:
        raise ValueError(f"unsupported XUVDB flags {flags}")
    grids = []
    for _ in range(n_grids):
        (name_len,) = struct.unpack("<I", r.read(4))
        name = r.read(name_len).decode("utf-8")
        type_code, leaf_log2, class_code, _res = struct.unpack("<BBBB", r.read(4))
        voxel_size = np.frombuffer(r.read(24), dtype="<f8")
        origin_world = np.frombuffer(r.read(24), dtype="<f8")
        if type_code == 0:
            (background,) = struct.unpack("<f", r.read(4))
        elif type_code == 1:
            (background,) = struct.unpack("<d", r.read(8))
        else:
            background = np.frombuffer(r.read(12), dtype="<f4").copy()
        grid = VdbGrid(
            dtype=_TYPE_BY_CODE.get(type_code, None) if type_code != 2
            else np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32)]),
            background=background,
            voxel_size=voxel_size,
            origin_world=origin_world,
            leaf_log2=leaf_log2,
            name=name,
            grid_class=GRID_CLASSES[class_code],
        )
        (n_leaves,) = struct.unpack("<I", r.read(4))
        dim = 1 << leaf_log2
        words = dim**3 // 64
        per_voxel = 3 if type_code == 2 else 1
        item = np.dtype("<f4") if type_code != 1 else np.dtype("<f8")
        for _leaf in range(n_leaves):
            origin = np.frombuffer(r.read(12), dtype="<i4").astype(np.int64)
            mask_words = np.frombuffer(r.read(words * 8), dtype="<u8")
            active = _mask_from_words(mask_words, dim**3)
            values_flat = np.frombuffer(r.read(dim**3 * per_voxel * item.itemsize), dtype=item)
            leaf = grid._get_or_create_leaf((origin[0] >> leaf_log2, origin[1] >> leaf_log2, origin[2] >> leaf_log2))
            if grid.is_vec:
                leaf.values[...] = values_flat.reshape(dim, dim, dim, 3)
            else:
                leaf.values[...] = values_flat.reshape(dim, dim, dim)
            leaf.active[...] = active.reshape(dim, dim, dim)
            leaf.invalidate()
        grids.append(grid)
    return grids
