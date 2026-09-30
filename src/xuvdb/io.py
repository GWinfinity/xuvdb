"""Serialization of XUVDB grids to the native `.xuvdb` binary format.

The extension is deliberately not `.vdb`: Houdini, Blender, Cycles and Arnold all parse `.vdb` as
OpenVDB, and foreign bytes under that suffix fail silently or crash - `save()` refuses such paths
outright. Interop with OpenVDB is an explicit, separate export (`openvdb_file.write_vdb`), which
emits a real OpenVDB stream.

Format (little-endian throughout; `u` = unsigned, `i` = signed, `f` = IEEE float):

    magic    5 bytes "XUVDB"
    version  u8       (3; v1 and v2 files read exactly as before)
    flags    u8       bit0 = payload is zlib-deflated, bit1 = CRC32 trailer present
    n_grids  u32

    payload (optionally deflated when flags bit0):
    per grid:
      name           u32 length + utf-8 bytes
      type_code      u8     (0 = float32, 1 = float64, 2 = vec3 float32, 3 = float16)
      leaf_log2      u8
      class_code     u8     (0 unknown, 1 level set, 2 fog volume, 3 staggered)
      grid_flags     u8     bit0 = rotation matrix (f64[9], row-major R) follows
      reserved       u8
      voxel_size     f64[3]
      origin_world   f64[3]
      [rotation      f64[9] when grid_flags bit0]  (world = R @ (index * voxel_size) + origin)
      background     f32 / f64 / f32[3]  (per type_code)
      n_leaves       u32
      per leaf, sorted by leaf origin (i, then j, then k):
        origin      i32[3]                 (in voxels, of the leaf's (0,0,0) corner)
        kind        u8                     (v3: 0 = dense, 1 = constant; v1/v2: always dense)
        dense:      mask u64[dim³/64] + values value_type[dim³]   (full dense, z-fastest)
        constant:   u8 active + one value  (the OpenVDB-tile leaf equivalent: a whole block
                    collapses to a single value; ~400x smaller on disk for filled regions)

    trailer (flags bit1): u32 CRC32 of the UNCOMPRESSED payload bytes.

`save(path, grids, compress=True)` sets both bits; `save(path, grids, compress=False)` writes
bit1 only (checksummed, uncompressed). Version-1 files (no compression, no trailer, no rotation)
remain readable.

The z-fastest voxel order (linear index `n = x*dim^2 + y*dim + z`) matches the OpenVDB leaf layout,
so round trips through `openvdb_file.py` are transpose-free. Like OpenVDB leaf nodes, a leaf's full
dense buffer is stored - inactive values (e.g. a level set's -background interior) survive a save /
load cycle.
"""

import struct
import zlib

import numpy as np

from .tree import GRID_CLASSES, VdbGrid

MAGIC = b"XUVDB"
VERSION = 3

_FLAG_COMPRESSED = 0x01
_FLAG_CRC = 0x02
_GRID_FLAG_ROTATION = 0x01

_TYPE_BY_CODE = {0: np.float32, 1: np.float64, 3: np.float16}
_CODE_BY_TYPE = {np.dtype(np.float32): 0, np.dtype(np.float64): 1, np.dtype(np.float16): 3}


def _pack_str(s):
    b = s.encode("utf-8")
    return struct.pack("<I", len(b)) + b


def _mask_words(active):
    """Pack a bool (dim,dim,dim) mask into u64 words, bit n = voxel n (z-fastest C order)."""
    bits = np.packbits(active.reshape(-1), bitorder="little")
    return np.pad(bits, (0, (-len(bits)) % 8)).view("<u8")  # explicit LE, matching the format


def _mask_from_words(words, count):
    bits = words.view(np.uint8)
    active = np.unpackbits(bits[: (count + 7) // 8], count=count, bitorder="little")
    return active.astype(bool)


def save(path, grids, compress=False):
    """Write one or more grids to a `.xuvdb` file (format v2: CRC32 checksum, optional zlib).

    `compress=True` zlib-deflates the payload as well; the CRC32 trailer covers the UNCOMPRESSED
    payload bytes either way, so corruption is caught independently of compression.
    """
    path = str(path)
    if path.lower().endswith(".vdb"):
        raise ValueError(
            f"refusing to write the XUVDB native format to {path!r}: every major volume tool "
            "(Houdini, Blender, Cycles, Arnold) parses '.vdb' as OpenVDB and fails opaquely on "
            "foreign bytes. Use the '.xuvdb' extension, or export a real OpenVDB stream with "
            "write_vdb()."
        )
    grids = list(grids)
    legacy_v1 = VERSION < 2  # compat mode: emit a genuine v1 stream (no crc/rotation/kind byte)
    has_kind = VERSION >= 3
    flags = 0 if legacy_v1 else _FLAG_CRC | (_FLAG_COMPRESSED if compress else 0)
    buf = bytearray()
    buf += MAGIC
    buf += struct.pack("<BBI", VERSION, flags, len(grids))
    for grid in grids:
        buf += _pack_str(grid.name)
        bg = np.asarray(grid.background, dtype=np.float64).reshape(-1)
        grid_flags = 0 if (grid.is_axis_aligned or legacy_v1) else _GRID_FLAG_ROTATION
        buf += struct.pack("<BBBB", grid.type_code, grid.leaf_log2, GRID_CLASSES.index(grid.grid_class), grid_flags)
        buf += grid.voxel_size.astype("<f8").tobytes()
        buf += grid.origin_world.astype("<f8").tobytes()
        if grid_flags & _GRID_FLAG_ROTATION:
            buf += grid.rotation.astype("<f8").tobytes()
        if grid.type_code == 0:
            buf += struct.pack("<f", float(bg[0]))
        elif grid.type_code == 1:
            buf += struct.pack("<d", float(bg[0]))
        else:
            buf += bg.astype("<f4").tobytes()
        leaves = grid.leaves()
        buf += struct.pack("<I", len(leaves))
        wire = "<f8" if grid.type_code == 1 else ("<f2" if grid.type_code == 3 else "<f4")
        for leaf in leaves:
            buf += leaf.origin.astype("<i4").tobytes()
            if has_kind and leaf.active.all() and np.all(leaf.values == leaf.values.flat[0]):
                # v3 constant leaf: one active value for the whole block (tile equivalent)
                buf += struct.pack("<B", 1)
                buf += np.asarray(leaf.values.flat[0], dtype=wire).tobytes()
                continue
            if has_kind:
                buf += struct.pack("<B", 0)
            mask = _mask_words(leaf.active)
            buf += mask.astype("<u8").tobytes()
            vals = leaf.values.reshape(-1)  # z-fastest C order
            buf += vals.astype(wire).tobytes()
    if not legacy_v1:
        crc = zlib.crc32(bytes(buf[11:])) & 0xFFFFFFFF  # checksum covers the payload only
        if compress:
            buf = bytes(buf[:11]) + zlib.compress(bytes(buf[11:]))  # header stays plain
        buf += struct.pack("<I", crc)  # trailer covers the UNCOMPRESSED payload
    with open(path, "wb") as f:
        f.write(bytes(buf))
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
    """Read all grids from a `.xuvdb` file (formats v1 and v2)."""
    with open(path, "rb") as f:
        data = f.read()
    r = _Reader(data)
    if r.read(len(MAGIC)) != MAGIC:
        raise ValueError(f"{path} is not an XUVDB file")
    version, flags, n_grids = struct.unpack("<BBI", r.read(6))
    if version not in (1, 2, 3):
        raise ValueError(f"unsupported XUVDB version {version}")
    if version == 1:
        if flags != 0:
            raise ValueError(f"unsupported XUVDB v1 flags {flags}")
    else:
        crc = struct.unpack("<I", data[-4:])[0]
        payload = data[11:-4]
        if flags & _FLAG_COMPRESSED:
            payload = zlib.decompress(payload)
        if flags & _FLAG_CRC and zlib.crc32(payload) & 0xFFFFFFFF != crc:
            raise ValueError(f"{path}: XUVDB payload checksum mismatch (file corrupted)")
        r = _Reader(payload)
    grids = []
    for _ in range(n_grids):
        (name_len,) = struct.unpack("<I", r.read(4))
        name = r.read(name_len).decode("utf-8")
        type_code, leaf_log2, class_code, grid_flags = struct.unpack("<BBBB", r.read(4))
        voxel_size = np.frombuffer(r.read(24), dtype="<f8")
        origin_world = np.frombuffer(r.read(24), dtype="<f8")
        rotation = None
        if version >= 2 and grid_flags & _GRID_FLAG_ROTATION:
            rotation = np.frombuffer(r.read(72), dtype="<f8").reshape(3, 3)
        if type_code == 0:
            (background,) = struct.unpack("<f", r.read(4))
        elif type_code == 1:
            (background,) = struct.unpack("<d", r.read(8))
        else:
            background = np.frombuffer(r.read(12 if type_code == 2 else 4), dtype="<f4").copy()
        grid = VdbGrid(
            dtype=_TYPE_BY_CODE.get(type_code, None) if type_code != 2
            else np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32)]),
            background=background,
            voxel_size=voxel_size,
            origin_world=origin_world,
            leaf_log2=leaf_log2,
            name=name,
            grid_class=GRID_CLASSES[class_code],
            rotation=rotation,
        )
        (n_leaves,) = struct.unpack("<I", r.read(4))
        dim = 1 << leaf_log2
        words = dim**3 // 64
        per_voxel = 3 if type_code == 2 else 1
        item = _TYPE_BY_CODE.get(type_code, np.float32) if type_code != 2 else np.float32
        wire_item = np.dtype("<f8") if type_code == 1 else np.dtype("<f2") if type_code == 3 else np.dtype("<f4")
        for _leaf in range(n_leaves):
            origin = np.frombuffer(r.read(12), dtype="<i4").astype(np.int64)
            if version >= 3:
                kind = r.read(1)[0]
            else:
                kind = 0
            leaf = grid._get_or_create_leaf((origin[0] >> leaf_log2, origin[1] >> leaf_log2, origin[2] >> leaf_log2))
            if kind == 1:  # v3 constant leaf: whole block one value, uniformly active
                value = np.frombuffer(r.read(wire_item.itemsize), dtype=wire_item).astype(item)
                if per_voxel == 3:
                    leaf.values[...] = np.full((dim, dim, dim, 3), value.reshape(3), dtype=item)
                else:
                    leaf.values[...] = np.full((dim, dim, dim), value.reshape(1)[0], dtype=item)
                leaf.active[...] = True
                leaf.invalidate()
                continue
            mask_words = np.frombuffer(r.read(words * 8), dtype="<u8")
            active = _mask_from_words(mask_words, dim**3)
            values_flat = np.frombuffer(r.read(dim**3 * per_voxel * wire_item.itemsize), dtype=wire_item)
            if wire_item != item:
                values_flat = values_flat.astype(item)
            if grid.is_vec:
                leaf.values[...] = values_flat.reshape(dim, dim, dim, 3)
            else:
                leaf.values[...] = values_flat.reshape(dim, dim, dim)
            leaf.active[...] = active.reshape(dim, dim, dim)
            leaf.invalidate()
        grids.append(grid)
    return grids


def _parse_header(data, path):
    r = _Reader(data)
    if r.read(len(MAGIC)) != MAGIC:
        raise ValueError(f"{path} is not an XUVDB file")
    version, flags, n_grids = struct.unpack("<BBI", r.read(6))
    if version not in (1, 2, 3):
        raise ValueError(f"unsupported XUVDB version {version}")
    if version == 1 and flags != 0:
        raise ValueError(f"unsupported XUVDB v1 flags {flags}")
    if version >= 2:
        crc = struct.unpack("<I", data[-4:])[0]
        payload = data[11:-4]
        if flags & _FLAG_COMPRESSED:
            payload = zlib.decompress(payload)
        if flags & _FLAG_CRC and zlib.crc32(payload) & 0xFFFFFFFF != crc:
            raise ValueError(f"{path}: XUVDB payload checksum mismatch (file corrupted)")
        r = _Reader(payload)
    else:
        r = _Reader(data)
    return r, version, n_grids


def _skip_grid_records(r, version, leaf_log2, type_code, n_leaves):
    """Advance past one grid's leaf records (for streaming skippers)."""
    dim = 1 << leaf_log2
    per_voxel = 3 if type_code == 2 else 1
    wire_item = np.dtype("<f8") if type_code == 1 else np.dtype("<f2") if type_code == 3 else np.dtype("<f4")
    for _ in range(n_leaves):
        r.read(12)
        if version >= 3:
            kind = r.read(1)[0]
        else:
            kind = 0
        if kind == 1:
            r.read(wire_item.itemsize)
            continue
        r.read(dim**3 // 64 * 8)
        r.read(dim**3 * per_voxel * wire_item.itemsize)


def iter_leaves(path, grid_index=0):
    """Lazily yield `(origin i64[3], values ndarray, active ndarray)` per leaf of a stored grid -
    the out-of-core primitive: leaves arrive in origin order without building the whole tree.

    Compressed (bit0) files are decompressed once up front (memory = one grid's payload);
    uncompressed files stream leaf-by-leaf from the buffer. Constant (v3) leaves materialize
    as dense arrays.
    """
    with open(path, "rb") as f:
        data = f.read()
    r, version, n_grids = _parse_header(data, str(path))
    if not 0 <= grid_index < n_grids:
        raise IndexError(f"grid_index {grid_index} out of range ({n_grids} grids)")
    for gi in range(grid_index):
        (name_len,) = struct.unpack("<I", r.read(4))
        r.read(name_len)
        type_code, leaf_log2, class_code, grid_flags = struct.unpack("<BBBB", r.read(4))
        r.read(24 + 24)
        if version >= 2 and grid_flags & _GRID_FLAG_ROTATION:
            r.read(72)
        bg_bytes = 24 if type_code == 2 else (8 if type_code == 1 else 4)
        r.read(bg_bytes)
        (n_leaves,) = struct.unpack("<I", r.read(4))
        _skip_grid_records(r, version, leaf_log2, type_code, n_leaves)
    (name_len,) = struct.unpack("<I", r.read(4))
    name = r.read(name_len).decode("utf-8")
    type_code, leaf_log2, class_code, grid_flags = struct.unpack("<BBBB", r.read(4))
    r.read(24 + 24)
    if version >= 2 and grid_flags & _GRID_FLAG_ROTATION:
        r.read(72)
    bg_bytes = 24 if type_code == 2 else (8 if type_code == 1 else 4)
    r.read(bg_bytes)
    (n_leaves,) = struct.unpack("<I", r.read(4))
    del name
    dim = 1 << leaf_log2
    per_voxel = 3 if type_code == 2 else 1
    wire_item = np.dtype("<f8") if type_code == 1 else np.dtype("<f2") if type_code == 3 else np.dtype("<f4")
    item = _TYPE_BY_CODE.get(type_code, np.float32) if type_code != 2 else np.float32
    for _leaf in range(n_leaves):
        origin = np.frombuffer(r.read(12), dtype="<i4").astype(np.int64)
        if version >= 3:
            kind = r.read(1)[0]
        else:
            kind = 0
        if kind == 1:
            value = np.frombuffer(r.read(wire_item.itemsize), dtype=wire_item).astype(item)
            if per_voxel == 3:
                values = np.full((dim, dim, dim, 3), value.reshape(3), dtype=item)
            else:
                values = np.full((dim, dim, dim), value.reshape(-1)[0], dtype=item)
            active = np.ones((dim, dim, dim), dtype=bool)
        else:
            mask_words = np.frombuffer(r.read(dim**3 // 64 * 8), dtype="<u8")
            active = _mask_from_words(mask_words, dim**3)
            values_flat = np.frombuffer(r.read(dim**3 * per_voxel * wire_item.itemsize), dtype=wire_item)
            if wire_item.itemsize != np.dtype(item).itemsize:
                values_flat = values_flat.astype(item)
            if per_voxel == 3:
                values = values_flat.reshape(dim, dim, dim, 3)
            else:
                values = values_flat.reshape(dim, dim, dim)
        yield origin, values, active


def iter_slabs(path, grid_index=0, axis=0, slab_leaves=1):
    """Yield `(ijk_min, dense_slab, active_slab)` slab-by-slab along `axis`, `slab_leaves` leaf
    blocks per slab - the out-of-core residence primitive: peak memory is one slab plus one
    leaf, never the grid. Leaf origins are i-major sorted, so `axis=0` slabs flush as the
    stream passes them (one open buffer); other axes buffer until the stream ends. Slabs cover
    their leaf-block extent exactly.
    """
    leaves = iter_leaves(path, grid_index=grid_index)
    slab_leaves = max(int(slab_leaves), 1)
    per_slab = {}

    def build(ents, slab_leaves):
        leaf_dim = ents[0][1].shape[0]
        tail = list(ents[0][1].shape[3:])
        mins = np.minimum.reduce([e[0] for e in ents])
        maxs = np.maximum.reduce([e[0] for e in ents])
        dims = (((maxs - mins) // leaf_dim + 1) * leaf_dim).astype(int)  # per-axis extent
        slab = np.zeros(list(dims) + tail, dtype=np.float32)
        slab_active = np.zeros(dims, dtype=bool)
        for origin, values, active in ents:
            dst = tuple(slice(int(origin[a] - mins[a]), int(origin[a] - mins[a]) + leaf_dim)
                        for a in range(3))
            slab[dst] = values.reshape([leaf_dim] * 3 + tail)
            slab_active[dst] = active.reshape([leaf_dim] * 3)
        return mins.copy(), slab, slab_active  # ijk_min = the slab's own content min

    for origin, values, active in leaves:
        dim = values.shape[0]
        kb = int(origin[axis]) // (dim * slab_leaves)
        per_slab.setdefault(kb, []).append((origin, values, active))
        if axis == 0 and len(per_slab) > 1:
            done = min(per_slab)
            if done < kb:
                yield build(per_slab.pop(done), slab_leaves)
    for kb in sorted(per_slab):
        yield build(per_slab[kb], slab_leaves)
