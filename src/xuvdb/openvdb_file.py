"""Native OpenVDB `.vdb` file reader and writer, implemented directly against the published stream
format (no OpenVDB installation required).

This is the interop path for the quadrants kernel world: Houdini / Blender / OpenVDB grids can be
brought in as editable XUVDB trees, and edited grids can be handed back to any OpenVDB consumer.

Wire format implemented (verified against the OpenVDB sources `io/Archive.cc`, `io/GridDescriptor.cc`,
`io/Compression.h`, `tree/{Tree,RootNode,InternalNode,LeafNode}.h`, `math/{Transform,Maps}.cc`,
`Metadata.h`):

- 57-byte file header: int64 magic `0x56444220`, u32 file version, u32 library major/minor, one-byte
  grid-offsets flag, 36-char UUID.
- file-level metadata map, then i32 grid count.
- per grid: descriptor (name / type / instance-parent strings + three i64 offsets), then at the grid
  offset: u32 compression flags, grid metadata map, transform (ScaleTranslate family), tree topology
  (Root -> InternalNode(5) -> InternalNode(4) -> LeafNode(3)), then leaf buffers.
- leaf and internal tables are z-fastest (`n = x * dim^2 + y * dim + z`), masks are u64 words with
  LSB-first bit order.

Written files use file version 224 with `COMPRESS_ACTIVE_MASK` only (no zip/blosc), so every value
block on disk is either metadata-0 (only active values, inactive equal the background) or
metadata-6 (full array) when a leaf carries irregular inactive values. Reading supports
COMPRESS_NONE, COMPRESS_ZIP (via the stdlib zlib), COMPRESS_ACTIVE_MASK and `_HalfFloat` grids;
Blosc-compressed grids raise an informative error.

Root-level tiles are materialized into dense leaves when reading (bounded by `max_tile_voxels`),
and never emitted when writing.
"""

import struct
import uuid as _uuid
import warnings
import zlib

import numpy as np

from .tree import GRID_CLASSES, VdbGrid

try:
    import blosc
except ImportError:  # blosc-compressed .vdb files need the optional `pip install blosc`
    blosc = None

OPENVDB_MAGIC = 0x56444220
FILE_VERSION = 224
LIB_MAJOR, LIB_MINOR = 11, 0

COMPRESS_NONE = 0x0
COMPRESS_ZIP = 0x1
COMPRESS_ACTIVE_MASK = 0x2
COMPRESS_BLOSC = 0x4

_LEAF_LOG, _INT4_LOG, _INT5_LOG = 3, 4, 5  # bottom-up log dims of the canonical 5_4_3 tree
_LEAF_DIM = 1 << _LEAF_LOG
_INT4_DIM = 1 << (_INT4_LOG + _LEAF_LOG)  # 128
_INT5_DIM = 1 << (_INT5_LOG + _INT4_LOG + _LEAF_LOG)  # 4096
_INT4_TABLE = 1 << (3 * _INT4_LOG)  # 4096 entries
_INT5_TABLE = 1 << (3 * _INT5_LOG)  # 32768 entries

_NO_MASK_OR_INACTIVE_VALS = 0
_NO_MASK_AND_MINUS_BG = 1
_NO_MASK_AND_ONE_INACTIVE_VAL = 2
_MASK_AND_NO_INACTIVE_VALS = 3
_MASK_AND_ONE_INACTIVE_VAL = 4
_MASK_AND_TWO_INACTIVE_VALS = 5
_NO_MASK_AND_ALL_VALS = 6

_TREE_TYPES = {
    "Tree_float_5_4_3": (np.float32, 1),
    "Tree_double_5_4_3": (np.float64, 1),
    "Tree_vec3s_5_4_3": (np.float32, 3),
}
_HALF_SUFFIX = "_HalfFloat"

_META_FIXED = {
    "bool": 1,
    "int32_t": 4,
    "int64_t": 8,
    "float": 4,
    "double": 8,
    "vec2i": 8, "vec2s": 8, "vec2d": 16,
    "vec3i": 12, "vec3s": 12, "vec3d": 24,
    "vec4i": 16, "vec4s": 16, "vec4d": 32,
    "mat4s": 64, "mat4d": 128,
}


# ------------------------------------------------------------------ stream helpers


class _WriteStream:
    def __init__(self):
        self.buf = bytearray()

    def u8(self, v):
        self.buf += struct.pack("<B", v)

    def i32(self, v):
        self.buf += struct.pack("<i", v)

    def u32(self, v):
        self.buf += struct.pack("<I", v)

    def i64(self, v):
        self.buf += struct.pack("<q", v)

    def f32(self, v):
        self.buf += struct.pack("<f", v)

    def f64(self, v):
        self.buf += struct.pack("<d", v)

    def string(self, s):
        b = s.encode("utf-8")
        self.u32(len(b))
        self.buf += b

    def raw(self, b):
        self.buf += b

    def vec3d(self, xyz):
        for v in xyz:
            self.f64(float(v))

    def tell(self):
        return len(self.buf)

    def patch_i64(self, pos, value):
        struct.pack_into("<q", self.buf, pos, value)


class _ReadStream:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def _take(self, n):
        if self.pos + n > len(self.data):
            raise ValueError("truncated .vdb stream")
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def u8(self):
        return self._take(1)[0]

    def i32(self):
        return struct.unpack("<i", self._take(4))[0]

    def u32(self):
        return struct.unpack("<I", self._take(4))[0]

    def i64(self):
        return struct.unpack("<q", self._take(8))[0]

    def f32(self):
        return struct.unpack("<f", self._take(4))[0]

    def f64(self):
        return struct.unpack("<d", self._take(8))[0]

    def string(self):
        (n,) = struct.unpack("<I", self._take(4))
        return self._take(n).decode("utf-8")

    def raw(self, n):
        return self._take(n)

    def vec3d(self):
        return np.array(struct.unpack("<3d", self._take(24)))

    def words(self, count):
        return np.frombuffer(self._take(count * 8), dtype="<u8")


def _pack_mask(bits_bool):
    """bool array -> u64 words, bit n = element n (LSB-first, matching NodeMask::save)."""
    bits = np.packbits(bits_bool.reshape(-1), bitorder="little")
    bits = np.pad(bits, (0, (-len(bits)) % 8))
    return bits.view(np.uint64)


def _unpack_mask(words, count):
    bits = words.view(np.uint8)
    return np.unpackbits(bits[: (count + 7) // 8], count=count, bitorder="little").astype(bool)


def _mask_words_for(count):
    return (count + 63) // 64


# ------------------------------------------------------------------ writer


def write_vdb(path, grids, blosc=False):
    """Write one or more `VdbGrid`s as a spec-conformant `.vdb` file readable by OpenVDB/Houdini.

    This is the explicit interop export: it emits a real OpenVDB stream, so the `.vdb` suffix is
    correct here - and expected, since the suffix is what volume tools key on. `blosc=True` emits
    Blosc-compressed value blocks (clevel 9, byte shuffle; OpenVDB's own framing) and needs the
    optional 'blosc' package. Half grids are upcast to float32 on write.
    """
    grids = [_as_f32(g) if g.type_code == 3 else g for g in grids]
    if not str(path).lower().endswith(".vdb"):
        warnings.warn(f"writing an OpenVDB stream to {str(path)!r} without the '.vdb' suffix; "
                      "volume tools identify OpenVDB files by extension")
    grids = list(grids)
    s = _WriteStream()
    s.i64(OPENVDB_MAGIC)
    s.u32(FILE_VERSION)
    s.u32(LIB_MAJOR)
    s.u32(LIB_MINOR)
    s.u8(1)  # has grid offsets (seekable)
    s.raw(str(_uuid.uuid4()).upper().encode("ascii"))  # 36 chars

    s.u32(0)  # empty file-level metadata map
    s.i32(len(grids))
    for grid in grids:
        if grid.type_code == 2:
            grid_type = "Tree_vec3s_5_4_3"
        else:
            grid_type = "Tree_float_5_4_3" if grid.type_code == 0 else "Tree_double_5_4_3"
        s.string(grid.name)  # unique name; duplicates would need \x1e suffixes
        s.string(grid_type)
        s.string("")  # instance parent name
        offset_pos = s.tell()
        s.i64(0)
        s.i64(0)
        s.i64(0)
        grid_pos = s.tell()
        block_pos = _write_grid_stream(s, grid, use_blosc=blosc)
        end_pos = s.tell()
        s.patch_i64(offset_pos, grid_pos)
        s.patch_i64(offset_pos + 8, block_pos)
        s.patch_i64(offset_pos + 16, end_pos)

    with open(path, "wb") as f:
        f.write(bytes(s.buf))
    return len(s.buf)


def _write_grid_stream(s, grid, use_blosc=False):
    """Emit one grid's metadata, transform, topology and buffers; returns the block (buffers) offset."""
    s.u32(COMPRESS_ACTIVE_MASK | (COMPRESS_BLOSC if use_blosc else 0))

    bbox = grid.bbox()
    meta = [("name", "string", grid.name.encode("utf-8")),
            ("class", "string", grid.grid_class.encode("utf-8")),
            ("file_compression", "string", b"active values"),
            ("file_bbox_min", "vec3i", struct.pack("<3i", *bbox[0]) if bbox is not None else struct.pack("<3i", 0, 0, 0)),
            ("file_bbox_max", "vec3i", struct.pack("<3i", *bbox[1]) if bbox is not None else struct.pack("<3i", -1, -1, -1)),
            ("file_voxel_count", "int64_t", struct.pack("<q", grid.active_voxel_count))]
    s.u32(len(meta))
    for name, type_name, payload in meta:
        s.string(name)
        s.string(type_name)
        s.u32(len(payload))
        s.raw(payload)

    # transform: AffineMap for rotated grids (row-major Mat4d), ScaleTranslateMap otherwise
    if grid.is_axis_aligned:
        s.string("ScaleTranslateMap")
        t = grid.origin_world
        vs = grid.voxel_size
        s.vec3d(t)
        s.vec3d(vs)
        s.vec3d(vs)
        s.vec3d(1.0 / vs)
        s.vec3d(1.0 / vs**2)
        s.vec3d(0.5 / vs)
    else:
        s.string("AffineMap")
        m = np.eye(4)
        m[:3, :3] = grid.rotation * grid.voxel_size  # world = R @ (index * s) + t
        m[:3, 3] = grid.origin_world
        s.raw(m.astype("<f8").tobytes())

    # ---- tree topology
    tree = _build_vdb_tree(grid)
    s.i32(1)  # buffer count
    if grid.type_code == 0:
        s.f32(float(grid.background))
    elif grid.type_code == 1:
        s.f64(float(grid.background))
    else:
        s.raw(np.asarray(grid.background).reshape(3).astype("<f4").tobytes())
    s.u32(0)  # root tiles
    s.u32(len(tree))  # root children (InternalNode level 5)
    for int5 in tree:
        s.raw(int5["origin"].astype("<i4").tobytes())
        s.raw(_pack_mask(int5["child_mask"]).astype("<u8").tobytes())
        s.raw(np.zeros(_mask_words_for(_INT5_TABLE), dtype="<u8").tobytes())  # value mask: no active tiles
        s.u8(_NO_MASK_OR_INACTIVE_VALS)  # all inactive values are +background
        for int4 in int5["children"]:
            s.raw(_pack_mask(int4["child_mask"]).astype("<u8").tobytes())
            s.raw(np.zeros(_mask_words_for(_INT4_TABLE), dtype="<u8").tobytes())
            s.u8(_NO_MASK_OR_INACTIVE_VALS)
            for leaf in int4["children"]:
                s.raw(_pack_mask(leaf["active"]).astype("<u8").tobytes())

    block_pos = s.tell()

    # ---- leaf buffers
    for int5 in tree:
        for int4 in int5["children"]:
            for leaf in int4["children"]:
                s.raw(_pack_mask(leaf["active"]).astype("<u8").tobytes())
                _write_compressed_values(s, leaf["values"], leaf["active"], grid, use_blosc)

    return block_pos


def _write_compressed_values(s, values, active, grid, use_blosc=False):
    """One leaf value block under COMPRESS_ACTIVE_MASK: metadata 0 (or 2/5/6) + active values."""
    flat = values.reshape(-1) if not grid.is_vec else values.reshape(-1, 3)
    inactive = flat[~active.reshape(-1)]
    if grid.is_vec:
        distinct = np.unique(inactive.reshape(-1, 3), axis=0) if len(inactive) else np.zeros((0, 3))
        bg = np.asarray(grid.background).reshape(3)
    else:
        distinct = np.unique(inactive) if len(inactive) else np.zeros(0)
        bg = float(grid.background)
    wire_dtype = np.dtype("<f8") if grid.type_code == 1 else np.dtype("<f4")

    if len(distinct) == 0 or (grid.is_vec and np.all(distinct == bg)) or (not grid.is_vec and np.all(distinct == bg)):
        metadata = _NO_MASK_OR_INACTIVE_VALS
    elif not grid.is_vec and len(distinct) == 1 and float(distinct[0]) == -bg:
        metadata = _NO_MASK_AND_MINUS_BG
    elif len(distinct) == 1:
        metadata = _NO_MASK_AND_ONE_INACTIVE_VAL
    elif len(distinct) == 2:
        metadata = _MASK_AND_TWO_INACTIVE_VALS
    else:
        metadata = _NO_MASK_AND_ALL_VALS

    s.u8(metadata)
    if metadata in (_NO_MASK_AND_ONE_INACTIVE_VAL, _MASK_AND_TWO_INACTIVE_VALS):
        v = distinct[0]
        s.raw(np.asarray(v, dtype=wire_dtype).tobytes())
    if metadata == _MASK_AND_TWO_INACTIVE_VALS:
        s.raw(np.asarray(distinct[1], dtype=wire_dtype).tobytes())
        selection = (flat == distinct[1]).all(axis=-1) if grid.is_vec else (flat == distinct[1])
        s.raw(_pack_mask(selection & ~active.reshape(-1)).astype("<u8").tobytes())
    payload = (flat if metadata == _NO_MASK_AND_ALL_VALS else flat[active.reshape(-1)]).astype(wire_dtype)
    _emit_value_payload(s, np.ascontiguousarray(payload).tobytes(), use_blosc)


def _emit_value_payload(s, payload, use_blosc):
    """The leaf's value buffer with bloscToStream semantics: i64 length (positive = Blosc frame,
    negative = -raw size) then the bytes; falls back to raw when compression does not shrink.
    Frame parameters mirror OpenVDB's bloscCompress (clevel 9, byte shuffle)."""
    if not use_blosc:
        s.raw(payload)
        return
    if blosc is None:
        raise ImportError("blosc output requires the optional 'blosc' package (pip install blosc)")
    frame = blosc.compress(payload, typesize=4 if len(payload) % 4 == 0 else 1,
                           clevel=9, shuffle=blosc.SHUFFLE)
    if len(frame) < len(payload):
        s.raw(struct.pack("<q", len(frame)))
        s.raw(frame)
    else:
        s.raw(struct.pack("<q", -len(payload)))
        s.raw(payload)


def _as_f32(grid):
    """Upcast a half grid to float32 (same structure) for the .vdb writer."""
    out = VdbGrid(background=grid.background, voxel_size=grid.voxel_size,
                  origin_world=grid.origin_world, leaf_log2=grid.leaf_log2,
                  name=grid.name, grid_class=grid.grid_class, rotation=grid.rotation)
    for key, leaf in grid._leaves.items():
        new = out._get_or_create_leaf(key)
        new.values[...] = leaf.values
        new.active[...] = leaf.active
    return out


def _build_vdb_tree(grid):
    """Group the grid's 2^log2 leaves into the canonical 5/4/3 node hierarchy.

    Each XUVDB leaf of dim `2**leaf_log2` splits into `(2**leaf_log2 / 8)^3` OpenVDB 8^3 leaf nodes;
    a sub-leaf is emitted only when it has at least one active voxel.
    """
    vdb_leaves = {}
    for leaf in grid.leaves():
        n_split = leaf.dim // _LEAF_DIM
        for sx in range(n_split):
            for sy in range(n_split):
                for sz in range(n_split):
                    sl = (
                        slice(sx * _LEAF_DIM, (sx + 1) * _LEAF_DIM),
                        slice(sy * _LEAF_DIM, (sy + 1) * _LEAF_DIM),
                        slice(sz * _LEAF_DIM, (sz + 1) * _LEAF_DIM),
                    )
                    active = leaf.active[sl]
                    if not active.any():
                        continue
                    origin = tuple(int(v) for v in leaf.origin + np.array([sx, sy, sz]) * _LEAF_DIM)
                    vdb_leaves[origin] = {"origin": origin, "active": active, "values": leaf.values[sl]}

    int4_table, int5_table = {}, {}
    for origin in sorted(vdb_leaves):
        int4_origin = tuple(o & ~(_INT4_DIM - 1) for o in origin)
        int5_origin = tuple(o & ~(_INT5_DIM - 1) for o in origin)
        int4_table.setdefault(int4_origin, []).append(vdb_leaves[origin])
        int5_table.setdefault(int5_origin, []).append(int4_origin)

    tree = []
    for int5_origin in sorted(int5_table):
        child_mask = np.zeros(_INT5_TABLE, dtype=bool)
        int4_nodes = []
        for int4_origin in sorted(set(int5_table[int5_origin])):
            local = np.array(int4_origin) - np.array(int5_origin)
            n = ((local[0] // _INT4_DIM) << (2 * _INT5_LOG)) + ((local[1] // _INT4_DIM) << _INT5_LOG) + (local[2] // _INT4_DIM)
            child_mask[n] = True
            leaf_mask = np.zeros(_INT4_TABLE, dtype=bool)
            for leaf in int4_table[int4_origin]:
                local_leaf = np.array(leaf["origin"]) - np.array(int4_origin)
                m = ((local_leaf[0] // _LEAF_DIM) << (2 * _INT4_LOG)) + ((local_leaf[1] // _LEAF_DIM) << _INT4_LOG) + (local_leaf[2] // _LEAF_DIM)
                leaf_mask[m] = True
            int4_nodes.append({"origin": np.array(int4_origin), "child_mask": leaf_mask, "children": int4_table[int4_origin]})
        tree.append({"origin": np.array(int5_origin), "child_mask": child_mask, "children": int4_nodes})
    return tree


# ------------------------------------------------------------------ reader


def read_vdb(path, grid_name=None, leaf_log2=3, max_tile_voxels=1 << 24):
    """Read grids from a `.vdb` file into `VdbGrid`s.

    `grid_name` selects a single grid by name; `leaf_log2` re-blocks the incoming 8^3 OpenVDB leaves
    (pass 3 for a 1:1 mapping). Root/interior active tiles are materialized into dense leaves unless
    a single tile exceeds `max_tile_voxels` voxels, in which case it is skipped with a warning.
    """
    with open(path, "rb") as f:
        data = f.read()
    s = _ReadStream(data)

    magic = s.i64()
    if magic != OPENVDB_MAGIC:
        raise ValueError(f"{path} is not an OpenVDB .vdb file")
    file_version = s.u32()
    if file_version < 221:
        raise ValueError(f".vdb file version {file_version} predates the current format (< 221)")
    if file_version > 225:
        warnings.warn(f".vdb file version {file_version} is newer than the tested range (<= 225)")
    s.u32()  # library major
    s.u32()  # library minor
    s.u8()  # has grid offsets
    s.raw(36)  # uuid
    _read_meta_map(s)  # file-level metadata (ignored)

    (grid_count,) = struct.unpack("<i", s.raw(4))
    # Descriptors interleave with their grid data: descriptor, grid stream, next descriptor...
    grids = []
    for _ in range(grid_count):
        unique_name = s.string()
        grid_type = s.string()
        instance_parent = s.string()
        grid_pos, block_pos, end_pos = s.i64(), s.i64(), s.i64()

        name = unique_name.split("\x1e")[0]
        wanted = grid_name is None or name == grid_name
        if wanted and instance_parent:
            raise NotImplementedError(
                f"grid {name!r} is an instance of {instance_parent!r}; instances are not supported")
        if wanted and grid_type.endswith(_HALF_SUFFIX):
            from_half = True
            grid_type = grid_type[: -len(_HALF_SUFFIX)]
        else:
            from_half = False
        if wanted and grid_type not in _TREE_TYPES:
            raise NotImplementedError(f"grid {name!r} has unsupported tree type {grid_type!r}")

        if wanted:
            if grid_pos > 0:
                s.pos = grid_pos
            grids.append(_read_grid_stream(s, name, grid_type, from_half, leaf_log2, max_tile_voxels))
        if end_pos > 0:  # non-seekable streams (all offsets zero) are already sequential
            s.pos = end_pos
    if grid_name is not None and not grids:
        raise KeyError(f"no grid named {grid_name!r} in {path}")
    return grids


def _read_meta_map(s):
    """Parse a MetaMap; known fixed-size types and strings are decoded, the rest skipped by size."""
    out = {}
    count = s.u32()
    for _ in range(count):
        name = s.string()
        type_name = s.string()
        size = s.u32()
        payload = s.raw(size)
        if type_name == "string":
            out[name] = payload.decode("utf-8")
        elif type_name in _META_FIXED:
            fmt = {"bool": "<?", "int32_t": "<i", "int64_t": "<q", "float": "<f", "double": "<d"}.get(type_name)
            if fmt and size == struct.calcsize(fmt):
                out[name] = struct.unpack(fmt, payload)[0]
            elif type_name.startswith("vec"):
                n = int(type_name[3])
                kind = type_name[4]
                out[name] = np.frombuffer(payload, dtype={"i": "<i4", "s": "<f4", "d": "<f8"}[kind]).reshape(n)
        # unknown types: value already skipped via its size prefix
    return out


def _read_grid_stream(s, name, grid_type, from_half, leaf_log2, max_tile_voxels):
    compression = s.u32()
    if compression & COMPRESS_BLOSC and blosc is None:
        raise ImportError(
            f"grid {name!r} uses Blosc compression; install the optional 'blosc' package "
            "(pip install blosc) to read it"
        )
    meta = _read_meta_map(s)

    map_type = s.string()
    voxel_size, origin_world, rotation = _read_transform(s, map_type)

    s.i32()  # buffer count (1, the pre-222 multi-buffer layout is gone)

    value_dtype, components = _TREE_TYPES[grid_type]
    background = _read_wire_value(s, value_dtype, components)
    if components == 3:
        dtype = np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32)])
        background = background.reshape(3)
    else:
        dtype = value_dtype
        background = float(background.reshape(-1)[0])
    grid_class = meta.get("class", "unknown")
    if grid_class not in GRID_CLASSES:
        grid_class = "unknown"
    grid = VdbGrid(dtype=dtype, background=background, voxel_size=voxel_size, origin_world=origin_world,
                   leaf_log2=leaf_log2, name=meta.get("name", name), grid_class=grid_class,
                   rotation=rotation)

    n_tiles = s.u32()
    n_int5 = s.u32()
    tile_regions = []
    for _ in range(n_tiles):
        origin = np.frombuffer(s.raw(12), dtype="<i4").astype(np.int64)
        value = _read_wire_value(s, value_dtype, components)
        active = bool(s.raw(1)[0])
        if active:
            tile_regions.append((origin, value))

    int5_nodes = []
    for _ in range(n_int5):
        origin = np.frombuffer(s.raw(12), dtype="<i4").astype(np.int64)
        child_mask = _unpack_mask(s.words(_mask_words_for(_INT5_TABLE)), _INT5_TABLE)
        value_mask = _unpack_mask(s.words(_mask_words_for(_INT5_TABLE)), _INT5_TABLE)
        values = _read_compressed_values(s, _INT5_TABLE, value_mask, value_dtype, components, background,
                                         compression, from_half)
        tile_regions.extend(_active_tiles_from_table(origin, values, value_mask, _INT4_DIM))
        int4_nodes = []
        for n in np.nonzero(child_mask)[0]:
            cx, cy, cz = _decode_index(int(n), _INT5_LOG)
            int4_origin = origin + np.array([cx, cy, cz]) * _INT4_DIM
            int4_nodes.append(_read_int4(s, int4_origin, value_dtype, components, background, compression,
                                         from_half, tile_regions))
        int5_nodes.append(int4_nodes)

    # leaf buffers follow in the exact child-on order recorded during topology
    for int4_nodes in int5_nodes:
        for int4 in int4_nodes:
            for leaf in int4["children"]:
                mask_words = s.words(_mask_words_for(_LEAF_DIM**3))
                active = _unpack_mask(mask_words, _LEAF_DIM**3)
                values = _read_compressed_values(s, _LEAF_DIM**3, active, value_dtype, components, background,
                                                 compression, from_half)
                _store_leaf(grid, leaf["origin"], active, values, components)

    for origin, value in tile_regions:
        if _INT4_DIM**3 > max_tile_voxels:
            warnings.warn(f"skipping an active tile at {tuple(origin)} larger than max_tile_voxels")
            continue
        _fill_tile(grid, origin, _INT4_DIM, value, components)
    return grid


def _read_int4(s, origin, value_dtype, components, background, compression, from_half, tile_regions):
    child_mask = _unpack_mask(s.words(_mask_words_for(_INT4_TABLE)), _INT4_TABLE)
    value_mask = _unpack_mask(s.words(_mask_words_for(_INT4_TABLE)), _INT4_TABLE)
    values = _read_compressed_values(s, _INT4_TABLE, value_mask, value_dtype, components, background,
                                     compression, from_half)
    tile_regions.extend(_active_tiles_from_table(origin, values, value_mask, _LEAF_DIM))
    children = []
    for n in np.nonzero(child_mask)[0]:
        cx, cy, cz = _decode_index(int(n), _INT4_LOG)
        children.append({"origin": origin + np.array([cx, cy, cz]) * _LEAF_DIM})
        # leaf topology: just the value mask, re-read at the start of each leaf buffer
        s.words(_mask_words_for(_LEAF_DIM**3))
    return {"origin": origin, "children": children}


def _decode_index(n, log2dim):
    """Inverse of the z-fastest table index `n = cx * d^2 + cy * d + cz` (d = 2^log2dim)."""
    d = 1 << log2dim
    return n >> (2 * log2dim), (n >> log2dim) & (d - 1), n & (d - 1)


def _active_tiles_from_table(origin, values, value_mask, tile_dim):
    out = []
    shape = (-1, 3) if values.ndim == 2 else (-1,)
    flat = values.reshape(shape)
    for n in np.nonzero(value_mask)[0]:
        cx, cy, cz = _decode_index(int(n), _log2_of_table(len(flat)))
        out.append((origin + np.array([cx, cy, cz]) * tile_dim, flat[n].copy()))
    return out


def _log2_of_table(table_size):
    log2 = 0
    while (1 << log2) ** 2 * (1 << log2) < table_size:
        log2 += 1
    return log2


def _read_wire_value(s, dtype, components):
    if components == 1:
        return np.array(s.f64() if dtype == np.float64 else s.f32(), dtype=np.float64)
    return np.frombuffer(s.raw(4 * components), dtype="<f4").astype(np.float64)


def _read_transform(s, map_type):
    """`(voxel_size, origin_world, rotation-or-None)`; AffineMap/UnitaryMap decompose the Mat4d
    (row-major, translation in the last column) into `R @ diag(scale)` when it is rigid."""
    if map_type in ("ScaleTranslateMap", "UniformScaleTranslateMap"):
        t = s.vec3d()
        scale = s.vec3d()
        for _ in range(4):  # voxel size, inverse scale, inverse square, half-inverse
            s.vec3d()
        return scale, t, None
    if map_type in ("ScaleMap", "UniformScaleMap"):
        scale = s.vec3d()
        for _ in range(4):
            s.vec3d()
        return scale, np.zeros(3), None
    if map_type == "TranslationMap":
        t = s.vec3d()
        return np.ones(3), t, None
    if map_type in ("AffineMap", "UnitaryMap"):
        m = np.frombuffer(s.raw(16 * 8), dtype="<f8").reshape(4, 4)
        linear = m[:3, :3]
        scale = np.linalg.norm(linear, axis=0)  # columns of R*diag(s) have length s_a
        rot = linear / scale
        if not np.allclose(rot @ rot.T, np.eye(3), atol=1e-6):
            rot = None  # shear / general affine: keep only the diagonal part
        return scale, m[:3, 3].copy(), rot
    raise NotImplementedError(f"unsupported transform map type {map_type!r}")


def _read_compressed_values(s, count, value_mask, value_dtype, components, background, compression, from_half):
    """Decode one `io::readCompressedValues` block into a (count,) or (count, 3) array."""
    metadata = s.u8()
    bg = np.asarray(background, dtype=np.float64).reshape(-1)
    inactive_val0 = bg.copy()
    inactive_val1 = bg.copy()
    if metadata != _NO_MASK_OR_INACTIVE_VALS:
        inactive_val0 = -bg
    wire = np.dtype("<f2") if from_half else (np.dtype("<f8") if value_dtype == np.float64 else np.dtype("<f4"))
    item = wire.itemsize * components

    if metadata in (_NO_MASK_AND_ONE_INACTIVE_VAL, _MASK_AND_ONE_INACTIVE_VAL, _MASK_AND_TWO_INACTIVE_VALS):
        inactive_val0 = np.frombuffer(s.raw(item), dtype=wire).astype(np.float64).reshape(-1)
    if metadata == _MASK_AND_TWO_INACTIVE_VALS:
        inactive_val1 = np.frombuffer(s.raw(item), dtype=wire).astype(np.float64).reshape(-1)

    selection = None
    if metadata in (_MASK_AND_NO_INACTIVE_VALS, _MASK_AND_ONE_INACTIVE_VAL, _MASK_AND_TWO_INACTIVE_VALS):
        selection = _unpack_mask(s.words(_mask_words_for(count)), count)

    mask_compressed = compression & COMPRESS_ACTIVE_MASK
    temp_count = int(value_mask.sum()) if (mask_compressed and metadata != _NO_MASK_AND_ALL_VALS) else count

    if temp_count == 0:
        blob = b""  # empty value tables (no active tiles) carry no data block at all
    else:
        blob = _read_data_blob(s, temp_count * item, compression)
    values = np.frombuffer(blob, dtype=wire).astype(np.float64)
    if components == 3:
        values = values.reshape(temp_count, 3)

    if temp_count == count:
        return values

    out = np.zeros((count, components) if components == 3 else (count,), dtype=np.float64)
    active_flat = value_mask
    out[active_flat] = values[: int(active_flat.sum())]
    inactive_flat = ~active_flat
    if selection is not None:
        out[inactive_flat & selection] = inactive_val1
        out[inactive_flat & ~selection] = inactive_val0
    else:
        out[inactive_flat] = inactive_val0
    return out


def _read_data_blob(s, num_bytes, compression):
    if compression & COMPRESS_BLOSC:
        if blosc is None:
            raise ImportError(
                "this .vdb uses Blosc compression; install the optional 'blosc' package "
                "(pip install blosc) to read it"
            )
        (nbytes,) = struct.unpack("<q", s.raw(8))
        if nbytes < 0:  # OpenVDB writes a negative length when the block stayed uncompressed
            return s.raw(-nbytes)
        raw = blosc.decompress(s.raw(nbytes))
        if len(raw) != num_bytes:
            raise ValueError(f"blosc block decompressed to {len(raw)} bytes, expected {num_bytes}")
        return raw
    if compression & COMPRESS_ZIP:
        (zsize,) = struct.unpack("<q", s.raw(8))
        if zsize <= 0:
            return s.raw(-zsize)
        blob = s.raw(zsize)
        return zlib.decompress(blob)
    return s.raw(num_bytes)


def _store_leaf(grid, origin, active_flat, values, components):
    dim = _LEAF_DIM
    leaf = grid._get_or_create_leaf((int(origin[0]) >> grid.leaf_log2, int(origin[1]) >> grid.leaf_log2,
                                     int(origin[2]) >> grid.leaf_log2))
    local = np.asarray(origin) - leaf.origin
    sl = tuple(slice(local[a], local[a] + dim) for a in range(3))
    if grid.is_vec:
        leaf.values[sl] = values.reshape(dim, dim, dim, 3)
    else:
        leaf.values[sl] = values.reshape(dim, dim, dim)
    leaf.active[sl] = active_flat.reshape(dim, dim, dim)


def _fill_tile(grid, origin, dim, value, components):
    half = grid.leaf_dim
    base = np.asarray(origin, dtype=np.int64)
    for kx in range(int(base[0]) >> grid.leaf_log2, (int(base[0]) + dim - 1 >> grid.leaf_log2) + 1):
        for ky in range(int(base[1]) >> grid.leaf_log2, (int(base[1]) + dim - 1 >> grid.leaf_log2) + 1):
            for kz in range(int(base[2]) >> grid.leaf_log2, (int(base[2]) + dim - 1 >> grid.leaf_log2) + 1):
                leaf = grid._get_or_create_leaf((kx, ky, kz))
                lo = np.maximum(base, leaf.origin) - leaf.origin
                hi = np.minimum(base + dim - 1, leaf.origin + half - 1) - leaf.origin
                sl = tuple(slice(lo[a], hi[a] + 1) for a in range(3))
                leaf.values[sl] = value
                leaf.active[sl] = True
