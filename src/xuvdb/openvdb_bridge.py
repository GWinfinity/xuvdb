"""Optional in-memory bridge to `pyopenvdb` (the OpenVDB Python bindings).

Used when the bindings are importable (Houdini's hython, Blender's bundled Python, conda-forge).
The file-level interop path that needs no bindings at all is `openvdb_file.py`.
"""

import numpy as np

try:
    import pyopenvdb
except ImportError:  # the module stays importable; the converters raise on use
    pyopenvdb = None


def _require_bindings():
    if pyopenvdb is None:
        raise ImportError(
            "pyopenvdb is not installed; use xuvdb.openvdb_file.read_vdb/write_vdb for "
            "file-level interop instead"
        )


def _openvdb_grid_class(name):
    _require_bindings()
    return {
        "unknown": pyopenvdb.GridClass.UNKNOWN,
        "level set": pyopenvdb.GridClass.LEVEL_SET,
        "fog volume": pyopenvdb.GridClass.FOG_VOLUME,
        "staggered": pyopenvdb.GridClass.STAGGERED,
    }[name]


def _xuvdb_grid_class(ovdb_value):
    _require_bindings()
    for name, value in (
        ("unknown", pyopenvdb.GridClass.UNKNOWN),
        ("level set", pyopenvdb.GridClass.LEVEL_SET),
        ("fog volume", pyopenvdb.GridClass.FOG_VOLUME),
        ("staggered", pyopenvdb.GridClass.STAGGERED),
    ):
        if int(value) == int(ovdb_value):
            return name
    return "unknown"


def to_openvdb(grid):
    """Convert a `VdbGrid` into a live `pyopenvdb` grid (FloatGrid / DoubleGrid / Vec3SGrid)."""
    _require_bindings()
    if grid.is_vec:
        ovdb = pyopenvdb.Vec3SGrid(background=tuple(float(v) for v in grid.background))
    elif grid.type_code == 1:
        ovdb = pyopenvdb.DoubleGrid(background=float(grid.background))
    else:
        ovdb = pyopenvdb.FloatGrid(background=float(grid.background))
    ovdb.name = grid.name
    ovdb.gridClass = _openvdb_grid_class(grid.grid_class)

    matrix = np.eye(4)
    matrix[:3, :3] = np.diag(grid.voxel_size)
    matrix[:3, 3] = grid.origin_world
    ovdb.transform = pyopenvdb.createLinearTransform(matrix=matrix)

    acc = ovdb.getAccessor()
    for coord, value in grid.iter_voxels():
        acc.setValueOn(coord, tuple(float(v) for v in value) if grid.is_vec else float(value))
    return ovdb


def from_openvdb(ovdb_grid, leaf_log2=3):
    """Convert a `pyopenvdb` grid into a `VdbGrid` (active tiles are materialized into leaves)."""
    _require_bindings()
    from .tree import VdbGrid

    value_type = ovdb_grid.valueType
    if value_type == "vec3s":
        dtype = np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32)])
        background = np.asarray(ovdb_grid.backgroundValue, dtype=np.float64).reshape(3)
    elif value_type == "double":
        dtype = np.float64
        background = float(ovdb_grid.backgroundValue)
    else:
        dtype = np.float32
        background = float(ovdb_grid.backgroundValue)

    matrix = np.array(ovdb_grid.transform.matrix)
    voxel_size = np.array([matrix[0, 0], matrix[1, 1], matrix[2, 2]])
    origin_world = np.array([matrix[0, 3], matrix[1, 3], matrix[2, 3]])
    grid_class = _xuvdb_grid_class(ovdb_grid.gridClass)

    grid = VdbGrid(dtype=dtype, background=background, voxel_size=voxel_size, origin_world=origin_world,
                   leaf_log2=leaf_log2, name=ovdb_grid.name or "grid", grid_class=grid_class)

    acc = ovdb_grid.getConstAccessor()
    for bbox in ovdb_grid.iterActiveTiles():
        lo, hi = np.array(bbox.min), np.array(bbox.max)
        value = acc.getValue(tuple(int(v) for v in lo))
        grid.fill_box(lo, hi, value, active=True)
    for coord in ovdb_grid.iterActiveVoxels():
        value = acc.getValue(tuple(coord))
        grid.set_value(tuple(coord), tuple(float(v) for v in value) if grid.is_vec else float(value))
    return grid
