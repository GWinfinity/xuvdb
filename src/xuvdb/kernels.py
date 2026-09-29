"""Quadrants kernels for GPU-side sampling and mutation of packed XUVDB volumes.

The host packs a scalar grid's leaves into a flat, key-sorted table (`gpu.GpuVolume`): sorted
`int64` leaf keys, one flat float32 value buffer, and the world-space transform scalars. Kernels
resolve a world point to its leaf by packing the leaf coordinates into a 60-bit key and binary
searching the sorted key array - no hash table, so the same buffer is valid on CPU and GPU
backends.

Value mutation (`kernel_write_voxels`) writes the packed buffer in place: unlike NanoVDB grids,
values are writable from kernels. Topology stays fixed for the lifetime of a packed volume; the
host re-packs after structural edits (leaf allocation, prune, load).
"""

import quadrants as qd

_KEY_OFF = 1 << 19  # per-axis leaf coordinate offset; supports +-524288 leaves per axis


@qd.func
def xuvdb_pack_key(kx: qd.i64, ky: qd.i64, kz: qd.i64) -> qd.i64:
    return (kx + qd.i64(_KEY_OFF)) * qd.i64(1 << 40) + (ky + qd.i64(_KEY_OFF)) * qd.i64(1 << 20) + (
        kz + qd.i64(_KEY_OFF)
    )


@qd.func
def xuvdb_find_leaf(
    keys: qd.types.ndarray(ndim=1),
    n_leaves: int,
    kx: qd.i64,
    ky: qd.i64,
    kz: qd.i64,
) -> int:
    key = xuvdb_pack_key(kx, ky, kz)
    lo = 0
    hi = n_leaves
    while lo < hi:
        mid = (lo + hi) // 2
        if keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    found = lo
    if lo >= n_leaves or keys[lo] != key:
        found = -1
    return found


@qd.func
def xuvdb_voxel_value(
    keys: qd.types.ndarray(ndim=1),
    values: qd.types.ndarray(ndim=1),
    n_leaves: int,
    leaf_log2: int,
    background: float,
    xi: qd.i64,
    yi: qd.i64,
    zi: qd.i64,
) -> float:
    dim = 1 << leaf_log2
    leaf_idx = xuvdb_find_leaf(keys, n_leaves, xi >> leaf_log2, yi >> leaf_log2, zi >> leaf_log2)
    lx = xi & qd.i64(dim - 1)
    ly = yi & qd.i64(dim - 1)
    lz = zi & qd.i64(dim - 1)
    n = lx * qd.i64(dim * dim) + ly * qd.i64(dim) + lz
    value = background
    if leaf_idx >= 0:
        value = values[leaf_idx * qd.i64(dim * dim * dim) + n]
    return value


@qd.func
def xuvdb_sample_nearest(
    keys: qd.types.ndarray(ndim=1),
    values: qd.types.ndarray(ndim=1),
    n_leaves: int,
    leaf_log2: int,
    background: float,
    o_x: float,
    o_y: float,
    o_z: float,
    s_x: float,
    s_y: float,
    s_z: float,
    x: float,
    y: float,
    z: float,
) -> float:
    xi = qd.i64(qd.floor((x - o_x) / s_x + 0.5))
    yi = qd.i64(qd.floor((y - o_y) / s_y + 0.5))
    zi = qd.i64(qd.floor((z - o_z) / s_z + 0.5))
    return xuvdb_voxel_value(keys, values, n_leaves, leaf_log2, background, xi, yi, zi)


@qd.func
def xuvdb_sample_linear(
    keys: qd.types.ndarray(ndim=1),
    values: qd.types.ndarray(ndim=1),
    n_leaves: int,
    leaf_log2: int,
    background: float,
    o_x: float,
    o_y: float,
    o_z: float,
    s_x: float,
    s_y: float,
    s_z: float,
    x: float,
    y: float,
    z: float,
) -> float:
    cx = (x - o_x) / s_x
    cy = (y - o_y) / s_y
    cz = (z - o_z) / s_z
    bx = qd.i64(qd.floor(cx))
    by = qd.i64(qd.floor(cy))
    bz = qd.i64(qd.floor(cz))
    fx = cx - qd.floor(cx)
    fy = cy - qd.floor(cy)
    fz = cz - qd.floor(cz)
    out = qd.f32(0.0)
    for dx in range(2):
        wx = fx if dx == 1 else 1.0 - fx
        for dy in range(2):
            wy = fy if dy == 1 else 1.0 - fy
            for dz in range(2):
                wz = fz if dz == 1 else 1.0 - fz
                out += qd.f32(
                    wx * wy * wz
                ) * qd.f32(
                    xuvdb_voxel_value(keys, values, n_leaves, leaf_log2, background,
                                     bx + qd.i64(dx), by + qd.i64(dy), bz + qd.i64(dz))
                )
    return out


@qd.kernel
def kernel_sample_nearest(
    keys: qd.types.ndarray(ndim=1),
    values: qd.types.ndarray(ndim=1),
    n_leaves: int,
    leaf_log2: int,
    background: float,
    o_x: float,
    o_y: float,
    o_z: float,
    s_x: float,
    s_y: float,
    s_z: float,
    points: qd.types.ndarray(ndim=2),
    out: qd.types.ndarray(ndim=1),
):
    for i in range(points.shape[0]):
        out[i] = xuvdb_sample_nearest(
            keys, values, n_leaves, leaf_log2, background, o_x, o_y, o_z, s_x, s_y, s_z,
            points[i, 0], points[i, 1], points[i, 2],
        )


@qd.kernel
def kernel_sample_linear(
    keys: qd.types.ndarray(ndim=1),
    values: qd.types.ndarray(ndim=1),
    n_leaves: int,
    leaf_log2: int,
    background: float,
    o_x: float,
    o_y: float,
    o_z: float,
    s_x: float,
    s_y: float,
    s_z: float,
    points: qd.types.ndarray(ndim=2),
    out: qd.types.ndarray(ndim=1),
):
    for i in range(points.shape[0]):
        out[i] = xuvdb_sample_linear(
            keys, values, n_leaves, leaf_log2, background, o_x, o_y, o_z, s_x, s_y, s_z,
            points[i, 0], points[i, 1], points[i, 2],
        )


@qd.kernel
def kernel_sdf_normal(
    keys: qd.types.ndarray(ndim=1),
    values: qd.types.ndarray(ndim=1),
    n_leaves: int,
    leaf_log2: int,
    background: float,
    o_x: float,
    o_y: float,
    o_z: float,
    s_x: float,
    s_y: float,
    s_z: float,
    points: qd.types.ndarray(ndim=2),
    out: qd.types.ndarray(ndim=2),
):
    """Central-difference surface normal of the SDF field at each point (world-space, normalized)."""
    for i in range(points.shape[0]):
        x = points[i, 0]
        y = points[i, 1]
        z = points[i, 2]
        nx = xuvdb_sample_linear(keys, values, n_leaves, leaf_log2, background, o_x, o_y, o_z, s_x, s_y, s_z,
                                x + s_x, y, z) - xuvdb_sample_linear(keys, values, n_leaves, leaf_log2,
                                                                    background, o_x, o_y, o_z, s_x, s_y, s_z,
                                                                    x - s_x, y, z)
        ny = xuvdb_sample_linear(keys, values, n_leaves, leaf_log2, background, o_x, o_y, o_z, s_x, s_y, s_z,
                                x, y + s_y, z) - xuvdb_sample_linear(keys, values, n_leaves, leaf_log2,
                                                                    background, o_x, o_y, o_z, s_x, s_y, s_z,
                                                                    x, y - s_y, z)
        nz = xuvdb_sample_linear(keys, values, n_leaves, leaf_log2, background, o_x, o_y, o_z, s_x, s_y, s_z,
                                x, y, z + s_z) - xuvdb_sample_linear(keys, values, n_leaves, leaf_log2,
                                                                    background, o_x, o_y, o_z, s_x, s_y, s_z,
                                                                    x, y, z - s_z)
        norm = qd.sqrt(nx * nx + ny * ny + nz * nz)
        if norm > 1e-12:
            out[i, 0] = nx / norm
            out[i, 1] = ny / norm
            out[i, 2] = nz / norm
        else:
            out[i, 0] = 0.0
            out[i, 1] = 0.0
            out[i, 2] = 0.0


@qd.kernel
def kernel_write_voxels(
    keys: qd.types.ndarray(ndim=1),
    values: qd.types.ndarray(ndim=1),
    n_leaves: int,
    leaf_log2: int,
    o_x: float,
    o_y: float,
    o_z: float,
    s_x: float,
    s_y: float,
    s_z: float,
    points: qd.types.ndarray(ndim=2),
    new_values: qd.types.ndarray(ndim=1),
):
    """Overwrite the value of the voxel nearest each point (in-place; topology unchanged).

    The active mask is host-side state and is not touched: deactivating/reactivating voxels or
    allocating leaves requires re-packing on the host.
    """
    dim = 1 << leaf_log2
    for i in range(points.shape[0]):
        xi = qd.i64(qd.floor((points[i, 0] - o_x) / s_x + 0.5))
        yi = qd.i64(qd.floor((points[i, 1] - o_y) / s_y + 0.5))
        zi = qd.i64(qd.floor((points[i, 2] - o_z) / s_z + 0.5))
        leaf_idx = xuvdb_find_leaf(keys, n_leaves, xi >> leaf_log2, yi >> leaf_log2, zi >> leaf_log2)
        if leaf_idx >= 0:
            lx = xi & qd.i64(dim - 1)
            ly = yi & qd.i64(dim - 1)
            lz = zi & qd.i64(dim - 1)
            n = lx * qd.i64(dim * dim) + ly * qd.i64(dim) + lz
            values[leaf_idx * qd.i64(dim * dim * dim) + n] = new_values[i]
