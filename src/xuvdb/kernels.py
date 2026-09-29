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


@qd.func
def xuvdb_block_exit_t(c0x: qd.f64, c0y: qd.f64, c0z: qd.f64,
                       dix: qd.f64, diy: qd.f64, diz: qd.f64,
                       vx: qd.i64, vy: qd.i64, vz: qd.i64,
                       log2: int, dim: int) -> qd.f64:
    """World t at which the ray leaves the leaf block containing (vx,vy,vz); 1e30 = no exit.

    Voxels span [k - 0.5, k + 0.5) in continuous index space, so block faces sit half a voxel
    outside the first/last voxel centers.
    """
    block_x = (vx >> log2) << log2
    block_y = (vy >> log2) << log2
    block_z = (vz >> log2) << log2
    best = qd.f64(1e30)
    if dix != 0.0:
        if dix > 0.0:
            best = qd.min(best, (block_x + dim - 0.5 - c0x) / dix)
        else:
            best = qd.min(best, (block_x - 0.5 - c0x) / dix)
    if diy != 0.0:
        if diy > 0.0:
            best = qd.min(best, (block_y + dim - 0.5 - c0y) / diy)
        else:
            best = qd.min(best, (block_y - 0.5 - c0y) / diy)
    if diz != 0.0:
        if diz > 0.0:
            best = qd.min(best, (block_z + dim - 0.5 - c0z) / diz)
        else:
            best = qd.min(best, (block_z - 0.5 - c0z) / diz)
    return best


@qd.kernel
def kernel_ray_surface_hit(
    keys: qd.types.ndarray(ndim=1),
    values: qd.types.ndarray(ndim=1),
    n_leaves: int,
    leaf_log2: int,
    background: qd.f64,
    bbox_lo: qd.types.ndarray(ndim=2),  # (n_leaves, 3) i32 local active-bbox min; lo > hi = empty
    bbox_hi: qd.types.ndarray(ndim=2),  # (n_leaves, 3) i32 local active-bbox max
    o_x: qd.f64, o_y: qd.f64, o_z: qd.f64,
    s_x: qd.f64, s_y: qd.f64, s_z: qd.f64,
    origins: qd.types.ndarray(ndim=2),  # (n_rays, 3) f64 world origins
    dirs: qd.types.ndarray(ndim=2),     # (n_rays, 3) f64 unit world directions
    tmax: qd.f64,
    isovalue: qd.f64,
    out: qd.types.ndarray(ndim=2),      # (n_rays, 6): flag, t, px, py, pz, value
    refine_steps: qd.template(),
    max_steps: qd.template(),
):
    """Batched kernel-side DDA mirroring `ray.py`: unallocated blocks and empty leaves are jumped,
    and inside a leaf the march advances straight to the leaf's cached active-bbox entry face. One
    march per ray, rays in parallel; sign changes between consecutive voxel-center samples are
    refined against trilinear samples by bisection."""
    for r in range(origins.shape[0]):
        ro_x, ro_y, ro_z = origins[r, 0], origins[r, 1], origins[r, 2]
        rd_x, rd_y, rd_z = dirs[r, 0], dirs[r, 1], dirs[r, 2]
        # index space: c0 = continuous ray origin, di = direction (t stays in world units)
        c0x, c0y, c0z = (ro_x - o_x) / s_x, (ro_y - o_y) / s_y, (ro_z - o_z) / s_z
        dix, diy, diz = rd_x / s_x, rd_y / s_y, rd_z / s_z
        di_len = qd.sqrt(dix * dix + diy * diy + diz * diz)
        s_min = qd.min(s_x, qd.min(s_y, s_z))
        eps_t = 1e-6 * s_min / qd.max(di_len, 1e-30)
        dim = 1 << leaf_log2

        t = qd.f64(0.0)
        vx = qd.i64(qd.ceil(c0x - 0.5))
        vy = qd.i64(qd.ceil(c0y - 0.5))
        vz = qd.i64(qd.ceil(c0z - 0.5))
        prev_valid = 0
        prev_v = qd.f64(background)
        prev_tc = qd.f64(0.0)
        hit = 0
        step = 0
        running = 1
        while running == 1:
            if step >= max_steps:
                running = 0
            else:
                step += 1
                leaf_idx = xuvdb_find_leaf(keys, n_leaves, vx >> leaf_log2, vy >> leaf_log2, vz >> leaf_log2)
                blo_x, bhi_x = qd.i32(1), qd.i32(0)
                blo_y, bhi_y = qd.i32(1), qd.i32(0)
                blo_z, bhi_z = qd.i32(1), qd.i32(0)
                t0 = qd.f64(0.0)
                have_bbox = 0
                if leaf_idx >= 0:
                    blo_x = bbox_lo[leaf_idx, 0]
                    bhi_x = bbox_hi[leaf_idx, 0]
                    blo_y = bbox_lo[leaf_idx, 1]
                    bhi_y = bbox_hi[leaf_idx, 1]
                    blo_z = bbox_lo[leaf_idx, 2]
                    bhi_z = bbox_hi[leaf_idx, 2]
                    if blo_x <= bhi_x and blo_y <= bhi_y and blo_z <= bhi_z:
                        # slab test against the active bbox (faces at voxel-center +/- 0.5)
                        base_x = (vx >> leaf_log2) << leaf_log2
                        base_y = (vy >> leaf_log2) << leaf_log2
                        base_z = (vz >> leaf_log2) << leaf_log2
                        t0 = qd.f64(-1e30)
                        t1 = qd.f64(1e30)
                        miss = 0
                        for a in qd.static(range(3)):
                            dia = dix if a == 0 else (diy if a == 1 else diz)
                            c0a = c0x if a == 0 else (c0y if a == 1 else c0z)
                            base = base_x if a == 0 else (base_y if a == 1 else base_z)
                            lo_a = qd.f64(blo_x if a == 0 else (blo_y if a == 1 else blo_z))
                            hi_a = qd.f64(bhi_x if a == 0 else (bhi_y if a == 1 else bhi_z))
                            fa_lo = base + lo_a - 0.5
                            fa_hi = base + hi_a + 0.5
                            if dia == 0.0:
                                if c0a < fa_lo or c0a > fa_hi:
                                    miss = 1
                            else:
                                ta = (fa_lo - c0a) / dia
                                tb = (fa_hi - c0a) / dia
                                if ta > tb:
                                    tmp = ta
                                    ta = tb
                                    tb = tmp
                                t0 = qd.max(t0, ta)
                                t1 = qd.min(t1, tb)
                        if miss == 0 and t0 <= t1:
                            have_bbox = 1

                if leaf_idx < 0 or have_bbox == 0:
                    # unallocated block, or a leaf with no active voxels / no bbox intersection
                    t_skip = xuvdb_block_exit_t(c0x, c0y, c0z, dix, diy, diz, vx, vy, vz, leaf_log2, dim)
                    if t_skip > tmax or t_skip >= 1e29:
                        running = 0
                    else:
                        t = t_skip + eps_t
                        vx = qd.i64(qd.ceil(c0x + t * dix - 0.5))
                        vy = qd.i64(qd.ceil(c0y + t * diy - 0.5))
                        vz = qd.i64(qd.ceil(c0z + t * diz - 0.5))
                        running = 1
                else:
                    if t < t0:
                        # advance straight to the bbox entry face, pushed a hair inside
                        t = t0 + eps_t
                        vx = qd.i64(qd.ceil(c0x + t * dix - 0.5))
                        vy = qd.i64(qd.ceil(c0y + t * diy - 0.5))
                        vz = qd.i64(qd.ceil(c0z + t * diz - 0.5))
                        running = 1
                    else:
                        # sample the voxel center; detect and refine a sign change
                        lx = vx & qd.i64(dim - 1)
                        ly = vy & qd.i64(dim - 1)
                        lz = vz & qd.i64(dim - 1)
                        n = lx * qd.i64(dim * dim) + ly * qd.i64(dim) + lz
                        v = qd.f64(background)
                        if leaf_idx >= 0:
                            v = qd.f64(values[leaf_idx * qd.i64(dim * dim * dim) + n])
                        tc = ((vx * s_x + o_x - ro_x) * rd_x + (vy * s_y + o_y - ro_y) * rd_y
                              + (vz * s_z + o_z - ro_z) * rd_z)
                        if prev_valid == 1 and (prev_v - isovalue) * (v - isovalue) <= 0.0 \
                                and not (prev_v == v and prev_tc == tc):
                            lo_t = prev_tc
                            hi_t = tc
                            px, py, pz = ro_x + lo_t * rd_x, ro_y + lo_t * rd_y, ro_z + lo_t * rd_z
                            v_lo = xuvdb_sample_linear(keys, values, n_leaves, leaf_log2, background,
                                                       o_x, o_y, o_z, s_x, s_y, s_z, px, py, pz) \
                                - qd.f32(isovalue)
                            for _r in range(refine_steps):
                                mid = 0.5 * (lo_t + hi_t)
                                px = ro_x + mid * rd_x
                                py = ro_y + mid * rd_y
                                pz = ro_z + mid * rd_z
                                v_mid = xuvdb_sample_linear(keys, values, n_leaves, leaf_log2, background,
                                                            o_x, o_y, o_z, s_x, s_y, s_z, px, py, pz) \
                                    - qd.f32(isovalue)
                                if v_lo * v_mid <= 0.0:
                                    hi_t = mid
                                else:
                                    lo_t = mid
                                    v_lo = v_mid
                            t_star = 0.5 * (lo_t + hi_t)
                            px = ro_x + t_star * rd_x
                            py = ro_y + t_star * rd_y
                            pz = ro_z + t_star * rd_z
                            out[r, 0] = 1.0
                            out[r, 1] = t_star
                            out[r, 2] = px
                            out[r, 3] = py
                            out[r, 4] = pz
                            out[r, 5] = xuvdb_sample_linear(keys, values, n_leaves, leaf_log2, background,
                                                            o_x, o_y, o_z, s_x, s_y, s_z, px, py, pz)
                            hit = 1
                            running = 0  # first crossing found: stop the march
                        prev_valid = 1
                        prev_v = v
                        prev_tc = tc
                        if hit == 0:
                            # Amanatides & Woo: exit the current voxel (explicit per axis)
                            best_t = qd.f64(1e30)
                            best_axis = -1
                            t_axis = qd.f64(0.0)
                            if dix != 0.0:
                                if dix > 0.0:
                                    t_axis = (vx + 0.5 - c0x) / dix
                                else:
                                    t_axis = (vx - 0.5 - c0x) / dix
                                if t_axis < t:
                                    t_axis = t + 1e-12
                                if t_axis < best_t:
                                    best_t = t_axis
                                    best_axis = 0
                            if diy != 0.0:
                                if diy > 0.0:
                                    t_axis = (vy + 0.5 - c0y) / diy
                                else:
                                    t_axis = (vy - 0.5 - c0y) / diy
                                if t_axis < t:
                                    t_axis = t + 1e-12
                                if t_axis < best_t:
                                    best_t = t_axis
                                    best_axis = 1
                            if diz != 0.0:
                                if diz > 0.0:
                                    t_axis = (vz + 0.5 - c0z) / diz
                                else:
                                    t_axis = (vz - 0.5 - c0z) / diz
                                if t_axis < t:
                                    t_axis = t + 1e-12
                                if t_axis < best_t:
                                    best_t = t_axis
                                    best_axis = 2
                            if best_axis == -1 or best_t > tmax or best_t >= 1e29:
                                running = 0  # degenerate direction, budget spent, or left the data
                            else:
                                if best_axis == 0:
                                    if dix > 0.0:
                                        vx += 1
                                    else:
                                        vx -= 1
                                elif best_axis == 1:
                                    if diy > 0.0:
                                        vy += 1
                                    else:
                                        vy -= 1
                                else:
                                    if diz > 0.0:
                                        vz += 1
                                    else:
                                        vz -= 1
                                t = best_t
                                running = 1
        if hit == 0:
            out[r, 0] = 0.0


@qd.func
def xuvdb_find_leaf_seg(keys: qd.types.ndarray(ndim=1), lo: int, hi: int,
                        kx: qd.i64, ky: qd.i64, kz: qd.i64) -> int:
    """Binary search restricted to keys[lo:hi] - the segment of one volume in a batch."""
    key = xuvdb_pack_key(kx, ky, kz)
    l = lo
    h = hi
    while l < h:
        mid = (l + h) // 2
        if keys[mid] < key:
            l = mid + 1
        else:
            h = mid
    found = l
    if l >= hi or keys[l] != key:
        found = -1
    return found


@qd.func
def xuvdb_voxel_value_seg(keys: qd.types.ndarray(ndim=1), values: qd.types.ndarray(ndim=1),
                          lo: int, hi: int, leaf_log2: int, background: qd.f64,
                          xi: qd.i64, yi: qd.i64, zi: qd.i64) -> qd.f64:
    dim = 1 << leaf_log2
    leaf_idx = xuvdb_find_leaf_seg(keys, lo, hi, xi >> leaf_log2, yi >> leaf_log2, zi >> leaf_log2)
    lx = xi & qd.i64(dim - 1)
    ly = yi & qd.i64(dim - 1)
    lz = zi & qd.i64(dim - 1)
    n = lx * qd.i64(dim * dim) + ly * qd.i64(dim) + lz
    value = background
    if leaf_idx >= 0:
        value = values[leaf_idx * qd.i64(dim * dim * dim) + n]
    return value


@qd.func
def xuvdb_sample_linear_seg(keys: qd.types.ndarray(ndim=1), values: qd.types.ndarray(ndim=1),
                            lo: int, hi: int, leaf_log2: int, background: qd.f64,
                            o_x: qd.f64, o_y: qd.f64, o_z: qd.f64,
                            s_x: qd.f64, s_y: qd.f64, s_z: qd.f64,
                            x: qd.f64, y: qd.f64, z: qd.f64) -> qd.f64:
    cx = (x - o_x) / s_x
    cy = (y - o_y) / s_y
    cz = (z - o_z) / s_z
    bx = qd.floor(cx)
    by = qd.floor(cy)
    bz = qd.floor(cz)
    fx = cx - bx
    fy = cy - by
    fz = cz - bz
    out = qd.f64(0.0)
    for dx in qd.static(range(2)):
        wx = fx if dx == 1 else 1.0 - fx
        for dy in qd.static(range(2)):
            wy = fy if dy == 1 else 1.0 - fy
            for dz in qd.static(range(2)):
                wz = fz if dz == 1 else 1.0 - fz
                out += wx * wy * wz * xuvdb_voxel_value_seg(
                    keys, values, lo, hi, leaf_log2, background,
                    qd.i64(bx) + qd.i64(dx), qd.i64(by) + qd.i64(dy), qd.i64(bz) + qd.i64(dz))
    return out


@qd.kernel
def kernel_sample_batch(keys: qd.types.ndarray(ndim=1),
                        values: qd.types.ndarray(ndim=1),
                        segs: qd.types.ndarray(ndim=1),        # (n_vol+1,) i32 volume segments
                        leaf_log2s: qd.types.ndarray(ndim=1),  # (n_vol,) i32
                        origins: qd.types.ndarray(ndim=2),     # (n_vol, 3) f64
                        steps: qd.types.ndarray(ndim=2),       # (n_vol, 3) f64 voxel sizes
                        backgrounds: qd.types.ndarray(ndim=1),  # (n_vol,) f64
                        vol_ids: qd.types.ndarray(ndim=1),     # (m,) i32
                        points: qd.types.ndarray(ndim=2),      # (m, 3) f32
                        out: qd.types.ndarray(ndim=1)):        # (m,) f32
    """Batched trilinear sampling across many packed volumes: each point picks its volume via
    `vol_ids`, the binary search is confined to that volume's key segment. One launch total."""
    for i in range(points.shape[0]):
        v = int(vol_ids[i])
        lo = int(segs[v])
        hi = int(segs[v + 1])
        out[i] = xuvdb_sample_linear_seg(
            keys, values, lo, hi, int(leaf_log2s[v]), qd.f64(backgrounds[v]),
            origins[v, 0], origins[v, 1], origins[v, 2],
            steps[v, 0], steps[v, 1], steps[v, 2],
            qd.f64(points[i, 0]), qd.f64(points[i, 1]), qd.f64(points[i, 2]))


@qd.func
def xuvdb_sample_quadratic(keys: qd.types.ndarray(ndim=1),
                           values: qd.types.ndarray(ndim=1),
                           n_leaves: int,
                           leaf_log2: int,
                           background: float,
                           o_x: float, o_y: float, o_z: float,
                           s_x: float, s_y: float, s_z: float,
                           x: float, y: float, z: float) -> float:
    """Triquadratic sample (3x3x3 quadratic B-spline; weights [0.5(1-u)^2, 0.5+u-u^2, 0.5u^2]
    per axis over taps floor(x)-1 .. floor(x)+1), matching the host `_sample_quadratic`."""
    cx, cy, cz = (x - o_x) / s_x, (y - o_y) / s_y, (z - o_z) / s_z
    bx, by, bz = qd.floor(cx), qd.floor(cy), qd.floor(cz)
    fx, fy, fz = cx - bx, cy - by, cz - bz
    out = qd.f32(0.0)
    wj = qd.f32(0.0)
    wk = qd.f32(0.0)
    wl = qd.f32(0.0)
    for j in qd.static(range(3)):
        if j == 0:
            wj = 0.5 * (1.0 - fx) * (1.0 - fx)
        elif j == 1:
            wj = 0.5 + fx - fx * fx
        else:
            wj = 0.5 * fx * fx
        for k in qd.static(range(3)):
            if k == 0:
                wk = 0.5 * (1.0 - fy) * (1.0 - fy)
            elif k == 1:
                wk = 0.5 + fy - fy * fy
            else:
                wk = 0.5 * fy * fy
            for l in qd.static(range(3)):
                if l == 0:
                    wl = 0.5 * (1.0 - fz) * (1.0 - fz)
                elif l == 1:
                    wl = 0.5 + fz - fz * fz
                else:
                    wl = 0.5 * fz * fz
                out += qd.f32(wj * wk * wl) * xuvdb_voxel_value(
                    keys, values, n_leaves, leaf_log2, background,
                    qd.i64(bx) - 1 + j, qd.i64(by) - 1 + k, qd.i64(bz) - 1 + l)
    return out


@qd.kernel
def kernel_sample_quadratic(keys: qd.types.ndarray(ndim=1),
                            values: qd.types.ndarray(ndim=1),
                            n_leaves: int,
                            leaf_log2: int,
                            background: float,
                            o_x: float, o_y: float, o_z: float,
                            s_x: float, s_y: float, s_z: float,
                            points: qd.types.ndarray(ndim=2),
                            out: qd.types.ndarray(ndim=1)):
    for i in range(points.shape[0]):
        out[i] = xuvdb_sample_quadratic(
            keys, values, n_leaves, leaf_log2, background, o_x, o_y, o_z, s_x, s_y, s_z,
            points[i, 0], points[i, 1], points[i, 2])


@qd.kernel
def kernel_reduce(mask: qd.types.ndarray(ndim=2),  # (n_leaves, dim^3/32) u32 active bits, z-fastest
                  values: qd.types.ndarray(ndim=1),
                  per_leaf: qd.types.ndarray(ndim=2),  # (n_leaves, 4) f64: sum, min, max, count
                  words: qd.template(),
                  dim3: qd.template()):
    """Per-leaf reduction over ACTIVE voxels (host combines the partials)."""
    for leaf_idx in range(mask.shape[0]):
        s = qd.f64(0.0)
        mn = qd.f64(1e30)
        mx = qd.f64(-1e30)
        c = 0
        base = qd.i64(leaf_idx) * dim3
        for w in range(words):
            word = mask[leaf_idx, w]
            if word != 0:
                for b in range(32):
                    if (word >> b) & 1:
                        v = qd.f64(values[base + w * 32 + b])
                        s += v
                        mn = qd.min(mn, v)
                        mx = qd.max(mx, v)
                        c += 1
        per_leaf[leaf_idx, 0] = s
        per_leaf[leaf_idx, 1] = mn
        per_leaf[leaf_idx, 2] = mx
        per_leaf[leaf_idx, 3] = qd.f64(c)

