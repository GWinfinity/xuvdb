"""Hierarchical DDA ray marching over a `VdbGrid` for level-set surface hits.

Mirrors the role of the binary exit search in `genesis/utils/sdf.py`, but on the sparse tree: the
traversal walks voxel boundaries (Amanatides & Woo, voxel centers at integer indices) and skips
empty space at two granularities - unallocated leaf blocks are jumped straight past, and inside an
allocated leaf the ray is advanced to the leaf's cached active-bbox entry face (NanoVDB-style
per-node bbox culling), so inactive interior never costs DDA steps. The march itself is pure
Python scalar math; numpy appears only on hits (trilinear refinement).
"""

import math

import numpy as np


def ray_surface_hit(grid, origin, direction, tmax=None, isovalue=0.0, refine_steps=24, use_bbox=True):
    """First crossing of `isovalue` along the ray, for scalar (SDF-style) grids.

    Returns `(t, point_world, value)` at the refined crossing, or None when the ray leaves the
    active data (or exceeds `tmax`) without a sign change. Crossings are refined against trilinear
    samples, so the reported point is sub-voxel accurate. `use_bbox=False` falls back to the plain
    leaf-block walk (the v0.1 reference behavior, kept for benchmarking/parity tests).
    """
    if grid.is_vec:
        raise TypeError("ray marching is defined on scalar grids")
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(d))
    if norm == 0.0:
        raise ValueError("direction must be nonzero")
    d = d / norm
    if tmax is None:
        bbox = grid.bbox()
        tmax = 1e30 if bbox is None else float(2.0 * np.linalg.norm(grid.index_to_world(bbox[1]) - o)) + 1.0
    tmax = float(tmax)

    # pure-scalar march state: continuous index origin, index-space direction (t stays in world)
    # index-space quantities: c0 = R^T (o - t) / s, di = R^T d / s (R = grid rotation)
    sx, sy, sz = (float(v) for v in grid.voxel_size)
    R = grid.rotation
    o_idx = (o - grid.origin_world) @ R
    d_idx = d @ R
    c0 = (o_idx[0] / sx, o_idx[1] / sy, o_idx[2] / sz)
    di = (d_idx[0] / sx, d_idx[1] / sy, d_idx[2] / sz)
    log2, dim = grid.leaf_log2, grid.leaf_dim
    eps = 1e-6 * min(sx, sy, sz) / max(math.sqrt(di[0] ** 2 + di[1] ** 2 + di[2] ** 2), 1e-30)
    leaves = grid._leaves
    # world position of a voxel center: t + R @ (v * s); per-axis world steps for center_t
    wxs = (float(R[0, 0]) * sx, float(R[0, 1]) * sy, float(R[0, 2]) * sz)
    wys = (float(R[1, 0]) * sx, float(R[1, 1]) * sy, float(R[1, 2]) * sz)
    wzs = (float(R[2, 0]) * sx, float(R[2, 1]) * sy, float(R[2, 2]) * sz)
    oxw, oyw, ozw = (float(v) for v in grid.origin_world)

    def center_t(voxel):
        """World ray parameter at which the ray is closest to the voxel center's plane."""
        wx = oxw + voxel[0] * wxs[0] + voxel[1] * wxs[1] + voxel[2] * wxs[2] - o[0]
        wy = oyw + voxel[0] * wys[0] + voxel[1] * wys[1] + voxel[2] * wys[2] - o[1]
        wz = ozw + voxel[0] * wzs[0] + voxel[1] * wzs[1] + voxel[2] * wzs[2] - o[2]
        return wx * d[0] + wy * d[1] + wz * d[2]

    def leaf_entry(key):
        """Per-leaf, per-ray cached march plan: `(t_enter, t_exit)` of the active bbox, or None."""
        leaf = leaves.get(key)
        if leaf is None:
            return None
        if not use_bbox:
            return False  # marker: march this leaf voxel by voxel (reference walk)
        blo, bhi = leaf.active_bbox()
        if blo is None:
            return None
        t0, t1 = -math.inf, math.inf
        for a in range(3):
            dia = di[a]
            c0a = c0[a]
            base = key[a] << log2
            if dia == 0.0:
                fa = float(base + blo[a]) - 0.5
                fb = float(base + bhi[a]) + 0.5
                if c0a < fa or c0a > fb:
                    return None
            else:
                ta = (base + blo[a] - 0.5 - c0a) / dia
                tb = (base + bhi[a] + 0.5 - c0a) / dia
                if ta > tb:
                    ta, tb = tb, ta
                t0 = t0 if t0 > ta else ta
                t1 = t1 if t1 < tb else tb
        return None if t0 > t1 else (t0, t1)

    def block_exit(voxel):
        """World t at which the ray leaves the leaf block containing `voxel`, or None."""
        best = math.inf
        for a in range(3):
            if di[a] != 0.0:
                block_lo = voxel[a] >> log2
                boundary = (block_lo << log2) + (dim - 0.5 if di[a] > 0 else -0.5)
                best = min(best, (boundary - c0[a]) / di[a])
        return best if math.isfinite(best) else None

    t = 0.0
    cur = _voxel_at(*c0)
    first = leaves.get((cur[0] >> log2, cur[1] >> log2, cur[2] >> log2))
    prev_v = float(first.values[tuple(cur[a] - first.origin[a] for a in range(3))]) if first is not None \
        else float(grid.background)
    prev_tc = center_t(cur)

    for _ in range(1 << 22):  # hard cap on visited voxels/blocks
        key = (cur[0] >> log2, cur[1] >> log2, cur[2] >> log2)
        ent = leaf_entry(key)
        if ent is None or (ent is not False and t > ent[1]):
            # unallocated block, empty leaf, or the active bbox is behind us: jump the block
            t_skip = block_exit(cur)
            if t_skip is None or t_skip > tmax:
                return None
            t = t_skip + eps
            cur = _voxel_at(c0[0] + t * di[0], c0[1] + t * di[1], c0[2] + t * di[2])
            continue
        if ent is not False and t < ent[0]:
            # advance straight to the bbox entry face instead of marching inactive voxels
            t = ent[0] + eps
            cur = _voxel_at(c0[0] + t * di[0], c0[1] + t * di[1], c0[2] + t * di[2])
            continue
        leaf = leaves[key]
        v = float(leaf.values[cur[0] - leaf.origin[0], cur[1] - leaf.origin[1], cur[2] - leaf.origin[2]])
        tc = center_t(cur)
        if (prev_v - isovalue) * (v - isovalue) <= 0.0 and not (prev_v == v and prev_tc == tc):
            # refine between the two voxel centers, where the trilinear field equals the
            # center values, so the sign change brackets a true root
            t_star = _refine(grid, o, d, prev_tc, tc, isovalue, refine_steps)
            point = o + t_star * d
            return t_star, point, float(grid.sample_linear(point))
        prev_v, prev_tc = v, tc
        nxt, t_next = _dda_next(c0, di, cur, t)
        if t_next > tmax or not math.isfinite(t_next):
            return None
        cur, t = nxt, t_next
    return None


def _voxel_at(cx, cy, cz):
    # voxel k spans [k - 0.5, k + 0.5); ceil(c - 0.5) is its exact inverse, mapping a point that
    # sits exactly on a face to the voxel being entered (floor(c + 0.5) would map it back to the
    # one being left, which stalls block skips)
    return (int(math.ceil(cx - 0.5)), int(math.ceil(cy - 0.5)), int(math.ceil(cz - 0.5)))


def _dda_next(c0, di, cur, t):
    """Exit time and next voxel when leaving voxel `cur` (entered at world parameter `t`)."""
    best_t = math.inf
    best_axis = -1
    for a in range(3):
        if di[a] == 0.0:
            continue
        boundary = cur[a] + (0.5 if di[a] > 0 else -0.5)
        t_axis = (boundary - c0[a]) / di[a]
        if t_axis < t:  # float dust from the entry computation
            t_axis = t + 1e-12
        if t_axis < best_t:
            best_t, best_axis = t_axis, a
    nxt = list(cur)
    if best_axis >= 0:
        nxt[best_axis] += 1 if di[best_axis] > 0 else -1
    return tuple(nxt), best_t


def _refine(grid, o, d, t_lo, t_hi, isovalue, steps):
    """Bisection against trilinear samples between two boundary crossings."""
    v_lo = float(grid.sample_linear(o + t_lo * d)) - isovalue
    for _ in range(steps):
        t_mid = 0.5 * (t_lo + t_hi)
        v_mid = float(grid.sample_linear(o + t_mid * d)) - isovalue
        if v_lo * v_mid <= 0.0:
            t_hi = t_mid
        else:
            t_lo, v_lo = t_mid, v_mid
    return 0.5 * (t_lo + t_hi)
