"""Hierarchical DDA ray marching over a `VdbGrid` for level-set surface hits.

Mirrors the role of the binary exit search in `genesis/utils/sdf.py`, but on the sparse tree: the
traversal walks voxel boundaries (Amanatides & Woo, voxel centers at integer indices) and, whenever
the current voxel's leaf block is not allocated, jumps straight to that block's exit face - empty
space is skipped at leaf granularity, not voxel granularity.
"""

import math

import numpy as np


def ray_surface_hit(grid, origin, direction, tmax=None, isovalue=0.0, refine_steps=24):
    """First crossing of `isovalue` along the ray, for scalar (SDF-style) grids.

    Returns `(t, point_world, value)` at the refined crossing, or None when the ray leaves the
    active data (or exceeds `tmax`) without a sign change. Crossings are refined against trilinear
    samples, so the reported point is sub-voxel accurate.
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

    c0 = (o - grid.origin_world) / grid.voxel_size  # continuous index of the ray origin
    di = d / grid.voxel_size  # index-space ray direction (t stays in world units)
    log2, dim = grid.leaf_log2, grid.leaf_dim

    def center_t(voxel):
        """World ray parameter at which the ray is closest to the voxel center's plane."""
        return float(np.dot(grid.index_to_world(voxel) - o, d))

    t = 0.0
    cur = _voxel_at(c0)
    prev_v = _leaf_value(grid, cur) if _has_leaf(grid, cur) else grid.background
    prev_tc = center_t(cur)

    for _ in range(1 << 22):  # hard cap on visited voxels/blocks
        if not _has_leaf(grid, cur):
            t_skip = _block_exit_t(c0, di, cur, log2, dim)
            if t_skip is None or t_skip > tmax:
                return None
            # land strictly past the exit face: a point exactly on a face maps back into the
            # old block under the [k-0.5, k+0.5) partition, which would stall the walk
            eps = 1e-6 * float(np.min(grid.voxel_size)) / max(float(np.linalg.norm(di)), 1e-30)
            t = t_skip + eps
            cur = _voxel_at(c0 + t * di)
            continue
        v = _leaf_value(grid, cur)
        if (prev_v - isovalue) * (v - isovalue) <= 0.0 and not (prev_v == v and prev_tc == center_t(cur)):
            # refine between the two voxel centers, where the trilinear field equals the
            # center values, so the sign change brackets a true root
            t_star = _refine(grid, o, d, prev_tc, center_t(cur), isovalue, refine_steps)
            point = o + t_star * d
            return t_star, point, float(grid.sample_linear(point))
        prev_v, prev_tc = v, center_t(cur)
        nxt, t_next = _dda_next(c0, di, cur, t)
        if t_next > tmax or not math.isfinite(t_next):
            return None
        cur, t = nxt, t_next
    return None


def _voxel_at(c):
    # voxel k spans [k - 0.5, k + 0.5); ceil(c - 0.5) is its exact inverse, mapping a point that
    # sits exactly on a face to the voxel being entered (floor(c + 0.5) would map it back to the
    # one being left, which stalls block skips)
    return (int(math.ceil(c[0] - 0.5)), int(math.ceil(c[1] - 0.5)), int(math.ceil(c[2] - 0.5)))


def _has_leaf(grid, voxel):
    return (voxel[0] >> grid.leaf_log2, voxel[1] >> grid.leaf_log2, voxel[2] >> grid.leaf_log2) in grid._leaves


def _leaf_value(grid, voxel):
    return float(grid.get_value(voxel))


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


def _block_exit_t(c0, di, voxel, log2, dim):
    """World t at which the ray leaves the dim^3 leaf block containing `voxel`, or None.

    Voxels span [k - 0.5, k + 0.5) in continuous index space, so the block's faces sit half a voxel
    outside its first/last voxel centers.
    """
    best = math.inf
    for a in range(3):
        if di[a] == 0.0:
            continue
        block_lo = (voxel[a] >> log2) << log2
        boundary = block_lo + dim - 0.5 if di[a] > 0 else block_lo - 0.5
        best = min(best, (boundary - c0[a]) / di[a])
    return best if math.isfinite(best) else None


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
