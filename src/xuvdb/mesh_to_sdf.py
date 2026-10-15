"""Triangle mesh -> narrow-band signed level sets.

Converts a triangle soup into a `VdbGrid` of ``grid_class="level set"``:

* **Unsigned distance** - exact point-to-triangle distance (Ericson,
  *Real-Time Collision Detection*, closest-point-on-triangle), vectorized in
  numpy and accelerated by a uniform-grid broadphase so only voxels that can
  fall inside the narrow band ever touch triangle geometry.
* **Sign** - ray parity along +X: a voxel center is inside when the ray
  ``(x > x0, y, z)`` crosses an odd number of triangles. The Möller-Trumbore
  test collapses, for an axis-aligned ray, to a 2-D point-in-triangle test in
  the projected yz-plane plus a comparison of the intersection x. Points
  sitting exactly on a projected triangle edge are counted once per adjacent
  triangle, so shared edges stay parity-neutral.

Caveats (also noted on the functions): only isotropic voxel sizes are
supported, and the parity sign rule assumes a watertight (closed, manifold)
mesh; open or self-intersecting meshes can produce wrong signs in places.
"""

import numpy as np

from .tree import VdbGrid

_MAX_DOMAIN_VOXELS = 200_000_000  # refuse absurdly large dense staging arrays
_BARY_TOL = 1e-9  # barycentric boundary tolerance for the parity test
# Deterministic sub-voxel jitter applied to the parity ray origin in y/z. Voxel
# centers and mesh vertices often share grid alignment (e.g. an axis-aligned
# cube on a 0.01 grid), which puts rays exactly through vertices/edges and
# breaks crossing parity. Nudging the ray by ~1e-7 voxel removes the degeneracy
# without crossing the surface (sign is constant between crossings).
_RAY_JITTER = np.array([0.0, 1.6180339887498949e-7, 1.4142135623730951e-7])


# --------------------------------------------------------------------------- helpers

def _isotropic_voxel_size(voxel_size):
    """Accept a scalar (or length-1) voxel size; reject anisotropic sizes."""
    arr = np.asarray(voxel_size, dtype=np.float64).reshape(-1)
    if arr.size != 1:
        if arr.size == 3 and np.allclose(arr, arr[0]):
            arr = arr[:1]
        else:
            raise ValueError("mesh_to_level_set supports isotropic voxel_size only")
    vs = float(arr[0])
    if not np.isfinite(vs) or vs <= 0:
        raise ValueError("voxel_size must be a positive scalar")
    return vs


def _as_mesh(vertices, faces):
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(v) == 0 or len(f) == 0:
        raise ValueError("mesh must contain at least one triangle")
    if f.min() < 0 or f.max() >= len(v):
        raise ValueError("faces index out of range for vertices")
    return v, f


def _point_tri_dist2(p, a, b, c):
    """Squared point-to-triangle distances, Ericson closest-point regions.

    `p, a, b, c` are matching `(N, 3)` point/triangle-vertex arrays (one row per
    point-triangle pair); returns the `(N,)` squared distances.
    """
    ab = b - a
    ac = c - a
    ap = p - a
    bp = p - b
    cp = p - c
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    d_a = np.einsum("ij,ij->i", ap, ap)
    d_b = np.einsum("ij,ij->i", bp, bp)
    d_c = np.einsum("ij,ij->i", cp, cp)
    t_ab = d1 / np.where(d1 > d3, d1 - d3, 1.0)
    q_ab = ap - ab * t_ab[:, None]
    d_ab = np.einsum("ij,ij->i", q_ab, q_ab)
    t_ac = d2 / np.where(d2 > d6, d2 - d6, 1.0)
    q_ac = ap - ac * t_ac[:, None]
    d_ac = np.einsum("ij,ij->i", q_ac, q_ac)
    t_bc = d4 / np.where(d4 > d5, d4 - d5, 1.0)
    q_bc = (ac - ab) * t_bc[:, None] - bp
    d_bc = np.einsum("ij,ij->i", q_bc, q_bc)
    denom = np.where(va + vb + vc != 0.0, va + vb + vc, 1.0)
    q_f = ap - (ab * vb[:, None] + ac * vc[:, None]) / denom[:, None]
    d_f = np.einsum("ij,ij->i", q_f, q_f)
    return np.select(
        [(d1 <= 0) & (d2 <= 0),
         (d3 >= 0) & (d4 <= d3),
         (vc <= 0) & (d1 >= 0) & (d3 <= 0),
         (d6 >= 0) & (d5 <= d6),
         (vb <= 0) & (d2 >= 0) & (d6 <= 0),
         va <= 0],
        [d_a, d_b, d_ab, d_c, d_ac, d_bc],
        default=d_f,
    )


def _count_x_ray_crossings(p, a, b, c):
    """Number of +X ray crossings per point, for an axis-aligned ray.

    `p, a, b, c` are matching `(N, 3)` arrays (one row per point-triangle pair);
    returns a `(N,)` 0/1 crossing flag per pair. A crossing is counted when the
    yz-projected point lies inside the projected triangle (barycentric test
    with a small boundary tolerance, so shared edges are counted once per
    adjacent face) and the intersection x is strictly greater than the point's x.
    """
    v0y = b[:, 1] - a[:, 1]
    v0z = b[:, 2] - a[:, 2]
    v1y = c[:, 1] - a[:, 1]
    v1z = c[:, 2] - a[:, 2]
    v2y = p[:, 1] - a[:, 1]
    v2z = p[:, 2] - a[:, 2]
    d00 = v0y * v0y + v0z * v0z
    d01 = v0y * v1y + v0z * v1z
    d11 = v1y * v1y + v1z * v1z
    d20 = v2y * v0y + v2z * v0z
    d21 = v2y * v1y + v2z * v1z
    denom = d00 * d11 - d01 * d01
    valid = np.abs(denom) > 1e-12
    safe = np.where(valid, denom, 1.0)
    u = (d11 * d20 - d01 * d21) / safe
    v = (d00 * d21 - d01 * d20) / safe
    inside = valid & (u >= -_BARY_TOL) & (v >= -_BARY_TOL) & (u + v <= 1.0 + _BARY_TOL)
    x_hit = a[:, 0] + u * (b[:, 0] - a[:, 0]) + v * (c[:, 0] - a[:, 0])
    return (inside & (x_hit > p[:, 0])).astype(np.int64)


def _build_bins(box_lo, box_hi, grid_lo, cell_size, dims):
    """Uniform-grid broadphase: map cells to the items whose boxes overlap them.

    `box_lo`/`box_hi` are `(T, k)` float coordinates in the same space as
    `grid_lo` (voxel-index space); `dims` is the `(k,)` cell count. Returns
    `(cell_ids, cell_ptr, item_ids)` - CSR-like: item ids overlapping cell
    `cell_ids[j]` live in `item_ids[cell_ptr[j]:cell_ptr[j+1]]`.
    """
    dims = np.asarray(dims, dtype=np.int64)
    ndim = len(dims)
    lo = np.floor((box_lo - grid_lo) / cell_size).astype(np.int64)
    hi = np.floor((box_hi - grid_lo) / cell_size).astype(np.int64)
    np.clip(lo, 0, dims - 1, out=lo)
    np.clip(hi, 0, dims - 1, out=hi)
    span = hi - lo + 1  # (T, ndim)
    sizes = np.prod(span, axis=1)
    offsets = np.zeros(len(sizes) + 1, dtype=np.int64)
    np.cumsum(sizes, out=offsets[1:])
    total = int(offsets[-1])
    # Vectorized enumeration of every (item, cell) overlap: for item t, the
    # cells of its box are the lexicographic range [0, sizes[t]).
    q = np.arange(total, dtype=np.int64) - np.repeat(offsets[:-1], sizes)
    t_of = np.repeat(np.arange(len(sizes), dtype=np.int64), sizes)
    flat = np.zeros(total, dtype=np.int64)
    rem = q
    for k in range(ndim):
        stride = np.prod(span[t_of][:, k + 1:], axis=1) if k + 1 < ndim else np.ones(total, dtype=np.int64)
        coord = lo[t_of, k] + rem // stride
        rem = rem % stride
        flat = flat * dims[k] + coord
    order = np.argsort(flat, kind="stable")
    flat = flat[order]
    items = t_of[order]
    uniq, counts = np.unique(flat, return_counts=True)
    ptr = np.zeros(len(uniq) + 1, dtype=np.int64)
    np.cumsum(counts, out=ptr[1:])
    return uniq, ptr, items


def _group_points_by_cell(points, grid_lo, cell_size, dims):
    """Sort point ids by their containing cell; mirror of `_build_bins` output."""
    dims = np.asarray(dims, dtype=np.int64)
    coords = np.floor((points - grid_lo) / cell_size).astype(np.int64)
    np.clip(coords, 0, dims - 1, out=coords)
    flat = coords[:, 0]
    for k in range(1, len(dims)):
        flat = flat * dims[k] + coords[:, k]
    order = np.argsort(flat, kind="stable")
    uniq, counts = np.unique(flat[order], return_counts=True)
    ptr = np.zeros(len(uniq) + 1, dtype=np.int64)
    np.cumsum(counts, out=ptr[1:])
    return uniq, ptr, order


def _candidate_pairs(point_bins, item_bins):
    """All (point, item) index pairs that share a broadphase cell.

    Returns `(pair_pt, pair_item)` int arrays with rows grouped by point
    (unsorted within that); reduce with `_segment_reduce`.
    """
    p_uniq, p_ptr, p_order = point_bins
    i_uniq, i_ptr, i_items = item_bins
    pos = np.searchsorted(i_uniq, p_uniq)
    hit = (pos < len(i_uniq)) & (i_uniq[np.minimum(pos, len(i_uniq) - 1)] == p_uniq)
    cells = np.nonzero(hit)[0]
    p_counts = (p_ptr[cells + 1] - p_ptr[cells]).astype(np.int64)
    i_counts = (i_ptr[pos[cells] + 1] - i_ptr[pos[cells]]).astype(np.int64)
    pair_counts = p_counts * i_counts
    offsets = np.zeros(len(cells) + 1, dtype=np.int64)
    np.cumsum(pair_counts, out=offsets[1:])
    pair_pt = np.empty(int(offsets[-1]), dtype=np.int64)
    pair_item = np.empty(int(offsets[-1]), dtype=np.int64)
    for row, c in enumerate(cells):
        pids = p_order[p_ptr[c]:p_ptr[c + 1]]
        itids = i_items[i_ptr[pos[c]]:i_ptr[pos[c] + 1]]
        o = int(offsets[row])
        n = int(pair_counts[row])
        pair_pt[o:o + n] = np.repeat(pids, len(itids))
        pair_item[o:o + n] = np.tile(itids, len(pids))
    return pair_pt, pair_item


def _segment_reduce(ufunc, values, keys, n_keys, identity):
    """Reduce `values` per integer segment id in `keys` (CSR-style row reduce)."""
    order = np.argsort(keys, kind="stable")
    k_sorted = keys[order]
    starts = np.flatnonzero(np.r_[True, k_sorted[1:] != k_sorted[:-1]])
    reduced = ufunc.reduceat(values[order], starts)
    out = np.full(n_keys, identity, dtype=values.dtype)
    out[k_sorted[starts]] = reduced
    return out


_PAIR_CHUNK = 4_000_000  # bound peak memory of the flat pair computations


def _binned_min_distance(points, tri_verts, dims, band_cells, cell_size, grid_lo):
    """Exact narrow-band point-to-mesh distances via broadphase + Ericson.

    Returns per-point min distance in index units; `inf` where no triangle
    comes within `band_cells` (those voxels cannot be active). Triangle boxes
    are expanded by `band_cells` so coverage is exact. Candidate pairs are
    built per broadphase cell and processed in cell groups: a cheap
    point-vs-AABB squared-distance test prunes each group, and only the
    survivors take the exact (expensive) point-triangle test.
    """
    lo = tri_verts.min(axis=1) - band_cells  # (T, 3) band-expanded for coverage
    hi = tri_verts.max(axis=1) + band_cells
    i_uniq, i_ptr, i_items = _build_bins(lo, hi, grid_lo, cell_size, dims)
    p_uniq, p_ptr, p_order = _group_points_by_cell(points, grid_lo, cell_size, dims)
    pos = np.searchsorted(i_uniq, p_uniq)
    hit = (pos < len(i_uniq)) & (i_uniq[np.minimum(pos, len(i_uniq) - 1)] == p_uniq)

    dist = np.full(len(points), np.inf)
    band2 = float(band_cells) ** 2
    cells = np.nonzero(hit)[0]
    for g in range(0, len(cells), 1024):
        group = cells[g:g + 1024]
        # concatenated cross product (point, triangle) pairs for the group
        p_counts = (p_ptr[group + 1] - p_ptr[group]).astype(np.int64)
        i_counts = (i_ptr[pos[group] + 1] - i_ptr[pos[group]]).astype(np.int64)
        pair_counts = p_counts * i_counts
        if pair_counts.sum() == 0:
            continue
        offsets = np.zeros(len(group) + 1, dtype=np.int64)
        np.cumsum(pair_counts, out=offsets[1:])
        total = int(offsets[-1])
        pair_pt = np.empty(total, dtype=np.int64)
        pair_item = np.empty(total, dtype=np.int64)
        for row, c in enumerate(group):
            pids = p_order[p_ptr[c]:p_ptr[c + 1]]
            itids = i_items[i_ptr[pos[c]]:i_ptr[pos[c] + 1]]
            o = int(offsets[row])
            n = int(pair_counts[row])
            pair_pt[o:o + n] = np.repeat(pids, len(itids))
            pair_item[o:o + n] = np.tile(itids, len(pids))

        # stage 1: squared distance to each triangle's (band-expanded) AABB
        p = points[pair_pt]
        d = (np.maximum(lo[pair_item] - p, 0.0)
             + np.maximum(p - hi[pair_item], 0.0))
        d2 = np.einsum("ij,ij->i", d, d)
        keep = d2 <= band2
        if not keep.any():
            continue
        pt_k = pair_pt[keep]
        it_k = pair_item[keep]
        # stage 2: exact point-to-triangle distance on the survivors
        d2e = _point_tri_dist2(points[pt_k], tri_verts[it_k, 0],
                               tri_verts[it_k, 1], tri_verts[it_k, 2])
        chunk_best = _segment_reduce(np.minimum, np.sqrt(d2e), pt_k,
                                     len(points), np.inf)
        np.minimum(dist, chunk_best, out=dist)
    return dist


def _binned_parity(points, tri_verts, dims, cell_size, grid_lo):
    """Per-point +X ray-crossing parity via a 2-D (y, z) broadphase."""
    points = points + _RAY_JITTER  # avoid exactly-on-vertex ray degeneracies
    yz_dims = dims[1:]
    lo = tri_verts.min(axis=1)[:, 1:]  # (T, 2)
    hi = tri_verts.max(axis=1)[:, 1:]
    item_bins = _build_bins(lo, hi, grid_lo[1:], cell_size, yz_dims)
    point_bins = _group_points_by_cell(points[:, 1:], grid_lo[1:], cell_size, yz_dims)
    pair_pt, pair_item = _candidate_pairs(point_bins, item_bins)

    counts = np.zeros(len(points), dtype=np.int64)
    for s in range(0, len(pair_pt), _PAIR_CHUNK):
        e = min(s + _PAIR_CHUNK, len(pair_pt))
        pt = pair_pt[s:e]
        it = pair_item[s:e]
        flags = _count_x_ray_crossings(points[pt], tri_verts[it, 0],
                                       tri_verts[it, 1], tri_verts[it, 2])
        counts += _segment_reduce(np.add, flags, pt, len(points), 0)
    return counts


# --------------------------------------------------------------------------- public API

def mesh_to_level_set(vertices, faces, voxel_size, band=3.0, origin_world=(0.0, 0.0, 0.0),
                      name="mesh", _fill_interior=True):
    """Triangle mesh -> narrow-band signed level set grid.

    Unsigned distance is the exact point-to-triangle distance (Ericson's
    closest-point-on-triangle), vectorized in chunks; the sign comes from +X
    ray parity (an odd number of triangle crossings means inside, so the SDF
    is negative in the interior). Only voxels with ``|sdf| <= band * voxel_size``
    are activated; everything else stays inactive with the background value
    ``band * voxel_size``.

    Parameters
    ----------
    vertices : (n, 3) float array-like
        Mesh vertex positions in world coordinates.
    faces : (m, 3) int array-like
        Triangle vertex indices.
    voxel_size : positive float
        Isotropic voxel edge length (anisotropic sizes are rejected).
    band : float
        Half-width of the narrow band, in voxels.
    origin_world : (3,) float
        World position of voxel index (0, 0, 0).
    name : str
        Grid name.

    Returns
    -------
    VdbGrid
        ``grid_class="level set"``, ``background = band * voxel_size``.

    Notes
    -----
    The parity sign rule assumes a watertight mesh; open or self-intersecting
    meshes may get wrong signs. Rays passing exactly through a vertex can also
    flip sign on a measure-zero set of voxels. The staging domain spans the
    mesh bbox expanded by ``(band + 1) * voxel_size``; domains above 2e8 voxels
    raise ``ValueError`` - lower `band` or coarsen `voxel_size`.
    """
    vs = _isotropic_voxel_size(voxel_size)
    ow = np.broadcast_to(np.asarray(origin_world, dtype=np.float64).reshape(-1), (3,)).copy()
    v, f = _as_mesh(vertices, faces)
    band = float(band)
    if not np.isfinite(band) or band <= 0:
        raise ValueError("band must be a positive number of voxels")
    bw = band * vs
    margin = bw + vs

    # Work in voxel-index units: index = (world - origin_world) / voxel_size.
    v_idx = (v - ow) / vs
    idx_lo = np.floor(v_idx.min(axis=0) - margin / vs).astype(np.int64)
    idx_hi = np.ceil(v_idx.max(axis=0) + margin / vs).astype(np.int64)
    dims = (idx_hi - idx_lo + 1).astype(np.int64)
    if np.any(dims <= 0) or int(np.prod(dims)) > _MAX_DOMAIN_VOXELS:
        raise ValueError(
            f"mesh level set domain of {int(np.prod(dims))} voxels is too large "
            f"(limit {_MAX_DOMAIN_VOXELS}); reduce band or coarsen voxel_size")

    tri_verts = v_idx[f]  # (T, 3, 3)

    # Candidate preselection: only voxels within band+1 of a triangle's tight
    # box can be active. Mark tight-box occupancy on the domain grid, dilate by
    # the band, and restrict the expensive distance stage to those voxels -
    # the full bbox^3 domain is never touched point-by-point.
    band_cells = band + 0.5
    tight_lo = tri_verts.min(axis=1)
    tight_hi = tri_verts.max(axis=1)
    t_uniq, _, _ = _build_bins(tight_lo, tight_hi, idx_lo.astype(np.float64),
                               1.0, dims)
    occ = np.zeros(tuple(dims), dtype=bool)
    occ[tuple(np.unravel_index(t_uniq, tuple(dims)))] = True
    for _ in range(int(np.ceil(band_cells)) + 1):
        grown = occ
        shifted = np.zeros_like(occ)
        shifted[1:, :, :] = occ[:-1, :, :]
        shifted[:-1, :, :] |= occ[1:, :, :]
        shifted[:, 1:, :] |= occ[:, :-1, :]
        shifted[:, :-1, :] |= occ[:, 1:, :]
        shifted[:, :, 1:] |= occ[:, :, :-1]
        shifted[:, :, :-1] |= occ[:, :, 1:]
        occ = shifted
    cand = np.nonzero(occ)
    points = (np.stack(cand, axis=1).astype(np.float64) + idx_lo)

    # Exact unsigned distance, broadphase-limited to the potential narrow band.
    dist_cells = max(1.0, float(dims.max()) / 128.0)
    dist = _binned_min_distance(points, tri_verts, dims, band_cells, dist_cells,
                                idx_lo.astype(np.float64)) * vs

    # Scatter candidate results back onto the full domain (flat, z-fastest).
    n_dom = int(np.prod(dims))
    cand_ids = np.ravel_multi_index(cand, tuple(dims))
    dist_dom = np.full(n_dom, np.inf)
    dist_dom[cand_ids] = dist
    active_flat = dist_dom <= bw
    signed = dist_dom.copy()
    if active_flat.any():
        # Sign by +X ray parity; only active voxels need a sign.
        act_ids = np.nonzero(active_flat)[0]
        # recover world-index coords of active voxels
        ai = np.unravel_index(act_ids, tuple(dims))
        act_pts = np.stack(ai, axis=1).astype(np.float64) + idx_lo
        parity_cell = max(2.0, float(dims.max()) / 32.0)
        crossings = _binned_parity(act_pts, tri_verts, dims, parity_cell,
                                   idx_lo.astype(np.float64))
        signed[act_ids] = np.where((crossings & 1) == 1, -dist_dom[act_ids],
                                   dist_dom[act_ids])

    # Fill the interior: voxels that parity never touched keep +dist, which would
    # read as "outside" deep inside the mesh and make marching cubes see a second
    # (spurious) surface at the inner band edge. The whole active band is a
    # watertight barrier (the unsigned distance field is continuous), so
    # flood-fill the exterior, blocked by ALL active voxels; unreached passable
    # voxels are enclosed, hence inside, and get the negative signed distance.
    # OpenVDB's meshToVolume stores the same valid-negative interior beyond the
    # active band; `_fill_interior=False` opts out (dilated shell semantics).
    if _fill_interior:
        blocked = active_flat.reshape(tuple(dims))
        passable = ~blocked
        exterior = np.zeros(tuple(dims), dtype=bool)
        exterior[0, :, :] = passable[0, :, :]
        exterior[-1, :, :] = passable[-1, :, :]
        exterior[:, 0, :] = passable[:, 0, :]
        exterior[:, -1, :] = passable[:, -1, :]
        exterior[:, :, 0] = passable[:, :, 0]
        exterior[:, :, -1] = passable[:, :, -1]
        for _ in range(int(dims.max())):
            shifted = np.zeros_like(exterior)
            shifted[1:, :, :] = exterior[:-1, :, :]
            shifted[:-1, :, :] |= exterior[1:, :, :]
            shifted[:, 1:, :] |= exterior[:, :-1, :]
            shifted[:, :-1, :] |= exterior[:, 1:, :]
            shifted[:, :, 1:] |= exterior[:, :, :-1]
            shifted[:, :, :-1] |= exterior[:, :, 1:]
            grown = shifted & passable
            if grown.sum() == exterior.sum():
                break
            exterior = grown
        interior_flat = (~exterior & passable).reshape(-1)
        depth = np.where(np.isfinite(signed[interior_flat]),
                         np.abs(signed[interior_flat]), bw)
        signed[interior_flat] = -depth
    else:
        interior_flat = np.zeros(n_dom, dtype=bool)

    values = np.full(tuple(dims), bw, dtype=np.float32)
    active = active_flat.reshape(tuple(dims))
    values[...] = signed.reshape(tuple(dims)).astype(np.float32)
    values[~active & ~interior_flat.reshape(tuple(dims))] = bw
    grid = VdbGrid.from_dense(values, origin=tuple(int(i) for i in idx_lo),
                              voxel_size=vs, origin_world=ow, background=bw,
                              name=name, grid_class="level set", active=active)
    # Interior blocks that from_dense skipped (no active voxels) still carry the
    # negative fill values - store them in inactive leaves so the grid behaves
    # like a stamped level set deep inside (OpenVDB meshToVolume does the same;
    # without this, marching cubes would see a spurious second surface at the
    # inner band edge).
    ll = grid.leaf_log2
    dim = grid.leaf_dim
    vals = values
    for kx in range(int(idx_lo[0] >> ll), (int(idx_hi[0]) >> ll) + 1):
        for ky in range(int(idx_lo[1] >> ll), (int(idx_hi[1]) >> ll) + 1):
            for kz in range(int(idx_lo[2] >> ll), (int(idx_hi[2]) >> ll) + 1):
                key = (kx, ky, kz)
                if key in grid._leaves:
                    continue
                block_lo = np.array([kx, ky, kz]) << ll
                data_lo = np.maximum(block_lo, idx_lo)
                data_hi = np.minimum(block_lo + dim - 1, idx_hi)
                src = tuple(slice(int(data_lo[a] - idx_lo[a]),
                                  int(data_hi[a] - idx_lo[a]) + 1) for a in range(3))
                if not data_lo[0] <= data_hi[0]:
                    continue
                block = vals[src]
                if not (block != bw).any():
                    continue
                leaf = grid._get_or_create_leaf(key)
                dst = tuple(slice(int(data_lo[a] - block_lo[a]),
                                  int(data_hi[a] - block_lo[a]) + 1) for a in range(3))
                leaf.values[dst] = block
                leaf.active[dst] = active[src]
                leaf.invalidate()
    return grid


def dilated_mesh_shell(vertices, faces, half_width, voxel_size,
                       origin_world=(0.0, 0.0, 0.0), name="mesh_shell"):
    """Level set of a mesh dilated by `half_width` (OpenVDB's
    ``createLevelSetDilatedMesh`` "mesh shell").

    The zero crossing sits at distance `half_width` outside the mesh surface;
    the narrow band follows the shell with a 3-voxel margin on each side, so
    sampled values approximate the signed distance to the dilated mesh. This
    is equivalent to shifting ``mesh_to_level_set``'s SDF by ``-half_width``.

    Sign caveats of `mesh_to_level_set` apply (watertight meshes only).
    """
    vs = _isotropic_voxel_size(voxel_size)
    half_width = float(half_width)
    if not np.isfinite(half_width):
        raise ValueError("half_width must be finite")
    band = 3.0 + abs(half_width) / vs
    grid = mesh_to_level_set(vertices, faces, vs, band=band,
                             origin_world=origin_world, name=name,
                             _fill_interior=False)
    shift = np.float32(half_width)
    for leaf in grid.leaves():
        leaf.values[...] = leaf.values - shift
        leaf.invalidate()
    grid.background = float(grid.background) - half_width
    return grid
