"""Chunked-slab utilities: tile a big index region into bounded-memory chunks.

The dense-slab operators (``offset``, ``smooth``) materialize the active
bounding box as float64 working arrays, which is what caps their scale. The
helpers here split that region into axis-aligned chunks so every per-chunk
slab stays under a voxel budget; each chunk carries a halo for the stencil,
and lockstep extract -> update -> write-back per chunk keeps the result
identical to the monolithic slab (every voxel reads previous-iteration
neighbor values).
"""

import numpy as np


def tile_ranges(lo, hi, max_voxels, min_width=1):
    """Tile the inclusive index range [lo, hi] into <= max_voxels chunks.

    Splits along the longest axis into contiguous ranges; each returned chunk
    is an interior region - callers add their own halo. ``min_width`` forces
    a minimum interior length along the split axis (operators need enough
    interior around each chunk for the data their stencils reach). Falls back
    to a single chunk (the monolithic behaviour) when the range fits or
    cannot be split further.
    """
    lo = np.asarray(lo, dtype=np.int64)
    hi = np.asarray(hi, dtype=np.int64)
    lengths = hi - lo + 1
    total = int(np.prod(lengths))
    if total <= max_voxels:
        return [(lo.copy(), hi.copy())]
    axis = int(np.argmax(lengths))
    n = int(np.ceil(total / max_voxels))
    n = max(1, min(n, int(lengths[axis] // max(min_width, 1))))
    if n <= 1:
        return [(lo.copy(), hi.copy())]
    bounds = np.linspace(int(lo[axis]), int(hi[axis]) + 1, n + 1).astype(np.int64)
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        clo = lo.copy()
        chi = hi.copy()
        clo[axis] = a
        chi[axis] = b - 1
        out.append((clo, chi))
    return out


def write_region(grid, phi, active, lo, hi, dlo, leaves=None):
    """Write a chunk's dense slab back over its interior [lo, hi], leaf by leaf.

    Only the interior receives new data; the halo and neighbouring chunks keep
    their contents. ``active`` may be None (value-only write, e.g. smoothing).
    ``leaves`` may be a precomputed overlap list (see :func:`leaves_overlapping`)
    - passing it avoids scanning the whole leaf dict on billion-voxel grids.
    """
    if leaves is None:
        leaves = list(grid._leaves.values())
    for leaf in leaves:
        glo = np.maximum(leaf.origin, lo)
        ghi = np.minimum(leaf.origin + grid.leaf_dim - 1, hi)
        if np.any(ghi < glo):
            continue
        slab = tuple(slice(int(glo[a] - dlo[a]), int(ghi[a] - dlo[a]) + 1)
                     for a in range(3))
        loc = tuple(slice(int(glo[a] - leaf.origin[a]), int(ghi[a] - leaf.origin[a]) + 1)
                    for a in range(3))
        leaf.values[loc] = phi[slab]
        if active is not None:
            leaf.active[loc] = active[slab]
        leaf.invalidate()


def leaves_overlapping(grid, lo, hi):
    """Leaves overlapping the inclusive index range [lo, hi], via leaf-key ranges
    (O(overlapping) dict lookups instead of an O(n_leaves) scan)."""
    lg = grid.leaf_log2
    klo = np.asarray(lo, dtype=np.int64) >> lg
    khi = np.asarray(hi, dtype=np.int64) >> lg
    out = []
    for kx in range(int(klo[0]), int(khi[0]) + 1):
        for ky in range(int(klo[1]), int(khi[1]) + 1):
            for kz in range(int(klo[2]), int(khi[2]) + 1):
                leaf = grid._leaves.get((kx, ky, kz))
                if leaf is not None:
                    out.append(leaf)
    return out
