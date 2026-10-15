"""Narrow-band level-set smoothing for level-set grids.

The equivalent of OpenVDB's ``LevelSetFilter::smooth``: relax the stored SDF
values with a discrete Laplacian (Jacobi) iteration,

    phi <- phi + alpha * (mean of the 6 face neighbors - phi),

which is a convex combination for ``alpha`` in (0, 1] and therefore
unconditionally stable. The zero level set moves only second-order (surface
rounding/sharpening of curvature irregularities); the active band is left
untouched - no re-stamping, no topology change.

Implementation notes: works on a dense slab around the active bounding box with
a one-voxel halo. Only voxels inside the active bounding box are relaxed (Jacobi,
double-buffered); everything outside - the halo and all inactive stored values
beyond the box - is Dirichlet-frozen at the extracted values for the whole run,
which keeps the far field blind to the iteration count and makes chained calls
(``smooth(smooth(g, 1), 1)``) exactly equivalent to one call with the summed
iteration count. The slab is written back leaf by leaf; leaves outside the box
are untouched and the original active mask is preserved.
"""

import numpy as np


def smooth(grid, iterations=1, alpha=0.5, max_slab_voxels=None):
    """Smooth a narrow-band level set in place (OpenVDB ``LevelSetFilter::smooth``).

    Each iteration relaxes the stored value of every voxel inside the active
    bounding box towards the mean of its 6 face neighbors by ``alpha`` (a convex
    combination for 0 < alpha <= 1, so the iteration is unconditionally stable
    and values stay bounded). Sharp creases round off; already-smooth regions
    barely change. The active mask, the band width and the leaf structure are
    preserved; voxels outside the active bounding box (halo and far field) are
    frozen at their extracted values, so the semantics do not depend on how
    many iterations are requested per call.

    Parameters
    ----------
    grid : VdbGrid
        Scalar, axis-aligned level set. Modified in place.
    iterations : int
        Number of relaxation sweeps (OpenVDB's ``iterations``). Values <= 0
        are a no-op.
    alpha : float
        Relaxation weight in (0, 1]; 1.0 applies the full Laplacian step.
    max_slab_voxels : int, optional
        Memory ceiling for the dense working slab, in voxels (default 64M,
        ~1GB of float64 temporaries). The active bounding box is swept in
        lockstep chunks of at most this many voxels (1-voxel halo), so peak
        memory stays bounded at billion-active scales. Chunks are exact: every
        voxel still reads previous-iteration neighbor values, so results are
        identical to the monolithic slab.

    Returns
    -------
    VdbGrid
        The same grid, for chaining.
    """
    if grid.is_vec:
        raise TypeError("smooth is defined on scalar grids only")
    if not grid.is_axis_aligned:
        raise ValueError("smooth requires an axis-aligned grid (identity rotation)")
    iterations = int(iterations)
    if iterations < 1:
        return grid
    if not (0.0 < float(alpha) <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    if max_slab_voxels is None:
        max_slab_voxels = 64_000_000
    max_slab_voxels = int(max_slab_voxels)
    if max_slab_voxels < 1_000_000:
        raise ValueError("max_slab_voxels must be >= 1,000,000")

    bbox = grid.bbox()
    if bbox is None:
        return grid
    dlo_full = np.asarray(bbox[0], dtype=np.int64)
    dhi_full = np.asarray(bbox[1], dtype=np.int64)

    from .slab import tile_ranges, write_region, leaves_overlapping
    chunks = tile_ranges(dlo_full, dhi_full, max_slab_voxels, min_width=1)
    chunk_leaves = [leaves_overlapping(grid, lo - 1, hi + 1) for lo, hi in chunks]

    # Jacobi sweeps, lockstep across chunks: every chunk extracts the previous
    # iteration's state with a 1-voxel halo, relaxes its interior, writes it
    # back. Interior voxels always read previous-iteration neighbor values, so
    # the chunked sweep is identical to the monolithic double-buffered sweep.
    for _ in range(iterations):
        for leaves, (lo, hi) in zip(chunk_leaves, chunks):
            slo, shi = lo - 1, hi + 1
            phi = _dense_view(grid, slo, shi, leaves)
            c = phi[1:-1, 1:-1, 1:-1]
            avg = (phi[2:, 1:-1, 1:-1] + phi[:-2, 1:-1, 1:-1]
                   + phi[1:-1, 2:, 1:-1] + phi[1:-1, :-2, 1:-1]
                   + phi[1:-1, 1:-1, 2:] + phi[1:-1, 1:-1, :-2]) / 6.0
            phi[1:-1, 1:-1, 1:-1] = c + alpha * (avg - c)
            write_region(grid, phi, None, lo, hi, slo, leaves)
    return grid


def _dense_view(grid, dlo, dhi, leaves=None):
    """Dense float64 slab [dlo, dhi] of the stored values; background outside.

    ``leaves`` may be a precomputed overlap list (see slab.leaves_overlapping)."""
    shape = tuple(int(v) for v in (dhi - dlo + 1))
    vals = np.full(shape, float(grid.background), dtype=np.float64)
    if leaves is None:
        leaf_slices = grid._for_each_leaf_slice(dlo, dhi, create=False)
    else:
        dim = grid.leaf_dim
        def _gen():
            for leaf in leaves:
                llo = np.maximum(leaf.origin, dlo)
                lhi = np.minimum(leaf.origin + dim - 1, dhi)
                if np.any(lhi < llo):
                    continue
                sl = tuple(slice(int(llo[a] - leaf.origin[a]), int(lhi[a] - leaf.origin[a]) + 1)
                           for a in range(3))
                yield leaf, sl
        leaf_slices = _gen()
    for leaf, sl in leaf_slices:
        dst = tuple(slice(int(leaf.origin[a] + sl[a].start - dlo[a]),
                          int(leaf.origin[a] + sl[a].stop - dlo[a])) for a in range(3))
        vals[dst] = leaf.values[sl]
    return vals
