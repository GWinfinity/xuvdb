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


def smooth(grid, iterations=1, alpha=0.5):
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

    bbox = grid.bbox()
    if bbox is None:
        return grid
    dlo = np.asarray(bbox[0], dtype=np.int64) - 1
    dhi = np.asarray(bbox[1], dtype=np.int64) + 1

    phi = _dense_view(grid, dlo, dhi)
    # target block = the active bounding box; everything else stays frozen
    t0 = tuple(int(bbox[0][a] - dlo[a]) for a in range(3))
    t1 = tuple(int(bbox[1][a] - dlo[a]) + 1 for a in range(3))
    new = phi.copy()
    for _ in range(iterations):
        c = phi[t0[0]:t1[0], t0[1]:t1[1], t0[2]:t1[2]]
        avg = (phi[t0[0] + 1:t1[0] + 1, t0[1]:t1[1], t0[2]:t1[2]]
               + phi[t0[0] - 1:t1[0] - 1, t0[1]:t1[1], t0[2]:t1[2]]
               + phi[t0[0]:t1[0], t0[1] + 1:t1[1] + 1, t0[2]:t1[2]]
               + phi[t0[0]:t1[0], t0[1] - 1:t1[1] - 1, t0[2]:t1[2]]
               + phi[t0[0]:t1[0], t0[1]:t1[1], t0[2] + 1:t1[2] + 1]
               + phi[t0[0]:t1[0], t0[1]:t1[1], t0[2] - 1:t1[2] - 1]) / 6.0
        new[t0[0]:t1[0], t0[1]:t1[1], t0[2]:t1[2]] = c + alpha * (avg - c)
        phi, new = new, phi  # swap buffers; `phi` always holds the current field

    _write_back(grid, phi, bbox, dlo)
    return grid


def _dense_view(grid, dlo, dhi):
    """Dense float64 slab [dlo, dhi] of the stored values; background outside."""
    shape = tuple(int(v) for v in (dhi - dlo + 1))
    vals = np.full(shape, float(grid.background), dtype=np.float64)
    for leaf, sl in grid._for_each_leaf_slice(dlo, dhi, create=False):
        dst = tuple(slice(int(leaf.origin[a] + sl[a].start - dlo[a]),
                          int(leaf.origin[a] + sl[a].stop - dlo[a])) for a in range(3))
        vals[dst] = leaf.values[sl]
    return vals


def _write_back(grid, phi, bbox, dlo):
    """Write the smoothed slab back over the active bounding box, leaf by leaf.

    Only the region the grid already covers ([bbox]) receives new values; the
    halo and every leaf outside it keep their previous contents. The active
    mask is not modified - smoothing moves the surface only second-order.
    """
    for leaf in list(grid._leaves.values()):
        # overlap of this leaf with the active bounding box, in global coords
        glo = np.maximum(leaf.origin, bbox[0])
        ghi = np.minimum(leaf.origin + grid.leaf_dim - 1, bbox[1])
        if np.any(ghi < glo):
            continue
        slab = tuple(slice(int(glo[a] - dlo[a]), int(ghi[a] - dlo[a]) + 1)
                     for a in range(3))
        loc = tuple(slice(int(glo[a] - leaf.origin[a]), int(ghi[a] - leaf.origin[a]) + 1)
                    for a in range(3))
        leaf.values[loc] = phi[slab]
        leaf.invalidate()
    # NOTE: no prune() here - the active mask is unchanged, and fringe leaves
    # that hold meaningful beyond-band values (which prune would discard) must
    # survive so repeated calls see the same far field.
