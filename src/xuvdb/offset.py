"""Narrow-band SDF offset (morphological dilation / erosion) for level-set grids.

The equivalent of OpenVDB's ``LevelSetFilter::offset``: move the zero level set by
a signed world-space distance by solving the eikonal PDE ``dphi/dt = sign(d) * |grad phi|``
with upwind (Godunov) differences on the active band, then re-stamping the band around
the new surface so activity follows it.

Implementation notes: the solver works on a dense slab cut around the active bounding
box (padded by ``|distance| + band + 3`` voxels, so everything the offset can reach is
inside the slab), advances with CFL-sized signed steps ``dt = distance / n_steps`` and,
per step, extends the field one voxel ahead of the motion before the upwind update so
the band never runs into stale far-field values. The slab is written back leaf by leaf
(vectorized slices); leaves left entirely outside the new band are deactivated and pruned.
"""

import numpy as np


def offset(grid, distance, band=None, n_steps=None, max_slab_voxels=None):
    """Offset a narrow-band level set by `distance` (world units), in place.

    Solves the eikonal PDE dphi/dt = sign(distance) * |grad phi| with upwind
    (Godunov) differences on the active band, CFL-stepped: each step advances
    phi by up to one voxel diagonal. Positive distance dilates, negative erodes.
    The band is re-stamped around the new zero level set so active voxels follow
    the surface. Without SDF reinitialization (see ROADMAP) repeated/large offsets
    drift from true distance — accuracy is good for |distance| <= a few voxels.
    Returns the grid.

    Parameters
    ----------
    grid : VdbGrid
        Scalar, axis-aligned narrow-band level set. Modified in place.
    distance : float
        World-space offset; positive dilates the surface, negative erodes it.
        ``distance == 0`` is a no-op.
    band : float, optional
        Narrow-band half-width in *voxel* units for the re-stamped active set
        (defaults to ``grid.background / mean(voxel_size)``, preserving the
        stamped band width). Accuracy requires ``|distance|`` to stay inside the
        band (plus a couple of voxels); larger offsets sweep through far-field
        voxels whose values are not meaningful.
    n_steps : int, optional
        Number of CFL steps (defaults to ``ceil(|distance| / (0.9*min(voxel_size)))``,
        i.e. each step moves the surface by at most ~0.9 voxels; every step is
        sub-cycled twice internally, so the effective advance is ~0.45 voxels).
    max_slab_voxels : int, optional
        Memory ceiling for the dense working slab, in voxels (default 64M,
        ~2.5GB of float64 temporaries). The padded active bounding box is
        processed in lockstep chunks of at most this many voxels, so peak memory
        stays bounded at billion-active scales; each chunk carries a 1-voxel
        halo and interiors are kept >= ``|distance| + band + 3`` so results are
        identical to the monolithic slab within the accuracy contract above.
    """
    if grid.is_vec:
        raise TypeError("offset is defined on scalar grids only")
    if not grid.is_axis_aligned:
        raise ValueError("offset requires an axis-aligned grid (identity rotation)")
    distance = float(distance)
    if not np.isfinite(distance):
        raise ValueError("distance must be finite")
    if distance == 0.0:
        return grid

    vs = grid.voxel_size.astype(np.float64)
    h_min = float(vs.min())
    mean_vs = float(np.mean(vs))
    if band is None:
        band = float(grid.background) / mean_vs
    band = float(band)
    if band <= 0.0:
        raise ValueError("band must be positive (voxel units)")
    band_world = band * mean_vs
    if n_steps is None:
        n_steps = int(np.ceil(abs(distance) / (0.9 * h_min)))
    n_steps = max(1, int(n_steps))
    # Each CFL step is sub-cycled twice: the first-order Godunov flux overshoots on
    # curved fronts by ~1 voxel at a full 0.9-voxel step, which breaks the <1.5-voxel
    # accuracy budget; at ~0.45 voxels/step the error saturates at the spatial limit.
    n_steps *= 2
    dt = abs(distance) / n_steps
    s = 1.0 if distance > 0.0 else -1.0  # dphi/dt = -s * |grad phi|: dilation shrinks phi
    if max_slab_voxels is None:
        max_slab_voxels = 64_000_000
    max_slab_voxels = int(max_slab_voxels)
    if max_slab_voxels < 1_000_000:
        raise ValueError("max_slab_voxels must be >= 1,000,000")

    bbox = grid.bbox()
    if bbox is None:
        return grid
    pad = int(np.ceil(abs(distance) / h_min)) + int(np.ceil(band_world / h_min)) + 3
    dlo = np.asarray(bbox[0], dtype=np.int64) - pad
    dhi = np.asarray(bbox[1], dtype=np.int64) + pad
    # interiors must hold the full reach of extrapolated data (|distance| + band)
    chunks = _tile_ranges(dlo, dhi, max_slab_voxels, min_width=pad)
    from .slab import leaves_overlapping
    chunk_leaves = [leaves_overlapping(grid, lo - 1, hi + 1) for lo, hi in chunks]

    bg = float(grid.background)
    n_layers = int(np.ceil(abs(distance) / h_min)) + 2
    # leaves entirely outside the slab cannot hold band voxels of the offset
    # surface; deactivate them up front (the band never leaves the slab)
    dim = grid.leaf_dim
    for leaf in list(grid._leaves.values()):
        if np.any(leaf.origin + dim - 1 < dlo) or np.any(leaf.origin > dhi):
            leaf.active[...] = False
            leaf.values[...] = bg
            leaf.invalidate()

    if len(chunks) == 1:
        # Fast path for the common small/medium grid: one chunk's slab stays in
        # memory across the whole solve, skipping the per-step extract/write-back
        # rounds below. Numerically identical to the chunked path: re-extraction
        # would rebuild the interior from the latest leaf state (which is exactly
        # the in-memory interior) and the halo from untouched leaf values (which
        # is exactly what the in-memory halo still holds).
        leaves = chunk_leaves[0]
        lo, hi = chunks[0]
        slo, shi = lo - 1, hi + 1
        phi, known = _dense_view(grid, slo, shi, leaves)
        for _ in range(n_layers):
            ext = _extend(phi, known, vs, bg)
            if not ext.any():
                break
            known |= ext
        active = np.abs(phi) < band_world
        for _ in range(n_steps):
            active = _step(phi, active, vs, bg, band_world, abs(dt), s)
        _write_region(grid, phi, active, lo, hi, slo, leaves)
        grid.prune()
        return grid

    # Fill the near far field: breadth-first extrapolation sweeps so the voxels
    # the front can reach carry a meaningful signed distance. Each chunk runs the
    # sweeps on its own slab (1-voxel halo); only the interior is written back,
    # so halo approximations never reach the result.
    for leaves, (lo, hi) in zip(chunk_leaves, chunks):
        slo, shi = lo - 1, hi + 1
        phi, known = _dense_view(grid, slo, shi, leaves)
        known = known.copy()
        for _ in range(n_layers):
            ext = _extend(phi, known, vs, bg)
            if not ext.any():
                break
            known |= ext
        active = np.abs(phi) < band_world
        _write_region(grid, phi, active, lo, hi, slo, leaves)

    # CFL steps, lockstep across chunks: every chunk extracts the previous
    # iteration's state (halo included), advances one step, writes its interior
    for _ in range(n_steps):
        for leaves, (lo, hi) in zip(chunk_leaves, chunks):
            slo, shi = lo - 1, hi + 1
            phi, active = _dense_view(grid, slo, shi, leaves)
            active = _step(phi, active, vs, bg, band_world, abs(dt), s)
            _write_region(grid, phi, active, lo, hi, slo, leaves)
    grid.prune()
    return grid


def _tile_ranges(lo, hi, max_voxels, min_width=1):
    from .slab import tile_ranges
    return tile_ranges(lo, hi, max_voxels, min_width)


def _write_region(grid, phi, active, lo, hi, dlo, leaves=None):
    from .slab import write_region
    write_region(grid, phi, active, lo, hi, dlo, leaves)


def _dense_view(grid, dlo, dhi, leaves=None):
    """`(phi, active)` dense float64 views of the slab [dlo, dhi]; background outside.

    ``leaves`` may be a precomputed overlap list (see slab.leaves_overlapping)."""
    shape = tuple(int(v) for v in (dhi - dlo + 1))
    vals = np.full(shape, float(grid.background), dtype=np.float64)
    act = np.zeros(shape, dtype=bool)
    if leaves is None:
        it = ((leaf, sl) for leaf, sl in grid._for_each_leaf_slice(dlo, dhi, create=False))
    else:
        dim = grid.leaf_dim
        def _slices():
            for leaf in leaves:
                llo = np.maximum(leaf.origin, dlo)
                lhi = np.minimum(leaf.origin + dim - 1, dhi)
                if np.any(lhi < llo):
                    continue
                sl = tuple(slice(int(llo[a] - leaf.origin[a]), int(lhi[a] - leaf.origin[a]) + 1)
                           for a in range(3))
                yield leaf, sl
        it = _slices()
    for leaf, sl in it:
        dst = tuple(slice(int(leaf.origin[a] + sl[a].start - dlo[a]),
                          int(leaf.origin[a] + sl[a].stop - dlo[a])) for a in range(3))
        vals[dst] = leaf.values[sl]
        act[dst] = leaf.active[sl]
    return vals, act


def _extend(phi, known, vs, bg):
    """One breadth-first SDF-extrapolation sweep from the `known` mask.

    Every unknown 6-neighbor of a known voxel gets phi(c) = phi(a) + sign(phi(a)) * h
    from the known neighbor `a` closest to the surface (smallest |phi(a)|). Returns the
    mask of voxels written; the caller ORs it into `known` for the next sweep.
    """
    n = phi.shape
    P = np.pad(phi, 1, mode="constant", constant_values=bg)
    KP = np.pad(known, 1, mode="constant", constant_values=False)
    best = np.full(n, np.inf)  # value of the known neighbor closest to the surface
    abs_best = np.full(n, np.inf)  # |best|, tracked in place to skip per-direction temporaries
    best_h = np.ones(n)
    found = np.zeros(n, dtype=bool)
    for a in range(3):
        for t in (1, -1):
            sl = [slice(1, 1 + n[k]) for k in range(3)]
            sl[a] = slice(1 + t, 1 + t + n[a])
            v = P[tuple(sl)]
            kv = KP[tuple(sl)]
            av = np.abs(v)
            take = kv & (av < abs_best)
            np.copyto(best, v, where=take)
            np.copyto(abs_best, av, where=take)
            np.copyto(best_h, vs[a], where=take)
            found |= kv
    ext = found & ~known
    phi[ext] = best[ext] + np.sign(best[ext]) * best_h[ext]
    return ext


def _relax_eikonal(phi, exact, sign0, h, n_sweeps, big=1.0e7):
    """Gauss-Seidel-style upwind relaxation of |grad psi| = 1 on psi = |phi|.

    `exact` voxels keep their stamped values (Dirichlet); every other voxel is
    updated from the per-axis minima of its 6 neighbors with the standard Godunov
    eikonal solve (one-sided, two-sided or three-sided quadratic, whichever is
    causal). Converges outward ~1 voxel per sweep; monotone (values only shrink).
    """
    psi = np.abs(phi)
    for _ in range(n_sweeps):
        P = np.pad(psi, 1, mode="constant", constant_values=big)
        ax = np.minimum(P[:-2, 1:-1, 1:-1], P[2:, 1:-1, 1:-1])
        ay = np.minimum(P[1:-1, :-2, 1:-1], P[1:-1, 2:, 1:-1])
        az = np.minimum(P[1:-1, 1:-1, :-2], P[1:-1, 1:-1, 2:])
        a1, a2, a3 = np.sort(np.stack([ax, ay, az]), axis=0)
        u = a1 + h  # one-sided
        u2 = 0.5 * (a1 + a2) + 0.5 * np.sqrt(np.maximum(2.0 * h * h - (a1 - a2) ** 2, 0.0))
        u = np.where(u <= a2, u, u2)  # two-sided
        s3 = a1 + a2 + a3
        u3 = (s3 + np.sqrt(np.maximum(s3 * s3 - 3.0 * (a1 * a1 + a2 * a2 + a3 * a3 - h * h), 0.0))) / 3.0
        u = np.minimum(np.where(u > a3, u3, u), big)  # three-sided, if causal
        psi = np.where(exact, psi, np.minimum(u, psi))
    phi[...] = sign0 * psi


def _step(phi, active, vs, bg, band_world, dt, s):
    """One CFL step: extend the band one voxel outward, then the Godunov-upwind update.

    `s = sign(distance)`: the zero level set advances along `s * grad phi`, i.e.
    phi_t = -s * |grad phi|; a positive distance (s > 0) dilates. Per axis the Godunov
    upwind flux of that Hamiltonian is max(s*D-back, 0)^2 + min(s*D-forward, 0)^2
    (backward differences where phi increases along the motion, forward where it
    decreases), summed over axes and square-rooted. Per step the band also grows one
    voxel ahead of the motion (`_extend`), so the active set follows the surface.
    """
    n = phi.shape
    ext = _extend(phi, active, vs, bg)
    update = active | ext

    # --- Godunov upwind |grad phi|.
    P = np.pad(phi, 1, mode="constant", constant_values=bg)
    c = P[(slice(1, 1 + n[0]), slice(1, 1 + n[1]), slice(1, 1 + n[2]))]
    g2 = None
    for a in range(3):
        sl_b = [slice(1, 1 + n[k]) for k in range(3)]
        sl_f = [slice(1, 1 + n[k]) for k in range(3)]
        sl_b[a] = slice(0, n[a])
        sl_f[a] = slice(2, n[a] + 2)
        dback = (c - P[tuple(sl_b)]) / vs[a]
        dfwd = (P[tuple(sl_f)] - c) / vs[a]
        gpos = np.maximum(s * dback, 0.0)
        gneg = np.minimum(s * dfwd, 0.0)
        g = gpos * gpos + gneg * gneg
        g2 = g if g2 is None else g2 + g
    grad = np.sqrt(g2)

    np.multiply(grad, -s * dt, out=grad)
    grad += phi
    np.copyto(phi, grad, where=update)
    return np.abs(phi) < band_world  # re-stamped band follows the new surface
