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


def offset(grid, distance, band=None, n_steps=None):
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

    bbox = grid.bbox()
    if bbox is None:
        return grid
    pad = int(np.ceil(abs(distance) / h_min)) + int(np.ceil(band_world / h_min)) + 3
    dlo = np.asarray(bbox[0], dtype=np.int64) - pad
    dhi = np.asarray(bbox[1], dtype=np.int64) + pad

    phi, active = _dense_view(grid, dlo, dhi)

    # Fill the near far field: breadth-first extrapolation sweeps so the voxels the
    # front can reach carry a meaningful signed distance. Unallocated / untouched
    # voxels read the background, whose sign is blind (positive "far outside" even
    # deep inside the surface) - without this the eroding front would sweep into
    # wrong-sign values. The surface advances at most |distance| (+1 safety voxel),
    # so that many layers of extrapolation suffice; beyond them the per-step band
    # extension maintains one voxel of meaningful values ahead of the front.
    bg = float(grid.background)
    known = active.copy()
    n_layers = int(np.ceil(abs(distance) / h_min)) + 2
    for _ in range(n_layers):
        if not _extend(phi, known, vs, bg).any():
            break
    active = np.abs(phi) < band_world

    for _ in range(n_steps):
        active = _step(phi, active, vs, bg, band_world, abs(dt), s)
    _write_back(grid, phi, active, dlo, dhi)
    return grid


def _dense_view(grid, dlo, dhi):
    """`(phi, active)` dense float64 views of the slab [dlo, dhi]; background outside."""
    shape = tuple(int(v) for v in (dhi - dlo + 1))
    vals = np.full(shape, float(grid.background), dtype=np.float64)
    act = np.zeros(shape, dtype=bool)
    for leaf, sl in grid._for_each_leaf_slice(dlo, dhi, create=False):
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
    best_h = np.ones(n)
    found = np.zeros(n, dtype=bool)
    for a in range(3):
        for t in (1, -1):
            sl = [slice(1, 1 + n[k]) for k in range(3)]
            sl[a] = slice(1 + t, 1 + t + n[a])
            v = P[tuple(sl)]
            kv = KP[tuple(sl)]
            take = kv & (np.abs(v) < np.abs(best))
            best = np.where(take, v, best)
            best_h = np.where(take, vs[a], best_h)
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


def _write_back(grid, phi, active, dlo, dhi):
    """Write the slab back leaf by leaf; deactivate everything the offset left outside."""
    dim = grid.leaf_dim
    lg = grid.leaf_log2
    bg = float(grid.background)

    # Leaves entirely outside the slab cannot hold band voxels of the offset surface
    # (the slab is padded by |distance| + band + 3), so their activity dies here.
    for leaf in list(grid._leaves.values()):
        if np.any(leaf.origin + dim - 1 < dlo) or np.any(leaf.origin > dhi):
            leaf.active[...] = False
            leaf.values[...] = bg
            leaf.invalidate()

    for kx in range(int(dlo[0]) >> lg, (int(dhi[0]) >> lg) + 1):
        for ky in range(int(dlo[1]) >> lg, (int(dhi[1]) >> lg) + 1):
            for kz in range(int(dlo[2]) >> lg, (int(dhi[2]) >> lg) + 1):
                key = (kx, ky, kz)
                lo_l = [int(max(dlo[a], (key[a] << lg)) - (key[a] << lg)) for a in range(3)]
                hi_l = [int(min(dhi[a], (key[a] << lg) + dim - 1) - (key[a] << lg)) for a in range(3)]
                src = tuple(slice(lo_l[a], hi_l[a] + 1) for a in range(3))
                dst = tuple(slice(lo_l[a] + (key[a] << lg) - int(dlo[a]),
                                  hi_l[a] + (key[a] << lg) - int(dlo[a]) + 1) for a in range(3))
                if key not in grid._leaves and not active[dst].any():
                    continue  # never materialize a fully-inactive leaf
                leaf = grid._get_or_create_leaf(key)
                leaf.values[...] = bg
                leaf.active[...] = False
                leaf.values[src] = phi[dst]
                leaf.active[src] = active[dst]
                leaf.invalidate()
    grid.prune()
