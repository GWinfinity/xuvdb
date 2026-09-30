"""The XUVDB sparse volume tree: a mutable, two-level voxel structure on the quadrants kernel.

The design targets the gap between OpenVDB and NanoVDB: OpenVDB trees are fully mutable but live on
the CPU behind a C++ library, while NanoVDB grids are GPU-resident but read-only. A XUVDB grid keeps a
flat, dictionary-keyed set of dense leaf blocks that can be created, filled, carved and pruned from
Python at any time, and whose packed value buffers can additionally be written by quadrants kernels
(see `kernels.py` and `gpu.py`) without a host round trip.

Layout: the root is a hash map from leaf-block origin to `Leaf`; each leaf owns one dense
`(dim, dim, dim)` value array plus an active mask. `dim = 2**leaf_log2` (default 16). Memory layout
follows the OpenVDB leaf convention - index `n = x * dim**2 + y * dim + z`, i.e. z varies fastest -
so leaf data converts to and from `.vdb` files without transposes.

Index-to-world convention matches OpenVDB linear maps: voxel (0,0,0) sits at `origin_world`, and
`world = index * voxel_size + origin_world` (voxel centers at integer indices).
"""

import numpy as np

DEFAULT_LEAF_LOG2 = 4

GRID_CLASSES = ("unknown", "level set", "fog volume", "staggered")


class Leaf:
    """One dense `dim^3` block of voxels with a per-voxel active mask.

    `active_bbox()` and `value_range()` cache per-leaf derived state; every write to `active` (or
    to values that feed the stats) must call `leaf.invalidate()` - the mutation sites are all in
    `tree.py`/`io.py`, greppable by `leaf.invalidate()`.
    """

    __slots__ = ("origin", "values", "active", "_bbox", "_minmax")

    def __init__(self, origin, dim, value_shape, background):
        self.origin = np.asarray(origin, dtype=np.int64).reshape(3)
        self.values = np.full((dim, dim, dim) + value_shape, background, dtype=np.float32)
        self.active = np.zeros((dim, dim, dim), dtype=bool)
        self._bbox = None
        self._minmax = None

    @property
    def dim(self):
        return self.values.shape[0]

    @property
    def n_active(self):
        return int(self.active.sum())

    def invalidate(self):
        """Drop cached derived state (active bbox, value range); call after any mask edit."""
        self._bbox = None
        self._minmax = None

    def active_bbox(self):
        """`(lo, hi)` local bounds of active voxels (both `None` when the leaf is empty), cached."""
        if self._bbox is None:
            xs, ys, zs = np.nonzero(self.active)
            if xs.size == 0:
                self._bbox = (None, None)
            else:
                self._bbox = (np.array([xs.min(), ys.min(), zs.min()], dtype=np.int64),
                              np.array([xs.max(), ys.max(), zs.max()], dtype=np.int64))
        return self._bbox

    def value_range(self):
        """`(vmin, vmax)` over active voxels (scalar grids), or `None` when the leaf is empty."""
        if self._minmax is None:
            picked = self.values[self.active]
            if picked.size == 0:
                self._minmax = (None, None)
            else:
                self._minmax = (float(picked.min()), float(picked.max()))
        return self._minmax


class VdbGrid:
    """A named sparse voxel grid.

    Supported value types are float16, float32, float64 and 3-channel float32 (`vec3`);
    `background` is the value returned outside the active region. Structural edits (`set_value`,
    `fill_box`, `stamp_sphere`, CSG ops) allocate or free leaves on demand; `prune` drops leaves
    that carry no active voxels.

    The index->world map is the rigid affine `world = R @ (index * voxel_size) + origin_world`
    with `rotation` an orthonormal 3x3 matrix (default identity). The quadrants kernels only
    support axis-aligned grids (`rotation is identity`).
    """

    def __init__(
        self,
        dtype=np.float32,
        background=0.0,
        voxel_size=(1.0, 1.0, 1.0),
        origin_world=(0.0, 0.0, 0.0),
        leaf_log2=DEFAULT_LEAF_LOG2,
        name="grid",
        grid_class="unknown",
        rotation=None,
    ):
        dtype = np.dtype(dtype)
        if dtype == np.float32:
            self.value_shape, self.type_code = (), 0
        elif dtype == np.float16:
            self.value_shape, self.type_code = (), 3
        elif dtype == np.float64:
            self.value_shape, self.type_code = (), 1
        elif dtype == np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32)]):
            self.value_shape, self.type_code = (3,), 2
        else:
            raise TypeError("dtype must be float16, float32, float64 or the vec3 float32 record")
        self.dtype = dtype

        background = np.asarray(background, dtype=np.float64).reshape(-1)
        if self.is_vec_shape(self.value_shape):
            self.background = np.broadcast_to(background, (3,)).astype(np.float32).copy()
        else:
            self.background = float(background[0])

        if not (2 <= int(leaf_log2) <= 6):
            raise ValueError("leaf_log2 must be in [2, 6]")
        self.leaf_log2 = int(leaf_log2)
        self.leaf_dim = 1 << self.leaf_log2

        self.voxel_size = np.broadcast_to(np.asarray(voxel_size, dtype=np.float64).reshape(-1), (3,)).copy()
        if np.any(self.voxel_size <= 0):
            raise ValueError("voxel_size components must be positive")
        self.origin_world = np.broadcast_to(np.asarray(origin_world, dtype=np.float64).reshape(-1), (3,)).copy()

        if rotation is None:
            self.rotation = np.eye(3)
        else:
            self.rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3).copy()
            if not np.allclose(self.rotation @ self.rotation.T, np.eye(3), atol=1e-8):
                raise ValueError("rotation must be an orthonormal 3x3 matrix")
        self.is_axis_aligned = bool(np.allclose(self.rotation, np.eye(3)))

        self.name = str(name)
        if grid_class not in GRID_CLASSES:
            raise ValueError(f"unsupported grid_class: {grid_class!r}; use one of {GRID_CLASSES}")
        self.grid_class = grid_class
        self._leaves = {}

    @staticmethod
    def is_vec_shape(value_shape):
        return value_shape == (3,)

    # ------------------------------------------------------------------ basic properties

    @property
    def is_vec(self):
        return self.is_vec_shape(self.value_shape)

    @property
    def n_leaves(self):
        return len(self._leaves)

    leaf_count = property(lambda self: len(self._leaves))  # readability alias

    @property
    def active_voxel_count(self):
        return int(sum(leaf.n_active for leaf in self._leaves.values()))

    def leaves(self):
        """Leaves in canonical (origin-sorted) order; the order every serializer uses."""
        return [self._leaves[key] for key in sorted(self._leaves)]

    def iter_voxels(self):
        """Yield (coord, value) for every active voxel, leaf by leaf."""
        for leaf in self.leaves():
            ox, oy, oz = leaf.origin
            xs, ys, zs = np.nonzero(leaf.active)
            for x, y, z in zip(xs, ys, zs):
                yield (int(ox + x), int(oy + y), int(oz + z)), leaf.values[x, y, z]

    def active_indices(self):
        """`(n, 3)` int64 array of every active voxel's index, leaf-order (vectorized per leaf)."""
        chunks = []
        for leaf in self.leaves():
            xs, ys, zs = np.nonzero(leaf.active)
            chunks.append(np.stack([leaf.origin[0] + xs, leaf.origin[1] + ys, leaf.origin[2] + zs], axis=1))
        if not chunks:
            return np.zeros((0, 3), dtype=np.int64)
        return np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]

    def active_values(self):
        """`(n,)` / `(n, 3)` array of every active voxel's value, matching `active_indices` order."""
        chunks = [leaf.values[leaf.active] for leaf in self.leaves()]
        if not chunks:
            return np.zeros((0,) + self.value_shape, dtype=self._numpy_value_dtype())
        return np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]

    def probe_batch(self, indices):
        """Vectorized probe for an `(n, 3)` index array: `((n,)[, 3])` values + `(n,)` active flags.

        Unallocated voxels read as `(background, False)`; the per-leaf gather is vectorized, only
        the grouping by leaf key is a Python loop.
        """
        idxs = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
        vals = np.full((len(idxs),) + self.value_shape, self.background, dtype=self._numpy_value_dtype())
        act = np.zeros(len(idxs), dtype=bool)
        rows_by_key = {}
        for r, key in enumerate(map(tuple, idxs >> self.leaf_log2)):
            rows_by_key.setdefault(key, []).append(r)
        for key, rows in rows_by_key.items():
            leaf = self._leaves.get(key)
            if leaf is None:
                continue
            rows = np.asarray(rows, dtype=np.int64)
            local = idxs[rows] - leaf.origin
            vals[rows] = leaf.values[local[:, 0], local[:, 1], local[:, 2]]
            act[rows] = leaf.active[local[:, 0], local[:, 1], local[:, 2]]
        return vals, act

    def bbox(self):
        """Index-space integer (min, max) bounding box of active voxels, or None if empty."""
        lo = hi = None
        for leaf in self._leaves.values():
            llo, lhi = leaf.active_bbox()
            if llo is None:
                continue
            leaf_lo, leaf_hi = leaf.origin + llo, leaf.origin + lhi
            lo = leaf_lo if lo is None else np.minimum(lo, leaf_lo)
            hi = leaf_hi if hi is None else np.maximum(hi, leaf_hi)
        if lo is None:
            return None
        return lo, hi

    def value_range(self):
        """`(vmin, vmax)` over active voxels of a scalar grid, or None when no active voxels.

        Cached per leaf (see `Leaf.value_range`); O(n_leaves) to combine after the first call.
        """
        if self.is_vec:
            raise TypeError("value_range is defined on scalar grids only")
        lo = hi = None
        for leaf in self._leaves.values():
            lmin, lmax = leaf.value_range()
            if lmin is None:
                continue
            lo = lmin if lo is None else min(lo, lmin)
            hi = lmax if hi is None else max(hi, lmax)
        return None if lo is None else (lo, hi)

    def copy(self):
        grid = VdbGrid(
            dtype=self.dtype,
            background=self.background,
            voxel_size=self.voxel_size,
            origin_world=self.origin_world,
            leaf_log2=self.leaf_log2,
            name=self.name,
            grid_class=self.grid_class,
            rotation=self.rotation,
        )
        for key, leaf in self._leaves.items():
            new = Leaf(leaf.origin, leaf.dim, self.value_shape, self.background)
            new.values[...] = leaf.values
            new.active[...] = leaf.active
            grid._leaves[key] = new
        return grid

    # ------------------------------------------------------------------ leaf bookkeeping

    def _leaf_key(self, ijk):
        return (int(ijk[0]) >> self.leaf_log2, int(ijk[1]) >> self.leaf_log2, int(ijk[2]) >> self.leaf_log2)

    def _get_or_create_leaf(self, key):
        leaf = self._leaves.get(key)
        if leaf is None:
            leaf = Leaf(np.asarray(key) << self.leaf_log2, self.leaf_dim, self.value_shape, self.background)
            self._leaves[key] = leaf
        return leaf

    # ------------------------------------------------------------------ point access

    def set_value(self, ijk, value, active=True):
        """Set one voxel (creating its leaf on demand) and mark it active."""
        leaf = self._get_or_create_leaf(self._leaf_key(ijk))
        local = (int(ijk[0]) - leaf.origin[0], int(ijk[1]) - leaf.origin[1], int(ijk[2]) - leaf.origin[2])
        leaf.values[local] = value
        leaf.active[local] = bool(active)
        leaf.invalidate()

    def get_value(self, ijk):
        """Value at a voxel index; the background outside allocated leaves."""
        leaf = self._leaves.get(self._leaf_key(ijk))
        if leaf is None:
            return self.background
        local = (int(ijk[0]) - leaf.origin[0], int(ijk[1]) - leaf.origin[1], int(ijk[2]) - leaf.origin[2])
        return leaf.values[local]

    def probe(self, ijk):
        """Return (value, active) at a voxel index."""
        leaf = self._leaves.get(self._leaf_key(ijk))
        if leaf is None:
            return self.background, False
        local = (int(ijk[0]) - leaf.origin[0], int(ijk[1]) - leaf.origin[1], int(ijk[2]) - leaf.origin[2])
        return leaf.values[local], bool(leaf.active[local])

    # ------------------------------------------------------------------ bulk edits

    def _for_each_leaf_slice(self, ijk_min, ijk_max, create=True):
        """Yield (leaf, local_slice) covering the integer box [ijk_min, ijk_max]."""
        lo = np.minimum(ijk_min, ijk_max)
        hi = np.maximum(ijk_min, ijk_max)
        key_lo, key_hi = lo >> self.leaf_log2, hi >> self.leaf_log2
        for kx in range(key_lo[0], key_hi[0] + 1):
            for ky in range(key_lo[1], key_hi[1] + 1):
                for kz in range(key_lo[2], key_hi[2] + 1):
                    key = (kx, ky, kz)
                    leaf = self._get_or_create_leaf(key) if create else self._leaves.get(key)
                    if leaf is None:
                        continue
                    leaf_lo = np.maximum(lo, leaf.origin) - leaf.origin
                    leaf_hi = np.minimum(hi, leaf.origin + self.leaf_dim - 1) - leaf.origin
                    yield leaf, tuple(slice(leaf_lo[a], leaf_hi[a] + 1) for a in range(3))

    def fill_box(self, ijk_min, ijk_max, value, active=True):
        """Fill the axis-aligned integer index box [ijk_min, ijk_max] with a constant value."""
        for leaf, sl in self._for_each_leaf_slice(np.asarray(ijk_min, dtype=np.int64),
                                                  np.asarray(ijk_max, dtype=np.int64)):
            leaf.values[sl] = value
            leaf.active[sl] = bool(active)
            leaf.invalidate()

    def stamp_sphere(self, center, radius, value=None, band=3.0):
        """Stamp an analytic sphere (world-space `center`).

        With `value` given, voxels inside the sphere get that constant (a fog-volume style stamp;
        repeated fog stamps overwrite). Without `value`, this is a level-set stamp: values compose
        by MIN-union with the existing field (same semantics as `csg(other, 'union')`), so repeated
        SDF stamps build a true CSG union - two spheres stamped on one grid intersect cleanly
        instead of overwriting each other. The active set is the union of the existing band and
        the new sphere's `|distance| <= band * mean(voxel_size)` band.
        """
        center = np.asarray(center, dtype=np.float64).reshape(3)
        vs = self.voxel_size
        center_i = self.world_to_index(center)
        r_world = float(radius)
        sdf = value is None
        band_world = float(band) * float(np.mean(vs))
        # Index-space bounds generous enough to contain the band / the solid sphere.
        reach = vs if self.is_axis_aligned else np.full(3, vs.min())  # rotated: conservative bound
        reach_i = (r_world + (band_world if sdf else 0.0)) / reach + 1.0
        lo = np.floor(center_i - reach_i).astype(np.int64)
        hi = np.ceil(center_i + reach_i).astype(np.int64)
        for leaf, sl in self._for_each_leaf_slice(lo, hi):
            axes = [leaf.origin[a] + np.arange(sl[a].start, sl[a].stop) for a in range(3)]
            wx, wy, wz = _slice_world_positions(self, leaf, sl)
            dist = np.sqrt((wx - center[0]) ** 2 + (wy - center[1]) ** 2 + (wz - center[2]) ** 2)
            if sdf:
                # SDF stamps compose by MIN-union (identical to `csg(other, 'union')`): the
                # existing narrow band stays active, the new sphere's band joins it. Fog stamps
                # (a `value` given) overwrite - a constant fill has no union to compose.
                d = (dist - r_world).astype(leaf.values.dtype)
                leaf.values[sl] = np.minimum(leaf.values[sl], d)
                leaf.active[sl] = leaf.active[sl] | (np.abs(d) <= band_world)
            else:
                leaf.values[sl] = value
                leaf.active[sl] = dist <= r_world
            leaf.invalidate()

    def scatter_particles(self, points, h=None, weights=None, kernel="cubic"):
        """Splat particles onto the grid as a fog/density volume (additive SPH-style rasterization).

        Each particle accumulates `weights[i] * W(r)` into every voxel within its support, with
        W normalized to unit integral in 3D: `kernel='cubic'` is the SPH cubic spline, `'linear'` a
        trilinear hat; both have support radius `2*h` (`h` defaults to two voxel sizes). Repeated
        calls accumulate; voxels touched by a nonzero kernel weight become active. This is the
        particle -> volume bridge for SPH liquids: density export for volume rendering, per-phase
        mixture fractions (one grid per phase), or velocity rasterization (pass `(n, 3)` weights on
        a vec3 grid).

        Python loops over particles with vectorized leaf-slice kernels inside - adequate for
        thousands of particles; batch large particle sets by leaf for production use.
        """
        if kernel not in ("cubic", "linear"):
            raise ValueError("kernel must be 'cubic' or 'linear'")
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        n = len(points)
        if h is None:
            h = 2.0 * float(np.mean(self.voxel_size))
        h = float(h)
        support = 2.0 * h
        vs = self.voxel_size
        if weights is None:
            weights = np.ones(n)
        weights = np.asarray(weights, dtype=np.float64)
        if self.is_vec:
            if weights.ndim == 0:
                weights = np.full((n, 3), float(weights))
            elif weights.ndim == 1 and weights.size == 3 and n == 1:
                weights = weights.reshape(1, 3)
            elif weights.ndim == 1 and weights.size == n:
                weights = np.broadcast_to(weights[:, None], (n, 3))
            elif weights.shape != (n, 3):
                raise ValueError("vec3 grids need per-particle scalar or (n, 3) weights")
        elif weights.ndim == 0:
            weights = np.full(n, float(weights))
        elif weights.ndim != 1 or len(weights) != n:
            raise ValueError("weights must supply one scalar per particle")

        reach = vs if self.is_axis_aligned else np.full(3, vs.min())
        reach_i = support / reach + 1.0
        for i in range(n):
            center_i = self.world_to_index(points[i])
            lo = np.floor(center_i - reach_i).astype(np.int64)
            hi = np.ceil(center_i + reach_i).astype(np.int64)
            for leaf, sl in self._for_each_leaf_slice(lo, hi):
                dist = _slice_world_distance(self, leaf, sl, points[i])
                q = dist / h
                if kernel == "cubic":
                    w = np.where(q < 1.0, 1.0 - 1.5 * q * q + 0.75 * q * q * q, 0.25 * (2.0 - q) ** 3)
                    w = np.where(q <= 2.0, w, 0.0) / (np.pi * h**3)
                else:
                    w = np.where(q <= 2.0, 1.0 - 0.5 * q, 0.0) * 3.0 / (np.pi * support**3)
                touched = w > 0.0
                if self.is_vec:
                    leaf.values[sl] += w[..., None] * weights[i]
                else:
                    leaf.values[sl] += w * float(weights[i])
                leaf.active[sl] |= touched
                leaf.invalidate()
        return self

    def union_spheres(self, centers, radius, band=3.0):
        """Union of per-particle sphere SDFs: a particle-level-set style surface proxy.

        Writes `min(existing, |p - center_i| - radius_i)` across each sphere's reach, so repeated
        calls compose into one liquid surface; the active set is the `|d| <= band * mean(voxel_size)`
        narrow band. The min-of-spheres distance is an upper bound of the true union distance in
        concave bridges between overlapping particles - Lipschitz-accurate as a collision/rendering
        proxy, refine with a proper surface reconstructor offline if exactness matters.
        `radius` is a world-space scalar or one value per particle.
        """
        centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
        if self.is_vec:
            raise TypeError("union_spheres is defined on scalar grids only")
        radius = np.asarray(radius, dtype=np.float64).reshape(-1)
        if len(radius) == 1:
            radius = np.repeat(radius, len(centers))
        if len(radius) != len(centers):
            raise ValueError("radius must be a scalar or match the particle count")
        vs = self.voxel_size
        band_world = float(band) * float(np.mean(vs))
        for center, r_world in zip(centers, radius):
            r_world = float(r_world)
            center_i = self.world_to_index(center)
            reach = vs if self.is_axis_aligned else np.full(3, vs.min())
            reach_i = (r_world + band_world) / reach + 1.0
            lo = np.floor(center_i - reach_i).astype(np.int64)
            hi = np.ceil(center_i + reach_i).astype(np.int64)
            for leaf, sl in self._for_each_leaf_slice(lo, hi):
                dist = _slice_world_distance(self, leaf, sl, center)
                sdf = dist - r_world
                leaf.values[sl] = np.minimum(leaf.values[sl], sdf.astype(np.float32))
                leaf.active[sl] |= np.abs(sdf) <= band_world
                leaf.invalidate()
        return self

    def csg(self, other, op):
        """CSG combination of two scalar SDF grids: `op` in 'union' | 'diff' | 'intersect'.

        The result is stored in this grid. Where only one side has a leaf the other side reads as
        its background (for level sets, "far outside"). Activity is union-ed except for 'intersect',
        which intersects it.
        """
        if self.is_vec or other.is_vec:
            raise TypeError("CSG is defined on scalar grids only")
        if op not in ("union", "diff", "intersect"):
            raise ValueError("op must be 'union', 'diff' or 'intersect'")
        if not (np.allclose(self.voxel_size, other.voxel_size) and np.allclose(self.origin_world, other.origin_world)):
            raise ValueError("CSG requires identical voxel grids (voxel_size and origin_world)")
        for key in sorted(set(self._leaves) | set(other._leaves)):
            theirs = other._leaves.get(key)
            if theirs is None:
                continue
            mine = self._get_or_create_leaf(key)
            if op == "union":
                mine.values[...] = np.minimum(mine.values, theirs.values)
                mine.active[...] |= theirs.active
            elif op == "diff":
                mine.values[...] = np.maximum(mine.values, -theirs.values)
                mine.active[...] |= theirs.active
            else:
                mine.values[...] = np.maximum(mine.values, theirs.values)
                mine.active[...] &= theirs.active
            mine.invalidate()
        return self

    def prune(self):
        """Drop leaves with no active voxels; returns the number of leaves removed."""
        dead = [key for key, leaf in self._leaves.items() if leaf.n_active == 0]
        for key in dead:
            del self._leaves[key]
        return len(dead)

    def compress(self):
        """Reset inactive voxels to background, then prune: the RAM-level constant-region pass.

        Far-field junk (e.g. exact distances stamped into inactive voxels) is what keeps
        mostly-empty leaves alive; this normalizes it away so `prune` can drop every
        fully-background leaf. Lossy by design for inactive values - call `sample*` first if
        you need them. Returns (leaves_dropped, voxels_reset).
        """
        if self.is_vec:
            bg = np.asarray(self.background).reshape(3)
        else:
            bg = float(self.background)
        reset = 0
        for leaf in self._leaves.values():
            dead = ~leaf.active
            if leaf.values[dead].size and np.any(leaf.values[dead] != bg):
                leaf.values[dead] = bg
                reset += int(np.count_nonzero(dead))
            leaf.invalidate()
        return self.prune(), reset

    # ------------------------------------------------------------------ dense conversion

    def _numpy_value_dtype(self):
        if self.type_code == 1:
            return np.float64
        if self.type_code == 3:
            return np.float16
        return np.float32

    def to_dense(self, bbox=None, pad=0):
        """Dense `(x, y, z[, 3])` array covering `bbox` (default: the active bbox), plus `ijk_min`.

        The background fills everything outside allocated leaves. This is the bridge back into the
        engine's dense `qd.field` grids and renderers.
        """
        if bbox is None:
            bbox = self.bbox()
            if bbox is None:
                shape = (1, 1, 1) + self.value_shape
                return np.full(shape, self.background, dtype=self._numpy_value_dtype()), np.zeros(3, dtype=np.int64)
        lo = np.asarray(bbox[0], dtype=np.int64) - pad
        hi = np.asarray(bbox[1], dtype=np.int64) + pad
        out = np.full(tuple(hi - lo + 1) + self.value_shape, self.background, dtype=self._numpy_value_dtype())
        for leaf, src in self._for_each_leaf_slice(lo, hi, create=False):
            leaf_lo = np.maximum(leaf.origin, lo) - lo
            leaf_hi = np.minimum(leaf.origin + self.leaf_dim - 1, hi) - lo
            dst = tuple(slice(leaf_lo[a], leaf_hi[a] + 1) for a in range(3))
            out[dst] = leaf.values[src]
        return out, lo

    @classmethod
    def from_dense(cls, values, origin=(0, 0, 0), voxel_size=(1.0, 1.0, 1.0), origin_world=(0.0, 0.0, 0.0),
                   background=0.0, name="grid", grid_class="unknown", leaf_log2=DEFAULT_LEAF_LOG2,
                   active=None):
        """Build a sparse grid from a dense `(x, y, z)` or `(x, y, z, 3)` array.

        Voxels equal to `background` stay inactive; pass a boolean `active` array to control the
        mask instead. `origin` shifts the dense array's voxel (0,0,0) to that index. This is the
        bridge from the engine's dense `qd.field` SDF and smoke grids into the sparse world.
        """
        values = np.asarray(values)
        if values.ndim == 4 and values.shape[3] == 3:
            grid = cls(dtype=np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32)]),
                       background=background, voxel_size=voxel_size, origin_world=origin_world,
                       leaf_log2=leaf_log2, name=name, grid_class=grid_class)
            values = values.astype(np.float32, copy=False)
        else:
            if values.ndim != 3:
                raise ValueError("dense values must be (x, y, z) or (x, y, z, 3)")
            values = values.astype(np.float64 if values.dtype == np.float64 else np.float32, copy=False)
            grid = cls(dtype=values.dtype, background=background, voxel_size=voxel_size,
                       origin_world=origin_world, leaf_log2=leaf_log2, name=name, grid_class=grid_class)
        if active is None:
            if grid.is_vec:
                active_mask = np.any(values != np.asarray(background).reshape(3), axis=-1)
            else:
                active_mask = values != float(background)
        else:
            active_mask = np.asarray(active, dtype=bool).reshape(values.shape[:3])

        origin = np.asarray(origin, dtype=np.int64).reshape(3)
        nx, ny, nz = values.shape[:3]
        shape = np.array([nx, ny, nz])
        for kx in range(int(origin[0]) >> leaf_log2, (int(origin[0]) + nx - 1 >> leaf_log2) + 1):
            for ky in range(int(origin[1]) >> leaf_log2, (int(origin[1]) + ny - 1 >> leaf_log2) + 1):
                for kz in range(int(origin[2]) >> leaf_log2, (int(origin[2]) + nz - 1 >> leaf_log2) + 1):
                    block_lo = np.array([kx, ky, kz]) << leaf_log2
                    data_lo = np.maximum(block_lo, origin)
                    data_hi = np.minimum(block_lo + (1 << leaf_log2), origin + shape)
                    src = tuple(slice(int(data_lo[a] - origin[a]), int(data_hi[a] - origin[a])) for a in range(3))
                    if not active_mask[src].any():
                        continue
                    leaf = grid._get_or_create_leaf((kx, ky, kz))
                    dst = tuple(slice(int(data_lo[a] - block_lo[a]), int(data_hi[a] - block_lo[a])) for a in range(3))
                    leaf.values[dst] = values[src]
                    leaf.active[dst] = active_mask[src]
                    leaf.invalidate()
        return grid

    # ------------------------------------------------------------------ transforms & sampling

    def index_to_world(self, ijk):
        """`world = R @ (index * voxel_size) + origin_world` (rigid affine)."""
        scaled = np.asarray(ijk, dtype=np.float64).reshape(-1, 3) * self.voxel_size
        world = scaled @ self.rotation.T + self.origin_world
        return world[0] if np.asarray(ijk).ndim == 1 else world

    def world_to_index(self, xyz):
        """Inverse map: `index = (R^T @ (world - origin_world)) / voxel_size`."""
        pts = (np.asarray(xyz, dtype=np.float64).reshape(-1, 3) - self.origin_world) @ self.rotation
        idx = pts / self.voxel_size
        return idx[0] if np.asarray(xyz).ndim == 1 else idx

    def sample_nearest(self, xyz):
        """Value at the voxel whose center is nearest to the world point."""
        idx = np.rint(self.world_to_index(np.asarray(xyz, dtype=np.float64))).astype(np.int64).reshape(-1)
        return self.get_value((int(idx[0]), int(idx[1]), int(idx[2])))

    def sample_linear(self, xyz):
        """Host-side trilinear sample at world coordinates (background outside data)."""
        return _sample_linear(self, np.atleast_2d(np.asarray(xyz, dtype=np.float64)))[0]

    def sample_quadratic(self, xyz):
        """Host-side triquadratic (3x3x3 quadratic B-spline) sample at world coordinates.

        Weights per axis at fractional offset `u`: `[0.5(1-u)^2, 0.5 + u - u^2, 0.5 u^2]` over
        taps `floor(x)-1 .. floor(x)+1` - the standard C1-continuous quadratic B-spline, matching
        OpenVDB's `QuadraticSampler`. Unlike nearest/linear it does NOT reproduce the exact value
        at voxel centers (half weight on the neighbors there); the payoff is second-order
        smoothness for shading/collision queries. Scalar grids only. A 1-D input returns a scalar.
        """
        return _sample_quadratic(self, np.atleast_2d(np.asarray(xyz, dtype=np.float64)))[0]

    def sample_gradient(self, xyz, order=1):
        """Central-difference gradient `(n, 3)` at world points; `order` picks the sampler."""
        pts = np.atleast_2d(np.asarray(xyz, dtype=np.float64))
        sampler = _sample_linear if order == 1 else _sample_quadratic
        sx, sy, sz = self.voxel_size
        gx = sampler(self, pts + [sx, 0, 0]) - sampler(self, pts - [sx, 0, 0])
        gy = sampler(self, pts + [0, sy, 0]) - sampler(self, pts - [0, sy, 0])
        gz = sampler(self, pts + [0, 0, sz]) - sampler(self, pts - [0, 0, sz])
        return np.stack([gx / (2.0 * sx), gy / (2.0 * sy), gz / (2.0 * sz)], axis=1).astype(np.float64)

    _STENCIL7 = ((0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    _STENCIL19_EDGES = ((1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
                        (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
                        (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1))

    def stencil7_batch(self, indices):
        """7-point stencil values `(n, 7)` at voxel indices, order [c, +x, -x, +y, -y, +z, -z]."""
        idxs = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
        offsets = np.asarray(self._STENCIL7, dtype=np.int64)
        vals, _ = self.probe_batch((idxs[:, None, :] + offsets[None, :, :]).reshape(-1, 3))
        return np.asarray(vals).reshape(len(idxs), 7)

    def stencil19_batch(self, indices):
        """19-point stencil `(n, 19)`: the 7-point pattern + the 12 edge neighbors (second half).

        Enough for second derivatives with cross terms (e.g. Hessian diagonals via
        `(f(+e_i) + f(-e_i) - 2 f(c)) / h^2` and mixed terms via edge pairs).
        """
        idxs = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
        offsets = np.asarray(self._STENCIL7 + self._STENCIL19_EDGES, dtype=np.int64)
        vals, _ = self.probe_batch((idxs[:, None, :] + offsets[None, :, :]).reshape(-1, 3))
        return np.asarray(vals).reshape(len(idxs), 19)


def _quad_weight(u, j):
    """Quadratic B-spline tap weight at fractional offset `u` for tap `j` in {0, 1, 2}."""
    if j == 0:
        return 0.5 * (1.0 - u) ** 2
    if j == 1:
        return 0.5 + u - u * u
    return 0.5 * u * u


def _sample_quadratic(grid, points_world):
    """Vectorized triquadratic sampling of a scalar grid; one result per row of `points_world`."""
    coords = grid.world_to_index(points_world)
    base = np.floor(coords).astype(np.int64)
    frac = coords - base
    n = len(points_world)
    # gather all 27 taps for all points in ONE probe_batch call
    taps = np.stack([base + np.array([j - 1, k - 1, l - 1])
                     for j in range(3) for k in range(3) for l in range(3)])  # (27, n, 3)
    vals, _ = grid.probe_batch(taps.reshape(-1, 3))
    vals = np.asarray(vals, dtype=np.float64).reshape(3, 3, 3, n)
    out = np.zeros(n, dtype=grid._numpy_value_dtype())
    for j in range(3):
        wj = _quad_weight(frac[:, 0], j)
        for k in range(3):
            wk = _quad_weight(frac[:, 1], k)
            for l in range(3):
                wl = _quad_weight(frac[:, 2], l)
                out += (wj * wk * wl * vals[j, k, l]).astype(out.dtype)
    return out


def _slice_world_positions(grid, leaf, sl):
    """World coordinates (x, y, z arrays) of the voxel centers in a leaf slice, affine-aware."""
    axes = [leaf.origin[a] + np.arange(sl[a].start, sl[a].stop) for a in range(3)]
    idx = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3) * grid.voxel_size
    world = idx @ grid.rotation.T + grid.origin_world
    return [world[:, a].reshape(len(axes[0]), len(axes[1]), len(axes[2])) for a in range(3)]


def _slice_world_distance(grid, leaf, sl, center):
    """World-space euclidean distance from `center` to every voxel center in a leaf slice."""
    wx, wy, wz = _slice_world_positions(grid, leaf, sl)
    return np.sqrt((wx - center[0]) ** 2 + (wy - center[1]) ** 2 + (wz - center[2]) ** 2)


def _sample_linear(grid, points_world):
    """Vectorized trilinear sampling of a scalar grid; one result per row of `points_world`."""
    points_world = np.atleast_2d(np.asarray(points_world, dtype=np.float64))
    coords = grid.world_to_index(points_world)
    base = np.floor(coords).astype(np.int64)
    frac = coords - base
    out = np.zeros(len(points_world), dtype=grid._numpy_value_dtype())
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ijk = base + np.array([dx, dy, dz])
                w = ((frac[:, 0] if dx else 1.0 - frac[:, 0])
                     * (frac[:, 1] if dy else 1.0 - frac[:, 1])
                     * (frac[:, 2] if dz else 1.0 - frac[:, 2]))
                if len(points_world) >= 32:
                    vals, _ = grid.probe_batch(ijk)  # vectorized gather pays off on batches
                else:
                    vals = np.array([grid.get_value((int(i[0]), int(i[1]), int(i[2]))) for i in ijk],
                                    dtype=np.float64)
                out += (np.asarray(vals, dtype=np.float64).reshape(-1) * w).astype(out.dtype)
    return out
