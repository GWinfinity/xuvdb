"""Host-side packing of a `VdbGrid` into flat buffers consumed by `kernels.py`.

A `GpuVolume` is the read-mostly, kernel-writable view of a scalar grid - the XUVDB counterpart of
a NanoVDB device buffer, except that values are mutable from kernels. Topology is frozen at pack
time; call `sync_to_host()` to pull kernel-written values back into the host tree, and re-create
the volume after structural edits (leaf allocation, prune, load).
"""

import numpy as np

from . import kernels
from .runtime import init_runtime
from .tree import VdbGrid

_KEY_OFF = 1 << 19


def _pack_key(kx, ky, kz):
    return (int(kx) + _KEY_OFF) * (1 << 40) + (int(ky) + _KEY_OFF) * (1 << 20) + (int(kz) + _KEY_OFF)


class GpuVolume:
    """Flat, key-sorted leaf table of a scalar grid plus kernel entry points."""

    def __init__(self, grid):
        if not isinstance(grid, VdbGrid):
            raise TypeError("GpuVolume wraps an xuvdb.VdbGrid")
        if grid.is_vec or grid.type_code != 0:
            raise TypeError("kernel packing supports float32 scalar grids only")
        if not grid.is_axis_aligned:
            raise TypeError("kernel packing requires axis-aligned grids (identity rotation)")
        if grid.n_leaves == 0:
            raise ValueError("grid has no leaves")
        init_runtime()
        self.grid = grid
        leaves = grid.leaves()
        keys = np.array([_pack_key(*key) for key in sorted(grid._leaves)], dtype=np.int64)
        if not np.all(keys[:-1] < keys[1:]):
            raise ValueError("leaf keys out of range or duplicated")
        per_leaf = grid.leaf_dim**3
        values = np.empty(len(leaves) * per_leaf, dtype=np.float32)
        self.mask = np.empty((len(leaves), per_leaf // 32), dtype=np.uint32)
        for i, leaf in enumerate(leaves):
            values[i * per_leaf:(i + 1) * per_leaf] = leaf.values.ravel()  # z-fastest C order
            bits = np.packbits(leaf.active.reshape(-1), bitorder="little")  # bit n = voxel n
            self.mask[i] = bits.view(np.uint32)
        self.keys = keys
        self.values = values
        self.background = float(grid.background)
        self._dirty = False

    # ------------------------------------------------------------------ queries

    def sample(self, points, linear=False, order=None):
        """Sample the packed volume at world-space points (N,3).

        `order`: 0 = nearest, 1 = trilinear, 2 = triquadratic (3x3x3 quadratic B-spline); when
        given it overrides the boolean `linear` flag. Order 2 matches the host
        `VdbGrid.sample_quadratic` and does not reproduce voxel-center values (B-spline nature).
        """
        points = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
        out = np.empty(len(points), dtype=np.float32)
        if order is None:
            order = 1 if linear else 0
        transform = (int(self.grid.n_leaves), int(self.grid.leaf_log2), float(self.background),
                     *(float(v) for v in self.grid.origin_world), *(float(v) for v in self.grid.voxel_size))
        if order == 0:
            kernels.kernel_sample_nearest(self.keys, self.values, *transform, points, out)
        elif order == 1:
            kernels.kernel_sample_linear(self.keys, self.values, *transform, points, out)
        elif order == 2:
            kernels.kernel_sample_quadratic(self.keys, self.values, *transform, points, out)
        else:
            raise ValueError("order must be 0 (nearest), 1 (linear) or 2 (quadratic)")
        return out

    def sdf_normal(self, points):
        """Central-difference SDF gradient (world-space, normalized) at world-space points."""
        points = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
        out = np.empty((len(points), 3), dtype=np.float32)
        kernels.kernel_sdf_normal(
            self.keys, self.values, int(self.grid.n_leaves), int(self.grid.leaf_log2), float(self.background),
            *(float(v) for v in self.grid.origin_world), *(float(v) for v in self.grid.voxel_size),
            points, out,
        )
        return out

    # ------------------------------------------------------------------ rays

    def ray_surface_hit(self, origin, direction, tmax=None, isovalue=0.0, refine_steps=24,
                        max_steps=1 << 20):
        """Kernel-side DDA hit of `isovalue`: `(t, point(3,), value)` or None per ray.

        One kernel launch carries every ray (the marches run in parallel), so passing all rays at
        once (`(n, 3)` origin/direction arrays, returns a list) amortizes the launch cost that a
        per-ray call pays every time. Mirrors `ray.ray_surface_hit` (same block skips and cached
        active-bbox culling). Only float32 scalar grids (the `GpuVolume` constraint) are supported.
        """
        single = np.asarray(origin).ndim == 1
        origins = np.ascontiguousarray(origin, dtype=np.float64).reshape(-1, 3)
        dirs = np.ascontiguousarray(direction, dtype=np.float64).reshape(-1, 3)
        if len(origins) != len(dirs):
            raise ValueError("origin and direction must have the same ray count")
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        if np.any(norms == 0.0):
            raise ValueError("direction must be nonzero")
        dirs = dirs / norms
        if tmax is None:
            bbox = self.grid.bbox()
            if bbox is None:
                tmax = 1e30
            else:
                far = self.grid.index_to_world(bbox[1])
                # conservative per-ray default: the largest single-ray bound across the batch
                tmax = float(np.max(2.0 * np.linalg.norm(far[None, :] - origins, axis=1) + 1.0))

        n = len(self.keys)
        bbox_lo = np.full((n, 3), 1, dtype=np.int32)
        bbox_hi = np.full((n, 3), 0, dtype=np.int32)  # lo > hi marks a leaf with no active voxels
        for i, leaf in enumerate(self.grid.leaves()):
            llo, lhi = leaf.active_bbox()
            if llo is not None:
                bbox_lo[i] = llo
                bbox_hi[i] = lhi
        out = np.zeros((len(origins), 6), dtype=np.float64)
        kernels.kernel_ray_surface_hit(
            self.keys, self.values, int(self.grid.n_leaves), int(self.grid.leaf_log2), float(self.background),
            bbox_lo, bbox_hi,
            *(float(v) for v in self.grid.origin_world), *(float(v) for v in self.grid.voxel_size),
            origins, dirs, float(tmax), float(isovalue), out,
            refine_steps=refine_steps, max_steps=max_steps,
        )
        results = []
        for r in range(len(origins)):
            if out[r, 0] == 0.0:
                results.append(None)
            else:
                results.append((float(out[r, 1]), out[r, 2:5].copy(), float(out[r, 5])))
        return results[0] if single else results

    # ------------------------------------------------------------------ reduction

    def reduce(self):
        """`{'sum', 'min', 'max', 'count'}` over ACTIVE voxels, one kernel launch.

        The active mask is packed at pack time (`GpuVolume.mask`) and, like the values, frozen
        for the volume's lifetime - re-pack after structural edits.
        """
        per_leaf = np.zeros((len(self.keys), 4), dtype=np.float64)
        kernels.kernel_reduce(
            self.mask, self.values, per_leaf,
            words=int(self.mask.shape[1]), dim3=int(self.grid.leaf_dim**3),
        )
        valid = per_leaf[:, 3] > 0
        if not valid.any():
            return {"sum": 0.0, "min": None, "max": None, "count": 0}
        return {
            "sum": float(per_leaf[valid, 0].sum()),
            "min": float(per_leaf[valid, 1].min()),
            "max": float(per_leaf[valid, 2].max()),
            "count": int(per_leaf[valid, 3].sum()),
        }

    # ------------------------------------------------------------------ mutation

    def write_voxels(self, points, new_values):
        """Overwrite voxel values from kernels; call `sync_to_host` to publish into the tree."""
        points = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
        new_values = np.ascontiguousarray(new_values, dtype=np.float32).reshape(-1)
        if len(new_values) != len(points):
            raise ValueError("new_values must have one entry per point")
        kernels.kernel_write_voxels(
            self.keys, self.values, int(self.grid.n_leaves), int(self.grid.leaf_log2),
            *(float(v) for v in self.grid.origin_world), *(float(v) for v in self.grid.voxel_size),
            points, new_values,
        )
        self._dirty = True
        return self

    def sync_to_host(self, refresh_mask=False):
        """Copy kernel-written values back into the host grid's leaves.

        Kernel writes land in the dense leaf buffers, so they flow through `sample*`, `.xuvdb`
        save/load and `to_dense` regardless of the active mask - but `write_vdb` (active-mask
        format) and the `active_*` queries are mask-filtered. `refresh_mask=True` re-marks every
        voxel whose value differs from the background as active afterwards, so newly-written
        cells become visible to mask-filtered consumers and the next packing; structural edits
        (which the mask cannot express) still require a re-pack.
        """
        per_leaf = self.grid.leaf_dim**3
        for i, leaf in enumerate(self.grid.leaves()):
            leaf.values[...] = self.values[i * per_leaf:(i + 1) * per_leaf].reshape(
                self.grid.leaf_dim, self.grid.leaf_dim, self.grid.leaf_dim
            )
            if refresh_mask:
                new_mask = self.values[i * per_leaf:(i + 1) * per_leaf] != float(self.grid.background)
                leaf.active[...] = new_mask.reshape(
                    self.grid.leaf_dim, self.grid.leaf_dim, self.grid.leaf_dim
                )
            leaf.invalidate()  # kernel writes changed values: derived stats are stale
        self._dirty = False
        return self.grid


class VolumeBatch:
    """Many packed scalar grids sharing one key/value buffer - the GridBatch-style transfer.

    Each member volume keeps its own transform (voxel size, origin, background, even leaf size);
    `sample(points, volume_ids)` resolves every point against its volume's key segment in a
    single kernel launch. Same packing constraints as `GpuVolume` (f32 scalar, axis-aligned)."""

    def __init__(self, grids):
        grids = list(grids)
        if not grids:
            raise ValueError("VolumeBatch needs at least one grid")
        init_runtime()
        self.volumes = [GpuVolume(g) for g in grids]  # validates the packing constraints
        self.grids = [v.grid for v in self.volumes]
        segments = [0]
        for vol in self.volumes:
            segments.append(segments[-1] + len(vol.keys))
        self.keys = np.concatenate([v.keys for v in self.volumes])
        self.values = np.concatenate([v.values for v in self.volumes])
        self.segments = np.array(segments, dtype=np.int32)
        self.leaf_log2s = np.array([g.leaf_log2 for g in self.grids], dtype=np.int32)
        self.origins = np.stack([g.origin_world for g in self.grids]).astype(np.float64)
        self.steps = np.stack([g.voxel_size for g in self.grids]).astype(np.float64)
        self.backgrounds = np.array([v.background for v in self.volumes], dtype=np.float64)

    def __len__(self):
        return len(self.volumes)

    def sample(self, points, volume_ids):
        """Trilinear sampling across the batch: `(m, 3)` points, per-point `(m,)` volume ids."""
        points = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
        vol_ids = np.ascontiguousarray(volume_ids, dtype=np.int32).reshape(-1)
        if len(vol_ids) != len(points):
            raise ValueError("volume_ids must supply one id per point")
        if len(vol_ids) and (vol_ids.min() < 0 or vol_ids.max() >= len(self.volumes)):
            raise IndexError("volume id out of range")
        out = np.empty(len(points), dtype=np.float32)
        kernels.kernel_sample_batch(
            self.keys, self.values, self.segments, self.leaf_log2s,
            self.origins, self.steps, self.backgrounds, vol_ids, points, out,
        )
        return out
