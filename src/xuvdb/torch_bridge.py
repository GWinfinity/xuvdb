"""Torch interop: leaf tables as tensors and a differentiable trilinear sampler.

The fVDB-direction bridge done the XUVDB way - interop, not a framework. Value buffers
round-trip as `torch.Tensor`s on any device, and `sample()` exposes the trilinear field as a
proper autograd node with analytic gradients w.r.t. BOTH the leaf values and the sample points
(scatter-add for values, corner-weight derivatives for points). Sparse convolutions, attention
and training operators stay out of scope - export to fvdb-core for those.

Axis-aligned float grids only, mirroring the kernel constraint.
"""

import numpy as np

from .gpu import _pack_key
from .tree import VdbGrid

try:
    import torch
except ImportError:  # torch interop needs an explicit `pip install xuvdb[torch]`
    torch = None

_KEY_OFF = 1 << 19
_CORNERS = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1))


def _require_torch():
    if torch is None:
        raise ImportError("torch interop requires PyTorch (pip install 'xuvdb[torch]')")


def _validate(grid):
    if grid.is_vec or grid.type_code not in (0, 1, 3):
        raise TypeError("tensor packing is defined on scalar float grids")
    if not grid.is_axis_aligned:
        raise TypeError("tensor packing requires axis-aligned grids (identity rotation)")
    if grid.n_leaves == 0:
        raise ValueError("grid has no leaves")


def grid_to_tensors(grid, device=None):
    """Pack a scalar grid's leaf table as tensors: dict of `keys` (n,) int64, `values`
    (n, dim^3) float32 (f16/f64 upcast), `active` (n, dim^3) bool. Optional `device` moves all."""
    _require_torch()
    _validate(grid)
    leaves = grid.leaves()
    keys = np.array([_pack_key(*k) for k in sorted(grid._leaves)], dtype=np.int64)
    values = np.stack([l.values.reshape(-1) for l in leaves]).astype(np.float32)
    active = np.stack([l.active.reshape(-1) for l in leaves])
    out = {"keys": torch.from_numpy(keys),
           "values": torch.from_numpy(values),
           "active": torch.from_numpy(active)}
    if device is not None:
        out = {k: v.to(device) for k, v in out.items()}
    return out


def tensors_to_grid(tensors, voxel_size, origin_world=(0.0, 0.0, 0.0), leaf_log2=4,
                    background=0.0, name="grid", grid_class="unknown"):
    """Rebuild a `VdbGrid` from `grid_to_tensors` output plus the grid meta (voxel_size etc.)."""
    _require_torch()
    keys = tensors["keys"].detach().cpu().numpy()
    values = tensors["values"].detach().cpu().numpy().astype(np.float32)
    active = tensors["active"].detach().cpu().numpy()
    grid = VdbGrid(background=background, voxel_size=voxel_size, origin_world=origin_world,
                   leaf_log2=leaf_log2, name=name, grid_class=grid_class)
    dim3 = grid.leaf_dim**3
    for row, key in enumerate(keys):
        kx = int(key >> 40) - _KEY_OFF
        ky = int((key >> 20) & ((1 << 20) - 1)) - _KEY_OFF
        kz = int(key & ((1 << 20) - 1)) - _KEY_OFF
        leaf = grid._get_or_create_leaf((kx, ky, kz))
        leaf.values[...] = values[row].reshape(grid.leaf_dim, grid.leaf_dim, grid.leaf_dim)
        leaf.active[...] = active[row].reshape(grid.leaf_dim, grid.leaf_dim, grid.leaf_dim)
        leaf.invalidate()
    return grid


# ------------------------------------------------------------------ differentiable sampling

def _corner_gather(values, keys, voxel_idx, log2, background):
    """Values at the (m, 8, 3) corner indices; `(v (m,8), rows (m,8) or -1, flat (m,8))`."""
    dim = 1 << log2
    m = voxel_idx.shape[0]
    k = (voxel_idx >> log2) + _KEY_OFF
    key = k[..., 0] * (1 << 40) + k[..., 1] * (1 << 20) + k[..., 2]
    key_flat = key.reshape(-1)
    idx = torch.searchsorted(keys, key_flat)
    idx = idx.clamp(max=len(keys) - 1)
    matched = keys[idx] == key_flat
    rows = torch.where(matched, idx, torch.full_like(idx, -1)).reshape(m, 8)
    local = voxel_idx & (dim - 1)
    flat = local[..., 0] * dim * dim + local[..., 1] * dim + local[..., 2]
    safe = rows.clamp_min(0).reshape(-1)
    v = values[safe, flat.reshape(-1)].reshape(m, 8)
    v = torch.where(rows >= 0, v, torch.full_like(v, background))
    return v, rows, flat


def _corner_weights(frac):
    """(m, 8) trilinear weights from (m, 3) fractional offsets."""
    w = torch.ones(len(frac), 8, dtype=frac.dtype, device=frac.device)
    for c, (dx, dy, dz) in enumerate(_CORNERS):
        w[:, c] = (frac[:, 0] if dx else 1.0 - frac[:, 0]) \
            * (frac[:, 1] if dy else 1.0 - frac[:, 1]) \
            * (frac[:, 2] if dz else 1.0 - frac[:, 2])
    return w


class _Trilinear(torch.autograd.Function):
    """out = trilinear(field(values), points); gradients w.r.t. values (scatter-add) and points
    (analytic corner-weight derivatives, scaled by 1/voxel_size)."""

    @staticmethod
    def forward(ctx, values, points, keys, o_x, o_y, o_z, s_x, s_y, s_z, log2, background):
        offs = torch.tensor(_CORNERS, dtype=torch.int64, device=points.device)
        coords = torch.stack([(points[:, 0] - o_x) / s_x,
                              (points[:, 1] - o_y) / s_y,
                              (points[:, 2] - o_z) / s_z], dim=1)
        base = torch.floor(coords).to(torch.int64)
        frac = coords - torch.floor(coords)
        idx = base.unsqueeze(1) + offs  # (m, 8, 3)
        v, rows, flat = _corner_gather(values, keys, idx, log2, background)
        w = _corner_weights(frac)
        out = (v * w).sum(dim=1)
        ctx.save_for_backward(values, keys, base, frac, v, rows, flat)
        ctx.log2 = log2
        ctx.scale = (s_x, s_y, s_z)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        values, keys, base, frac, v, rows, flat = ctx.saved_tensors
        dim3 = (1 << ctx.log2) ** 3
        found = rows >= 0
        grad_values = grad_points = None
        if ctx.needs_input_grad[0]:
            grad_values = torch.zeros_like(values)
            lin = (rows * dim3 + flat)[found]
            contrib = (_corner_weights(frac) * grad_out.unsqueeze(1))[found]
            grad_values.view(-1).index_put_((lin,), contrib, accumulate=True)
        if ctx.needs_input_grad[1]:
            offs = torch.tensor(_CORNERS, dtype=frac.dtype, device=frac.device)
            g = []
            for a in range(3):
                w_others = torch.ones_like(v)
                for b in range(3):
                    if b == a:
                        continue
                    sel = offs[:, b] == 1
                    w_b = frac[:, b].unsqueeze(1) * sel + (1.0 - frac[:, b]).unsqueeze(1) * (~sel)
                    w_others = w_others * w_b
                sign = offs[:, a] * 2.0 - 1.0  # +1 for the frac corner, -1 for the (1-frac) one
                g.append(((v * w_others * sign).sum(dim=1)) / ctx.scale[a])
            grad_points = torch.stack(g, dim=1)
        return (grad_values, grad_points, None, None, None, None, None, None, None, None, None)


def sample_t(tensors, points, voxel_size, origin_world=(0.0, 0.0, 0.0), leaf_log2=4,
             background=0.0, device=None):
    """Differentiable trilinear sample over an explicit (trainable) values tensor.

    `tensors` is `grid_to_tensors` output; `values` may carry `requires_grad`. Returns (m,)
    float32; gradients flow to `values` and to `points` (if it is a tensor with requires_grad).
    """
    _require_torch()
    values = tensors["values"].float()
    keys = tensors["keys"]
    if torch.is_tensor(points):
        pts = points.float()
    else:
        pts = torch.as_tensor(np.asarray(points), dtype=torch.float32)
    tgt = device or pts.device
    return _Trilinear.apply(values.to(tgt), pts.to(tgt), keys.to(tgt),
                            float(origin_world[0]), float(origin_world[1]), float(origin_world[2]),
                            float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2]),
                            int(leaf_log2), float(background))


def sample(grid, points, device=None):
    """Differentiable trilinear sample of a grid's field at world points (convenience wrapper).

    Packs the grid on the fly, so gradients land on the wrapper's internal tensor - use
    `sample_t` when you need them on your own values tensor.
    """
    _require_torch()
    _validate(grid)
    t = grid_to_tensors(grid)
    return sample_t(t, points, grid.voxel_size, grid.origin_world, grid.leaf_log2,
                    grid.background, device=device)
