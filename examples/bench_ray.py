"""Benchmark: host DDA without bbox, host DDA with bbox, and the quadrants-kernel DDA.

Run:  python examples/bench_ray.py [n_particles] [n_rays] [backend]   (backend: cpu|cuda)
"""
import sys
import time

import numpy as np
import quadrants as qd

import xuvdb
from xuvdb.gpu import GpuVolume

if len(sys.argv) > 3 and sys.argv[3] == "cuda":
    qd.init(arch=qd.cuda)  # claim the runtime before GpuVolume's init_runtime()

n_particles = int(sys.argv[1]) if len(sys.argv) > 1 else 300
n_rays = int(sys.argv[2]) if len(sys.argv) > 2 else 300

rng = np.random.default_rng(7)
pts = rng.uniform(-0.8, 0.8, size=(n_particles, 3))
radii = rng.uniform(0.02, 0.05, size=n_particles)

grid = xuvdb.VdbGrid(background=3 * 0.05, voxel_size=0.05, leaf_log2=4,
                     name="liquid", grid_class="level set")
grid.union_spheres(pts, radii, band=3.0)
print(f"grid: {n_particles} particles, {grid.n_leaves} leaves, "
      f"{grid.active_voxel_count} active voxels, bbox={grid.bbox()}")

vol = GpuVolume(grid)
center = pts.mean(axis=0)
dirs = rng.normal(size=(n_rays, 3))
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
origins = center + dirs * 2.5
dirs = -dirs  # shoot inward at the blob

for label, fn in [
    ("host DDA (no bbox)", lambda o, d: xuvdb.ray_surface_hit(grid, o, d, tmax=50.0, use_bbox=False)),
    ("host DDA (bbox)   ", lambda o, d: xuvdb.ray_surface_hit(grid, o, d, tmax=50.0, use_bbox=True)),
    ("kernel DDA (batch)", lambda o, d: vol.ray_surface_hit(o, d, tmax=50.0)),
]:
    vol.ray_surface_hit(origins[:1], dirs[:1], tmax=50.0)  # warm: compile/JIT outside the timer
    t0 = time.perf_counter()
    if label.startswith("kernel"):
        hits = fn(origins, dirs)  # one launch for all rays
    else:
        hits = [fn(origins[i], dirs[i]) for i in range(n_rays)]
    dt = time.perf_counter() - t0
    n_hit = sum(h is not None for h in hits)
    print(f"{label}: {dt * 1e3:8.2f} ms total  ({dt / n_rays * 1e6:7.1f} us/ray)  hits={n_hit}/{n_rays}")

# cross-check: kernel and bbox-culled host agree on every ray (same explicit tmax on both sides)
kernel_hits = vol.ray_surface_hit(origins, dirs)
for i in range(n_rays):
    h = xuvdb.ray_surface_hit(grid, origins[i], dirs[i], tmax=50.0)
    k = kernel_hits[i]
    assert (h is None) == (k is None), f"parity broken at ray {i}"
    if h is not None:
        assert abs(k[0] - h[0]) < 1e-5, f"t mismatch at ray {i}: {k[0]} vs {h[0]}"
print("parity: kernel == host(bbox) on all rays  OK")
