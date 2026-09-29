"""Benchmark suite: the published XUVDB performance baseline.

Run:  python examples/bench_suite.py [cuda]      -> prints a Markdown table
Results live in BENCHMARKS.md; regenerate there when the numbers change.
"""
import sys
import time

import numpy as np

if len(sys.argv) > 1 and sys.argv[1] == "cuda":
    import quadrants as qd
    qd.init(arch=qd.cuda)

import xuvdb
from xuvdb.gpu import GpuVolume, VolumeBatch

rng = np.random.default_rng(7)
N_PTS = 100_000
N_RAYS = 1000

grid = xuvdb.VdbGrid(background=0.15, voxel_size=0.05, leaf_log2=4,
                     name="bench", grid_class="level set")
pts_p = rng.uniform(-0.8, 0.8, size=(300, 3))
grid.union_spheres(pts_p, rng.uniform(0.02, 0.05, size=300), band=3.0)
vol = GpuVolume(grid)
inside = np.array([grid.bbox()[0], grid.bbox()[1]]) * 0.05
pts = rng.uniform(inside[0], inside[1], size=(N_PTS, 3)).astype(np.float32)
rows = []


def bench(label, fn, unit):
    fn()  # warm
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    rows.append((label, dt, unit, out))
    return dt


# --- sampling: host vs kernel (per 100k points)
bench("host sample_linear (100k pts)", lambda: np.array([grid.sample_linear(p) for p in pts[:2000]]),
      "per 2k pts")  # host is per-point; scale label honestly
bench("kernel sample order=0 (100k)", lambda: vol.sample(pts, order=0), "per 100k pts")
bench("kernel sample order=1 (100k)", lambda: vol.sample(pts, order=1), "per 100k pts")
bench("kernel sample order=2 (100k)", lambda: vol.sample(pts, order=2), "per 100k pts")
bench("kernel reduce (32k active)", lambda: vol.reduce(), "per call")

center = pts_p.mean(axis=0)
dirs = rng.normal(size=(N_RAYS, 3))
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
origins = center + dirs * 2.5
dirs = -dirs


def host_rays():
    return [xuvdb.ray_surface_hit(grid, origins[i], dirs[i], tmax=50.0) for i in range(N_RAYS)]


def kernel_rays():
    return vol.ray_surface_hit(origins, dirs, tmax=50.0)


bench(f"host DDA bbox ({N_RAYS} rays)", host_rays, "per 1000 rays")
bench(f"kernel DDA batch ({N_RAYS} rays)", kernel_rays, "per 1000 rays")

batch = VolumeBatch([grid, grid.copy()])
ids = rng.integers(0, 2, size=N_PTS)
bench("VolumeBatch sample (2 vols, 100k)", lambda: batch.sample(pts, ids), "per 100k pts")

host_hits = sum(h is not None for h in host_rays())
k_hits = sum(h is not None for h in kernel_rays())
assert host_hits == k_hits, "ray parity broken"

print(f"| benchmark | time | rate |")
print(f"|---|---|---|")
for label, dt, unit, _ in rows:
    n = N_RAYS if "rays" in unit else (2000 if "2k" in unit else N_PTS)
    print(f"| {label} | {dt * 1e3:.1f} ms | {n / dt / 1e3:.0f}k/s |")
print(f"\ngrid: 300-particle union_spheres, {grid.n_leaves} leaves, "
      f"{grid.active_voxel_count} active voxels; ray hits {host_hits}/{N_RAYS} (host==kernel)")
