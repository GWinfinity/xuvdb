"""GPU backend smoke test: run the packed-volume kernels on every quadrants backend available.

Run:  python examples/gpu_smoke.py
"""
import numpy as np
import quadrants as qd

import xuvdb


def try_backend(arch):
    try:
        qd.init(arch=arch)
    except Exception as e:  # noqa: BLE001 - report and continue
        print(f"{arch}: UNAVAILABLE ({type(e).__name__}: {e})")
        return
    grid = xuvdb.VdbGrid(background=0.15, voxel_size=0.05, leaf_log2=4,
                         name="g", grid_class="level set")
    grid.stamp_sphere((0.3, 0.2, 0.1), radius=0.25, band=3.0)
    vol = xuvdb.GpuVolume(grid)
    pts = np.array([[0.3, 0.45, 0.1], [0.3, 0.2, 0.1]], np.float32)
    d = vol.sample(pts, linear=True)
    hit = vol.ray_surface_hit((0.3, 0.2, 2.0), (0, 0, -1))
    assert abs(d[0]) < 2e-2 and abs(d[1] + 0.25) < 2e-2, d
    assert hit is not None and abs(hit[2]) < 5e-2, hit
    print(f"{arch}: sample={np.round(d, 4).tolist()}  ray t={hit[0]:.4f}  OK")


try_backend(qd.cpu)
try_backend(qd.cuda)
try_backend(qd.vulkan)
