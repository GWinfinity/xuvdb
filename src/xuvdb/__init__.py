"""XUVDB（太虚）: a sparse-volume format built on the quadrants kernel.

The name reads 太虚 (tàixū) - from《庄子·知北游》「不游乎太虚」and 张载《正蒙》「太虚无形，
气之本体」: the closest Chinese rendering of an unbounded, sparse index domain - unbounded (the
root hash domain has no extent), formless (unallocated space has no shape; sampling it returns the
background). The name lives at the concept layer only (docs, logs, visualization labels); API
identifiers stay English (`xuvdb.prune()`, never `xuvdb.sunyi()`), and the native extension
`.xuvdb` never uses `.vdb` - every major volume tool parses that suffix as OpenVDB and fails
opaquely on foreign bytes. Interop with OpenVDB is an explicit, separate export.

`VdbGrid` is a mutable sparse voxel tree (dictionary of dense leaf blocks) that fills the gap this
module was designed for: OpenVDB trees are CPU-bound and NanoVDB grids are GPU read-only, while an
XUVDB grid can be structurally edited from Python (allocate/fill/stamp/CSG/prune leaves at any time)
and its packed value buffer can additionally be *written* by quadrants kernels
(`GpuVolume.write_voxels`) without a host round trip.

Interoperability with OpenVDB is first-class and needs no OpenVDB installation:

- `write_vdb` / `read_vdb` read and write real `.vdb` files (the published stream format, verified
  against the OpenVDB sources; file version 224, active-mask compression, zip readable, blosc not)
- `to_openvdb` / `from_openvdb` convert live grids through `pyopenvdb` when the bindings exist
  (Houdini hython, Blender Python, conda-forge)
- `save` / `load` use the native compact `.xuvdb` format

Typical flows against the engine's existing dense solvers:

- rigid SDF (落点 1): bake `geom.sdf_val` dense grids with `VdbGrid.from_dense`, edit/CSG them
  sparsely, sample on device via `GpuVolume.sample` / `sdf_normal` and cast rays with
  `ray_surface_hit`
- MPM / smoke (落点 2, 3): pull density or velocity fields off the dense `qd.field` after a step,
  sparsify, and hand `.vdb` files to Houdini/Blender for volume rendering, or bring Houdini caches
  back in for initial conditions
- liquid & oil (落点 4, SPH): rasterize particles to fog density volumes with `scatter_particles`
  (unit-integral SPH kernels; one grid per phase gives mixture fractions) or build particle
  level-set surface proxies with `union_spheres`; import Houdini container/initial-surface `.vdb`
  caches as collision/initialization SDFs sampled on device

Kernels run on any quadrants backend; call `init_runtime()` once before the first kernel call if
`gs.init()` has not run (mirroring `genesis.occ`).
"""

from .gpu import GpuVolume, VolumeBatch
from .io import load, save
from .openvdb_bridge import from_openvdb, to_openvdb
from .openvdb_file import read_vdb, write_vdb
from .ray import ray_surface_hit
from .runtime import init_runtime
from . import torch_bridge
from .tree import Leaf, VdbGrid

__version__ = "1.1.0"

__all__ = [
    "GpuVolume",
    "Leaf",
    "VdbGrid",
    "VolumeBatch",
    "from_openvdb",
    "init_runtime",
    "load",
    "ray_surface_hit",
    "read_vdb",
    "save",
    "to_openvdb",
    "write_vdb",
]
