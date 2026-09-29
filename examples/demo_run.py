"""End-to-end run of the XUVDB project: every flow from the README quick-start, with checks.

Notes:
- since 1.0.0, SDF `stamp_sphere` composes by MIN-union, so two stamps on one grid ARE a
  CSG union; this demo keeps the explicit `csg(second, 'union')` form because it exercises
  more API surface (and the two forms are verified equivalent in tests/test_v10.py);
- `read_vdb` returns a list of grids;
- the `.vdb` roundtrip is compared on narrow-band points: the OpenVDB stream only stores
  ACTIVE voxel values (+ lazy +-background for inactive ones), so inactive exact-distance
  values outside the band are intentionally reduced to background.
"""
import os

import numpy as np

import xuvdb

os.makedirs("demo_output", exist_ok=True)
ok = lambda label, cond, detail="": print(f"  PASS  {label} {detail}") if cond else (_ for _ in ()).throw(AssertionError(label + " " + detail))

C1, R1 = np.array([0.3, 0.2, 0.1]), 0.25
C2, R2 = np.array([0.5, 0.2, 0.1]), 0.10
sdf_union = lambda p: min(np.linalg.norm(p - C1) - R1, np.linalg.norm(p - C2) - R2)

# ---- 1) structural editing: narrow-band level set, CSG union, fill, prune
print("[1] edit: stamp + csg union + fill_box + prune")
grid = xuvdb.VdbGrid(background=3 * 0.05, voxel_size=0.05, leaf_log2=4,
                     name="shield", grid_class="level set")
grid.stamp_sphere(C1, radius=R1, band=3.0)
second = grid.copy()
second.stamp_sphere(C2, radius=R2)
grid.csg(second, "union")
grid.fill_box((-2, -2, -2), (2, 2, 2), 0.0)
grid.prune()
print(f"    leaves={grid.n_leaves}  active_voxels={grid.active_voxel_count}  bbox={grid.bbox()}")
ok("grid has leaves", grid.n_leaves > 0)

# ---- 2) native .xuvdb roundtrip (lossless by construction)
print("[2] save/load (.xuvdb)")
xuvdb.save("demo_output/scene.xuvdb", [grid])
back = xuvdb.load("demo_output/scene.xuvdb")[0]
pts = np.random.default_rng(0).uniform(-0.1, 0.9, size=(512, 3))
v0 = np.array([grid.sample_linear(p) for p in pts])
v1 = np.array([back.sample_linear(p) for p in pts])
ok("host trilinear identical after roundtrip", np.allclose(v0, v1, atol=1e-6),
   f"(max diff {np.abs(v0 - v1).max():.2e})")

# ---- 3) OpenVDB interop: real .vdb bytes, write then read
print("[3] write_vdb / read_vdb (.vdb)")
xuvdb.write_vdb("demo_output/scene.vdb", [grid])
with open("demo_output/scene.vdb", "rb") as f:
    magic = f.read(8)
ok(".vdb magic bytes", magic == b"\x20\x42\x44\x56\x00\x00\x00\x00", repr(magic))
vb = xuvdb.read_vdb("demo_output/scene.vdb", grid_name="shield")[0]
# band-safe points: |sdf| <= 2 voxels -> the whole trilinear stencil sits in the active band
band_pts = pts[np.abs([sdf_union(p) for p in pts]) <= 0.10][:64]
v_band = np.array([grid.sample_linear(p) for p in band_pts])
v2 = np.array([vb.sample_linear(p) for p in band_pts])
ok("narrow-band trilinear matches after .vdb roundtrip", np.allclose(v_band, v2, atol=1e-5),
   f"(max diff {np.abs(v_band - v2).max():.2e}, {len(band_pts)} pts)")
# irregular inactive values survive only inside sub-leaves that hold active band voxels
out_idx = (0, 12, 8)
ok("inactive value preserved in active-bearing sub-leaf",
   abs(float(vb.get_value(out_idx)) - float(grid.get_value(out_idx))) < 1e-6,
   f"(host={float(grid.get_value(out_idx)):+.4f} -> vdb={float(vb.get_value(out_idx)):+.4f})")
# far-field voxels whose whole 8^3 sub-leaf is inactive are dropped to background (OpenVDB
# active-mask semantics); XUVDB host sampling reads them, so far-field interpolation diverges
far_idx = (-1, 13, 10)
print(f"    note: inactive-only sub-leaf folds to background on .vdb readback "
      f"(host={float(grid.get_value(far_idx)):+.4f} -> vdb={float(vb.get_value(far_idx)):+.4f})")

# ---- 4) kernel sampling / writing (GpuVolume)
print("[4] GpuVolume: sample / sdf_normal / write_voxels")
vol = xuvdb.GpuVolume(grid)
top = np.array([[0.3, 0.45, 0.1]], np.float32)      # sphere-1 surface, away from sphere-2
center = np.array([[0.3, 0.2, 0.1]], np.float32)
d_surf = vol.sample(top, linear=True)[0]
d_in = vol.sample(center, linear=True)[0]
ok("SDF ~ 0 on surface", abs(d_surf) < 2e-2, f"(d={d_surf:+.4f})")
ok("SDF ~ -r at center", abs(d_in + 0.25) < 2e-2, f"(d={d_in:+.4f})")
n = vol.sdf_normal(top)[0]
ok("normal ~ +y on sphere top", np.allclose(n, [0, 1, 0], atol=5e-2), f"(n={np.round(n, 3).tolist()})")
gpu_lin = vol.sample(band_pts.astype(np.float32), linear=True)
ok("kernel linear == host linear", np.allclose(gpu_lin, v_band, atol=1e-5),
   f"(max diff {np.abs(gpu_lin - v_band).max():.2e})")
vol.write_voxels(center, np.array([-0.05], np.float32)).sync_to_host()
ok("kernel write -> host tree", abs(grid.sample_linear(center[0]) + 0.05) < 1e-6,
   f"(d={grid.sample_linear(center[0]):+.4f})")

# ---- 5) particles <-> volume
print("[5] scatter_particles + union_spheres")
drops = np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
fog = xuvdb.VdbGrid(voxel_size=0.05, name="liquid", grid_class="fog volume")
fog.scatter_particles(drops, h=4 * 0.05, weights=1.0)
m = sum(l.values.sum() for l in fog.leaves()) * 0.05**3
ok("splat mass conserved (unit-integral kernel)", abs(m - 2.0) < 1e-4, f"(mass={m:.6f}, expect 2)")
surf = xuvdb.VdbGrid(background=3 * 0.05, voxel_size=0.05, grid_class="level set")
surf.union_spheres(drops, radius=0.03)
ok("surface grid built", surf.n_leaves > 0, f"(leaves={surf.n_leaves})")

# ---- 6) DDA ray
print("[6] ray_surface_hit (DDA + bisection refine)")
t, point, value = xuvdb.ray_surface_hit(grid, (0.3, 0.2, 2.0), (0, 0, -1))
z_expect = 0.1 + 0.25  # first crossing: sphere-1 front surface
ok("ray hits front surface", t is not None and abs(point[2] - z_expect) < 3e-2,
   f"(t={t:.4f}, point=({point[0]:.3f},{point[1]:.3f},{point[2]:.4f}), value={float(value):+.4f})")

# ---- bonus: dense <-> sparse bridge
print("[+] from_dense / to_dense")
dense = np.linspace(0.15, -0.15, 64).reshape(4, 4, 4).astype(np.float32)
sp = xuvdb.VdbGrid.from_dense(dense, origin=(0, 0, 0), voxel_size=0.05,
                              background=0.15, grid_class="level set")
dense2, _ = sp.to_dense()
ok("dense roundtrip", dense2.shape == dense.shape and np.allclose(np.asarray(dense2), dense, atol=1e-6))

print("\nALL DEMO CHECKS PASSED")
