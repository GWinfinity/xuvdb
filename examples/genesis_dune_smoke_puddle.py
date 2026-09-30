"""Dune + smoke + puddle on authored ground: an xuvdb-authored scene simulated in genesis-world and captured back into xuvdb.

One world (meters, z-up), both directions of the xuvdb <-> genesis bridge:

  pre-sim    author the ground and the dune as xuvdb level sets - an outdoor surface that
             reads flat but undulates by ~+-1 cm, an indoor floor plate and a doorway sill
             (门槛, ~4 cm) where the room meets the outside - and ray-query them to lay the
             scene out (dune seat height, puddle drop height, smoke jet footprint). Exports
             `dune.vdb` for DCC tools and a watertight OBJ that genesis fills with MPM sand
             particles; the ground goes to genesis as a visual heightfield Terrain plus
             analytic plane/box prims (same constants) carrying the physics;
  simulate   genesis on one clock: MPM sand (the dune, avalanching on its lee face),
             PBD.Liquid (a blob splashing into a puddle that drains against the sill) and
             the stable-fluids SFSolver (smoke rising from a jet). genesis's SF grid always
             covers the unit cube; the world-space box it maps to is recorded here and used
             as the grid transform on export. All solvers share the substep rate (a genesis
             constraint); substeps=16 hits the solver's CFL suggestion for grid_density=64.
  post-sim   pull the fields back into xuvdb: particle phases -> fog density
             (`scatter_particles`) plus a particle level-set surface (`union_spheres`), the
             dense smoke field -> sparse (`from_dense`, the README's "sparsify the engine's
             dense qd.field" flow). One multi-grid `desert.xuvdb` + one `.vdb` per field.

Run:  python examples/genesis_dune_smoke_puddle.py [--quick] [--frames N]
Outputs land in demo_output/genesis/.
"""
import argparse
import time
from pathlib import Path

import numpy as np

import xuvdb

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "demo_output" / "genesis"

# ---------------------------------------------------------------- dune shape (the one source of truth)
DUNE_CX, DUNE_CY = 0.55, 0.45
DUNE_H = 0.16                      # crest height above the local ground [m]
RX_UP, RX_DOWN, RY = 0.26, 0.15, 0.18   # stoss (-x, gentle) vs lee (+x, steep) radii [m]


def dune_height(x, y):
    """Barchan-ish profile: cos dome, stretched upwind so the lee face exceeds the sand's
    angle of repose and avalanches once released. Seats on top of the local ground."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rx = np.where(x < DUNE_CX, RX_UP, RX_DOWN)
    r = np.sqrt(((x - DUNE_CX) / rx) ** 2 + ((y - DUNE_CY) / RY) ** 2)
    h = DUNE_H * np.cos(0.5 * np.pi * np.clip(r, 0.0, 1.0)) ** 1.35
    return np.where(r < 1.0, h, 0.0)


# ---------------------------------------------------------------- ground: flat-looking outdoor undulation + indoor floor + doorway sill
GROUND_X0, GROUND_X1 = -1.00, 1.60
GROUND_Y0, GROUND_Y1 = -0.60, 1.40
GROUND_HS, GROUND_VS = 0.01, 0.0025     # terrain grid pitch / height quantization [m]
SILL_LO, SILL_HI = -0.05, 0.05     # sill strip across the doorway line x=0
SILL_TOP = 0.04                    # sill crest above the outdoor datum [m]
FLOOR_Z = 0.01                     # indoor floor sits a step above the outdoor datum
DOOR_Y0, DOOR_Y1 = 0.05, 0.85      # the sill spans the doorway width only


def ground_h(x, y):
    """Outdoor ground that looks flat but undulates ~+-1 cm on 0.2-0.7 m wavelengths, a flat
    indoor floor plate left of the doorway, and a raised sill strip bridging the two. The
    undulation flattens smoothly under the dune footprint: the dune hides that patch anyway,
    and a locally flat contact keeps the penalty-coupled sand from being shaken fluid."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    t = np.clip((x - SILL_HI) / 0.25, 0.0, 1.0)   # undulation fades out at the sill
    taper = t * t * (3.0 - 2.0 * t)
    r2 = ((x - DUNE_CX) / (RX_UP + 0.10)) ** 2 + ((y - DUNE_CY) / (RY + 0.10)) ** 2
    f = np.clip((r2 - 1.0) / 0.6, 0.0, 1.0)       # and under the dune footprint
    flat = f * f * (3.0 - 2.0 * f)
    und = (0.005 * np.sin(2 * np.pi * x / 0.73 + 1.3) * np.sin(2 * np.pi * y / 0.61 + 0.7)
           + 0.0035 * np.sin(2 * np.pi * x / 0.31 + 4.0) * np.cos(2 * np.pi * y / 0.41 + 2.1)
           + 0.002 * np.sin(2 * np.pi * (x + y) / 0.23 + 0.8))
    h = taper * flat * und
    h = np.where(x < SILL_LO, FLOOR_Z, h)          # indoor floor plate
    in_door = (y > DOOR_Y0) & (y < DOOR_Y1)        # sill ramps: outdoor side 0->top, indoor side floor->top
    ramp_hi = np.clip((SILL_HI - x) / 0.02, 0.0, 1.0)
    ramp_lo = np.clip((x - SILL_LO) / 0.02, 0.0, 1.0)
    sill = np.where(x >= 0.0, SILL_TOP * ramp_hi, FLOOR_Z + (SILL_TOP - FLOOR_Z) * ramp_lo)
    return np.where(in_door, np.maximum(h, sill), h)


def dune_top(x, y):
    """The dune seated on the ground (the one source of truth for sand geometry)."""
    return ground_h(x, y) + dune_height(x, y)


def heightfield_obj(path, x0, x1, y0, y1, top_fn, bottom_fn, nx=96, ny=84):
    """Watertight heightfield mesh between two height functions (top surface + skirt + bottom cap)."""
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = top_fn(X, Y)
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    vid = lambda i, j: i * ny + j
    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            faces.append((a, b, c))       # (b-a)x(c-a) points +z: top faces up
            faces.append((a, c, d))

    ring = ([(i, 0) for i in range(nx - 1)] + [(nx - 1, j) for j in range(ny - 1)]
            + [(i, ny - 1) for i in range(nx - 1, 0, -1)] + [(0, j) for j in range(ny - 1, 0, -1)])
    n_top = len(verts)
    skirt = np.stack([X.ravel()[[vid(i, j) for i, j in ring]],
                      Y.ravel()[[vid(i, j) for i, j in ring]],
                      bottom_fn(X.ravel()[[vid(i, j) for i, j in ring]],
                                Y.ravel()[[vid(i, j) for i, j in ring]])], axis=1)
    verts = np.concatenate([verts, skirt], axis=0)
    for a in range(len(ring)):                      # skirt quads: top ring -> bottom ring
        b = (a + 1) % len(ring)
        ta, tb = vid(*ring[a]), vid(*ring[b])
        ba, bb = n_top + a, n_top + b
        faces.append((ta, tb, bb))
        faces.append((ta, bb, ba))
    center = len(verts)                             # bottom cap: fan around the ring
    verts = np.concatenate([verts, [[0.5 * (x0 + x1), 0.5 * (y0 + y1),
                                     float(bottom_fn(np.array([0.5 * (x0 + x1)]), np.array([0.5 * (y0 + y1)]))[0])]]],
                           axis=0)
    for a in range(len(ring)):
        faces.append((n_top + a, n_top + (a + 1) % len(ring), center))

    import trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    trimesh.repair.fix_normals(mesh)
    if not mesh.is_watertight:
        raise RuntimeError(f"{path} is not watertight; particle fill would be wrong")
    mesh.export(path)
    return float(mesh.volume)


def write_dune_obj(path, nx=96, ny=84, half=0.30):
    """The dune seated on the ground: top follows ground_h + dune profile, skirt buried 4 mm
    into the ground so no gap opens under it."""
    return heightfield_obj(
        path, DUNE_CX - half, DUNE_CX + half, DUNE_CY - half, DUNE_CY + half,
        top_fn=dune_top, bottom_fn=lambda x, y: ground_h(x, y) - 0.004, nx=nx, ny=ny,
    )


def band_levelset(x0, x1, y0, y1, z0, z1, top_fn, voxel, band, name):
    """Author a heightfield as an xuvdb narrow-band level set. The signed value is the vertical
    distance scaled by the local surface slope, so it is exact along verticals (sign is globally
    correct); a real SDF would need a closest-point query, which the band and the vertical-ray
    queries here don't need."""
    xs = np.arange(x0, x1 + 1e-9, voxel)
    ys = np.arange(y0, y1 + 1e-9, voxel)
    zs = np.arange(z0, z1 + 1e-9, voxel)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    top = top_fn(X, Y)
    gx, gy = np.gradient(top, xs, ys, edge_order=2)
    scale = np.sqrt(1.0 + gx ** 2 + gy ** 2)
    dense = np.empty((len(xs), len(ys), len(zs)), dtype=np.float32)
    for k, z in enumerate(zs):
        dense[:, :, k] = ((z - top) / scale).astype(np.float32)
    dense = np.clip(dense, -band * voxel, band * voxel)
    return xuvdb.VdbGrid.from_dense(
        dense, voxel_size=(voxel, voxel, voxel), origin_world=(xs[0], ys[0], zs[0]),
        background=band * voxel, name=name, grid_class="level set",
    )


def build_dune_grid(voxel=0.0125, band=3.0):
    """The seated dune (ground + profile) as an xuvdb level set."""
    half = 0.34
    return band_levelset(
        DUNE_CX - half, DUNE_CX + half, DUNE_CY - half, DUNE_CY + half,
        -0.05, DUNE_H + FLOOR_Z + 6 * voxel, dune_top, voxel, band, "dune_sdf",
    )


def build_ground_grid(voxel=0.0125, band=3.0):
    """The whole authored ground (undulation + floor + sill) as an xuvdb level set."""
    return band_levelset(
        GROUND_X0, GROUND_X1, GROUND_Y0, GROUND_Y1,
        -4 * voxel, SILL_TOP + 6 * voxel, ground_h, voxel, band, "ground_sdf",
    )


def terrain_z(grid, x, y):
    """Vertical-ray surface query against the xuvdb dune (host DDA)."""
    hit = xuvdb.ray_surface_hit(grid, (float(x), float(y), 1.0), (0.0, 0.0, -1.0))
    return None if hit is None else float(hit[1][2])


def ok(label, cond, detail=""):
    if cond:
        print(f"  PASS  {label} {detail}")
    else:
        raise AssertionError(f"{label} {detail}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300, help="scene.step() calls (dt=5e-3 each)")
    ap.add_argument("--sf-res", type=int, default=64, help="stable-fluids grid resolution")
    ap.add_argument("--quick", action="store_true", help="tiny CI-style run (8 frames, res 32)")
    ap.add_argument("--no-render", action="store_true", help="skip genesis camera renders")
    ap.add_argument("--backend", choices=["cpu", "cuda"], default="cpu",
                    help="gs.init backend (default: cpu)")
    args = ap.parse_args()
    if args.quick:
        args.frames, args.sf_res = 8, 32

    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("render_*.png"):
        stale.unlink()

    # ===== 1) pre-sim: author the ground + dune in xuvdb, lay the scene out with ray queries
    print("[1] pre-sim: xuvdb ground/dune authoring + layout queries")
    t0 = time.perf_counter()
    ground = build_ground_grid()
    ground.prune()
    dune = build_dune_grid()
    dune.prune()
    crest = terrain_z(dune, DUNE_CX, DUNE_CY)
    # trilinear crossing across neighboring columns smears the isosurface by a fraction of a voxel
    ok("dune ray hit at crest", crest is not None and abs(crest - dune_top(DUNE_CX, DUNE_CY)) < 0.25 * 0.0125,
       f"(ray z={crest:.4f} m)")
    xuvdb.write_vdb(OUT / "dune.vdb", [dune])
    mesh_volume = write_dune_obj(OUT / "dune.obj")
    print(f"    dune_sdf: {dune.n_leaves} leaves, {dune.active_voxel_count} active voxels "
          f"[{time.perf_counter() - t0:.1f}s]")
    ok("dune.vdb written", (OUT / "dune.vdb").stat().st_size > 1000)

    # ground queries: the sill, the indoor floor, and the flat-looking outdoor undulation
    sill_z = terrain_z(ground, 0.0, 0.45)
    ok("sill ray hit", sill_z is not None and abs(sill_z - SILL_TOP) < 0.004,
       f"(ray z={sill_z:.4f} m)")
    floor_z = terrain_z(ground, -0.5, 0.45)
    ok("indoor floor ray hit", floor_z is not None and abs(floor_z - FLOOR_Z) < 0.004,
       f"(ray z={floor_z:.4f} m)")
    und_pts = np.linspace(0.2, 1.4, 24)      # south of the dune: the dune flattens its own patch
    und_z = np.array([terrain_z(ground, x, 0.12) for x in und_pts])
    ok("outdoor ground reads flat but undulates",
       float(np.std(und_z)) > 0.0015 and float(np.abs(und_z - FLOOR_Z).max()) < 0.015,
       f"(std {np.std(und_z) * 1e3:.1f} mm, max dev {np.abs(und_z - FLOOR_Z).max() * 1e3:.1f} mm)")

    PUDDLE_XY = (0.30, 0.10)      # in front of the dune, off its footprint
    ok("puddle site off-dune", terrain_z(dune, *PUDDLE_XY) is None)
    puddle_ground = terrain_z(ground, *PUDDLE_XY)
    WATER_LOWER = np.array([PUDDLE_XY[0] - 0.10, PUDDLE_XY[1] - 0.10, puddle_ground + 0.04])
    WATER_UPPER = np.array([PUDDLE_XY[0] + 0.10, PUDDLE_XY[1] + 0.10, puddle_ground + 0.14])
    # a ~4 cm drop onto the local ground; PBD (not MPM) for the puddle: genesis's inviscid MPM
    # liquid spreads until it is a one-particle-deep sheet (flood, and sub-voxel in the export),
    # while PBD's viscosity relaxation keeps the blob contained - a puddle that survives as a
    # level set at 1 cm voxels, draining against the doorway sill
    water_top0 = float(WATER_UPPER[2])
    SMOKE_LO = np.array([0.80, 0.15, 0.0])    # world box the SF unit cube maps to
    SMOKE_SIZE = np.array([0.60, 0.60, 0.60])
    jet_world = SMOKE_LO + np.array([0.5, 0.5, 0.03]) * SMOKE_SIZE
    jet_world[2] += float(terrain_z(ground, jet_world[0], jet_world[1]))   # sit on the local ground
    ok("jet site off-dune", terrain_z(dune, jet_world[0], jet_world[1]) is None)
    jet_unit = tuple((jet_world - SMOKE_LO) / SMOKE_SIZE)   # genesis SF lives in the unit cube

    # ===== 2) simulate in genesis-world
    print("[2] genesis: MPM sand (dune) + PBD liquid (puddle) + stable-fluids smoke")
    import genesis as gs
    from genesis.utils.misc import qd_to_numpy

    gs.init(backend=gs.cuda if args.backend == "cuda" else gs.cpu,
            seed=0, precision="32", logging_level="warning")

    import quadrants as qd

    @qd.data_oriented
    class UpJet:
        """Static vertical smoke jet following the SFSolver jet protocol (get_tan_dir/get_factor)."""

        def __init__(self, world_center, radius):
            self.center = qd.Vector(world_center)
            self.radius = float(radius)

        @qd.func
        def get_tan_dir(self, t: float):
            return qd.Vector([0.0, 0.0, 1.0])

        @qd.func
        def get_factor(self, i: int, j: int, k: int, dx: float, t: float):
            d = (qd.Vector([i, j, k], dt=gs.qd_float) * dx - self.center).norm(gs.EPS)
            return (1.0 - d / self.radius) if d < self.radius else 0.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=5e-3, substeps=16),
        # the MPM grid boundary carries a ~3-cell safety pad, so the domain floor sits below the
        # ground; the ground itself comes in as analytic prims (see below) plus a visual-only
        # heightfield terrain.
        # mpm_pbd coupling is cut: the puddle sits off-dune and one-way media keep both calm -
        # with it on, the spreading sheet kept pushing the whole pile (sand p99 |v| ~0.6 for the
        # entire run); without it the dune freezes (p99 < 0.01) and keeps 86% of its crest.
        mpm_options=gs.options.MPMOptions(lower_bound=(-0.1, -0.1, -0.08), upper_bound=(1.5, 1.0, 0.6)),
        pbd_options=gs.options.PBDOptions(
            max_density_solver_iterations=10,
            lower_bound=(-0.1, -0.1, -0.08), upper_bound=(1.5, 1.0, 0.6),
        ),
        coupler_options=gs.options.LegacyCouplerOptions(mpm_pbd=False),
        sf_options=gs.options.SFOptions(res=args.sf_res, solver_iters=25, decay=0.025),
        show_viewer=False,
    )
    hf_x = np.arange(GROUND_X0, GROUND_X1 + 1e-9, GROUND_HS)
    hf_y = np.arange(GROUND_Y0, GROUND_Y1 + 1e-9, GROUND_HS)
    height_field = ground_h(*np.meshgrid(hf_x, hf_y, indexing="ij")) / GROUND_VS
    # the authored ground renders as a heightfield terrain, but its PHYSICS ground is a set of
    # analytic prims derived from the same constants: the mesh-SDF penalty contact of a terrain
    # geom keeps agitating MPM sand (probed: the pile liquefies to ~15% of its crest even over a
    # locally flattened patch), while plane/box prims freeze it (crest retention ~97%).
    scene.add_entity(
        morph=gs.morphs.Terrain(height_field=height_field, horizontal_scale=GROUND_HS,
                                vertical_scale=GROUND_VS, pos=(GROUND_X0, GROUND_Y0, 0.0),
                                collision=False),
        surface=gs.surfaces.Default(color=(0.60, 0.56, 0.48, 1.0)),
    )
    scene.add_entity(morph=gs.morphs.Plane(visualization=False))
    scene.add_entity(
        morph=gs.morphs.Box(lower=(-1.0, GROUND_Y0, -0.05), upper=(SILL_LO, GROUND_Y1, FLOOR_Z),
                            fixed=True, visualization=False),
        material=gs.materials.Rigid(needs_coup=True, coup_friction=0.4),
    )
    scene.add_entity(
        morph=gs.morphs.Box(lower=(SILL_LO, DOOR_Y0, -0.05), upper=(SILL_HI, DOOR_Y1, SILL_TOP),
                            fixed=True, visualization=False),
        material=gs.materials.Rigid(needs_coup=True, coup_friction=0.4),
    )
    # genesis's MPM.Sand liquefies a static pile in this build (its strain-space projection
    # drops the plastic-memory term of the original taichi formulation, so the pressure-dependent
    # strength vanishes no matter the friction angle / E / particle density - probed 33-45 deg,
    # E up to 4e6, 2 particles per cell, dt down to 1.6e-4: the dune melts to a pancake).
    # ElastoPlastic's von Mises clamp gives the same visual for this scene: the pile slumps while
    # its shear stress ~ rho*g*H*sin(theta)/2 exceeds the yield stress and freezes near
    # sin(theta*) ~ 2*yield/(rho*g*H) - 500 Pa lets the lee face slump once and freezes the
    # decoupled pile near 86% of its initial crest.
    sand = scene.add_entity(
        morph=gs.morphs.Mesh(file=str(OUT / "dune.obj")),
        material=gs.materials.MPM.ElastoPlastic(E=1e6, von_mises_yield_stress=500.0),
        surface=gs.surfaces.Default(color=(0.85, 0.66, 0.4, 1.0), vis_mode="particle"),
    )
    water = scene.add_entity(
        morph=gs.morphs.Box(lower=tuple(WATER_LOWER), upper=tuple(WATER_UPPER)),
        material=gs.materials.PBD.Liquid(sampler="regular", viscosity_relaxation=0.3),
        surface=gs.surfaces.Default(color=(0.25, 0.5, 0.9, 0.5), vis_mode="particle"),
    )
    scene.sim.sf_solver.set_jets([UpJet(jet_unit, radius=0.09)])
    cam = None
    if not args.no_render:
        try:
            cam = scene.add_camera(res=(760, 460), pos=(2.0, -1.35, 0.95), lookat=(0.65, 0.42, 0.12), fov=38)
        except Exception as e:  # noqa: BLE001 - rendering is optional (headless servers)
            print(f"    camera unavailable ({type(e).__name__}: {e}); continuing without renders")
    scene.build()

    def host_xyz(t):
        """Particle getters return CUDA tensors on the gpu backend; move to host."""
        if hasattr(t, "cpu"):
            t = t.cpu().numpy()
        return np.asarray(t).reshape(-1, 3)

    sand_p0 = host_xyz(sand.get_particles_pos())
    print(f"    sand particles: {len(sand_p0)}  (mesh volume {mesh_volume * 1e3:.1f} L)")
    water_p0 = host_xyz(water.get_particles_pos())
    print(f"    water particles: {len(water_p0)}")

    sf = scene.sim.sf_solver
    n = args.frames
    smoke_caps = {}       # frame -> dense (res,res,res) smoke density snapshot
    capture_frames = {n // 3, n // 2, 2 * n // 3, n - 1}
    render_frames = {0, n // 3, 2 * n // 3, n - 1}
    t_sim = time.perf_counter()
    for f in range(n):
        scene.step()
        if f in capture_frames:
            q = qd_to_numpy(sf.grid.q).astype(np.float32)[..., 0]
            q[q < 2e-3] = 0.0
            smoke_caps[f] = q
        if cam is not None and f in render_frames:
            try:
                rgb, *_ = cam.render()
                from PIL import Image
                Image.fromarray(np.asarray(rgb)).save(OUT / f"render_f{f:04d}.png")
            except Exception as e:  # noqa: BLE001
                print(f"    render failed at f{f} ({type(e).__name__}: {e}); disabling camera")
                cam = None
        if (f + 1) % 50 == 0 or f == n - 1:
            print(f"    frame {f + 1}/{n}  t={float(scene.sim.cur_t):.2f}s  "
                  f"[{time.perf_counter() - t_sim:.0f}s]")
    t_sim = time.perf_counter() - t_sim

    sand_p1 = host_xyz(sand.get_particles_pos())
    sand_v1 = host_xyz(sand.get_particles_vel())
    water_p1 = host_xyz(water.get_particles_pos())
    water_v1 = host_xyz(water.get_particles_vel())

    # ===== 3) physics sanity checks on the raw particle data
    print("[3] physics checks")
    zs0 = np.sort(sand_p0[:, 2])
    zs1 = np.sort(sand_p1[:, 2])
    # the decoupled pile slumps once on release and freezes (measured: crest 0.120 of 0.139,
    # p99 speed < 0.01 m/s). what must NOT happen is the MPM.Sand-style liquefaction, which
    # flattens the pile to ~15% of its crest within 0.5 s with the whole pile still moving.
    ok("dune crest survived", zs1[-100].mean() > 0.7 * zs0[-100].mean(),
       f"(p100 z: {zs0[-100].mean():.3f} -> {zs1[-100].mean():.3f} m)")
    sp = np.linalg.norm(sand_v1, axis=1)
    full = float(scene.sim.cur_t) > 0.5     # the settle/drop/spread checks need real sim time
    if full:
        # the seated dune starts ~8 mm buried in the terrain and pops out over the first ~0.1 s
        ok("sand settled", float(np.quantile(sp, 0.99)) < 0.15,
           f"(p99 |v| = {np.quantile(sp, 0.99):.3f}, max = {sp.max():.3f} m/s)")
    if full:
        span0 = np.ptp(water_p0[:, 0]) + np.ptp(water_p0[:, 1])
        span1 = np.ptp(water_p1[:, 0]) + np.ptp(water_p1[:, 1])
        ok("water landed and spread", water_p1[:, 2].max() < water_top0 + 0.03 and span1 > span0,
           f"(z_max {water_p1[:, 2].max():.3f} m, x+y span {span0:.2f} -> {span1:.2f} m)")
        # ripples/jitter are legitimate; "ponded" means a thin, locally contained blob whose
        # bulk creeps slowly (PBD's surface layer jiggles, hence median not mean/max)
        z995 = float(np.quantile(water_p1[:, 2], 0.995))
        v_med = float(np.median(np.linalg.norm(water_v1, axis=1)))
        ok("puddle ponded thin", z995 < 0.12, f"(99.5th pct z = {z995:.3f} m)")
        ok("puddle mostly calm", v_med < 0.5, f"(median |v| = {v_med:.3f} m/s)")
        ok("smoke emitted", smoke_caps[n - 1].sum() > 0.5,
           f"(final integral {smoke_caps[n - 1].sum():.1f})")

    # ===== 4) post-sim: capture everything back into xuvdb
    print("[4] post-sim: particles/fields -> xuvdb grids")
    t1 = time.perf_counter()

    def phase_grids(points, name, voxel, radius):
        fog = xuvdb.VdbGrid(voxel_size=(voxel,) * 3, name=f"{name}_density", grid_class="fog volume")
        fog.scatter_particles(points, h=2.5 * voxel)
        surf = xuvdb.VdbGrid(background=3 * voxel, voxel_size=(voxel,) * 3,
                             name=f"{name}_surface", grid_class="level set")
        surf.union_spheres(points, radius=radius)
        surf.prune()
        return fog, surf

    sand_fog, sand_surf = phase_grids(sand_p1, "sand", voxel=0.015, radius=0.0075)
    water_fog, water_surf = phase_grids(water_p1, "puddle", voxel=0.010, radius=0.005)

    voxel_sf = float(SMOKE_SIZE[0] / args.sf_res)
    smoke_grids = []
    for f, q in smoke_caps.items():
        g = xuvdb.VdbGrid.from_dense(q, voxel_size=(voxel_sf,) * 3, origin_world=tuple(SMOKE_LO),
                                     background=0.0, name=f"smoke_f{f}", grid_class="fog volume")
        g.prune()
        smoke_grids.append(g)
    smoke_final = smoke_grids[-1]
    occ = smoke_final.active_voxel_count / args.sf_res ** 3
    print(f"    smoke occupancy {occ:.1%} (sparse stores {smoke_final.active_voxel_count} of "
          f"{args.sf_res ** 3} cells)  [{time.perf_counter() - t1:.1f}s]")

    # ===== 5) write outputs + roundtrip check
    print("[5] outputs")
    grids = [ground, dune, sand_fog, sand_surf, water_fog, water_surf] + smoke_grids
    xuvdb.save(OUT / "desert.xuvdb", grids, compress=True)
    xuvdb.write_vdb(OUT / "sand_surface.vdb", [sand_surf])
    xuvdb.write_vdb(OUT / "puddle_surface.vdb", [water_surf])
    xuvdb.write_vdb(OUT / "smoke.vdb", [smoke_final])

    back = {g.name: g for g in xuvdb.load(OUT / "desert.xuvdb")}
    ok(".xuvdb roundtrip keeps every grid", len(back) == len(grids), f"({len(back)} grids)")
    p_crest = (DUNE_CX, DUNE_CY, dune_height(DUNE_CX, DUNE_CY))
    ok(".xuvdb roundtrip preserves the dune SDF",
       abs(back["dune_sdf"].sample_linear(p_crest) - dune.sample_linear(p_crest)) < 1e-6,
       f"(crest sample {dune.sample_linear(p_crest):.2e} -> "
       f"{back['dune_sdf'].sample_linear(p_crest):.2e})")
    med = np.median([back["puddle_surface"].sample_linear(p) for p in water_p1[::37]])
    ok("puddle SDF straddles the particles", abs(med) < 0.01, f"(median d = {med * 1e3:.1f} mm)")
    for f in (OUT / "smoke.vdb", OUT / "sand_surface.vdb", OUT / "puddle_surface.vdb", OUT / "dune.vdb"):
        with open(f, "rb") as fh:
            ok(f"{f.name} magic", fh.read(8) == b"\x20\x42\x44\x56\x00\x00\x00\x00")
    print("    files:")
    for f in sorted(OUT.iterdir()):
        print(f"      {f.name:24s} {f.stat().st_size / 1e3:9.1f} kB")

    # ===== 6) preview png (matplotlib, best effort)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
        q = smoke_caps[n - 1]
        im = axs[0].imshow(q.max(axis=1).T, origin="lower", extent=(
            SMOKE_LO[0], SMOKE_LO[0] + SMOKE_SIZE[0], SMOKE_LO[2], SMOKE_LO[2] + SMOKE_SIZE[2]),
            cmap="inferno", aspect="equal")
        axs[0].set_title(f"smoke (max-y projection), frame {n - 1}")
        plt.colorbar(im, ax=axs[0], fraction=0.046)
        axs[1].scatter(sand_p1[::4, 0], sand_p1[::4, 1], s=1, c=sand_p1[::4, 2], cmap="copper")
        axs[1].scatter(water_p1[::4, 0], water_p1[::4, 1], s=2, c=water_p1[::4, 2], cmap="winter")
        axs[1].set_xlim(-0.1, 1.5); axs[1].set_ylim(-0.1, 1.0); axs[1].set_aspect("equal")
        axs[1].set_title("particles top-down (color = z)")
        xs = np.linspace(-0.9, 1.45, 500)
        axs[2].plot(xs, ground_h(xs, DUNE_CY), "-", color="gray", lw=1.5, label="ground")
        axs[2].plot(xs, dune_top(xs, DUNE_CY), "-", color="peru", lw=2, label="dune (seated)")
        axs[2].scatter(sand_p1[::8, 0], sand_p1[::8, 2], s=1, alpha=0.3, c="peru")
        axs[2].scatter(water_p1[::4, 0], water_p1[::4, 2], s=2, alpha=0.5, c="dodgerblue")
        axs[2].set_ylim(-0.05, 0.45); axs[2].set_xlim(-0.9, 1.45)
        axs[2].set_title("side view x-z (floor + sill at left)"); axs[2].legend(loc="upper right")
        fig.suptitle("genesis-world: MPM sand dune + PBD puddle + stable-fluids smoke -> xuvdb")
        fig.tight_layout()
        fig.savefig(OUT / "preview.png", dpi=110)
        print(f"    preview -> {OUT / 'preview.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"    preview skipped ({type(e).__name__}: {e})")

    print(f"\nALL CHECKS PASSED  (sim {t_sim:.0f}s, total {time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
