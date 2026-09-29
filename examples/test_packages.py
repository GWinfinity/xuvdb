"""Test that `quadrants` and `genesis-world` (installed via uv) work on this machine."""
import sys

print(f"Python: {sys.version}")

# ---------- Test 1: quadrants (Taichi fork, imported as `qd`) ----------
print("\n===== Test 1: quadrants =====", flush=True)
import numpy as np
import quadrants as qd

print(f"quadrants version: {qd.__version__}")
qd.init(arch=qd.cpu)

@qd.kernel
def double(src: qd.types.ndarray(), dst: qd.types.ndarray()):
    for i in src:
        dst[i] = 2.0 * src[i]

src = np.arange(8, dtype=np.float32)
dst = np.zeros_like(src)
double(src, dst)
assert np.allclose(dst, src * 2), f"quadrants kernel result wrong: {dst}"
print(f"kernel result: {dst.tolist()}  -> quadrants OK")

# ---------- Test 2: genesis-world ----------
print("\n===== Test 2: genesis-world =====", flush=True)
import genesis as gs

print(f"genesis version: {gs.__version__}")
gs.init(backend=gs.cpu, seed=0)

scene = gs.Scene(show_viewer=False)
scene.add_entity(gs.morphs.Plane())
box = scene.add_entity(gs.morphs.Box(size=(0.3, 0.3, 0.3), pos=(0.0, 0.0, 1.0)))
scene.build()

z0 = float(np.asarray(box.get_links_pos())[0, 2])
for _ in range(60):
    scene.step()
z1 = float(np.asarray(box.get_links_pos())[0, 2])
print(f"box height: start={z0:.3f} m, after 60 steps={z1:.3f} m")
assert z1 < z0, "box did not fall"
assert abs(z1 - 0.15) < 0.1, f"box did not settle near the ground: {z1}"
print("rigid-body sim (falling box) -> genesis OK")

print("\nALL TESTS PASSED")
