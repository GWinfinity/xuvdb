"""Tests for xuvdb.mesh_to_sdf: triangle mesh -> narrow-band level sets."""

import time

import numpy as np
import pytest

from xuvdb.mesh_to_sdf import dilated_mesh_shell, mesh_to_level_set


# --------------------------------------------------------------------------- meshes

def _uv_sphere(center, radius, n_lat=20, n_lon=40):
    """Lat-long sphere mesh (poles + rings), roughly 2 * n_lat * n_lon triangles."""
    center = np.asarray(center, dtype=np.float64)
    verts = [center + [0, 0, radius], center - [0, 0, radius]]  # +z / -z poles
    for i in range(1, n_lat):
        phi = np.pi * i / n_lat
        for j in range(n_lon):
            th = 2 * np.pi * j / n_lon
            verts.append(center + radius * np.array([np.sin(phi) * np.cos(th),
                                                       np.sin(phi) * np.sin(th),
                                                       np.cos(phi)]))
    faces = []
    base = 2 + np.arange(n_lat - 1) * n_lon  # first vertex of each ring
    for j in range(n_lon):
        j1 = (j + 1) % n_lon
        faces.append([0, base[0] + j1, base[0] + j])          # +z fan
        faces.append([1, base[-1] + j, base[-1] + j1])        # -z fan
    for i in range(n_lat - 2):
        for j in range(n_lon):
            j1 = (j + 1) % n_lon
            a, b = base[i] + j, base[i] + j1
            c, d = base[i + 1] + j, base[i + 1] + j1
            faces += [[a, b, d], [a, d, c]]
    return np.array(verts), np.array(faces, dtype=np.int64)


def _cube(center, half):
    c = np.asarray(center, dtype=np.float64)
    h = float(half)
    v = c + h * np.array([[sx, sy, sz] for sz in (-1, 1) for sy in (-1, 1)
                          for sx in (-1, 1)])
    # 12 triangles, two per face, outward winding.
    f = [[0, 2, 3], [0, 3, 1],    # z- face
         [4, 5, 7], [4, 7, 6],    # z+ face
         [0, 1, 5], [0, 5, 4],    # y- face
         [2, 6, 7], [2, 7, 3],    # y+ face
         [0, 4, 6], [0, 6, 2],    # x- face
         [1, 3, 7], [1, 7, 5]]    # x+ face
    return v, np.array(f, dtype=np.int64)


# --------------------------------------------------------------------------- tests

def test_sphere_level_set_sign_and_distance():
    radius, vs, band = 0.3, 0.01, 3.0
    center = np.array([0.05, -0.03, 0.08])
    verts, faces = _uv_sphere(center, radius)
    t0 = time.perf_counter()
    grid = mesh_to_level_set(verts, faces, vs, band=band)
    elapsed = time.perf_counter() - t0

    assert grid.grid_class == "level set"
    assert grid.background == pytest.approx(band * vs)
    assert elapsed < 30.0  # generous; binned broadphase is much faster

    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(300, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    # Sample just inside / just outside the surface so trilinear taps stay in-band.
    for offset in (-1.5 * vs, -0.5 * vs, 0.5 * vs, 1.5 * vs):
        pts = center + dirs * (radius + offset)
        sdf = np.asarray([grid.sample_linear(p) for p in pts], dtype=np.float64)
        analytic = np.linalg.norm(pts - center, axis=1) - radius
        assert np.all(np.sign(sdf) == np.sign(analytic))
        err = np.abs(np.abs(sdf) - np.abs(analytic))
        assert err.max() < 2.0 * vs, f"max distance error {err.max():.5f}"


def test_cube_octant_signs_and_face_distance():
    vs, band = 0.01, 3.0
    center = np.array([0.003, -0.007, 0.011])
    half = 0.15
    verts, faces = _cube(center, half)
    grid = mesh_to_level_set(verts, faces, vs, band=band)

    rng = np.random.default_rng(1)
    for sx, sy, sz in [[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]:
        s = np.array([sx, sy, sz])
        # inside, 0.02 from the sx face (kept asymmetric in y/z to avoid edges)
        p_in = center + s * np.array([half - 0.02, 0.05, 0.07])
        # outside, 0.05 from the sx face
        p_out = center + s * np.array([half + 0.05, 0.05, 0.07])
        assert grid.sample_linear(p_in) < 0, f"inside point {p_in} sampled >= 0"
        assert grid.sample_linear(p_out) > 0, f"outside point {p_out} sampled <= 0"
        # face midpoint at distance d along the face normal
        d = 0.025
        p_face = center + s * np.array([half + d, 0.0, 0.0])
        sdf = grid.sample_linear(p_face)
        assert abs(abs(sdf) - d) < 2.0 * vs, f"face distance error {abs(abs(sdf) - d):.5f}"
        assert np.sign(sdf) == np.sign(d)


def test_narrow_band_mask_and_background():
    vs, band = 0.01, 3.0
    center = np.array([0.02, 0.01, -0.01])
    verts, faces = _uv_sphere(center, 0.25)
    grid = mesh_to_level_set(verts, faces, vs, band=band)

    vals = grid.active_values()
    assert vals.size == grid.active_voxel_count > 0
    assert np.all(np.abs(vals) <= (band + 0.5) * vs)

    # Inactive voxels probe as (background, False) and sample as background.
    lo, hi = grid.bbox()
    corner = np.array([lo[0] - 3, lo[1] - 3, lo[2] - 3])
    value, active = grid.probe(tuple(int(i) for i in corner))
    assert not active
    assert value == pytest.approx(grid.background)
    assert grid.sample_linear(grid.index_to_world(corner)) == pytest.approx(grid.background)


def test_dilated_mesh_shell_cube():
    vs = 0.01
    center = np.array([0.003, -0.007, 0.011])
    half, hw = 0.15, 0.05
    verts, faces = _cube(center, half)
    shell = dilated_mesh_shell(verts, faces, hw, vs)
    assert shell.grid_class == "level set"

    # Equivalent to the SDF of a cube with half-size half + hw.
    big = half + hw
    rng = np.random.default_rng(2)
    for axis in range(3):
        for sgn in (-1, 1):
            d = 0.02
            off = np.zeros(3)
            off[axis] = sgn * (big + d)
            p = center + off
            sdf = shell.sample_linear(p)
            assert abs(sdf - d) < 2.0 * vs, f"shell distance error {abs(sdf - d):.5f}"
    # and just inside the shell surface the sign flips
    p_in = center + [half + hw - 0.01, 0.02, -0.03]
    q_in = center + [0.02, 0.03, half + hw - 0.015]
    assert shell.sample_linear(p_in) < 0
    assert shell.sample_linear(q_in) < 0
    assert shell.sample_linear(center) > 0  # deep interior of the original mesh is outside the shell
    assert shell.background == pytest.approx(3.0 * vs)


def test_sphere_volume_roundtrip():
    radius, vs = 0.3, 0.01
    center = np.array([0.05, -0.03, 0.08])
    verts, faces = _uv_sphere(center, radius, n_lat=16, n_lon=32)
    grid = mesh_to_level_set(verts, faces, vs, band=3.0)

    # Enclosed volume from the narrow band: flood-fill the connected component
    # of "not (active and > 0)" voxels that contains the negative band. The
    # positive band shell seals the interior off from the exterior (deep
    # interior samples as background, so slab zero-crossing integration cannot
    # see across it). tests/test_meshing.py does not exist, so we measure the
    # volume here instead of importing xuvdb.meshing.
    origins = np.array([leaf.origin for leaf in grid.leaves()])
    base = origins.min(axis=0) - 1
    dims = origins.max(axis=0) - base + 17
    values = np.full(tuple(dims), grid.background, dtype=np.float64)
    active = np.zeros(tuple(dims), dtype=bool)
    for leaf in grid.leaves():
        sl = tuple(slice(int(leaf.origin[a] - base[a]), int(leaf.origin[a] - base[a]) + leaf.dim)
                   for a in range(3))
        values[sl] = leaf.values
        active[sl] = leaf.active
    # Complement flood fill: grow the exterior inward from the array boundary,
    # blocked by the whole active band (watertight by continuity of the distance
    # field); everything unreachable is inside. Solid = enclosed inactive
    # interior plus the negative half of the band (robust to isolated parity
    # sign flips, unlike seeding from negative voxels).
    blocked = active
    passable = ~blocked
    exterior = np.zeros(tuple(dims), dtype=bool)
    exterior[0, :, :] = passable[0, :, :]
    exterior[-1, :, :] = passable[-1, :, :]
    exterior[:, 0, :] = passable[:, 0, :]
    exterior[:, -1, :] = passable[:, -1, :]
    exterior[:, :, 0] = passable[:, :, 0]
    exterior[:, :, -1] = passable[:, :, -1]
    while True:  # 6-connected dilation through passable voxels
        grown = exterior.copy()
        grown[:-1] |= exterior[1:] & passable[:-1]
        grown[1:] |= exterior[:-1] & passable[1:]
        grown[:, :-1] |= exterior[:, 1:] & passable[:, :-1]
        grown[:, 1:] |= exterior[:, :-1] & passable[:, 1:]
        grown[:, :, :-1] |= exterior[:, :, 1:] & passable[:, :, :-1]
        grown[:, :, 1:] |= exterior[:, :, :-1] & passable[:, :, 1:]
        if grown.sum() == exterior.sum():
            break
        exterior = grown
    inside = (~exterior & ~active) | (active & (values < 0))
    volume = float(inside.sum()) * vs ** 3
    exact = 4.0 / 3.0 * np.pi * radius ** 3
    rel_err = abs(volume - exact) / exact
    assert rel_err < 0.05, f"volume {volume:.5f} vs {exact:.5f} ({rel_err:.2%})"


def test_invalid_inputs():
    verts, faces = _cube([0, 0, 0], 0.1)
    with pytest.raises(ValueError):
        mesh_to_level_set(verts, faces, (0.01, 0.02, 0.01))  # anisotropic
    with pytest.raises(ValueError):
        mesh_to_level_set(verts, faces, 0.01, band=-1.0)
    faces_bad = faces.copy()
    faces_bad[0, 0] = len(verts) + 10  # out-of-range vertex index
    with pytest.raises(ValueError):
        mesh_to_level_set(verts, faces_bad, 0.01)
    with pytest.raises(ValueError):
        mesh_to_level_set(verts, faces, 1e-6)  # absurdly fine -> huge domain
