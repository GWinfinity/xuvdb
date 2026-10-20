"""Boundary-aware `sdf_normal` / `sample_gradient`.

Regression coverage for the narrow-band limitation found in the 1.3.3 benchmarks: stencil taps
crossing the band edge used to degrade into 0/0 noise whose sign was float-rounding luck
(CPU/GPU directions could flip). The contract is now:

- in-band points: classic central difference, byte-identical to the pre-fix kernel;
- band-edge points: one-sided differences against the query value (first-order, deterministic);
- outside the band: exact zero vector, `valid=0` (never NaN/garbage);
- `extend=k`: out-of-band points adopt the nearest active voxel's normal (Chebyshev distance
  <= k) - normal-field dilation for shading/collision shells.
"""
import numpy as np
import pytest

import xuvdb

C = np.array([0.3, 0.2, 0.1])
R = 0.25
H = 0.05              # voxel size
BAND = 3 * H          # background == band half-width (world units)


def make_sphere(radius=R):
    grid = xuvdb.VdbGrid(background=BAND, voxel_size=H, leaf_log2=4,
                         name="s", grid_class="level set")
    grid.stamp_sphere(C, radius=radius, band=3.0)
    return grid


def radial(pts, center=C):
    d = np.asarray(pts, dtype=np.float64) - center
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def on_sphere(rng, n, radius):
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return (C + dirs * radius).astype(np.float32), dirs


def test_inband_matches_classic_central_difference():
    """Well inside the band every tap is active: arithmetic must equal the old kernel exactly."""
    grid = make_sphere()
    vol = xuvdb.GpuVolume(grid)
    pts, _ = on_sphere(np.random.default_rng(0), 256, R - H)   # sdf ~= -1 voxel
    f = lambda q: vol.sample(np.ascontiguousarray(q, np.float32), linear=True)
    d = np.stack([
        f(pts + [H, 0, 0]) - f(pts - [H, 0, 0]),
        f(pts + [0, H, 0]) - f(pts - [0, H, 0]),
        f(pts + [0, 0, H]) - f(pts - [0, 0, H]),
    ], axis=1)
    ref = d / np.linalg.norm(d, axis=1, keepdims=True)
    got, valid = vol.sdf_normal(pts, return_valid=True)
    assert np.all(valid == 1)
    assert np.allclose(got, ref, atol=2e-5), np.abs(got - ref).max()


def test_inband_radial_unit_and_host_agreement():
    grid = make_sphere()
    vol = xuvdb.GpuVolume(grid)
    pts, dirs = on_sphere(np.random.default_rng(1), 128, R)    # on the surface
    n, valid = vol.sdf_normal(pts, return_valid=True)
    assert np.all(valid == 1)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-5)
    assert np.allclose((n * dirs).sum(axis=1), 1.0, atol=5e-2)
    host = grid.sample_gradient(pts)
    assert np.allclose(np.linalg.norm(host, axis=1), 1.0, atol=5e-2)
    assert np.allclose(host / np.linalg.norm(host, axis=1, keepdims=True), n, atol=5e-2)


def test_out_of_band_is_zero_deterministic_no_nan():
    grid = make_sphere()
    vol = xuvdb.GpuVolume(grid)
    pts, _ = on_sphere(np.random.default_rng(2), 512, R + 8 * H)   # sdf ~= +8 voxels > band
    n1, v1 = vol.sdf_normal(pts, return_valid=True)
    n2, v2 = vol.sdf_normal(pts, return_valid=True)
    assert np.array_equal(n1, n2) and np.array_equal(v1, v2)   # deterministic, not rounding luck
    assert np.all(n1 == 0.0)
    assert np.all(v1 == 0)
    assert not np.isnan(n1).any()


def test_extend_donates_nearest_inband_normal():
    # Large sphere (R = 40 voxels): normals vary slowly across the shell, so the direction at the
    # donor approximates the direction at the query - the assertion targets the extension code,
    # not the sphere's curvature. (On a 5-voxel sphere, a donor 4 voxels away legitimately
    # differs by ~30-45 deg.)
    grid = make_sphere(radius=40 * H)
    vol = xuvdb.GpuVolume(grid)
    pts, dirs = on_sphere(np.random.default_rng(3), 256, 40 * H + 5 * H)  # 2 voxels past the band edge (band = 3H)
    n0, v0 = vol.sdf_normal(pts, return_valid=True)
    assert np.all(n0 == 0.0) and np.all(v0 == 0)                     # without extend: zero vector
    n, v = vol.sdf_normal(pts, extend=4, return_valid=True)
    assert np.all(v == 1)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-5)
    assert np.allclose((n * dirs).sum(axis=1), 1.0, atol=5e-2)       # still the radial direction


def test_extend_range_limit_and_validation():
    grid = make_sphere()
    vol = xuvdb.GpuVolume(grid)
    pts, _ = on_sphere(np.random.default_rng(4), 64, R + 20 * H)     # 17 voxels past the band
    n, v = vol.sdf_normal(pts, extend=2, return_valid=True)
    assert np.all(n == 0.0) and np.all(v == 0)                       # no donor within range
    with pytest.raises(ValueError):
        vol.sdf_normal(pts, extend=-1)


def test_band_edge_outputs_are_clean():
    """Sweep radii across the band edge: every output is exactly zero or a unit vector with a
    sane (outward) direction - the 0/0 sign-flip garbage class is gone."""
    grid = make_sphere()
    vol = xuvdb.GpuVolume(grid)
    rng = np.random.default_rng(5)
    dirs = rng.normal(size=(8, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    rads = R + np.linspace(-3.5 * H, 3.5 * H, 29)                    # crosses the edge at +-3H
    pts = (C + dirs[:, None, :] * rads[None, :, None]).reshape(-1, 3).astype(np.float32)
    n, v = vol.sdf_normal(pts, return_valid=True)
    assert not np.isnan(n).any()
    norms = np.linalg.norm(n, axis=1)
    defined = v == 1
    assert np.all(norms[defined] > 0.9)                              # one-sided stays unit-ish
    assert np.all(n[~defined] == 0.0)
    cos = (n[defined] * radial(pts[defined])).sum(axis=1)
    assert np.all(cos > 0.9)                                         # no direction flips


def test_host_gradient_boundary_aware():
    grid = make_sphere()
    rng = np.random.default_rng(6)
    in_pts, dirs = on_sphere(rng, 64, R - H)
    out_pts, _ = on_sphere(rng, 64, R + 10 * H)
    g_in = grid.sample_gradient(in_pts)
    assert np.allclose(np.linalg.norm(g_in, axis=1), 1.0, atol=5e-2)
    assert np.allclose((g_in * dirs).sum(axis=1), 1.0, atol=5e-2)
    g_out = grid.sample_gradient(out_pts)
    assert np.all(g_out == 0.0)                                      # zero vector, not 0/0 noise
    # kernel/host parity across the whole sweep (both use the same boundary-aware rules)
    rads = R + np.linspace(-2.5 * H, 2.5 * H, 21)
    sweep = (C + dirs[:4, None, :] * rads[None, :, None]).reshape(-1, 3).astype(np.float32)
    kern, valid = xuvdb.GpuVolume(grid).sdf_normal(sweep, return_valid=True)
    host = grid.sample_gradient(sweep)
    host_norm = np.linalg.norm(host, axis=1, keepdims=True)
    host_unit = np.divide(host, host_norm, out=np.zeros_like(host), where=host_norm > 1e-12)
    assert np.allclose(host_unit[valid == 1], kern[valid == 1], atol=5e-2)
    assert np.all(host_norm[valid == 0] == 0.0)
    with pytest.raises(NotImplementedError):
        grid.sample_gradient(sweep, extend=1)
