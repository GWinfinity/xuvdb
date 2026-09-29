"""Transform dispatch regression: each OpenVDB map type parses with the Maps.h byte layout.

The review (P0-1) flagged the README for documenting only the ScaleTranslate family; these tests
lock the actual per-type dispatch (the code has dispatched per type since 0.4) against
hand-built payloads straight from the Maps.h `write()` layouts.
"""
import numpy as np
import pytest

from xuvdb.openvdb_file import _read_transform


class FakeStream:
    """Minimal stream exposing what _read_transform consumes."""

    def __init__(self, map_type, payload):
        self._type = map_type
        self._buf = payload
        self._pos = 0

    def string(self):
        return self._type  # reader already consumed the type; mirror it back

    def vec3d(self):
        v = np.frombuffer(self._buf[self._pos:self._pos + 24], dtype="<f8").copy()
        self._pos += 24
        return v

    def raw(self, n):
        b = self._buf[self._pos:self._pos + n]
        assert len(b) == n, "stream underrun: layout mismatch"
        self._pos += n
        return b


def pack_vec3(v):
    return np.asarray(v, dtype="<f8").tobytes()


def test_scale_translate_map():
    s = FakeStream("ScaleTranslateMap",
                   pack_vec3([1, 2, 3]) + pack_vec3([0.05, 0.05, 0.05])
                   + pack_vec3([0.05] * 3) + pack_vec3([20.0] * 3) + pack_vec3([400.0] * 3)
                   + pack_vec3([10.0] * 3))
    vs, t, rot = _read_transform(s, "ScaleTranslateMap")
    assert np.allclose(vs, 0.05) and np.allclose(t, [1, 2, 3]) and rot is None
    assert s._pos == 144  # exactly 6 x Vec3d, nothing left over


def test_uniform_scale_map_openvdb_default():
    """OpenVDB's createLinearTransform(voxelDim) emits UniformScaleMap: 5 x Vec3d."""
    s = FakeStream("UniformScaleMap",
                   pack_vec3([0.1] * 3) + pack_vec3([0.1] * 3) + pack_vec3([10.0] * 3)
                   + pack_vec3([100.0] * 3) + pack_vec3([5.0] * 3))
    vs, t, rot = _read_transform(s, "UniformScaleMap")
    assert np.allclose(vs, 0.1) and np.allclose(t, 0) and rot is None
    assert s._pos == 120  # 5 x Vec3d


def test_translation_map():
    s = FakeStream("TranslationMap", pack_vec3([4, 5, 6]))
    vs, t, rot = _read_transform(s, "TranslationMap")
    assert np.allclose(vs, 1.0) and np.allclose(t, [4, 5, 6]) and rot is None
    assert s._pos == 24


def test_affine_map_rigid_decomposition():
    """A rotated grid (Houdini-style AffineMap): linear part = R * diag(s), row-major Mat4d."""
    ang = np.pi / 6
    R = np.array([[np.cos(ang), -np.sin(ang), 0.0], [np.sin(ang), np.cos(ang), 0.0],
                  [0.0, 0.0, 1.0]])
    s_vec = np.array([0.05, 0.1, 0.2])
    t = np.array([1.0, -2.0, 3.0])
    m = np.eye(4)
    m[:3, :3] = R * s_vec
    m[:3, 3] = t
    s = FakeStream("AffineMap", m.astype("<f8").tobytes())
    vs, t_out, rot = _read_transform(s, "AffineMap")
    assert np.allclose(vs, s_vec, atol=1e-12)
    assert np.allclose(t_out, t, atol=1e-12)
    assert rot is not None and np.allclose(rot, R, atol=1e-9)
    assert s._pos == 128  # Mat4d


def test_affine_map_with_shear_raises_instead_of_degrading():
    """N-1: a sheared affine must fail loudly - silent degradation produced wrong geometry
    (non-rigid linear part, column norms are not the true diagonal)."""
    m = np.eye(4)
    m[:3, :3] = np.array([[1.0, 0.3, 0.0],   # 0.3 off-diagonal = shear
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]])
    m[:3, 3] = [1.0, 2.0, 3.0]
    s = FakeStream("AffineMap", m.astype("<f8").tobytes())
    with pytest.raises(NotImplementedError, match="non-rigid"):
        _read_transform(s, "AffineMap")


def test_unitary_map_consumes_full_mat4():
    """Regression: UnitaryMap payloads are a full Mat4d (128 B); the pre-0.4 reader ate only
    72 bytes and desynced every following grid."""
    ang = np.pi / 2
    R = np.array([[np.cos(ang), -np.sin(ang), 0.0], [np.sin(ang), np.cos(ang), 0.0],
                  [0.0, 0.0, 1.0]])
    m = np.eye(4)
    m[:3, :3] = R
    s = FakeStream("UnitaryMap", m.astype("<f8").tobytes())
    vs, t, rot = _read_transform(s, "UnitaryMap")
    assert np.allclose(vs, 1.0) and rot is not None and np.allclose(rot, R, atol=1e-9)
    assert s._pos == 128


def test_frustum_map_raises_explicitly():
    s = FakeStream("NonlinearFrustumMap", b"\x00" * 16)
    with pytest.raises(NotImplementedError, match="NonlinearFrustumMap"):
        _read_transform(s, "NonlinearFrustumMap")
