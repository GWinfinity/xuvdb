"""Grid metadata roundtrip through the native `.vdb` stream."""
import numpy as np
import pytest

import xuvdb


def _grid():
    grid = xuvdb.VdbGrid(background=3 * 0.05, voxel_size=0.05, leaf_log2=4,
                         name="shield", grid_class="level set")
    grid.stamp_sphere((0.3, 0.2, 0.1), radius=0.25, band=3.0)
    return grid


def test_metadata_vdb_roundtrip(tmp_path):
    grid = _grid()
    grid.metadata = {
        "designer": "test",
        "version": 7,
        "density": 2.5,
        "flagged": True,
        "gravity": np.array([0.0, -9.81, 0.0]),
    }
    path = str(tmp_path / "scene.vdb")
    xuvdb.write_vdb(path, [grid])
    back = xuvdb.read_vdb(path)[0]
    assert back.metadata["designer"] == "test"
    assert back.metadata["version"] == 7
    assert back.metadata["density"] == pytest.approx(2.5)
    assert back.metadata["flagged"] is True
    assert np.allclose(back.metadata["gravity"], [0.0, -9.81, 0.0])
    # the serializer's own keys are not mirrored into user metadata
    for key in ("name", "class", "file_bbox_min", "file_voxel_count"):
        assert key not in back.metadata


def test_metadata_defaults_empty():
    assert _grid().metadata == {}


def test_metadata_reserved_key_rejected(tmp_path):
    grid = _grid()
    grid.metadata = {"name": "nope"}
    with pytest.raises(ValueError, match="reserved"):
        xuvdb.write_vdb(str(tmp_path / "scene.vdb"), [grid])


def test_metadata_unsupported_type_rejected(tmp_path):
    grid = _grid()
    grid.metadata = {"bad": [1, 2, 3]}
    with pytest.raises(TypeError, match="unsupported type"):
        xuvdb.write_vdb(str(tmp_path / "scene.vdb"), [grid])


def test_metadata_copy_isolated():
    grid = _grid()
    grid.metadata = {"a": 1}
    clone = grid.copy()
    clone.metadata["a"] = 2
    assert grid.metadata["a"] == 1
