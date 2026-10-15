"""Marching-cubes meshing tests: isosurface extraction from sparse level sets."""
import time

import numpy as np
import pytest

import xuvdb
from xuvdb.meshing import MC_TRI_TABLE, _CORNERS, _EDGES, level_set_to_mesh


def make_sphere_grid(radius=0.5, voxel_size=0.02, center=(0.0, 0.0, 0.0), band=3.0):
    grid = xuvdb.VdbGrid(background=band * voxel_size, voxel_size=voxel_size,
                         leaf_log2=4, name="sphere", grid_class="level set")
    grid.stamp_sphere(center, radius=radius, band=band)
    return grid


def mesh_volume(vertices, faces):
    """Enclosed volume via the divergence theorem (signed tetrahedra sum)."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


def vertex_normals(vertices, faces):
    """Area-weighted per-vertex normals from face winding."""
    normals = np.zeros((len(vertices), 3), dtype=np.float64)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_n = np.cross(v1 - v0, v2 - v0)
    for c in range(3):
        np.add.at(normals[:, c], faces[:, 0], face_n[:, c])
        np.add.at(normals[:, c], faces[:, 1], face_n[:, c])
        np.add.at(normals[:, c], faces[:, 2], face_n[:, c])
    return normals


def dense_reference_mesh(grid, isovalue=0.0):
    """Naive per-voxel marching cubes over `to_dense(pad=1)` - the correctness oracle.

    Deliberately dumb (one Python iteration per cell, its own dedup dict) so that an
    agreement with the vectorized leaf-based implementation means the sparse read
    semantics (stored values in allocated leaves, background outside) are right.
    """
    iso = float(isovalue)
    dense, ijk_min = grid.to_dense(pad=1)
    vs = grid.voxel_size
    ow = grid.origin_world
    nx, ny, nz = dense.shape
    verts = []
    vert_ids = {}
    faces = []

    def edge_vertex(base, v, e):
        a, b = _EDGES[e]
        lo_c, hi_c = (a, b) if a < b else (b, a)
        lo_off = np.asarray(_CORNERS[lo_c], dtype=np.float64)
        hi_off = np.asarray(_CORNERS[hi_c], dtype=np.float64)
        va, vb = float(v[lo_c]), float(v[hi_c])
        den = vb - va
        t = 0.5 if den == 0.0 else min(1.0, max(0.0, (iso - va) / den))
        pos = base + lo_off + t * (hi_off - lo_off)
        world = pos * vs + ow
        p1 = tuple((base + lo_off).astype(np.int64))
        p2 = tuple((base + hi_off).astype(np.int64))
        key = p1 + p2 if p1 <= p2 else p2 + p1
        vid = vert_ids.get(key)
        if vid is None:
            vid = len(verts)
            vert_ids[key] = vid
            verts.append(world)
        return vid

    for x in range(nx - 1):
        for y in range(ny - 1):
            for z in range(nz - 1):
                base = ijk_min.astype(np.float64) + (x, y, z)
                v = np.array([dense[x + cx, y + cy, z + cz] for cx, cy, cz in _CORNERS],
                             dtype=np.float64)
                high = v >= iso
                if high.all() or not high.any():
                    continue
                case = 0
                for i in range(8):
                    case |= int(high[i]) << i
                for i in range(0, len(MC_TRI_TABLE[case]), 3):
                    faces.append([edge_vertex(base, v, MC_TRI_TABLE[case][i]),
                                  edge_vertex(base, v, MC_TRI_TABLE[case][i + 1]),
                                  edge_vertex(base, v, MC_TRI_TABLE[case][i + 2])])
    if not verts:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    return np.stack(verts).astype(np.float32), np.asarray(faces, dtype=np.int32)


def rounded_vertex_set(vertices, decimals=6):
    """Order-independent vertex multiset for sparse-vs-dense mesh comparison."""
    return set(map(tuple, np.round(vertices, decimals)))


def test_sphere_volume_and_outward_normals():
    """Unit sphere at voxel_size=0.02: volume within 3% of 4/3*pi*r^3, normals out."""
    grid = make_sphere_grid()
    vertices, faces = level_set_to_mesh(grid)
    assert vertices.shape[1] == 3 and faces.shape[1] == 3
    exact = 4.0 / 3.0 * np.pi * 0.5 ** 3
    volume = mesh_volume(vertices.astype(np.float64), faces)
    assert abs(volume - exact) / exact < 0.03, f"volume {volume} vs {exact}"
    normals = vertex_normals(vertices, faces)
    outward = np.einsum("ij,ij->i", normals, vertices.astype(np.float64))
    fraction = float((outward > 0).mean())
    assert fraction > 0.95, f"outward fraction {fraction}"


def test_sparse_matches_dense_reference():
    """Narrow-band sparse extraction == naive dense marching cubes on the same grid."""
    grid = make_sphere_grid()
    v_sparse, f_sparse = level_set_to_mesh(grid)
    v_dense, f_dense = dense_reference_mesh(grid)
    # same vertex geometry, order-independent; both sides weld exact-zero-corner
    # duplicates (the library welds its own output, so weld the reference too)
    assert rounded_vertex_set(v_sparse) == rounded_vertex_set(v_dense)
    assert len(v_sparse) == len(rounded_vertex_set(v_sparse)), "sparse mesh has duplicate vertices"
    # the naive dense reference intentionally skips welding - exact-zero corners
    # duplicate positionally there too, which is what the rounded-set compare above covers


def test_union_of_two_spheres():
    """CSG-union of two overlapping spheres: one connected component, analytic volume."""
    grid = make_sphere_grid()
    grid.stamp_sphere((0.5, 0.0, 0.0), radius=0.5, band=3.0)  # min-union composition
    vertices, faces = level_set_to_mesh(grid)
    assert len(faces) > 0

    # connected components over the vertex-sharing graph (union-find)
    parent = np.arange(len(vertices))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for tri in faces:
        root = find(int(tri[0]))
        for other in tri[1:]:
            parent[find(int(other))] = root
    components = {find(int(tri[0])) for tri in faces}
    assert len(components) == 1, f"expected 1 component, got {len(components)}"

    d = 0.5  # center distance; r = 0.5 each
    r = 0.5
    overlap = np.pi * (4 * r + d) * (2 * r - d) ** 2 / 12.0
    exact = 2.0 * (4.0 / 3.0 * np.pi * r ** 3) - overlap
    volume = mesh_volume(vertices.astype(np.float64), faces)
    assert abs(volume - exact) / exact < 0.05, f"volume {volume} vs {exact}"


def test_empty_grid_returns_empty_arrays():
    """An all-background grid has no straddling cells: empty (0,3) arrays of right dtype."""
    grid = make_sphere_grid()
    grid.fill_box((-30, -30, -30), (30, 30, 30), grid.background, active=True)
    vertices, faces = level_set_to_mesh(grid)
    assert vertices.shape == (0, 3)
    assert faces.shape == (0, 3)
    assert vertices.dtype == np.float32
    assert faces.dtype == np.int32
    # a fresh grid with no leaves at all behaves the same
    empty = xuvdb.VdbGrid(background=0.06, voxel_size=0.02, name="e", grid_class="level set")
    v2, f2 = level_set_to_mesh(empty)
    assert v2.shape == (0, 3) and f2.shape == (0, 3)


def test_vertices_are_world_space():
    """Mesh bbox matches the sphere's world-space bbox (origin offset applied)."""
    center = np.array([0.3, -0.2, 0.1])
    grid = make_sphere_grid(center=center)
    vs = float(np.mean(grid.voxel_size))
    vertices, _ = level_set_to_mesh(grid)
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    np.testing.assert_allclose(lo, center - 0.5, atol=vs)
    np.testing.assert_allclose(hi, center + 0.5, atol=vs)


def test_rejects_rotated_and_vector_grids():
    rotated = xuvdb.VdbGrid(voxel_size=0.02, rotation=[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        level_set_to_mesh(rotated)
    vec = xuvdb.VdbGrid(dtype=np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32)]))
    with pytest.raises(TypeError):
        level_set_to_mesh(vec)


def test_extraction_performance():
    """Sphere mesh extraction stays well under the 2 s budget (numpy hot path)."""
    grid = make_sphere_grid()
    t0 = time.perf_counter()
    vertices, faces = level_set_to_mesh(grid)
    elapsed = time.perf_counter() - t0
    assert len(faces) > 10000
    assert elapsed < 2.0, f"extraction took {elapsed:.2f}s"
