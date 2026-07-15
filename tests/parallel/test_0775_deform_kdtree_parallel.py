"""Parallel navigation after mesh motion / on overlapped nav DMs (issue #360).

Moving-mesh workflows rebuild the cell-search kd-tree every step
(``nuke_coords_and_rebuild`` -> ``_build_kd_tree_index``). On the 1-cell-
overlap navigation clone used for manifold (codim-1) meshes the local
coordinate array is laid out by the coordinate PetscSection AND
``distributeOverlap`` leaves the ghost-vertex coordinates unscattered, so the
old affine ``nav_coords[plex_point - pStart]`` map ran out of bounds
(``IndexError``) on every rank in parallel. These tests reproduce the crash
(pre-fix) and assert that navigation actually works — not merely that no
exception is raised.

Run with:
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_0775_deform_kdtree_parallel.py
    mpirun -n 4 python -m pytest --with-mpi tests/parallel/test_0775_deform_kdtree_parallel.py
"""

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [
    pytest.mark.level_1,
    pytest.mark.tier_a,
    pytest.mark.mpi(min_size=2),
    pytest.mark.timeout(120),
]


@pytest.mark.mpi(min_size=2)
def test_deform_box_navigation_parallel():
    """A distributed box mesh must survive a deform step and still locate
    points via the rebuilt kd-tree on every rank."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2, qdegree=2,
    )

    X = np.asarray(mesh.X.coords, dtype=float).copy()
    bump = 0.02 * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])
    new = X.copy()
    new[:, 0] += bump
    new[:, 1] += bump

    # Pre-fix this raised IndexError inside _build_kd_tree_index on every
    # rank the moment the overlapped/section-ordered coord array was indexed.
    mesh.deform(new)

    # Positive navigation check: evaluate the coordinate field at this rank's
    # own (deformed) vertices; point-location must return them exactly.
    pts = np.asarray(mesh.X.coords, dtype=float)
    vx = np.asarray(uw.function.evaluate(mesh.X[0], pts)).reshape(-1)
    vy = np.asarray(uw.function.evaluate(mesh.X[1], pts)).reshape(-1)
    err = max(np.abs(vx - pts[:, 0]).max(), np.abs(vy - pts[:, 1]).max())
    assert err < 1e-8, f"navigation broken after deform: max coord error {err}"


@pytest.mark.mpi(min_size=2)
def test_manifold_surface_navigation_parallel():
    """A codim-1 surface submesh forces the 1-cell-overlap navigation clone.
    Building it (kd-tree + face control points) must not crash in parallel,
    and every navigation coordinate row — including the overlap ghost rows —
    must carry the correct geometry (issue #360)."""
    r_outer = 1.0
    ann = uw.meshing.Annulus(
        radiusOuter=r_outer, radiusInner=0.5, cellSize=0.1, qdegree=2,
    )

    # Pre-fix: this raised IndexError in _build_kd_tree_index at construction
    # (the overlapped nav coord array is short by the ghost rows).
    surf = ann.extract_surface("Upper")
    assert surf.dm.getDimension() == 1

    nav_dm = surf._nav_dm if surf._nav_dm is not None else surf.dm
    nav_coords = surf._nav_coords

    # Completeness: a navigation coordinate row for every vertex the local
    # cell closures reference (pre-fix the array was short -> out of bounds).
    pStart, pEnd = nav_dm.getDepthStratum(0)
    assert nav_coords.shape[0] >= (pEnd - pStart), (
        f"nav coords ({nav_coords.shape[0]} rows) do not cover all "
        f"{pEnd - pStart} nav vertices"
    )

    # Correctness: the "Upper" surface loop sits at r = radiusOuter; every
    # navigation row (owned AND overlap ghost) must lie on that circle. A
    # zero/garbage ghost row (the pre-fix failure mode) would violate this.
    rows = surf._coord_rows_for_points(nav_dm, np.arange(pStart, pEnd))
    r = np.sqrt((nav_coords[rows] ** 2).sum(axis=1))
    assert np.allclose(r, r_outer, atol=1e-9), (
        f"nav vertex radii off the surface: [{r.min()}, {r.max()}] != {r_outer}"
    )

    # Exercise a real navigation-driven operation on the manifold.
    h = uw.discretisation.MeshVariable("h360", surf, 1, degree=1)
    h.data[:, 0] = 1.0
    uw.meshing.smooth_surface_field(h, n_iters=5, alpha=0.5, taubin=True)
    assert np.allclose(h.data[:, 0], 1.0, atol=1e-8), (
        "surface smoother did not preserve the constant field"
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--with-mpi"]))
