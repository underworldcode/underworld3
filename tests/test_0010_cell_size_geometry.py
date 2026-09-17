"""Issue #687: cell_size is an own-cell geometric quantity, including after deform.

The independent oracle reads vertex coordinates through the coordinate section;
it does not use the mesh's cached radii or centroid kd-tree. Run serial and MPI.
"""

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _vertex_rms(mesh):
    dm = mesh.dm
    section = dm.getCoordinateDM().getLocalSection()
    coordinates = dm.getCoordinatesLocal().array
    start, end = dm.getHeightStratum(0)
    first_vertex, last_vertex = dm.getDepthStratum(0)
    radii = []
    for cell in range(start, end):
        vertices = [int(point) for point in dm.getTransitiveClosure(cell)[0]
                    if first_vertex <= point < last_vertex]
        points = np.array([coordinates[section.getOffset(v):section.getOffset(v) + mesh.cdim]
                           for v in vertices])
        radii.append(np.sqrt(np.mean(np.sum((points - points.mean(axis=0)) ** 2, axis=1))))
    return np.asarray(radii)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("simplex", [True, False], ids=["simplex", "tensor"])
def test_cell_size_matches_own_vertices_and_tracks_deform(dim, simplex):
    geometry = dict(minCoords=(0.0,) * dim, maxCoords=(1.0,) * dim, qdegree=3)
    mesh = (uw.meshing.UnstructuredSimplexBox(**geometry, cellSize=0.25, regular=False)
            if simplex else uw.meshing.StructuredQuadBox(**geometry, elementRes=(4,) * dim))
    mesh.cell_size()
    field = mesh._cell_size_variable
    errors = []
    for phase in ("initial", "deformed"):
        if phase == "deformed":
            coordinates = np.array(mesh.X.coords)
            coordinates[:, 0] = 1.7 * coordinates[:, 0] + 0.2 * coordinates[:, 1]
            mesh.deform(coordinates)
        expected = _vertex_rms(mesh)
        actual = np.asarray(field.array[:, 0, 0])
        shapes_match = actual.shape == expected.shape
        assert all(uw.mpi.comm.allgather(shapes_match)), (actual.shape, expected.shape)
        local_error = float(np.abs(actual - expected).max(initial=0.0))
        error = max(uw.mpi.comm.allgather(local_error))
        errors.append(error)
        uw.pprint(f"CELL_SIZE_GEOMETRY dim={dim} simplex={simplex} phase={phase} "
                  f"ranks={uw.mpi.size} max_error={error:.12g}")
    assert max(errors) < 1e-12, errors


def test_regular_square_cell_size_keeps_global_radius():
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4), qdegree=2)
    legacy = np.array(mesh._radii)
    global_radius = mesh.get_min_radius()
    mesh.cell_size()
    expected = np.sqrt(2.0) / 8.0
    error = float(np.abs(np.asarray(mesh._cell_size_variable.array) - expected).max(initial=0.0))
    assert max(uw.mpi.comm.allgather(error)) < 1e-12
    assert global_radius == pytest.approx(expected, rel=1e-12)
    assert all(uw.mpi.comm.allgather(np.array_equal(mesh._radii, legacy)))


@pytest.mark.parametrize("h", [0.5, 0.25])
def test_regular_simplex_cell_size_is_the_closed_form(h):
    """cell_size on congruent right-isosceles cells is 2h/3, exactly.

    `test_cell_size_matches_own_vertices_and_tracks_deform` above checks the
    implementation against an independent reading of the same DEFINITION, so it
    stays true if the definition itself is changed on both sides. This pins the
    VALUE against geometry instead: `regular=True` tiles the box with congruent
    right-isosceles triangles of legs h, whose vertices sit at (0,0), (h,0),
    (0,h) up to rigid motion, so the RMS distance to the centroid is

        sqrt( ( 2(h/3)^2 + 2[(2h/3)^2 + (h/3)^2] ) / 3 ) = 2h/3

    A redefinition of `cell_size` fails here immediately, at the quantity that
    changed. #692 changed it for simplices and nothing failed, so the Nitsche
    penalty (gamma*mu/h) moved 43% unnoticed and surfaced months later as a
    5.67% miss on a spherical-shell benchmark (#734) — on one platform's
    triangulation only, which is the hardest kind of failure to read backwards.
    """
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=h, regular=True, qdegree=2,
    )
    mesh.cell_size()
    radii = np.asarray(mesh._cell_size_variable.array[:, 0, 0])

    error = float(np.abs(radii - 2.0 * h / 3.0).max(initial=0.0))
    assert max(uw.mpi.comm.allgather(error)) < 1e-12, (
        f"cell_size on congruent legs-{h} right-isosceles cells is "
        f"{radii.min():.12g}..{radii.max():.12g}, expected exactly {2.0 * h / 3.0:.12g}"
    )


def test_cell_size_and_min_radius_agree_on_tensor_cells_but_not_simplices():
    """The two mesh-size measures coincide on TENSOR cells only, by sqrt(2).

    `add_nitsche_bc` offers `local_h=False` to fall back from `mesh.cell_size()`
    to the global `mesh.get_min_radius()`, and its docstring used to say the two
    coincide "on a uniform mesh". They do on a regular quad box — which is
    presumably where that was checked — and they do NOT on a uniform SIMPLEX
    mesh, which is what every free-slip and fault model is built on.

    Both measures have closed forms on congruent right-isosceles cells of legs h:
    cell_size is 2h/3 (vertex RMS about the centroid) and get_min_radius is
    sqrt(2)h/3, so the ratio is exactly sqrt(2). Pinning it means neither measure
    can be redefined without this saying so, and says which way the Nitsche
    penalty moves when it is.
    """
    quad = uw.meshing.StructuredQuadBox(elementRes=(4, 4), qdegree=2)
    quad.cell_size()
    quad_local = np.asarray(quad._cell_size_variable.array[:, 0, 0])
    quad_ratio = float(quad_local.min()) / quad.get_min_radius()
    assert quad_ratio == pytest.approx(1.0, rel=1e-12), (
        f"on tensor cells the two measures must coincide; ratio {quad_ratio:.12g}")

    h = 0.25
    simplex = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=h, regular=True, qdegree=2,
    )
    simplex.cell_size()
    simplex_local = np.asarray(simplex._cell_size_variable.array[:, 0, 0])

    assert simplex.get_min_radius() == pytest.approx(np.sqrt(2.0) * h / 3.0, rel=1e-12)
    ratio = float(simplex_local.min()) / simplex.get_min_radius()
    assert ratio == pytest.approx(np.sqrt(2.0), rel=1e-12), (
        f"cell_size/get_min_radius on uniform simplices is {ratio:.12g}, "
        "expected sqrt(2) — the Nitsche penalty gamma*mu/h scales with this"
    )
