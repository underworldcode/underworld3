"""``mesh.cell_size()`` is each cell's ``volume**(1/dim)``, and it tracks deformation.

The oracle here is computed from the vertex coordinates, not from PETSc, so this
is a check and not a restatement of the implementation. For simplices that is
the determinant volume of the cell; for the structured boxes it is the analytic
cell volume, scaled by the determinant of the affine map when the mesh is
deformed.

Partition independence is NOT tested here -- it cannot be, at one rank count.
``tests/parallel/test_1079`` compares the field cell by cell against its own
serial answer, and ``test_1080`` does the same for the three radius accessors.
"""

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

# The deformation applied below, and its Jacobian determinant. Volumes scale by
# |det|, so lengths scale by |det|**(1/dim).
AFFINE = np.array([[1.7, 0.2], [0.0, 1.0]])
AFFINE_DET = 1.7


def _simplex_volume_from_vertices(mesh):
    """Each simplex's volume from its own vertex coordinates.

    Triangle: ``|det[v1-v0, v2-v0]| / 2``. Tetrahedron: ``|det[...]| / 6``.
    """
    dim = mesh.dim
    cell_start, cell_end = mesh.dm.getHeightStratum(0)
    point_start, _point_end = mesh.dm.getDepthStratum(0)

    volumes = np.empty(cell_end - cell_start)
    factorial = 2.0 if dim == 2 else 6.0
    for cell in range(cell_end - cell_start):
        corners = mesh.dm.getTransitiveClosure(cell)[0][-(dim + 1):]
        coords = mesh._coords[corners - point_start]
        edges = coords[1:] - coords[0]
        volumes[cell] = abs(np.linalg.det(edges)) / factorial
    return volumes


@pytest.mark.parametrize("dim", [2, 3])
def test_simplex_cell_size_is_the_cube_root_of_its_own_volume(dim):
    """Against a determinant volume computed from the cell's vertices."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0,) * dim, maxCoords=(1.0,) * dim, qdegree=3,
        cellSize=0.25, regular=False)
    mesh.cell_size()
    actual = np.asarray(mesh._cell_size_variable.array[:, 0, 0])
    expected = _simplex_volume_from_vertices(mesh) ** (1.0 / dim)

    error = float(np.abs(actual - expected).max(initial=0.0))
    uw.pprint(f"CELL_SIZE_GEOMETRY simplex dim={dim} max_error={error:.12g}")
    assert error < 1.0e-12, error


@pytest.mark.parametrize("dim", [2, 3])
def test_structured_cell_size_is_the_analytic_value_and_tracks_deformation(dim):
    """A regular box has an exact answer, and an affine deform scales it.

    ``elementRes=4`` on the unit box gives cells of side 0.25, so
    ``volume**(1/dim)`` is 0.25 whatever the dimension. Applying a linear map
    multiplies every cell volume by ``|det|``, hence every length by
    ``|det|**(1/dim)`` -- which the field must follow after ``deform``.
    """
    mesh = uw.meshing.StructuredQuadBox(
        minCoords=(0.0,) * dim, maxCoords=(1.0,) * dim, qdegree=3,
        elementRes=(4,) * dim)
    mesh.cell_size()
    field = mesh._cell_size_variable

    actual = np.asarray(field.array[:, 0, 0])
    error = float(np.abs(actual - 0.25).max(initial=0.0))
    assert max(uw.mpi.comm.allgather(error)) < 1.0e-12, error

    coordinates = np.array(mesh.X.coords)
    coordinates[:, 0] = AFFINE[0, 0] * coordinates[:, 0] + AFFINE[0, 1] * coordinates[:, 1]
    mesh.deform(coordinates)

    expected = 0.25 * AFFINE_DET ** (1.0 / dim)
    actual = np.asarray(field.array[:, 0, 0])
    error = float(np.abs(actual - expected).max(initial=0.0))
    uw.pprint(f"CELL_SIZE_GEOMETRY tensor dim={dim} deformed max_error={error:.12g}")
    assert max(uw.mpi.comm.allgather(error)) < 1.0e-12, (error, expected)


def test_the_global_minimum_agrees_with_the_field():
    """``get_min_radius()`` reduces the same field ``cell_size()`` exposes.

    On a regular box every cell is the same size, so the global minimum is that
    size -- which also pins the reduction to the analytic value rather than to
    whatever the field happens to hold.
    """
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4), qdegree=2)
    mesh.cell_size()
    assert mesh.get_min_radius() == pytest.approx(0.25, rel=1.0e-12)
    assert float(np.asarray(mesh._cell_size_variable.array).max()) == pytest.approx(
        0.25, rel=1.0e-12)


@pytest.mark.parametrize("h", [0.5, 0.25])
def test_regular_simplex_cell_size_is_the_closed_form(h):
    """cell_size on congruent right-isosceles cells is h/sqrt(2), exactly.

    `test_cell_size_matches_own_vertices_and_tracks_deform` above checks the
    implementation against an independent reading of the same DEFINITION, so it
    stays true if the definition itself is changed on both sides. This pins the
    VALUE against geometry instead: `regular=True` tiles the box with congruent
    right-isosceles triangles of legs h, whose area is h^2/2, and #694 defines
    the characteristic length as PETSc's volume**(1/dim):

        cell_size = (h^2 / 2)^(1/2) = h / sqrt(2)

    A redefinition of `cell_size` fails here immediately, at the quantity that
    changed. #692 changed it for simplices and nothing failed, so the Nitsche
    penalty (gamma*mu/h) moved 43% unnoticed and surfaced months later as a
    5.67% miss on a spherical-shell benchmark (#734) - on one platform's
    triangulation only, which is the hardest kind of failure to read backwards.
    That is why this is written as geometry rather than as whatever the code
    returns: the previous vertex-RMS definition gave 2h/3 here, and the number
    moving from 2h/3 to h/sqrt(2) is exactly the event this test announces.
    """
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=h, regular=True, qdegree=2,
    )
    mesh.cell_size()
    radii = np.asarray(mesh._cell_size_variable.array[:, 0, 0])

    expected = h / np.sqrt(2.0)
    error = float(np.abs(radii - expected).max(initial=0.0))
    assert max(uw.mpi.comm.allgather(error)) < 1e-12, (
        f"cell_size on congruent legs-{h} right-isosceles cells is "
        f"{radii.min():.12g}..{radii.max():.12g}, expected exactly {expected:.12g}"
    )


def test_cell_size_and_min_radius_are_now_one_measure():
    """``cell_size()`` and ``get_min_radius()`` read the SAME quantity (#694).

    ``add_nitsche_bc`` offers ``local_h=False`` to fall back from
    ``mesh.cell_size()`` to the global ``mesh.get_min_radius()``. Before #694
    those were different DEFINITIONS - vertex-RMS about the centroid versus
    PETSc's face-distance radius - so the switch silently rescaled the penalty
    ``gamma*mu/h`` by a factor that depended on the cell type: 1 on tensor cells
    and sqrt(2) on simplices. That is how #734 happened.

    #694 makes both read ``volume**(1/dim)``, so ``local_h`` is now a choice
    between the LOCAL cell and the GLOBAL minimum and nothing else. On a uniform
    mesh, where those two coincide, the measures must therefore agree EXACTLY -
    on simplices as well as on tensor cells, which is the half that used not to
    hold. Pinning both means neither can drift back apart silently.
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

    # both sides against the closed form, so a common drift cannot cancel
    assert simplex.get_min_radius() == pytest.approx(h / np.sqrt(2.0), rel=1e-12)
    ratio = float(simplex_local.min()) / simplex.get_min_radius()
    assert ratio == pytest.approx(1.0, rel=1e-12), (
        f"cell_size/get_min_radius on uniform simplices is {ratio:.12g}, "
        "expected 1 - before #694 this was sqrt(2), and the Nitsche penalty "
        "gamma*mu/h scaled with it"
    )
