"""Mesh coordinates are degree 1; a mesh refuses any other degree (#978).

``degree`` on a mesh constructor set the degree of the coordinate field. Particle
location, cell geometry, boundary normals and mesh motion all read the
coordinates as straight-sided cells, and a degree-2 mesh failed in each: the
FVM cell sizes read edge nodes as corners (#908), ``boundary_normal`` raised,
and ``deform`` broadcast vertex coordinates into the larger degree-2 coordinate
vector. It never curved a boundary either: a degree-2 annulus put its wall edge
nodes on the chords. So a mesh now refuses ``degree != 1``, each ``uw.meshing``
constructor refuses before it generates anything, ``mesh.degree`` is read-only,
and a variable keeps its own ``degree``.

The parallel half is ``tests/parallel/test_0782_mesh_degree_refused_on_every_rank.py``.
"""
import inspect

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _takes_a_coordinate_degree(entry):
    if not inspect.isfunction(entry):
        return False
    parameters = inspect.signature(entry).parameters
    return "degree" in parameters and "qdegree" in parameters


# Every constructor has a default for each argument, so ``degree`` alone calls it.
_CONSTRUCTORS = sorted(
    name for name in uw.meshing.__all__
    if _takes_a_coordinate_degree(getattr(uw.meshing, name)))


def test_the_constructors_under_test_are_the_ones_we_know():
    """The list is found by signature, so it could come back short."""
    assert len(_CONSTRUCTORS) >= 20, _CONSTRUCTORS
    assert {"UnstructuredSimplexBox", "StructuredQuadBox", "Annulus", "SphericalShell",
            "CubedSphere", "SphericalManifold", "RegionalGeographicBox",
            "SegmentedSphericalShell", "BoxInternalPatch"} <= set(_CONSTRUCTORS)


@pytest.mark.parametrize("name", _CONSTRUCTORS)
def test_a_constructor_refuses_degree_2_before_it_generates_a_mesh(name, monkeypatch):
    # development: every one of these built a degree-2 mesh. A constructor that
    # reaches ``Mesh`` has already run gmsh, so reaching it fails the test.
    def reached(*args, **kwargs):
        raise AssertionError(f"{name} went on to build a Mesh")

    monkeypatch.setattr(uw.discretisation.Mesh, "__init__", reached)
    with pytest.raises(ValueError, match=r"mesh coordinates are degree 1.*MeshVariable"):
        getattr(uw.meshing, name)(degree=2)


@pytest.mark.parametrize("degree", [0, 2, 3, 1.0, True, "1", None])
def test_a_mesh_from_a_plex_refuses_any_degree_but_the_integer_one(degree):
    # development: 2 and 3 built and failed later in cell sizes, normals or
    # deform; 1.0 and True failed in PETSc's option parser with error 63
    plex = uw.meshing.StructuredQuadBox(elementRes=(2, 2)).dm.clone()
    with pytest.raises(ValueError, match=r"mesh coordinates are degree 1.*MeshVariable") as refusal:
        uw.discretisation.Mesh(plex, degree=degree)
    assert f"degree={degree!r} " in str(refusal.value)


def test_the_degree_of_a_mesh_cannot_be_changed():
    mesh = uw.meshing.StructuredQuadBox(elementRes=(2, 2))
    with pytest.raises(AttributeError):
        mesh.degree = 2
    assert mesh.degree == 1


def test_a_degree_2_variable_on_a_degree_1_mesh_still_builds():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25, regular=True)
    v = uw.discretisation.MeshVariable("V0782", mesh, 2, degree=2)
    # a 4 x 4 grid of squares carries (2 * 4 + 1)**2 = 81 P2 nodes
    assert v.degree == 2
    assert np.asarray(v.coords).shape == (81, 2)


def test_each_cell_of_a_manifold_has_the_area_of_its_flat_facet():
    """A 2-D mesh in 3-D, where the Jacobian is not square: half the cross
    product of two edges is the closed form for each straight-sided cell."""
    mesh = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.4, qdegree=3)
    nodes = np.asarray(mesh._get_coords_for_basis(1, True))
    corners = nodes[np.asarray(mesh._cell_node_indices(1, True))]
    areas = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    radii = np.asarray(mesh._cell_radii).reshape(-1)
    assert areas.max() > 1.05 * areas.min()
    assert np.allclose(radii ** 2, areas, rtol=1e-12, atol=0.0)
