"""Every owned wall node is found at any rank count, by the rotated free-slip node list
with an analytic normal and by the free surface's node list (#935).

Each boundary facet is labelled on one rank, so a rank can own a wall node all of
whose labelled facets sit on neighbours. ``rotated_bc._boundary_velocity_nodes`` with
the geometric normal (``normal=None``) completes its node set over the point SF
(#560). With an analytic or constant normal it took only the nodes of the facets
labelled on the rank, so such a node got no constraint row and carried normal flow;
a production annulus showed it at np 24 to 52 with the radial normal.
``FreeSurface._surface_node_coords`` took its vertices the same way, and every
surface row it matches (the mesh deformation, the datum of the consistent solve)
missed that vertex on its owner.

A rank whose share of a wall is one node is what the label-local lists lost, so the
fixtures are coarse boxes. Measured on Linux with PETSc 3.25 and ParMETIS (the
default partitioner when it is built), one rank owns a corner at np 2 with no facet
of either wall beside it, on both fixtures; at np 4 the unstructured box holds such
a node and the regular box does not (``HOLDS_AT``).

- On every rank and wall, the analytic normal finds exactly the geometric path's nodes.
- Each fixture exercises the case at the rank counts it was measured at (np 2, CI's
  count, for both): some rank owns a wall node that no facet labelled on that rank
  reaches. If a partitioner change makes this fail, the fixture has lost its power,
  not the library its fix: choose a coarse mesh on which, at that rank count, some
  rank owns a wall node that only a neighbour's facets reach.
- A rotated free-slip solve with the constant wall normals leaves no normal flow at
  any wall node (exactly none serially; up to 0.67 of max |v| without the fix).
- The free surface's node list on Top holds every local vertex on Top.

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1140_rotated_analytic_normal_wall_nodes_parallel.py
    mpirun -n 4 python -m pytest --with-mpi tests/parallel/test_1140_rotated_analytic_normal_wall_nodes_parallel.py
"""
from types import SimpleNamespace

import numpy as np
import pytest
import sympy

import underworld3 as uw
import underworld3.utilities.rotated_bc as rbc
from underworld3.systems.free_surface import FreeSurface

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.timeout(300)]

WALL_NORMALS = {"Top": (0.0, 1.0), "Bottom": (0.0, -1.0),
                "Left": (-1.0, 0.0), "Right": (1.0, 0.0)}
FIXTURES = [(False, 0.5), (True, 0.4)]     # (regular, cellSize)
# the rank counts at which each fixture was measured to hold the case
HOLDS_AT = {(False, 0.5): (2, 4), (True, 0.4): (2,)}


def _box(regular, cell_size):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=cell_size,
        regular=regular, qdegree=2)


def _box_solver(regular, cell_size):
    mesh = _box(regular, cell_size)
    v = uw.discretisation.MeshVariable("V", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    return stokes, v


def _set_up(regular, cell_size):
    """The solver's DM, with no boundary condition: an essential condition on a wall
    takes its DOFs out of the global vector, and the geometric path then drops the
    node in question too."""
    stokes, _v = _box_solver(regular, cell_size)
    stokes._setup_pointwise_functions()
    stokes._setup_discretisation()
    return stokes


def _owned(dm):
    p0, p1 = dm.getChart()
    _nroots, leaves, _remote = dm.getPointSF().getGraph()
    ghost = set() if leaves is None else {int(q) for q in leaves}
    return {q for q in range(p0, p1) if q not in ghost}


def _reached_by_local_facets(stokes, wall):
    """Velocity points in the closure of a facet of ``wall`` labelled on this rank."""
    dm = stokes.dm
    lsec = dm.getLocalSection()
    sis = rbc._boundary_stratum_is(dm, stokes.mesh, wall)
    facets = [int(f) for f in sis.getIndices()] if (sis and sis.getSize() > 0) else []
    return {int(q) for f in facets for q in dm.getTransitiveClosure(f)[0]
            if lsec.getFieldDof(int(q), 0) > 0}


@pytest.mark.tier_b
@pytest.mark.parametrize("regular, cell_size", FIXTURES)
def test_the_analytic_normal_finds_the_geometric_nodes(regular, cell_size):
    stokes = _set_up(regular, cell_size)
    differ = []
    for wall, normal in WALL_NORMALS.items():
        geometric = {q for q, _n in rbc._boundary_velocity_nodes(stokes, wall)}
        analytic = {q for q, _n in rbc._boundary_velocity_nodes(stokes, wall, normal=normal)}
        if geometric != analytic:
            differ.append((wall, sorted(geometric - analytic), sorted(analytic - geometric)))
    every = uw.mpi.comm.allgather(differ)
    assert not any(every), (
        f"np {uw.mpi.size}: the analytic and geometric node sets differ, per rank "
        f"(wall, missed, extra): {every} (#935)")


@pytest.mark.tier_b
@pytest.mark.parametrize("regular, cell_size", FIXTURES)
def test_the_fixture_holds_an_owned_wall_node_no_local_facet_reaches(regular, cell_size):
    if uw.mpi.size not in HOLDS_AT[(regular, cell_size)]:
        pytest.skip(f"not measured to hold the case at np {uw.mpi.size}")
    stokes = _set_up(regular, cell_size)
    owned = _owned(stokes.dm)
    held = 0
    for wall in WALL_NORMALS:
        complete = {q for q, _n in rbc._boundary_velocity_nodes(stokes, wall)}
        held += len((complete - _reached_by_local_facets(stokes, wall)) & owned)
    total = sum(uw.mpi.comm.allgather(held))
    assert total > 0, (
        f"np {uw.mpi.size}: no rank owns a wall node that only a neighbour's facets "
        f"reach, so this fixture cannot see #935; the partition has changed under it "
        f"(see the module docstring)")


@pytest.mark.tier_b
@pytest.mark.parametrize("regular, cell_size", FIXTURES)
def test_no_normal_flow_at_any_wall_node(regular, cell_size):
    stokes, v = _box_solver(regular, cell_size)
    x, _y = stokes.mesh.X
    # buoyancy up the left wall and down the right: one cell, which moves a corner
    # that has lost its constraint rows
    stokes.bodyforce = sympy.Matrix([[0.0, sympy.cos(sympy.pi * x)]])
    for wall, normal in WALL_NORMALS.items():
        stokes.add_rotated_freeslip_bc(0, wall, normal=normal)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-8
    stokes.solve()
    assert stokes._rotated_freeslip_info["converged"]

    c = np.asarray(v.coords)
    u = np.asarray(v.array)[:, 0, :]
    scale = max(uw.mpi.comm.allgather(float(np.abs(u).max()) if u.size else 0.0))
    tol = 1.0e-12
    on_wall = {"Top": c[:, 1] > 1 - tol, "Bottom": c[:, 1] < tol,
               "Left": c[:, 0] < tol, "Right": c[:, 0] > 1 - tol}
    worst = 0.0
    for wall, mask in on_wall.items():
        if mask.any():
            un = u[mask] @ np.asarray(WALL_NORMALS[wall])
            worst = max(worst, float(np.abs(un).max()))
    worst = max(uw.mpi.comm.allgather(worst))
    assert scale > 0.0
    # exactly 0 serially (axis-aligned walls); on kaiju without the fix the missed
    # node carried 1.3e-10 to 0.67 of max |v| at np 2 and 4
    assert worst <= 1.0e-13 * scale, (
        f"np {uw.mpi.size}: normal flow {worst:.3e} at a wall node against "
        f"max |v| {scale:.3e} (#935)")


@pytest.mark.tier_b
@pytest.mark.parametrize("regular, cell_size", FIXTURES)
def test_the_free_surface_finds_every_surface_vertex(regular, cell_size):
    mesh = _box(regular, cell_size)
    found = FreeSurface._surface_node_coords(SimpleNamespace(mesh=mesh, surface="Top"))
    dm = mesh.dm
    v0, v1 = dm.getDepthStratum(0)
    csec = dm.getCoordinateSection()
    cvec = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, 2)
    local = cvec[[csec.getOffset(q) // 2 for q in range(v0, v1)]]
    on_top = {tuple(x) for x in local[local[:, 1] == 1.0]}
    missed = sorted(on_top - {tuple(x) for x in found})
    extra = sorted({tuple(x) for x in found} - on_top)
    every = uw.mpi.comm.allgather((missed, extra))
    assert not any(m or e for m, e in every), (
        f"np {uw.mpi.size}: the free surface's Top vertices miss or add local "
        f"vertices on Top, per rank (missed, extra): {every} (#935)")
