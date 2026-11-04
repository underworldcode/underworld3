"""An analytic rotated free-slip normal constrains every owned wall node at any rank
count (#935).

``rotated_bc._boundary_velocity_nodes`` takes a rank's wall nodes from the closures of
the boundary facets labelled on that rank. Each facet is labelled on one rank, so a
rank can own a wall node all of whose labelled facets sit on neighbours. The geometric
normal (``normal=None``) completes the node set over the point SF (#560). An analytic
or constant normal did not, and such a node got no constraint row. Kaiju met it on the
zg95 annulus at np 24 to 52 with the radial normal; np 16 happened to be clean.

What the analytic path lost is a rank whose share of a wall is one node, or a short
run of them, so the fixtures are coarse boxes. On kaiju (development 60dbf0e4) one rank
owns a corner at np 2, with no facet of either wall beside it: the unstructured box at
cellSize 0.5 and the regular box at cellSize 0.4, at np 2, 3 and 4.

- On every rank and wall, the analytic normal finds the geometric path's nodes.
- The fixtures exercise the case: some rank owns a wall node that no facet labelled on
  that rank reaches. If a partitioner change makes this fail, the fixtures have lost
  their power, not the library its fix; choose a mesh that shows it again with the
  campaigns probe ``probe_935_analytic_wall_nodes.py``.
- A rotated free-slip solve with the constant wall normals leaves no normal flow at any
  wall node. The node that got no row is a corner, which with its row on both walls
  carries no flow; without the rows it moves with the cell.

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1140_rotated_analytic_normal_wall_nodes_parallel.py
    mpirun -n 4 python -m pytest --with-mpi tests/parallel/test_1140_rotated_analytic_normal_wall_nodes_parallel.py
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
import underworld3.utilities.rotated_bc as rbc

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.timeout(300)]

WALL_NORMALS = {"Top": (0.0, 1.0), "Bottom": (0.0, -1.0),
                "Left": (-1.0, 0.0), "Right": (1.0, 0.0)}
FIXTURES = [(False, 0.5), (True, 0.4)]     # (regular, cellSize)


def _box_solver(regular, cell_size):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=cell_size,
        regular=regular, qdegree=2)
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
def test_the_analytic_normal_finds_every_wall_node(regular, cell_size):
    stokes = _set_up(regular, cell_size)
    missed = []
    for wall, normal in WALL_NORMALS.items():
        geometric = {q for q, _n in rbc._boundary_velocity_nodes(stokes, wall)}
        analytic = {q for q, _n in rbc._boundary_velocity_nodes(stokes, wall, normal=normal)}
        if geometric - analytic:
            missed.append((wall, sorted(geometric - analytic)))
    every = uw.mpi.comm.allgather(missed)
    assert not any(every), (
        f"np {uw.mpi.size}: the analytic normal misses wall nodes the geometric path "
        f"finds, per rank (wall, points): {every} (#935)")


@pytest.mark.tier_b
def test_the_fixtures_hold_an_owned_wall_node_no_local_facet_reaches():
    held = 0
    for regular, cell_size in FIXTURES:
        stokes = _set_up(regular, cell_size)
        owned = _owned(stokes.dm)
        for wall in WALL_NORMALS:
            complete = {q for q, _n in rbc._boundary_velocity_nodes(stokes, wall)}
            held += len((complete - _reached_by_local_facets(stokes, wall)) & owned)
    total = sum(uw.mpi.comm.allgather(held))
    assert total > 0, (
        f"np {uw.mpi.size}: no rank owns a wall node that only a neighbour's facets "
        f"reach, so this file cannot see #935; the partition has changed under the "
        f"fixtures (see the module docstring)")


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
    assert worst <= 1.0e-10 * scale, (
        f"np {uw.mpi.size}: normal flow {worst:.3e} at a wall node against "
        f"max |v| {scale:.3e} (#935)")
