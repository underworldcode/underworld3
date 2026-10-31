"""Interpolation within one mesh on a rank that holds cells but owns none of their DOFs.

PETSc's ``DMPlexComputeInterpolatorNested`` builds the matrix between two DMs on
one plex (a coordinate DM and a basis DM, a variable's sub-DM and a scratch
clone). Since v3.24.0 it skips its preallocation pass on a rank with no local
rows and no local columns, but runs its fill pass over that rank's cells all
the same. A
shared point belongs to the highest rank that holds it, so a rank whose cells
sit wholly inside higher ranks' regions owns nothing. Its entries then reach
owners that never preallocated them: the owner raises "New nonzero caused a
malloc" inside ``MatAssemblyEnd``, and the rank that did not raise waits in the
assembly's own collective for ever. Found on kaiju at np 32 and 52 on the
124-cell coarsest level of the faulted annulus, where custom-P multigrid reads
each level's node coordinates (#902), and on a 3-D notch mesh at np 16.

A shell partition gives rank 0 one interior cell of a 4 x 4 grid of triangles
and the other ranks the rest, so rank 0 owns none of its cell's vertices and
edges. The test checks that case is set up, then that the node coordinates
(degree 1 to 3, the third with two nodes on each edge) and the projection
between degrees come back right on every rank, through UW3's copy of the
interpolator (``petsc_dm_create_interpolation_within_plex``). A custom-P FMG
solve whose coarse level is that grid, under its refinement, pins the same
case inside a real multigrid build. Before the fix the degree-2 coordinates
hung, so each test carries a thread timeout, which fails the run instead of
stalling it.

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1107_interpolation_rank_owning_nothing_parallel.py
"""
from enum import Enum

import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

import underworld3 as uw
from underworld3.function import project_to_degree
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(120, method="thread")]

N = 4
_WALLS = Enum("Boundaries", {"Walls": 1})


def _owned_nothing_dm():
    """The 4 x 4 grid of triangles, all on rank 0, partitioned so rank 0 keeps one
    interior cell (every edge shared with a cell of another rank). Its outer
    edges carry the label ``Walls``."""
    comm = PETSc.COMM_WORLD
    rank, size = comm.getRank(), comm.getSize()
    if rank == 0:
        xs = np.linspace(0.0, 1.0, N + 1)
        coords = np.array([[x, y] for y in xs for x in xs])
        cells = []
        for j in range(N):
            for i in range(N):
                a = j * (N + 1) + i
                cells += [[a, a + 1, a + N + 2], [a, a + N + 2, a + N + 1]]
        cells = np.asarray(cells, dtype=PETSc.IntType)
    else:
        coords = np.zeros((0, 2))
        cells = np.zeros((0, 3), dtype=PETSc.IntType)
    dm = PETSc.DMPlex().createFromCellList(2, cells, coords, interpolate=True, comm=comm)
    dm.createLabel("Walls")
    dm.markBoundaryFaces("Walls", 1)
    dm.labelComplete(dm.getLabel("Walls"))
    part = dm.getPartitioner()
    part.setType(PETSc.Partitioner.Type.SHELL)
    if rank == 0:
        cS, cE = dm.getHeightStratum(0)
        interior = [c for c in range(cS, cE)
                    if all(dm.getSupportSize(e) == 2 for e in dm.getCone(c))]
        c0 = interior[len(interior) // 2]
        rest = np.array_split(np.asarray([c for c in range(cS, cE) if c != c0]), size - 1)
        part.setShellPartition(size, [1] + [len(r) for r in rest],
                               [c0] + [int(c) for r in rest for c in r])
    else:
        part.setShellPartition(size, [0] * size, [])
    dm.distribute()
    return dm


@pytest.fixture(scope="module")
def mesh():
    return uw.discretisation.Mesh(_owned_nothing_dm(), degree=1, qdegree=2)


def _all(flag):
    return uw.mpi.comm.allreduce(bool(flag), op=MPI.LAND)


def _cell_vertex_coords(mesh):
    """Each local cell's vertex coordinates, read from the coordinate section."""
    dm = mesh.dm
    vS, vE = dm.getDepthStratum(0)
    sec = dm.getCoordinateSection()
    x = dm.getCoordinatesLocal().array
    cS, cE = dm.getHeightStratum(0)
    out = []
    for c in range(cS, cE):
        pts = dm.getTransitiveClosure(c)[0]
        out.append(np.array([x[sec.getOffset(v):sec.getOffset(v) + 2]
                             for v in pts if vS <= v < vE]))
    return out


def _sorted_rows(a):
    a = np.round(np.asarray(a), 12)
    return a[np.lexsort(a.T[::-1])]


def _rank_zero_owns_nothing(mesh):
    """Rank 0 holds one cell and owns no vertex or edge, as one verdict for every
    rank: a rank-0-only assert would leave the others waiting in the next
    collective until the timeout."""
    cS, cE = mesh.dm.getHeightStratum(0)
    leaves = set(np.asarray(mesh.dm.getPointSF().getGraph()[1]).tolist())
    owned = 0
    for d in (0, 1):
        pS, pE = mesh.dm.getDepthStratum(d)
        owned += sum(1 for p in range(pS, pE) if p not in leaves)
    return _all(uw.mpi.rank != 0 or (cE - cS == 1 and owned == 0))


def test_rank_zero_holds_a_cell_and_owns_nothing(mesh):
    """The case under test."""
    assert _rank_zero_owns_nothing(mesh)


def _expected_nodes(verts, degree, t=None):
    """A triangle's Lagrange nodes: its vertices, then (degree 2) its edge
    midpoints, or (degree 3) two points on each edge, at ``t`` and ``1 - t``
    from either end, and the centroid."""
    a, b = verts[[0, 1, 2]], verts[[1, 2, 0]]
    if degree == 1:
        return verts
    if degree == 2:
        return np.vstack([verts, 0.5 * (a + b)])
    return np.vstack([verts, (1 - t) * a + t * b, t * a + (1 - t) * b,
                      verts.mean(axis=0, keepdims=True)])


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_node_coordinates_on_every_rank(mesh, degree):
    """Each cell's nodes are its vertices and, by degree, its edge midpoints,
    or two points per edge and its centroid. Degree 3 places its edge points by
    PETSc's node rule (``node_endpoints`` False), so ``t`` is read from the
    first cell and must then hold on every edge of every cell, on every rank."""
    coords = mesh._get_coords_for_basis(degree, True)
    rows = mesh._cell_node_indices(degree, True)
    cells = _cell_vertex_coords(mesh)
    t = None
    if degree == 3 and cells:
        v, nodes = cells[0], coords[rows[0]]
        on_edge = [np.dot(x - v[0], v[1] - v[0]) / np.dot(v[1] - v[0], v[1] - v[0])
                   for x in nodes if abs(np.cross(v[1] - v[0], x - v[0])) < 1e-12]
        t = min(s for s in on_edge if s > 1e-12)
    t_all = uw.mpi.comm.allgather(t)
    ok = all(abs(s - t_all[0]) < 1e-12 for s in t_all if s is not None) if degree == 3 else True
    for k, verts in enumerate(cells):
        got = coords[rows[k]]
        ok &= np.allclose(_sorted_rows(got), _sorted_rows(_expected_nodes(verts, degree, t)),
                          atol=1e-12)
    assert _all(ok)


def test_projection_between_degrees_on_every_rank(mesh):
    """P1 to P2 of a linear field and P2 to P1 of a quadratic one are exact at the
    nodes."""
    s1 = uw.discretisation.MeshVariable("s1_1107", mesh, 1, degree=1)
    s2 = uw.discretisation.MeshVariable("s2_1107", mesh, 1, degree=2)
    x1 = np.asarray(s1.coords)
    x2 = np.asarray(s2.coords)
    s1.array[...] = (x1[:, 0] + 2.0 * x1[:, 1]).reshape(-1, 1, 1)
    s2.array[...] = (x2[:, 0] ** 2).reshape(-1, 1, 1)

    up = project_to_degree(s1, target_degree=2)
    down = project_to_degree(s2, target_degree=1)
    n2 = mesh._get_coords_for_basis(2, True)
    n1 = mesh._get_coords_for_basis(1, True)
    ok = (np.allclose(up[:, 0], n2[:, 0] + 2.0 * n2[:, 1], atol=1e-12)
          and np.allclose(down[:, 0], n1[:, 0] ** 2, atol=1e-12))
    assert _all(ok)


def _poisson_on(dm, name):
    """A degree-2 Poisson problem, zero on the walls with a unit source."""
    mesh = uw.discretisation.Mesh(dm.clone(), degree=1, qdegree=3, boundaries=_WALLS)
    T = uw.discretisation.MeshVariable(name, mesh, 1, degree=2)
    p = uw.systems.Poisson(mesh, T)
    p.constitutive_model = uw.constitutive_models.DiffusionModel
    p.constitutive_model.Parameters.diffusivity = 1.0
    p.f = 1.0
    p.add_dirichlet_bc(0.0, "Walls")
    p.petsc_options["ksp_rtol"] = 1.0e-10
    return p


def test_custom_fmg_over_a_level_where_a_rank_owns_nothing():
    """Custom-P FMG reads every level's node coordinates. Its coarse level is the
    grid on which rank 0 owns nothing, under the grid's refinement. The build
    installs multigrid, and the solve agrees with GAMG's."""
    base = _owned_nothing_dm()
    fine = base.refine()
    coarse = uw.discretisation.Mesh(base.clone(), degree=1, qdegree=3, boundaries=_WALLS)
    assert _rank_zero_owns_nothing(coarse)                      # the case under test

    reference = _poisson_on(fine, "Tg_1107")
    reference.preconditioner = "gamg"
    reference.solve()

    solver = _poisson_on(fine, "Tm_1107")
    custom_mg.set_custom_fmg(solver, [coarse], builder="barycentric")
    solver.solve()

    assert solver.snes.getKSP().getPC().getType() == "mg"
    assert solver.snes.getConvergedReason() > 0
    assert "custom_mg.build" not in solver.pc_fallbacks
    u, r = np.asarray(solver.Unknowns.u.data), np.asarray(reference.Unknowns.u.data)
    num = uw.mpi.comm.allreduce(float(np.sum((u - r) ** 2)))
    den = uw.mpi.comm.allreduce(float(np.sum(r ** 2)))
    assert (num / den) ** 0.5 < 1.0e-6
