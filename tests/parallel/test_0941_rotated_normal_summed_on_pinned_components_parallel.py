"""The geometric wall normal is summed over every rank in every component (#941).

``rotated_bc._sum_node_vectors_across_ranks`` completes a node's geometric normal,
the sum of |f| n_f over the wall's facets, across the ranks. It rode the velocity
field's local-to-global scatter, which carries no DOF an essential condition
constrains: at a node where a rotated wall meets a wall pinning one component,
that component kept each rank's own share, zero on a rank holding none of the
node's facets. The normal there depended on the partition, and so did the
traction read at the node (H1 measured 0.5 % at a stranded corner) and the
value the corner's reduced constraint takes from the pinned part of the normal
(#939). The sum now rides the DM's point star forest, every component.

The fixture is a box whose top wall is tilted, so its normal has a component
along the pinned one, with rotated free slip on the top (the geometric normal)
and the vertical velocity pinned on the left wall. Every rank that holds a node
of the top wall must hold the same normal for it, compared by the node's global
point number, and the top-left corner's normal must be the tilted wall's,
(-1, 4) / sqrt(17), wherever the corner's facets live.

The partition is built, not left to the partitioner, so the corner is stranded
at any rank count: the top-left square is split along the diagonal from the
corner, and its lower triangle, which holds the corner and a Left facet but no
Top facet, goes to rank 1 with the column below it; rank 0 holds the triangle
with the corner's Top facet. (The default partitions did not strand it: at np 2
the old fixture shared no Top node, and at np 3 it passed on development, H7's
review of #1265. H1's shell partitions c941a/b, which strand it, failed on
development and passed on the branch.) If the distributed mesh does not
strand the corner, the tests skip.

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_0941_rotated_normal_summed_on_pinned_components_parallel.py
"""
from enum import Enum

import numpy as np
import pytest
import sympy
from petsc4py import PETSc

import underworld3 as uw
from underworld3.utilities import rotated_bc

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(300)]

TILT = 0.25                                       # the top wall rises by TILT over the width


N = 8                                             # cells along a side


class _Walls(Enum):
    Bottom = 11
    Top = 12
    Right = 13
    Left = 14
    All_Boundaries = 1001


def _stranded_corner_dm():
    """The tilted box as triangles, distributed so that rank 1 holds the top-left
    corner and none of its Top facets. Collective."""
    comm = PETSc.COMM_WORLD
    rank, size = comm.getRank(), comm.getSize()
    coords = np.zeros((0, 2))
    cells = np.zeros((0, 3), dtype=PETSc.IntType)
    if rank == 0:
        vid = lambda i, j: j * (N + 1) + i
        coords = np.array([[i / N, (j / N) * (1.0 + TILT * i / N)]
                           for j in range(N + 1) for i in range(N + 1)])
        tris = []
        for j in range(N):
            for i in range(N):
                a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
                # every square split along the diagonal from its top-left corner
                tris += [[a, b, d], [b, c, d]]
        cells = np.asarray(tris, dtype=PETSc.IntType)
    dm = PETSc.DMPlex().createFromCellList(2, cells, coords, interpolate=True, comm=comm)

    walls = {"Bottom": lambda x, y: y < 1.0e-12,
             "Top": lambda x, y: abs(y - (1.0 + TILT * x)) < 1.0e-12,
             "Left": lambda x, y: x < 1.0e-12,
             "Right": lambda x, y: x > 1.0 - 1.0e-12}
    for wall in _Walls:
        dm.createLabel(wall.name)
    if rank == 0:
        vStart, vEnd = dm.getDepthStratum(0)
        fStart, fEnd = dm.getDepthStratum(1)
        X = dm.getCoordinatesLocal().array.reshape(-1, 2)
        for f in range(fStart, fEnd):
            if dm.getSupportSize(f) != 1:
                continue
            ends = [v for v in dm.getCone(f)]
            xy = [X[v - vStart] for v in ends]
            for wall in ("Bottom", "Top", "Left", "Right"):
                if all(walls[wall](*p) for p in xy):
                    label = dm.getLabel(wall)
                    for q in [f] + ends:
                        label.setValue(q, _Walls[wall].value)
            dm.getLabel("All_Boundaries").setValue(f, _Walls.All_Boundaries.value)

    part = dm.getPartitioner()
    part.setType(PETSc.Partitioner.Type.SHELL)
    if rank == 0:
        # rank 1: the corner's lower triangle (square (0, N-1), first triangle)
        # and the left column below it; ranks 2+: shares of the rest; rank 0: the rest
        cell = lambda i, j, k: 2 * (j * N + i) + k
        to_one = [cell(0, j, k) for j in range(N - 1) for k in (0, 1)] + [cell(0, N - 1, 0)]
        rest = [c for c in range(len(cells)) if c not in set(to_one)]
        if size == 1:
            owner = [list(range(len(cells)))]
        else:
            shares = [rest] if size == 2 else np.array_split(rest, size - 1)
            owner = [list(shares[0]), to_one] + [list(s) for s in shares[1:]]
        part.setShellPartition(size, [len(o) for o in owner],
                               [int(c) for o in owner for c in o])
    else:
        part.setShellPartition(size, [0] * size, [])
    dm.distribute()
    return dm


def _corner_is_stranded(dm):
    """Whether some rank holds the top-left corner and none of its Top facets.
    Collective."""
    vStart, vEnd = dm.getDepthStratum(0)
    X = dm.getCoordinatesLocal().array.reshape(-1, 2)
    top = dm.getLabel("Top")
    stranded = False
    for v in range(vStart, vEnd):
        if np.abs(X[v - vStart] - (0.0, 1.0)).max() < 1.0e-12:
            stranded = not any(top.getValue(f) == _Walls.Top.value for f in dm.getSupport(v))
    return bool(uw.mpi.comm.allreduce(int(stranded)))


def _tilted_box_solver():
    dm = _stranded_corner_dm()
    if not _corner_is_stranded(dm):
        pytest.skip("the distributed mesh does not strand the corner")
    mesh = uw.discretisation.Mesh(dm, degree=1, qdegree=3, boundaries=_Walls)
    v = uw.discretisation.MeshVariable("V941", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P941", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, _y = mesh.X
    stokes.bodyforce = sympy.Matrix([[0, sympy.cos(sympy.pi * x)]])
    stokes.add_rotated_freeslip_bc(0, "Top")
    stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
    stokes.add_dirichlet_bc((None, 0.0), "Left")
    stokes.add_dirichlet_bc((0.0, 0.0), "Right")
    stokes.tolerance = 1.0e-8
    stokes.solve()
    return mesh, stokes


def test_every_rank_holds_the_same_normal_at_every_wall_node():
    mesh, stokes = _tilted_box_solver()
    dm = stokes.dm
    pStart, _pEnd = dm.getChart()
    numbering = dm.createPointNumbering().getIndices()
    nodes = rotated_bc._boundary_velocity_nodes(stokes, "Top")
    mine = []
    for q, n in nodes:
        g = int(numbering[q - pStart])
        mine.append((g if g >= 0 else -(g + 1), tuple(float(c) for c in n)))
    by_point = {}
    for entries in uw.mpi.comm.allgather(mine):
        for g, n in entries:
            by_point.setdefault(g, []).append(n)
    shared = {g: ns for g, ns in by_point.items() if len(ns) > 1}
    assert shared, "no wall node is held by more than one rank: the fixture shows nothing"
    differ = {g: ns for g, ns in shared.items()
              if np.abs(np.array(ns) - np.array(ns[0])).max() > 1.0e-14}
    assert not differ, differ


def test_the_corner_takes_the_tilted_walls_normal_on_every_rank():
    mesh, stokes = _tilted_box_solver()
    dm = stokes.dm
    left = dm.getLabel("Left")
    # the corner's one Top facet is the tilted line y = 1 + TILT x
    expected = np.array([-TILT, 1.0]) / np.hypot(TILT, 1.0)
    found = []
    for q, n in rotated_bc._boundary_velocity_nodes(stokes, "Top"):
        # the corner, by topology: the Top node the Left label also holds
        if left.getValue(q) >= 0:
            found.append(np.asarray(n))
    for n in found:
        assert np.abs(n - expected).max() < 1.0e-12, (n, expected)
    assert uw.mpi.comm.allreduce(len(found)) >= 1
