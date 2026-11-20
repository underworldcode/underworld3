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

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_0941_rotated_normal_summed_on_pinned_components_parallel.py
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.utilities import rotated_bc

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(300)]

TILT = 0.25                                       # the top wall rises by TILT over the width


def _tilted_box_solver():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=1.0 / 8, qdegree=3)
    X = np.asarray(mesh.X.coords).copy()
    X[:, 1] *= 1.0 + TILT * X[:, 0]
    mesh.deform(X)
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
