"""The direct-LU rotated solve pins one pressure DOF at any rank count (#348).

Rotated free slip's opt-in direct path (``solver._rotated_use_lu``) has no null space,
so it fixes the pressure gauge by pinning one pressure DOF. Each rank used to pin the
first pressure DOF of its own chart: at np > 1 every rank pinned one, the gauge was
over-constrained, and the answer depended on the partition. The pin is now the lowest
global pressure row, chosen over all ranks.

- exactly one pressure row is pinned, whatever the rank count;
- the direct solve agrees with the iterative one (which fixes the gauge with the
  constant-pressure null space) in velocity, and in pressure up to its constant.

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1156_rotated_lu_pressure_pin_parallel.py
"""
import numpy as np
import pytest
import sympy
from petsc4py import PETSc

import underworld3 as uw
from underworld3.utilities import rotated_bc

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b,
              pytest.mark.timeout(300, method="thread"),
              pytest.mark.skipif(not PETSc.Sys.hasExternalPackage("mumps"),
                                 reason="the direct rotated path factors with MUMPS")]


def _box(tag, use_lu):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.1, qdegree=3)
    v = uw.discretisation.MeshVariable(f"V{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P{tag}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, y = mesh.X
    stokes.bodyforce = sympy.Matrix([[0, sympy.cos(sympy.pi * x) * sympy.sin(sympy.pi * y)]])
    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_rotated_freeslip_bc(0, wall)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-10
    if use_lu:
        stokes._rotated_use_lu = True
    stokes.solve()
    return stokes, v, p


def _global(values):
    """The locally owned-and-ghost values as one array on every rank (an
    allgather), so a norm is the same everywhere; ghosts count twice, alike in
    both solves, which a relative difference does not mind."""
    return np.concatenate(uw.mpi.comm.allgather(np.asarray(values).ravel()))


def test_one_pressure_row_is_pinned_at_any_rank_count():
    stokes, _v, _p = _box("pin", use_lu=True)
    row, pinned = rotated_bc._global_pressure_pin(stokes.dm)
    assert pinned
    # development: one per rank
    assert uw.mpi.comm.allreduce(int(row is not None)) == 1
    assert len(set(uw.mpi.comm.allgather(pinned))) == 1


def test_the_direct_solve_matches_the_iterative_one():
    _s, v_lu, p_lu = _box("lu", use_lu=True)
    _s, v_it, p_it = _box("it", use_lu=False)
    vl, vi = _global(v_lu.data), _global(v_it.data)
    pl, pi = _global(p_lu.data), _global(p_it.data)
    pl, pi = pl - pl.mean(), pi - pi.mean()
    # development at np 2: two pressure DOFs pinned to zero, a different problem
    assert np.linalg.norm(vl - vi) < 1.0e-6 * np.linalg.norm(vi)
    assert np.linalg.norm(pl - pi) < 1.0e-6 * np.linalg.norm(pi)
