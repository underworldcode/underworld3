"""The direct-LU rotated solve pins one pressure DOF at any rank count (#348).

Rotated free slip's opt-in direct path (``solver._rotated_use_lu``) has no null space,
so it fixes the pressure gauge by pinning one pressure DOF. Each rank used to pin the
first pressure DOF of its own chart: at np > 1 every rank pinned one, the gauge was
over-constrained, and the answer depended on the partition. The pin is now the lowest
global pressure row, chosen over all ranks.

- exactly one pressure row is pinned, whatever the rank count;
- the pinned row is a pressure row, the lowest one (an off-by-one row is an
  eliminated velocity row and would pass the other checks);
- the direct solve agrees with the iterative one (which fixes the gauge with the
  constant-pressure null space) in velocity, and in pressure up to its constant;
- a nonlinear direct solve takes a second Newton increment at np > 1: the pin
  used to compress its row out of the rotated operator's pattern, and the next
  increment's in-place PtAP crashed in malloc on the owning rank;
- a pressure Dirichlet condition already fixes the gauge, so nothing is pinned
  and the direct solve matches the iterative one.

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1156_rotated_lu_pressure_pin_parallel.py
"""
import numpy as np
import pytest
import sympy
from mpi4py import MPI
from petsc4py import PETSc

import underworld3 as uw
from underworld3.utilities import rotated_bc

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b,
              pytest.mark.timeout(300, method="thread"),
              pytest.mark.skipif(not PETSc.Sys.hasExternalPackage("mumps"),
                                 reason="the direct rotated path factors with MUMPS")]


def _box(tag, use_lu, nonlinear=False, pressure_top=False):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.1, qdegree=3)
    v = uw.discretisation.MeshVariable(f"V{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P{tag}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = (
        1 + 50 * v.sym.dot(v.sym) if nonlinear else 1.0)
    x, y = mesh.X
    stokes.bodyforce = sympy.Matrix([[0, sympy.cos(sympy.pi * x) * sympy.sin(sympy.pi * y)]])
    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_rotated_freeslip_bc(0, wall)
    if pressure_top:
        stokes.add_condition(f_id=1, c_type="dirichlet", conds=0.0, label="Top")
    stokes.petsc_use_pressure_nullspace = not pressure_top
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
    owned = stokes._subdict["pressure"][0].getIndices()
    lowest = uw.mpi.comm.allreduce(int(owned.min()) if owned.size else np.iinfo(np.int64).max,
                                   op=MPI.MIN)
    # every collective before any assert: a rank that fails alone hangs the others
    found = uw.mpi.comm.allreduce(row if row is not None else -1, op=MPI.MAX)
    owner_ok = uw.mpi.comm.allreduce(row is None or bool(np.isin(row, owned)), op=MPI.LAND)
    assert owner_ok
    assert found == lowest


def test_the_direct_solve_matches_the_iterative_one():
    _s, v_lu, p_lu = _box("lu", use_lu=True)
    _s, v_it, p_it = _box("it", use_lu=False)
    vl, vi = _global(v_lu.data), _global(v_it.data)
    pl, pi = _global(p_lu.data), _global(p_it.data)
    pl, pi = pl - pl.mean(), pi - pi.mean()
    # development at np 2: two pressure DOFs pinned to zero, a different problem
    assert np.linalg.norm(vl - vi) < 1.0e-6 * np.linalg.norm(vi)
    assert np.linalg.norm(pl - pi) < 1.0e-6 * np.linalg.norm(pi)


def test_a_nonlinear_direct_solve_takes_a_second_increment():
    stokes, v_lu, _p = _box("nllu", use_lu=True, nonlinear=True)
    info = stokes._rotated_freeslip_info
    # development at np 2: a malloc error in MatPtAP on the pin's owner, a hang elsewhere
    assert info["converged"]
    assert info["nonlinear_iterations"] >= 2
    _s, v_it, _p = _box("nlit", use_lu=False, nonlinear=True)
    vl, vi = _global(v_lu.data), _global(v_it.data)
    assert np.linalg.norm(vl - vi) < 1.0e-6 * np.linalg.norm(vi)


def test_a_pressure_dirichlet_condition_is_the_gauge():
    stokes, v_lu, p_lu = _box("pdlu", use_lu=True, pressure_top=True)
    row, pinned = rotated_bc._global_pressure_pin(stokes.dm)
    _s, v_it, p_it = _box("pdit", use_lu=False, pressure_top=True)
    vl, vi = _global(v_lu.data), _global(v_it.data)
    pl, pi = _global(p_lu.data), _global(p_it.data)
    # a pin on a real pressure row: 0.9-3.6 % off in velocity (#348 review 2, N4)
    assert np.linalg.norm(vl - vi) < 1.0e-6 * np.linalg.norm(vi)
    assert np.linalg.norm(pl - pi) < 1.0e-6 * np.linalg.norm(pi)
    assert not pinned and row is None
