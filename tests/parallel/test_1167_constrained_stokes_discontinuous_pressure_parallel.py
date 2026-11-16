"""A constrained Stokes solve with a discontinuous pressure completes (#1167).

The local index sets that copy the solution back into the variables took every
unknown of a field only when the PRESSURE was continuous, whatever field was
asked for, and then one index per point. With a discontinuous pressure the
multipliers lost their pinned interior unknowns, the velocity's complement grew
by them (306 entries for a variable of 162 on this mesh), and the copy-back
stopped with a broadcast error after the solve. Now every unknown of each point
is taken for any field.

The constraint holds on the walls (normal velocity ~0), and the solution agrees
with the continuous-pressure one to the discretisation difference.

    python scripts/mpi_supervisor.py --silence 300 -- mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1167_constrained_stokes_discontinuous_pressure_parallel.py
"""
import numpy as np
import pytest
import sympy
from mpi4py import MPI

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(300)]


def _solve(continuous):
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4), qdegree=3)
    v = uw.discretisation.MeshVariable(f"V1167{int(continuous)}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P1167{int(continuous)}", mesh, 1, degree=1,
                                       continuous=continuous)
    stokes = uw.systems.Stokes_Constrained(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, _y = mesh.X
    stokes.bodyforce = sympy.Matrix([[0.0, sympy.cos(sympy.pi * x)]])
    stokes.add_essential_bc((0.0, 0.0), "Bottom")
    stokes.add_constraint_bc(0.0, "Left", normal=sympy.Matrix([[-1.0, 0.0]]))
    stokes.add_constraint_bc(0.0, "Right", normal=sympy.Matrix([[1.0, 0.0]]))
    stokes.solve()
    X = np.asarray(v.coords)
    V = np.asarray(v.array)[:, 0, :]
    side = (np.abs(X[:, 0]) < 1e-9) | (np.abs(X[:, 0] - 1) < 1e-9)
    largest = uw.mpi.comm.allreduce(float(np.abs(V).max(initial=0.0)), op=MPI.MAX)
    normal = uw.mpi.comm.allreduce(float(np.abs(V[side, 0]).max(initial=0.0)), op=MPI.MAX)
    return stokes.snes.getConvergedReason(), largest, normal


def test_a_discontinuous_pressure_solves_with_constraint_conditions():
    reason, largest, normal = _solve(continuous=False)
    assert reason > 0, reason
    assert normal < 1e-5, normal
    _, reference, _ = _solve(continuous=True)
    assert largest == pytest.approx(reference, rel=1e-3), (largest, reference)
