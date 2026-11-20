"""Stokes_Constrained with a P3 velocity gives the essential-wall solution.

Holding the side walls' normal velocity by a constraint and by an essential
condition poses the same problem, and with a P2 velocity the two agree to 7
digits. With a P3 velocity the constrained solve came out about 26% low (max |v|
0.0860 against 0.1162 here, 0.0593 on simplices), on either wall and with either
multiplier degree, and refining did not close the gap.

The cause was the section rebuilt to constrain the multipliers' interior dofs: it
was built field by field and did not carry the fields' symmetries, which orient
the dofs of a point a cell sees reversed. A P2 edge has one dof and needs none; a
P3 edge has two, so every cell next to a reversed edge read its velocity in the
wrong order (found in the review of #1200).

    python scripts/mpi_supervisor.py --silence 300 -- mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1200_constrained_stokes_p3_velocity_parallel.py
"""
import numpy as np
import pytest
import sympy
from mpi4py import MPI

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(300)]

_count = [0]


def _solve(simplex, constrained, multiplier_degree=None, velocity_degree=3):
    _count[0] += 1
    if simplex:
        mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.25, qdegree=velocity_degree + 1)
    else:
        mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4), qdegree=velocity_degree + 1)
    v = uw.discretisation.MeshVariable(f"V1200p3_{_count[0]}", mesh, 2, degree=velocity_degree)
    p = uw.discretisation.MeshVariable(f"P1200p3_{_count[0]}", mesh, 1,
                                       degree=velocity_degree - 1)
    solver = (uw.systems.Stokes_Constrained if constrained else uw.systems.Stokes)(
        mesh, velocityField=v, pressureField=p)
    solver.constitutive_model = uw.constitutive_models.ViscousFlowModel
    solver.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, y = mesh.X
    solver.bodyforce = sympy.Matrix([[0.0, sympy.cos(sympy.pi * x) * (1 + y)]])
    solver.tolerance = 1.0e-10
    solver.petsc_options.delValue("ksp_monitor")
    solver.add_essential_bc((0.0, 0.0), "Bottom")
    for wall, nx in (("Left", -1.0), ("Right", 1.0)):
        if constrained:
            solver.add_constraint_bc(0.0, wall, normal=sympy.Matrix([[nx, 0.0]]),
                                     degree=multiplier_degree)
        else:
            solver.add_essential_bc((0.0, None), wall)
    solver.solve()
    assert solver.snes.getConvergedReason() > 0
    return np.asarray(v.array)[:, 0, :].copy()


def _max(a):
    return uw.mpi.comm.allreduce(float(np.abs(a).max(initial=0.0)), op=MPI.MAX)


@pytest.mark.parametrize("simplex", [False, True], ids=["quads", "simplices"])
@pytest.mark.parametrize("multiplier_degree", [None, 3], ids=["P2 multiplier", "P3 multiplier"])
def test_a_p3_velocity_held_by_the_constraint_is_the_essential_solution(simplex,
                                                                        multiplier_degree):
    essential = _solve(simplex, constrained=False)
    constrained = _solve(simplex, constrained=True, multiplier_degree=multiplier_degree)
    scale = _max(essential)
    # The gap was 26%. A P3 multiplier holds the P3 trace as the essential condition
    # does (1e-7 of the scale, as P2 against P2). The default P2 multiplier holds
    # it only weakly (the wall normal velocity is 7e-6), and gives 2e-6.
    tolerance = 1.0e-6 if multiplier_degree == 3 else 1.0e-5
    assert _max(constrained - essential) < tolerance * scale, (
        f"max |v| constrained {_max(constrained):.8f}, essential {scale:.8f}")
