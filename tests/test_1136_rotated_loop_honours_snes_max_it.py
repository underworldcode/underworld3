"""The rotated Newton/Picard loop honours a user-set ``snes_max_it`` (#920).

A solve with rotated free slip or a split fault runs its own outer loop,
``rotated_bc.solve_rotated_freeslip``, whose cap defaults to 50. The solver
called it without passing a cap, so a user's ``snes_max_it`` never reached it,
while the standard path resolves the same key (#483). On the add-on matrix's
yielding-lid box, Coulomb and rate-and-state faults stopped at 50 iterations
still in their Picard phase, and only wrapping the function in the driver raised
the cap.

The fixture is test_1018's power-law box (eta = eps_II^(1/3 - 1)) under rotated
free slip on every wall, with the frozen (Picard) tangent: 43 increments at
``tolerance`` 1e-7 and 54 at 1e-9, measured on macOS. The tests check that the
cap can be raised past 50 and lowered, that the loop tests the residual after its
last increment as PETSc's SNES does (a linear solve converges under a cap of 1,
and a cap of 0 is one residual evaluation), and that the cap reaches the vector
solvers' resolver and ``fault_contact.solve_with_fault``.
"""
import numpy as np
import pytest
import sympy
from petsc4py import PETSc

import underworld3 as uw
from underworld3.utilities import fault_contact

from test_1018_rotated_freeslip import _powerlaw_stokes

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

DIVERGED_MAX_IT = PETSc.SNES.ConvergedReason.DIVERGED_MAX_IT


def _box():
    return uw.meshing.StructuredQuadBox(
        elementRes=(8, 8), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)


def _rotated_powerlaw_box(prefix, tolerance):
    s, _v, _p = _powerlaw_stokes(_box(), prefix, cj=False)
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.tolerance = tolerance
    return s


def test_the_cap_can_be_raised_past_50():
    s = _rotated_powerlaw_box("raised", 1e-9)
    s.petsc_options["snes_max_it"] = 100
    s.solve()
    info = s._rotated_freeslip_info
    assert info["converged"], f"not converged in {info['nonlinear_iterations']} increments"
    assert info["nonlinear_iterations"] > 50, (
        "the fixture converges within the default cap, so the test cannot fail")


def test_the_cap_can_be_lowered():
    s = _rotated_powerlaw_box("lowered", 1e-7)
    s.petsc_options["snes_max_it"] = 3
    s.solve()
    info = s._rotated_freeslip_info
    assert info["nonlinear_iterations"] == 3, (
        f"the rotated loop ran {info['nonlinear_iterations']} increments under "
        f"snes_max_it = 3 (#920)")
    assert not info["converged"]
    assert s.snes.getConvergedReason() == DIVERGED_MAX_IT


def test_a_linear_solve_converges_under_a_cap_of_one():
    mesh = _box()
    x, _y = mesh.X
    s = uw.systems.Stokes(mesh)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    s.bodyforce = sympy.Matrix([[0.0, -sympy.cos(sympy.pi * x)]])
    s.petsc_use_pressure_nullspace = True
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_rotated_freeslip_bc(0, wall)
    s.petsc_options["snes_max_it"] = 1
    s.solve()
    info = s._rotated_freeslip_info
    assert info["nonlinear_iterations"] == 1
    assert info["converged"], "the residual after the last increment was not tested"


def test_a_cap_of_zero_evaluates_the_residual_only():
    s = _rotated_powerlaw_box("zero", 1e-7)
    s.petsc_options["snes_max_it"] = 0
    s.solve()
    info = s._rotated_freeslip_info
    assert info["nonlinear_iterations"] == 0
    assert not info["converged"]
    assert s.snes.getConvergedReason() == DIVERGED_MAX_IT


def test_a_vector_solver_reads_the_user_cap():
    mesh = _box()
    u = uw.discretisation.MeshVariable("u_cap", mesh, mesh.dim, degree=2)
    projection = uw.systems.Vector_Projection(mesh, u)
    assert projection._rotated_max_it() == 50
    projection.petsc_options["snes_max_it"] = 7
    assert projection._rotated_max_it() == 7


def test_solve_with_fault_takes_the_cap():
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1, refinement=1,
        qdegree=2)
    mesh = base.add_fault([("Flt", np.array([[0.30, 0.45], [0.70, 0.55]]))])
    _x, y = mesh.X
    s = uw.systems.Stokes(mesh)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    s.petsc_use_pressure_nullspace = True
    for wall in ("Top", "Bottom", "Left", "Right"):
        s.add_dirichlet_bc((y - 0.5, 0.0), wall)
    s.add_fault_bc(0, "Flt")
    s.petsc_options["snes_max_it"] = 0
    info = fault_contact.solve_with_fault(s)
    assert info["nonlinear_iterations"] == 0
    assert not info["converged"]
