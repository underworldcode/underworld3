"""The rotated workspace watches the coefficient variables of the kernels it runs (#546, m3).

The rotated free-slip workspace keeps the operator between solves and reassembles
it when a variable the operator depends on changes. It reads which variables those
are from the constitutive parameters once, and used to reuse that list whenever
the geometry matched. A rewire that adds a variable (a viscosity that now depends
on a temperature) was safe only because every rewire also tore the workspace down.
The list is now reused only for the same compiled kernels (the JIT key).

The test leaves a workspace that claims older kernels with no coefficients, as a
rewire that kept the workspace would. The next solve must read the list again, so
that a later change of the new variable reassembles the operator. Unwatched, the
first increment rode the old operator and the later ones reassembled: the answer
was right, but a linear solve took more than one increment.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _annulus_stokes(tag, mesh, T):
    v = uw.discretisation.MeshVariable(f"V{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P{tag}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0 + T.sym[0]
    X = mesh.X
    r = sympy.sqrt(X[0] ** 2 + X[1] ** 2)
    th = sympy.atan2(X[1], X[0])
    stokes.bodyforce = (sympy.cos(3 * th) * sympy.sin(sympy.pi * (r - 0.5) / 0.5)
                        * sympy.Matrix([X[0] / r, X[1] / r]))
    stokes.add_rotated_freeslip_bc(0, "Lower")
    stokes.add_rotated_freeslip_bc(0, "Upper")
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-8
    return v, stokes


def test_a_workspace_from_other_kernels_reads_its_coefficients_again():
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2, qdegree=3)
    T = uw.discretisation.MeshVariable("T1158", mesh, 1, degree=1)
    v, stokes = _annulus_stokes("a", mesh, T)
    stokes.solve()
    cache = stokes._rotated_linear_cache
    assert [var.name for var in cache["coefficient_variables"]] == ["T1158"]

    # a workspace built for older kernels with no coefficients
    cache["coefficient_variables"] = ()
    cache["coefficient_states"] = ()
    cache["operator_sig"] = ("older kernels", cache["operator_sig"][1])
    T.array[...] = 1.0                 # work for this solve, so it assembles and keeps a key
    stokes.solve(zero_init_guess=False)
    assert [var.name for var in stokes._rotated_linear_cache["coefficient_variables"]] \
        == ["T1158"]

    # development: T unwatched, the first increment rode the operator for viscosity 1
    T.array[...] = 10.0
    stokes.solve(zero_init_guess=False)
    v_reused = np.asarray(v.array).copy()

    v_fresh, fresh = _annulus_stokes("b", mesh, T)
    fresh.solve()
    reference = np.asarray(v_fresh.array)
    info = stokes._rotated_freeslip_info
    # a linear problem: one increment on the operator the new T gives
    assert info["converged"] and info["nonlinear_iterations"] == 1, info["nonlinear_iterations"]
    assert np.linalg.norm(v_reused - reference) < 1.0e-6 * np.linalg.norm(reference)
