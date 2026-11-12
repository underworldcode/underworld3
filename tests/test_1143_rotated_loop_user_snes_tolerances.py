"""The rotated Newton loop honours a user-set ``snes_rtol`` and ``snes_atol`` (#924).

``solve_rotated_freeslip`` took ``rtol`` from ``solver.tolerance`` and fixed ``atol`` at
1e-11, so a user who set ``petsc_options["snes_rtol"]`` or ``["snes_atol"]`` changed
nothing on a solve with rotated free slip. The keys now reach the loop: ``snes_rtol``
through the saddle point's owned-option record (as ``snes_max_it`` does, #920), and
``snes_atol`` and ``snes_stol`` when the user sets them.

The fixture is a power law in an annulus with rotated free slip on the outer wall, at
``tolerance`` 1e-10, so the loop takes several increments. Each test sets one key
looser than the tolerance would make it and checks the loop stops where that key says.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _rotated_power_law(name):
    mesh = uw.meshing.Annulus(radiusInner=0.5, radiusOuter=1.0, cellSize=0.2, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x**2 + y**2)
    unit_r = sympy.Matrix([[x / r, y / r]])
    th = sympy.atan2(y, x)
    v = uw.discretisation.MeshVariable(f"V{name}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P{name}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    edot = mesh.vector.strain_tensor(v.sym)
    eII = sympy.sqrt(sympy.Rational(1, 2) * (edot[0, 0] ** 2 + edot[1, 1] ** 2)
                     + edot[0, 1] ** 2)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = (
        (sympy.Float(0.01) + eII) ** sympy.Rational(-2, 3))
    stokes.bodyforce = 1.0e2 * sympy.cos(3 * th) * (r - 0.5) / 0.5 * unit_r
    stokes.add_dirichlet_bc((0.0, 0.0), "Lower")
    stokes.add_rotated_freeslip_bc(0.0, "Upper")
    stokes.tolerance = 1.0e-10
    return stokes


def test_a_user_snes_rtol_is_the_loop_s_rtol():
    stokes = _rotated_power_law("1143a")
    stokes.petsc_options["snes_rtol"] = 1.0e-3
    stokes.solve(zero_init_guess=True)
    info = stokes._rotated_freeslip_info
    history = np.asarray(info["rnorm_history"])
    target = 1.0e-3 * info["ref"] + 1.0e-11
    assert info["converged"], history
    # it stops at the FIRST iterate inside the user's target; at the tolerance's 1e-10
    # the iterate before the last is far inside it
    assert history[-1] <= target, (history, target)
    assert np.all(history[:-1] > target), (history, target)


def test_a_user_snes_atol_is_the_loop_s_atol():
    probe = _rotated_power_law("1143b")
    probe.solve(zero_init_guess=True)
    rest = probe._rotated_freeslip_info["rnorm0"]
    assert probe._rotated_freeslip_info["nonlinear_iterations"] > 1

    stokes = _rotated_power_law("1143c")
    stokes.petsc_options["snes_atol"] = 10.0 * rest
    stokes.solve(zero_init_guess=True)
    info = stokes._rotated_freeslip_info
    # the cold residual is already inside the user's absolute target: no increment
    assert info["converged"], info["rnorm_history"]
    assert info["nonlinear_iterations"] == 0, info["rnorm_history"]
