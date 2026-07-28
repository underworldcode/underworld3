"""Test that `GenericFluxModel.Parameters.flux` properly propagates to the solver.

Verifies that the flux expression is stored as raw sympy (not wrapped in
UWexpression) so the JIT sees field-variable dependencies correctly.

Problem:  -div(k·grad(phi)) = f  with phi=0 on all walls, f=1.
Solve with k=1, then change k → 10 via Parameters.flux.
Solution scales as 1/k.
"""

import sympy
import pytest
import numpy as np
import underworld3 as uw
from underworld3.systems import Poisson


@pytest.fixture(scope="function")
def mesh():
    uw.reset_default_model()
    return uw.meshing.StructuredQuadBox(elementRes=(4, 4))


def _setup_solver(u, mesh, flux_expr):
    """Helper: configure Poisson with GenericFluxModel and direct solve."""
    solver = Poisson(mesh, u_Field=u)
    solver.petsc_options["ksp_type"] = "preonly"
    solver.petsc_options["pc_type"] = "lu"
    solver.petsc_options["pc_factor_mat_solver_type"] = "mumps"
    solver.constitutive_model = uw.constitutive_models.GenericFluxModel(solver.Unknowns)
    solver.constitutive_model.Parameters._solver = solver
    solver.constitutive_model.Parameters.flux = flux_expr
    solver.f = 1.0
    solver.add_essential_bc(0.0, "Bottom")
    solver.add_essential_bc(0.0, "Top")
    solver.add_essential_bc(0.0, "Left")
    solver.add_essential_bc(0.0, "Right")
    return solver


@pytest.mark.tier_a
@pytest.mark.level_1
def test_flux_update_via_force_setup(mesh):
    """Change flux between solves using _force_setup=True."""
    u = uw.discretisation.MeshVariable("u", mesh, 1, degree=2)
    x, y = mesh.X

    grad = sympy.Matrix([u.sym.diff(x)[0], u.sym.diff(y)[0]])
    solver = _setup_solver(u, mesh, grad)

    solver.solve(_force_setup=True, zero_init_guess=True)
    peak_a = float(np.max(np.abs(u.data)))
    assert peak_a > 1e-10, f"Solution is zero (k=1)"

    # Change flux: k=10
    solver.constitutive_model.Parameters.flux = \
        sympy.Matrix([10 * u.sym.diff(x)[0], 10 * u.sym.diff(y)[0]])
    solver.solve(_force_setup=True, zero_init_guess=False)
    peak_b = float(np.max(np.abs(u.data)))
    assert peak_b > 1e-10, f"Solution is zero (k=10)"

    ratio = peak_a / peak_b if peak_b > 0 else 0
    print(f"  force_setup: peak_a={peak_a:.6f}, peak_b={peak_b:.6f}, ratio={ratio:.3f}")
    assert abs(ratio - 10.0) < 2.0


@pytest.mark.tier_a
@pytest.mark.level_1
def test_flux_update_via_solver_link(mesh):
    """Change flux WITHOUT _force_setup, relying on _solver link + _reset()."""
    u = uw.discretisation.MeshVariable("u", mesh, 1, degree=2)
    x, y = mesh.X

    grad = sympy.Matrix([u.sym.diff(x)[0], u.sym.diff(y)[0]])
    solver = _setup_solver(u, mesh, grad)

    solver.solve(_force_setup=True, zero_init_guess=True)
    peak_a = float(np.max(np.abs(u.data)))
    assert peak_a > 1e-10

    # Change flux via _solver link (no _force_setup)
    solver.constitutive_model.Parameters.flux = \
        sympy.Matrix([10 * u.sym.diff(x)[0], 10 * u.sym.diff(y)[0]])

    assert not solver.constitutive_model._solver_is_setup
    assert solver._needs_function_rewire

    solver.solve(zero_init_guess=False)
    peak_b = float(np.max(np.abs(u.data)))
    assert peak_b > 1e-10

    ratio = peak_a / peak_b if peak_b > 0 else 0
    print(f"  solver_link: peak_a={peak_a:.6f}, peak_b={peak_b:.6f}, ratio={ratio:.3f}")
    assert abs(ratio - 10.0) < 2.0


@pytest.mark.tier_a
@pytest.mark.level_1
def test_flux_stores_raw_sympy(mesh):
    """Flux must NOT be wrapped in UWexpression — JIT must see field variables."""
    u = uw.discretisation.MeshVariable("u", mesh, 1, degree=2)
    x, y = mesh.X

    solver = Poisson(mesh, u_Field=u)
    solver.constitutive_model = uw.constitutive_models.GenericFluxModel(solver.Unknowns)
    solver.constitutive_model.Parameters.flux = \
        sympy.Matrix([u.sym.diff(x)[0], u.sym.diff(y)[0]])

    flux = solver.constitutive_model.Parameters.flux

    # Check it's a plain sympy Matrix, not a UWexpression
    assert isinstance(flux, sympy.Matrix), f"Expected sympy.Matrix, got {type(flux)}"
    # The elements should be raw sympy, not UWexpressions
    # UWexpressions show as "q_0 = ..." in str representation
    flux_str = str(flux)
    assert "q_" not in flux_str, (
        f"Flux elements are wrapped in UWexpressions. "
        f"Got: {flux_str}"
    )
