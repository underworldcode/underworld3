"""Poisson solver with unit-aware boundary conditions (Notebook 13 workflow).

Solve del^2 T = 0 on a box with quantity-valued Dirichlet BCs and check the
solution: constant gradient dT/dy = Delta_T / L_y and the BC values honoured
at the boundaries.

Coordinate note: ``uw.function.evaluate()`` takes coordinates as plain
arrays in model (non-dimensional) units. Quantity-valued coordinate lists
(``[(x_qty, y_qty)]``) are NOT supported — that proposed extension was
declared unsupported (LE-08 / BF-12, units-family ruling D7, 2026-07-06).
Physical locations are converted here with ``uw.scaling.non_dimensionalise``,
and expectations are stated in the same non-dimensional frame so the test is
independent of the model's internal scale choices.
"""

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _nd(qty):
    """Model-frame (non-dimensional) value of a quantity."""
    return float(uw.scaling.non_dimensionalise(qty))


def _setup_and_solve(L_x, L_y, T_bottom, T_top):
    """Build the mesh, solve the Poisson problem with quantity BCs, and
    return (T, gradT) after a gradient projection."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(10, 10), minCoords=(0.0, 0.0), maxCoords=(L_x, L_y)
    )

    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2, units="kelvin")
    # Create the gradient variable BEFORE solving (creating it after
    # corrupts the DM state — the "Batman" regression, test_0813).
    gradT = uw.discretisation.MeshVariable("gradT", mesh, mesh.dim, degree=1)

    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1
    poisson.f = 0.0

    # Quantity-valued BCs — the point of this test
    poisson.add_dirichlet_bc(T_bottom, "Bottom")
    poisson.add_dirichlet_bc(T_top, "Top")

    poisson.solve()
    assert poisson.snes.getConvergedReason() > 0, "Solver did not converge"

    gradient_proj = uw.systems.Vector_Projection(mesh, gradT)
    gradient_proj.uw_function = mesh.vector.gradient(T.sym)
    gradient_proj.solve()

    return T, gradT


def _check_linear_solution(L_x, L_y, T_bottom, T_top):
    """Common assertions, all in the model (non-dimensional) frame."""
    uw.reset_default_model()
    model = uw.get_default_model()
    model.set_reference_quantities(
        domain_depth=uw.quantity(500, "m"),  # Matches L_y
        material_density=uw.quantity(3300, "kg/m**3"),
    )

    T, gradT = _setup_and_solve(L_x, L_y, T_bottom, T_top)

    Lx_nd, Ly_nd = _nd(L_x), _nd(L_y)
    Tb_nd, Tt_nd = _nd(T_bottom), _nd(T_top)

    # BC values honoured at the boundaries. Evaluate a whisker inside the
    # domain — point-location exactly ON the boundary is unreliable.
    eps = 1e-6 * Ly_nd
    T_at_bottom = uw.function.evaluate(
        T.sym, np.array([[Lx_nd / 2, eps]], dtype=np.float64)
    )
    T_at_top = uw.function.evaluate(
        T.sym, np.array([[Lx_nd / 2, Ly_nd - eps]], dtype=np.float64)
    )
    span = abs(Tt_nd - Tb_nd)
    assert abs(float(T_at_bottom[0, 0, 0]) - Tb_nd) < 1e-3 * span, (
        f"Bottom BC not applied: {float(T_at_bottom[0, 0, 0])} != {Tb_nd}"
    )
    assert abs(float(T_at_top[0, 0, 0]) - Tt_nd) < 1e-3 * span, (
        f"Top BC not applied: {float(T_at_top[0, 0, 0])} != {Tt_nd}"
    )

    # Constant gradient dT/dy = Delta_T / L_y (model frame); dT/dx = 0
    expected_gradient = (Tt_nd - Tb_nd) / Ly_nd
    grad = uw.function.evaluate(
        gradT.sym, np.array([[Lx_nd / 2, Ly_nd / 2]], dtype=np.float64)
    )
    dT_dx = float(grad[0, 0, 0])
    dT_dy = float(grad[0, 0, 1])

    assert abs(dT_dx) < 1e-3 * abs(expected_gradient), (
        f"dT/dx should be ~0, got {dT_dx}"
    )
    assert abs(dT_dy - expected_gradient) < 1e-3 * abs(expected_gradient), (
        f"dT/dy should be {expected_gradient}, got {dT_dy}"
    )


def test_poisson_linear_gradient_with_pint_quantities():
    """Pint Quantity BCs (``value * uw.units(...)``) produce the correct
    linear solution."""
    _check_linear_solution(
        L_x=1000 * uw.units("m"),
        L_y=500 * uw.units("m"),
        T_bottom=300 * uw.units("K"),
        T_top=1600 * uw.units("K"),
    )


def test_poisson_linear_gradient_with_uwquantity():
    """Same problem with ``uw.quantity()`` BCs."""
    _check_linear_solution(
        L_x=uw.quantity(1000, "m"),
        L_y=uw.quantity(500, "m"),
        T_bottom=uw.quantity(300, "K"),
        T_top=uw.quantity(1600, "K"),
    )
