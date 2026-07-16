"""
Test Poisson solver with unit-aware boundary conditions.

This test replicates the workflow from Notebook 13 to ensure unit-aware BCs
produce correct results, not just that they are accepted.

Coordinate forms (2026-07 audit, BF-12 / maintainer ruling D7): quantity-valued
coordinate LISTS — ``[(x_qty, y_qty)]`` — are an UNSUPPORTED input to
``uw.function.evaluate()`` and raise ``TypeError``. These tests use the
supported form: plain numpy arrays of model-unit (non-dimensional)
coordinates. ``test_quantity_coordinate_lists_are_rejected`` pins the
rejection contract.
"""

import pytest

import underworld3 as uw
import numpy as np

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _nd(quantity):
    """Model-unit (non-dimensional) float for a UWQuantity."""
    result = uw.non_dimensionalise(quantity)
    return float(getattr(result, "value", result))


def _set_reference_quantities():
    uw.reset_default_model()
    orchestration_model = uw.get_default_model()
    orchestration_model.set_reference_quantities(
        domain_depth=uw.quantity(500, "m"),  # Matches L_y
        material_density=uw.quantity(3300, "kg/m**3"),
    )


def _setup_and_solve(L_x, L_y, T_bottom, T_top, gradient_variable=True):
    """Build the Notebook-13 Poisson problem with unit-aware BCs and solve it.

    Returns (mesh, T, gradT). All four physical inputs are quantities
    (Pint or UWQuantity) — passing them through the BC path IS the test.
    """
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(10, 10), minCoords=(0.0, 0.0), maxCoords=(L_x, L_y), units="metre"
    )
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2, units="kelvin")
    gradT = (
        uw.discretisation.MeshVariable("gradT", mesh, mesh.dim, degree=1)
        if gradient_variable
        else None
    )

    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1
    poisson.f = 0.0
    poisson.add_dirichlet_bc(T_bottom, "Bottom")
    poisson.add_dirichlet_bc(T_top, "Top")
    poisson.solve()
    assert poisson.snes.getConvergedReason() > 0, "Solver did not converge"
    return mesh, T, gradT


def _project_gradient(mesh, T, gradT):
    gradient_proj = uw.systems.Vector_Projection(mesh, gradT)
    gradient_proj.uw_function = mesh.vector.gradient(T.sym)
    gradient_proj.solve()


def _check_linear_gradient(mesh, gradT, delta_T_kelvin, L_y_model):
    """Evaluate the projected gradient at the domain centre (supported
    coordinate form: plain model-unit array) and compare against the exact
    linear solution dT/dy = ΔT / L_y in the model frame.

    The temperature variable carries kelvin with no temperature reference
    scale, so model-frame temperature values ARE kelvin values and the
    expected model-frame gradient is ΔT[K] / L_y[model units].
    """
    coords = np.asarray(mesh._coords)
    centre = np.array([[0.5 * (coords[:, 0].min() + coords[:, 0].max()),
                        0.5 * (coords[:, 1].min() + coords[:, 1].max())]])
    grad = np.asarray(uw.function.evaluate(gradT.sym, centre))

    dT_dx = grad[0, 0, 0]
    dT_dy = grad[0, 0, 1]
    expected = delta_T_kelvin / L_y_model  # 1300 K over 1 model unit

    assert abs(dT_dx) < 1e-2 * abs(expected), f"dT/dx should be ~0, got {dT_dx}"
    assert abs(dT_dy - expected) < 1e-2 * abs(expected), (
        f"dT/dy should be {expected:.3f}, got {dT_dy:.3f}"
    )


def test_poisson_linear_gradient_with_pint_quantities():
    """Poisson with Pint Quantity BCs produces the exact linear gradient."""
    _set_reference_quantities()

    L_x = 1000 * uw.units("m")
    L_y = 500 * uw.units("m")
    T_bottom = 300 * uw.units("K")
    T_top = 1600 * uw.units("K")

    mesh, T, gradT = _setup_and_solve(L_x, L_y, T_bottom, T_top)
    _project_gradient(mesh, T, gradT)

    L_y_model = _nd(uw.quantity(500.0, "m"))
    _check_linear_gradient(mesh, gradT, (T_top - T_bottom).magnitude, L_y_model)


def test_poisson_linear_gradient_with_uwquantity():
    """Same problem with uw.quantity() BCs instead of Pint quantities."""
    _set_reference_quantities()

    L_x = uw.quantity(1000, "m")
    L_y = uw.quantity(500, "m")
    T_bottom = uw.quantity(300, "K")
    T_top = uw.quantity(1600, "K")

    mesh, T, gradT = _setup_and_solve(L_x, L_y, T_bottom, T_top)
    _project_gradient(mesh, T, gradT)

    L_y_model = _nd(L_y)
    _check_linear_gradient(mesh, gradT, (T_top - T_bottom).value, L_y_model)


def test_poisson_check_bc_values():
    """The unit-aware BC values must appear in the solution at the boundaries."""
    _set_reference_quantities()

    L_x = 1000 * uw.units("m")
    L_y = 500 * uw.units("m")
    T_bottom = 300 * uw.units("K")
    T_top = 1600 * uw.units("K")

    mesh, T, _ = _setup_and_solve(L_x, L_y, T_bottom, T_top, gradient_variable=False)

    coords = np.asarray(mesh._coords)
    x_mid = 0.5 * (coords[:, 0].min() + coords[:, 0].max())
    y_min = coords[:, 1].min()
    y_max = coords[:, 1].max()

    T_at_bottom = np.asarray(uw.function.evaluate(T.sym, np.array([[x_mid, y_min]])))
    # NOTE: a point exactly ON the top boundary is not claimed by
    # point-location (evaluate returns its 1e-18 sentinel), so probe one
    # part in 1e9 inside the domain.
    T_at_top = np.asarray(uw.function.evaluate(T.sym, np.array([[x_mid, y_max * (1 - 1e-9)]])))

    assert abs(T_at_bottom.ravel()[0] - 300.0) < 1.0, (
        f"Bottom BC not applied correctly: {T_at_bottom.ravel()[0]} != 300"
    )
    assert abs(T_at_top.ravel()[0] - 1600.0) < 1.0, (
        f"Top BC not applied correctly: {T_at_top.ravel()[0]} != 1600"
    )


def test_quantity_coordinate_lists_are_rejected():
    """Regression (BF-12): evaluate() must reject coordinate lists — plain or
    quantity-valued — with a TypeError that names the supported forms."""
    _set_reference_quantities()

    L_x = 1000 * uw.units("m")
    L_y = 500 * uw.units("m")

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(L_x, L_y)
    )
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1, units="kelvin")
    T.data[:, 0] = 1.0

    with pytest.raises(TypeError, match="not supported"):
        uw.function.evaluate(T.sym, [(L_x / 2, L_y / 2)])

    with pytest.raises(TypeError, match="not supported"):
        uw.function.evaluate(T.sym, [(uw.quantity(500, "m"), uw.quantity(250, "m"))])

    with pytest.raises(TypeError, match="not supported"):
        uw.function.evaluate(T.sym, [(1.0, 0.5)])
