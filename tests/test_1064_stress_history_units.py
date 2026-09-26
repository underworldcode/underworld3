"""A stress history in a units-aware model is the non-dimensional history, read in Pa.

The Maxwell shear box twice: once with reference quantities (1 km, 1e21 Pa s,
1 Myr) and every input given with units -- the timestep in kyr, deliberately not
the reference time unit -- and once as the same problem in plain numbers
(eta = G = 1, dt = 0.1, speed 0.5). Both start from the steady shear profile
already in place, so each history's first level is written from the flux of a
moving flow. The history's stores are non-dimensional work arrays: they must
agree between the two runs to solver precision, at order 1 and at order 2 (where
the order-2 weights depend on the ratio of the current step to the previous
one). Read back through the variable, the stress is in Pa: its non-dimensional
value times the stress scale, 1e21 Pa s / 1 Myr (#788).
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

MYR_S = 3.15576e13
STRESS_SCALE_PA = 1.0e21 / MYR_S
STEPS = 12
ORIGIN = np.array([[0.0, 0.0]])
CASES = [(t, 1) for t in ("semi_lagrangian", "integration_point", "forward", "lagrangian", "eulerian")] \
    + [(t, 2) for t in ("semi_lagrangian", "integration_point", "eulerian")]


def _shear_box(transport, order, with_units):
    uw.reset_default_model()
    if with_units:
        uw.get_default_model().set_reference_quantities(
            length=uw.quantity(1.0, "km"), viscosity=uw.quantity(1.0e21, "Pa*s"),
            time=uw.quantity(1.0, "Myr"))
    mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 8), minCoords=(-1.0, -0.5), maxCoords=(1.0, 0.5))
    tag = f"{transport[:3]}{order}{'u' if with_units else 'n'}"
    v = uw.discretisation.MeshVariable(f"U_{tag}", mesh, 2, degree=2, units="km/Myr" if with_units else None)
    p = uw.discretisation.MeshVariable(f"P_{tag}", mesh, 1, degree=1, units="Pa" if with_units else None)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.stress_transport = transport
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(stokes.Unknowns, order=order)
    parameters = stokes.constitutive_model.Parameters
    if with_units:
        parameters.shear_viscosity_0 = uw.quantity(1.0e21, "Pa*s")
        parameters.shear_modulus = uw.quantity(STRESS_SCALE_PA, "Pa")
        dt = uw.quantity(100.0, "kyr")
        speed = uw.quantity(0.5, "km/Myr")
    else:
        parameters.shear_viscosity_0 = 1.0
        parameters.shear_modulus = 1.0
        dt, speed = 0.1, 0.5
    parameters.dt_elastic = dt
    stokes.add_dirichlet_bc((speed, 0.0), "Top")
    stokes.add_dirichlet_bc((-speed, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    stokes.tolerance = 1.0e-8
    stokes.petsc_options["snes_type"] = "newtonls"
    stokes.petsc_options["ksp_type"] = "fgmres"
    # The steady shear profile already in place, written non-dimensionally (the
    # coordinates come back in metres in a units model; 1 km/Myr is the velocity
    # scale, so the non-dimensional values are the same in both runs).
    X = np.asarray(uw.non_dimensionalise(v.coords) if with_units else v.coords)
    v.data[:, 0], v.data[:, 1] = X[:, 1], 0.0
    for _ in range(STEPS):
        stokes.solve(timestep=dt, zero_init_guess=False)
    return stokes


@pytest.mark.parametrize("transport, order", CASES)
def test_units_model_history_is_the_nondimensional_history(transport, order):
    plain = _shear_box(transport, order, with_units=False)
    plain_store = np.array(plain.DFDt.psi_star[0].data)
    plain_xy = float(np.asarray(uw.function.evaluate(plain.DFDt.psi_star[0].sym[0, 1], ORIGIN)).reshape(-1)[0])

    stokes = _shear_box(transport, order, with_units=True)
    store = np.asarray(stokes.DFDt.psi_star[0].data)
    assert np.abs(store - plain_store).max() < 1.0e-6 * np.abs(plain_store).max(), (transport, order)

    carried = uw.function.evaluate(stokes.DFDt.psi_star[0].sym[0, 1], ORIGIN)
    value_pa = float(uw.units.Quantity(float(np.asarray(carried).reshape(-1)[0]), carried.units).to("Pa").magnitude)
    expected_pa = plain_xy * STRESS_SCALE_PA
    assert abs(value_pa - expected_pa) < 1.0e-6 * abs(expected_pa), (transport, order, value_pa, expected_pa)
