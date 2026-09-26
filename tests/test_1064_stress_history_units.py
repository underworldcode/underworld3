"""Every stress history carries a viscoelastic stress through a units-aware model.

The Maxwell shear box of test_1059, with reference quantities set: the
velocity is given in km/Myr, the viscosity in Pa s, the modulus in Pa, the
timestep in Myr. The history stores are non-dimensional work arrays behind
the units boundary; what enters them must be reduced on the way in, and
what a user reads back through ``.array`` or ``evaluate`` must come out in
Pa. The analytic loading curve
:math:`\\sigma_{xy} = \\eta\\dot{\\gamma}\\,(1 - e^{-t\\mu/\\eta})`
is the baseline, in Pa (#788).
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

VISCOSITY_PA_S = 1.0e21
RELAXATION_MYR = 1.0
SPEED_KM_MYR, HEIGHT_KM, WIDTH_KM = 0.5, 1.0, 2.0
MYR_S = 3.15576e13

TRANSPORTS = ["semi_lagrangian", "integration_point", "forward", "lagrangian", "eulerian"]


def _units_model():
    uw.reset_default_model()
    orchestration_model = uw.get_default_model()
    orchestration_model.set_reference_quantities(
        length=uw.quantity(1.0, "km"),
        viscosity=uw.quantity(VISCOSITY_PA_S, "Pa*s"),
        time=uw.quantity(1.0, "Myr"),
    )
    return orchestration_model


def _maxwell_shear_with_units(transport, steps=20, dt_myr=0.1):
    _units_model()
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(16, 8), minCoords=(-WIDTH_KM / 2, -HEIGHT_KM / 2),
        maxCoords=(WIDTH_KM / 2, HEIGHT_KM / 2))
    v = uw.discretisation.MeshVariable(f"U_{transport}", mesh, mesh.dim, degree=2, units="km/Myr")
    p = uw.discretisation.MeshVariable(f"P_{transport}", mesh, 1, degree=1, units="Pa")
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.stress_transport = transport
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=1)
    parameters = stokes.constitutive_model.Parameters
    parameters.shear_viscosity_0 = uw.quantity(VISCOSITY_PA_S, "Pa*s")
    parameters.shear_modulus = uw.quantity(VISCOSITY_PA_S / (RELAXATION_MYR * MYR_S), "Pa")
    dt = uw.quantity(dt_myr, "Myr")
    parameters.dt_elastic = dt
    stokes.add_dirichlet_bc((uw.quantity(SPEED_KM_MYR, "km/Myr"), 0.0), "Top")
    stokes.add_dirichlet_bc((uw.quantity(-SPEED_KM_MYR, "km/Myr"), 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    stokes.tolerance = 1.0e-6
    stokes.petsc_options["snes_type"] = "newtonls"
    stokes.petsc_options["ksp_type"] = "fgmres"
    for _ in range(steps):
        stokes.solve(timestep=dt, zero_init_guess=False)
    return stokes, dt


def _exact_pa(t_myr):
    shear_rate_per_s = 2.0 * SPEED_KM_MYR / HEIGHT_KM / MYR_S
    return VISCOSITY_PA_S * shear_rate_per_s * (1.0 - np.exp(-t_myr / RELAXATION_MYR))


def _carried_pa(stokes):
    """The history's first level at the origin, read through ``carried``, in Pa."""
    carried = uw.function.evaluate(stokes.DFDt.carried(0)[0, 1], np.array([[0.0, 0.0]]))
    quantity = uw.units.Quantity(float(np.asarray(carried).reshape(-1)[0]), carried.units)
    return float(quantity.to("Pa").magnitude)


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_stress_history_loads_in_pascals(transport):
    """The carried stress, read back through ``DFDt.carried``, is the Maxwell
    curve in Pa: the same 2% order-1 tolerance as the non-dimensional box
    (test_1059), on 2.74e7 Pa at t = 2 Myr. The store itself is a
    non-dimensional work array (sigma/G of order one here)."""
    steps, dt_myr = 20, 0.1
    stokes, dt = _maxwell_shear_with_units(transport, steps, dt_myr)
    exact_pa = _exact_pa(steps * dt_myr)
    assert abs(_carried_pa(stokes) - exact_pa) / exact_pa < 0.02, transport
    assert np.abs(np.asarray(stokes.DFDt.psi_star[0].data)).max() < 10.0
