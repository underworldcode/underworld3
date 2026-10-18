"""The yield-stress floor is scaled like the yield stress in a units model (#758).

``ViscoPlasticFlowModel`` floors the yield stress, :math:`\\tau_y \\ge \\tau_{y,min}`,
with ``uw.maths.smooth_max(tau_y, tau_y_min, 0)``, i.e.
:math:`\\tfrac12(a + b + |a - b|)`. The floor can be written as a plain
non-dimensional number, as a ``uw.quantity`` in any stress unit, or as a product of a
dimensional expression and a number. All are the same physical stress, so all must
give the same effective viscosity: the exact :math:`\\max(\\tau_y, \\tau_{y,min})`.

Before the fix, ``UWexpression + UWexpression`` added the operands' dimensional
magnitudes eagerly (``self.value + other.value``) whenever the left operand was not
a bare quantity, so in :math:`a + b` the floor arrived as 0.35 (MPa) or 3.5e5 (Pa)
in a non-dimensional sum while :math:`|a - b|` scaled it correctly. A floor 83 times
below the yield stress raised it (quantity) or stopped yielding entirely (product).

Baseline: pure shear with a linear velocity field, which the degree-2 velocity
represents exactly, so :math:`\\dot\\varepsilon_{II} = R` and the yielded viscosity is
:math:`\\eta = \\max(\\tau_y, \\tau_{y,min}) / (2R)` in closed form.
"""

import numpy as np
import pytest

import underworld3 as uw
from underworld3.function.expressions import unwrap_expression

pytestmark = [pytest.mark.level_2]

H = 10.4e3                                  # m
V = 44.496e-3 / (365.25 * 86400)            # m/s
ETA = 5.8333e19                             # Pa s
STRESS_SCALE = ETA * V / H                  # Pa

FLOOR_PA = 0.35e6

# every spelling of the same 0.35 MPa floor; C0 is the cohesion expression
FLOOR_SPELLINGS = {
    "nondimensional_float": lambda C0, tau_pa: FLOOR_PA / STRESS_SCALE,
    "quantity_Pa": lambda C0, tau_pa: uw.quantity(FLOOR_PA, "Pa"),
    "quantity_MPa": lambda C0, tau_pa: uw.quantity(FLOOR_PA / 1.0e6, "MPa"),
    "product_with_expression": lambda C0, tau_pa: (FLOOR_PA / tau_pa) * C0,
}


def _pure_shear_viscosity(tau_pa, floor_spelling):
    """Effective viscosity, in model units (eta / eta_0), of a ViscoPlasticFlowModel
    in pure shear at strain rate R = 29.1 MPa / eta_0, the yield stress being the
    cohesion expression times a unit field. Returns the evaluate() value, the
    compiled (projected) nodal values, and R."""
    uw.reset_default_model()
    uw.get_default_model().set_reference_quantities(
        length=uw.quantity(H, "m"),
        velocity=uw.quantity(V, "m/s"),
        viscosity=uw.quantity(ETA, "Pa*s"),
    )
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4), minCoords=(0.0, 0.0),
        maxCoords=(uw.quantity(H, "m"), uw.quantity(H, "m")),
    )
    v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2, units="m/s")
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, units="Pa")
    seed = uw.discretisation.MeshVariable("seed", mesh, 1, degree=1)
    seed.array[...] = 1.0

    rate = 29.1e6 / ETA                     # 1/s
    x = np.asarray(v.coords.to("m"))
    v.array[...] = uw.utilities.UnitAwareArray(
        (rate * np.stack([x[:, 0], -x[:, 1]], axis=-1)).reshape(-1, 1, 2), units="m/s")

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    constitutive_model = stokes.constitutive_model
    constitutive_model.Parameters.shear_viscosity_0 = uw.quantity(ETA, "Pa*s")
    C0 = uw.expression(r"C_0", uw.quantity(tau_pa, "Pa"), "cohesion")
    constitutive_model.Parameters.yield_stress = C0 * seed.sym[0]
    constitutive_model.yield_mode = "min"
    if floor_spelling is not None:
        constitutive_model.Parameters.yield_stress_min = (
            FLOOR_SPELLINGS[floor_spelling](C0, tau_pa))

    viscosity = constitutive_model.viscosity
    centre = np.array([[0.5, 0.5]])
    evaluated = float(np.asarray(uw.function.evaluate(viscosity, centre)).ravel()[0])

    eta_field = uw.discretisation.MeshVariable("eta", mesh, 1, degree=1)
    projection = uw.systems.Projection(mesh, eta_field)
    projection.uw_function = viscosity
    projection.smoothing = 0.0
    projection.tolerance = 1.0e-12
    projection.solve()
    compiled = np.asarray(eta_field.array).ravel()
    return evaluated, compiled, rate


@pytest.mark.tier_b
@pytest.mark.parametrize("floor_spelling", [None, *FLOOR_SPELLINGS])
def test_floor_far_below_the_yield_stress_is_inert(floor_spelling):
    """tau_y = 29.1 MPa, floor 0.35 MPa: eta = tau_y / (2R) = eta_0 / 2 exactly."""
    tau_pa = 29.1e6
    evaluated, compiled, rate = _pure_shear_viscosity(tau_pa, floor_spelling)
    expected = tau_pa / (2.0 * rate) / ETA
    assert evaluated == pytest.approx(expected, rel=1.0e-10)
    assert compiled == pytest.approx(np.full_like(compiled, expected), rel=1.0e-8)


@pytest.mark.tier_b
@pytest.mark.parametrize("floor_spelling", list(FLOOR_SPELLINGS))
def test_floor_above_the_yield_stress_is_honoured(floor_spelling):
    """tau_y = 0.2 MPa, floor 0.35 MPa: eta = floor / (2R), for every spelling."""
    tau_pa = 0.2e6
    evaluated, compiled, rate = _pure_shear_viscosity(tau_pa, floor_spelling)
    expected = FLOOR_PA / (2.0 * rate) / ETA
    assert evaluated == pytest.approx(expected, rel=1.0e-10)
    assert compiled == pytest.approx(np.full_like(compiled, expected), rel=1.0e-8)


@pytest.mark.level_1
@pytest.mark.tier_b
def test_sum_of_field_expression_and_quantity_expression_scales_each_operand():
    """The root cause, directly: a sum of a field expression and a dimensional
    constant (expression or bare quantity) non-dimensionalises each term."""
    uw.reset_default_model()
    uw.get_default_model().set_reference_quantities(
        length=uw.quantity(H, "m"),
        velocity=uw.quantity(V, "m/s"),
        viscosity=uw.quantity(ETA, "Pa*s"),
    )
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(2, 2), minCoords=(0.0, 0.0),
        maxCoords=(uw.quantity(H, "m"), uw.quantity(H, "m")),
    )
    seed = uw.discretisation.MeshVariable("seed_sum", mesh, 1, degree=1)
    C0 = uw.expression(r"C_0", uw.quantity(29.1, "MPa"), "cohesion")
    tau = uw.expression(r"\tau", C0 * seed.sym[0], "yield stress")
    floor = uw.expression(r"f", uw.quantity(0.35, "MPa"), "floor")
    for total in (tau + floor, floor + tau, tau + uw.quantity(0.35, "MPa"),
                  tau - (-1 * uw.quantity(0.35, "MPa")),
                  -(uw.quantity(-0.35, "MPa") - tau)):
        nondimensional = unwrap_expression(total, mode="nondimensional")
        at_unit_seed = float(nondimensional.subs(seed.sym[0], 1))
        assert at_unit_seed == pytest.approx(
            (29.1e6 + 0.35e6) / STRESS_SCALE, rel=1.0e-12)
