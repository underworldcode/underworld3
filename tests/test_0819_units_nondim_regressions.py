"""Regressions at the units <-> non-dimensional boundary.

Covers two crashes found with an active scaling model:

- issue #328: ``uw.non_dimensionalise(pint.Quantity)`` raised ``TypeError``
  (the protocol-5 branch constructed UWQuantity with an invalid
  ``dimensionality=`` keyword whenever a scale was resolvable);
- issue #271: ``set_pressure_gauge`` computed the surface-mean pressure as a
  dimensional quantity but applied it to the non-dimensional ``p.data``
  inside the SNES update callback, crashing the solve
  (``_UFuncOutputCastingError``).
"""

import numpy as np
import pytest
import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


@pytest.fixture()
def scaling_model():
    """Active reference scaling covering length, time and mass."""
    uw.reset_default_model()
    orchestration_model = uw.get_default_model()
    orchestration_model.set_reference_quantities(
        length=uw.quantity(500, "m"),
        velocity=uw.quantity(1, "cm/year"),
        viscosity=uw.quantity(1e21, "Pa*s"),
    )
    yield orchestration_model
    uw.reset_default_model()


def test_non_dimensionalise_plain_pint_quantity(scaling_model):
    # issue #328: a raw pint quantity with a resolvable scale must
    # non-dimensionalise to a plain float, not raise TypeError.
    result = uw.non_dimensionalise(250 * uw.units("m"))
    assert isinstance(result, float)
    assert result == pytest.approx(0.5)  # length scale is 500 m


def test_non_dimensionalise_pint_quantity_non_si_units(scaling_model):
    # The scale is computed in SI, so a non-SI input must reduce through SI:
    # 0.25 km == 250 m == 0.5 length-scales.
    assert uw.non_dimensionalise(0.25 * uw.units("km")) == pytest.approx(0.5)


def test_pressure_gauge_solves_under_units_model(scaling_model):
    # issue #271: the gauge callback fires inside the SNES non-dimensional
    # cordon; the dimensional surface-mean must be non-dimensionalised
    # before it is subtracted from p.data. Before the fix this solve
    # crashed with an object-dtype ufunc casting error.
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(8, 8), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        units="metre", qdegree=3)
    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2, units="cm/year")
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, units="Pa")

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = uw.quantity(1e21, "Pa*s")
    stokes.add_dirichlet_bc((uw.quantity(1.0, "cm/year"), 0.0), "Top")
    stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
    stokes.add_dirichlet_bc((0.0, 0.0), "Left")
    stokes.add_dirichlet_bc((0.0, 0.0), "Right")
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-8

    stokes.set_pressure_gauge("Top")
    stokes.solve()
    assert stokes.snes.getConvergedReason() > 0

    # The gauge held: the non-dimensional mean pressure on Top is zero.
    area = uw.maths.BdIntegral(mesh, 1.0, "Top").evaluate()
    mean_top = uw.maths.BdIntegral(mesh, p.sym[0, 0], "Top").evaluate() / area
    if isinstance(mean_top, uw.function.quantities.UWQuantity):
        mean_top = mean_top.data
    assert np.isclose(float(mean_top), 0.0, atol=1.0e-6), f"top-mean pressure = {mean_top}"
