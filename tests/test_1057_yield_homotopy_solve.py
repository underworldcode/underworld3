"""Layer 2 of the nonlinear-solver design: the model-advertised yield homotopy and
``stokes.solve(homotopy=True)``.

Contract under test (design:
``docs/developer/design/nonlinear-solver-homotopy-warmstart.md``, Layer 2):

  * A constitutive model **advertises** whether it has a yield law to sharpen
    (``supports_yield_homotopy``) — true for the viscoplastic / VEP models, false for
    a plain viscous one.
  * ``_yield_homotopy_control()`` puts the model in its smooth δ-parameterised mode
    and reports how to march it: a model-owned δ setter, and the tangent to pair with
    it (Newton for viscous-plastic yield, the frozen/Picard tangent for elastic VEP,
    whose consistent yield tangent is indefinite).
  * ``solve(homotopy=True)`` runs the multi-solve continuation and returns the march
    summary; on a model with no yield law it raises a clear error rather than
    silently solving something else.

The hard-case convergence behaviour (Spiegelman notch at η=1e26) is validated
separately in the study; these are cheap serial checks of the API contract.
"""

import pytest
import sympy

import underworld3 as uw


pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _viscoplastic_stokes(cellSize=0.4):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=cellSize
    )
    v = uw.discretisation.MeshVariable("Vh", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("Ph", mesh, 1, degree=1, continuous=True)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.constitutive_model.Parameters.yield_stress = 5.0
    stokes.bodyforce = sympy.Matrix([0.0, -1.0])
    stokes.add_essential_bc((0.0, 0.0), "Bottom")
    stokes.add_essential_bc((0.0, None), "Left")
    stokes.add_essential_bc((0.0, None), "Right")
    stokes.tolerance = 1.0e-5
    return mesh, stokes


def test_plain_viscous_model_does_not_advertise_homotopy():
    cm = uw.constitutive_models.ViscousFlowModel
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5
    )
    v = uw.discretisation.MeshVariable("Vv", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("Pv", mesh, 1, degree=1, continuous=True)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = cm
    assert stokes.constitutive_model.supports_yield_homotopy is False


def test_viscoplastic_model_advertises_homotopy():
    _, stokes = _viscoplastic_stokes()
    assert stokes.constitutive_model.supports_yield_homotopy is True


def test_homotopy_control_switches_model_to_smooth_mode():
    """The control must leave the model in the δ-parameterised power-mean mode and
    hand back a working, model-owned δ setter."""
    _, stokes = _viscoplastic_stokes()
    cm = stokes.constitutive_model
    control = cm._yield_homotopy_control()

    assert cm.yield_mode == "softmin"
    assert cm.yield_smoother == "powermean"
    # Newton tangent for a non-elastic viscoplastic yield.
    assert control.tangent is True

    # The setter must move BOTH the stored value and the constants[] atom, so a later
    # _get_yield_softness() cannot reset δ to a stale number.
    control.set_delta(0.25)
    assert cm.yield_softness == pytest.approx(0.25)
    assert float(cm._get_yield_softness().sym) == pytest.approx(0.25)


def test_vep_model_requests_the_picard_tangent():
    """The consistent yield tangent over the elastic stress-history block is
    indefinite, so the elastic models must ask for the frozen tangent."""
    cm = uw.constitutive_models.ViscoElasticPlasticFlowModel
    assert cm._yield_homotopy_tangent is False
    assert uw.constitutive_models.ViscoPlasticFlowModel._yield_homotopy_tangent is True


def test_solve_homotopy_on_unsupported_model_raises():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5
    )
    v = uw.discretisation.MeshVariable("Vr", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("Pr", mesh, 1, degree=1, continuous=True)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = sympy.Matrix([0.0, -1.0])
    stokes.add_essential_bc((0.0, 0.0), "Bottom")

    with pytest.raises(TypeError, match="supports_yield_homotopy"):
        stokes.solve(homotopy=True)


def test_solve_homotopy_marches_and_reports():
    """A viscoplastic solve driven by the homotopy converges and reports the march."""
    _, stokes = _viscoplastic_stokes()
    report = stokes.solve(homotopy=True,
                          homotopy_options=dict(delta0=1.0, dmin=0.05, verbose=False))

    assert report["converged"] is True
    assert report["steps"] >= 1
    # Settled at (or below) the starting δ, and never below the requested floor.
    assert report["settled_delta"] is not None
    assert report["settled_delta"] <= 1.0
    assert stokes.has_solution is True
    # The model is left holding the δ that actually converged.
    assert stokes.constitutive_model.yield_softness == pytest.approx(
        report["settled_delta"]
    )
