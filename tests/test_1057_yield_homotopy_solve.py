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


@pytest.mark.parametrize(
    "mode,smoother",
    [("min", None), ("softmin", "sqrt"), ("softmin", "powermean")],
)
def test_cold_viscoplastic_solve_survives_zero_strain_rate(mode, smoother):
    """A COLD (v=0) viscoplastic solve must not produce NaN in ANY yield mode.

    Regression: at v=0 the strain-rate invariant is 0, so the plastic viscosity
    tau_y/(2 edot_II) is +inf. That is fine — the soft-min should carry it to the
    viscous branch. But the power-mean form computed its harmonic mean as
    eta_ve*eta_pl/(eta_ve+eta_pl), which is inf/inf = NaN, so a cold power-mean solve
    died with DIVERGED_FNORM_NAN while `min` and `sqrt` converged. Rewriting that
    mean as eta_ve/(1+f) — algebraically identical — keeps the infinite-eta_pl limit
    finite. The same singularity occurs at any rigid (unyielded) point, not only on a
    cold start.
    """
    _, stokes = _viscoplastic_stokes(cellSize=0.5)
    cm = stokes.constitutive_model
    cm.yield_mode = mode
    if smoother is not None:
        cm.yield_smoother = smoother

    stokes.solve()  # cold: zero_init_guess defaults True
    reason = int(stokes.snes.getConvergedReason())
    assert reason > 0, (
        f"cold viscoplastic solve failed for yield_mode={mode!r} "
        f"smoother={smoother!r} (reason={reason}; -4 is FNORM_NAN)"
    )


@pytest.mark.parametrize("zero_init_guess", [True, False])
def test_consistent_newton_never_assembles_at_zero_strain_rate(zero_init_guess):
    """A viscoplastic Jacobian is NaN at zero strain rate, so the warm/cold machinery
    must make that state unreachable — including when the caller asks for a WARM start
    on a solution that has never been written (adversarial review, M9).

    The residual survives v=0 (the soft-min carries the infinite plastic branch to the
    viscous one); the tangent does not. Only the interposed Picard step keeps the
    consistent-Newton path off it.
    """
    _, stokes = _viscoplastic_stokes(cellSize=0.5)
    stokes.consistent_jacobian = True
    assert stokes._solution_is_trivially_zero() is True   # never solved

    stokes.solve(zero_init_guess=zero_init_guess)
    reason = int(stokes.snes.getConvergedReason())
    assert reason > 0, (
        f"consistent-Newton solve from an all-zero field failed with reason={reason} "
        f"(-4 is FNORM_NAN) for zero_init_guess={zero_init_guess}"
    )


def test_yield_stress_is_floored_at_zero_by_default():
    """The yield stress is compared against the second invariant of the stress, so a
    negative tau_y is meaningless. `yield_stress_min` therefore defaults to 0 and the
    floor is active without the user asking.

    Regression: the default used to be the "unset" sentinel -oo (so sympy could cancel
    the term away), which left a pressure-dependent Drucker-Prager yield
    C + sin(phi)*p free to go negative in tension. Hard `Min` hid that (it discarded
    the resulting negative eta_pl); the power-mean the homotopy switches to raised a
    negative base to a non-integer power and produced NaN.
    """
    _, stokes = _viscoplastic_stokes()
    cm = stokes.constitutive_model
    assert cm.Parameters.yield_stress_min.sym == 0


@pytest.mark.parametrize("mode,smoother",
                         [("min", None), ("softmin", "powermean")])
def test_pressure_dependent_yield_survives_tension(mode, smoother):
    """A Drucker-Prager yield that goes NEGATIVE in tension must not poison the
    viscosity in any yield mode — the zero floor must catch it first."""
    import numpy as np

    mesh, stokes = _viscoplastic_stokes(cellSize=0.5)
    cm = stokes.constitutive_model
    x, y = mesh.X
    # C + sin(phi)*p with the pressure driven strongly negative (tension) over part
    # of the domain, so the raw yield stress changes sign inside the mesh.
    cm.Parameters.yield_stress = 1.0 + 0.5 * stokes.p.sym[0]
    cm.yield_mode = mode
    if smoother is not None:
        cm.yield_smoother = smoother

    eta = cm.viscosity.sym
    vals = uw.function.evaluate(eta, mesh.X.coords)
    assert np.all(np.isfinite(vals)), (
        f"viscosity is non-finite for yield_mode={mode!r} smoother={smoother!r} "
        f"where the pressure-dependent yield stress goes negative"
    )


def test_solve_homotopy_marches_and_reports():
    """A viscoplastic solve driven by the homotopy converges and reports the march."""
    _, stokes = _viscoplastic_stokes()
    report = stokes.solve(homotopy=True,
                          homotopy_options=dict(delta0=1.0, dmin=0.05, verbose=False))

    assert report["converged"] is True
    # The march must actually DESCEND, not just solve once at delta0 and stop.
    assert report["steps"] > 1, "the march did not take a second step"
    assert report["settled_delta"] is not None
    assert report["settled_delta"] < 1.0, (
        f"delta never descended below delta0 (settled at {report['settled_delta']})"
    )
    assert report["reached_dmin"] is True
    assert stokes.has_solution is True
    # The model is left holding the δ that actually converged.
    assert stokes.constitutive_model.yield_softness == pytest.approx(
        report["settled_delta"]
    )


def test_homotopy_restores_the_solver_tangent():
    """The march sets the tangent the model asks for, but must hand the solver back
    as it found it (adversarial review, M5)."""
    _, stokes = _viscoplastic_stokes()
    stokes.consistent_jacobian = False
    stokes.solve(homotopy=True, homotopy_options=dict(delta0=1.0, dmin=0.1))
    assert stokes.consistent_jacobian is False, (
        "solve(homotopy=True) left the user's tangent permanently changed"
    )


@pytest.mark.parametrize("kwargs", [dict(zero_init_guess=True), dict(picard=2)])
def test_homotopy_rejects_arguments_it_would_have_to_ignore(kwargs):
    """The march decides cold-vs-warm and its own warm-up per step, so an argument
    that contradicts it is refused rather than silently dropped (review, m7)."""
    _, stokes = _viscoplastic_stokes(cellSize=0.5)
    with pytest.raises(ValueError):
        stokes.solve(homotopy=True, **kwargs)


def test_homotopy_refuses_a_stress_history_solver():
    """A march is several solves; on a VEP solver each one would advance the elastic
    stress history by a full timestep (adversarial review, C3). Refuse loudly rather
    than silently integrating N steps for one requested dt."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5
    )
    v = uw.discretisation.MeshVariable("Vve", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("Pve", mesh, 1, degree=1, continuous=True)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel
    cm = stokes.constitutive_model
    cm.Parameters.shear_viscosity_0 = 1.0
    cm.Parameters.shear_modulus = 10.0
    cm.Parameters.dt_elastic = 0.1
    cm.Parameters.yield_stress = 5.0
    stokes.bodyforce = sympy.Matrix([0.0, -1.0])
    stokes.add_essential_bc((0.0, 0.0), "Bottom")

    with pytest.raises(NotImplementedError, match="stress history"):
        stokes.solve(homotopy=True, timestep=0.1)
