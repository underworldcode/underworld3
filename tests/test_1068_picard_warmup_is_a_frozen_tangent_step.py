"""A Picard step is a Newton iteration with the FROZEN tangent — not a residual sweep (#791).

The Layer 1 design (docs/developer/design/nonlinear-solver-homotopy-warmstart.md) specifies the
cold-start warm-up and ``solve(picard=N)`` as Picard (frozen-coefficient) iterations: linear
Stokes solves — velocity block and Schur complement — with the viscosity held at the current
state. The standard path instead ran SNES ``nrichardson`` with no nonlinear preconditioner,
``x <- x - lambda F(x)``: a residual step with no linear solve, nearly inert, and not a Picard
step. The rotated free-slip path already had the right semantics; these tests hold the standard
path to the same contract, per tangent mode:

  * ``consistent_jacobian=False`` — every iteration uses the frozen tangent, so ``picard`` is
    satisfied by the solve itself and must change NOTHING (bit-identical result);
  * ``"continuation"`` — ``picard=N`` guarantees at least N frozen-tangent iterations;
  * ``True`` — the pure-Newton compile has no frozen tangent: ``picard>0`` raises on a nonlinear
    residual, and is a no-op on a linear one.

It also pins where the removed automatic warm-up is genuinely unnecessary: when the rest state is
exactly zero (homogeneous BCs, no stress history) the consistent and frozen tangents coincide, so
Newton's first iteration IS the Picard step. That does NOT extend to boundary-driven problems.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _yielding_box(tag, tangent, tau_y=0.30, cellSize=0.25):
    """Sheared hard enough to yield. The body force MUST vary in x: a uniform one is
    hydrostatic, nothing moves and the yield law never engages."""
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=cellSize)
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("Vpw" + tag, mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("Ppw" + tag, mesh, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    cm = s.constitutive_model
    cm.Parameters.shear_viscosity_0 = 1.0
    cm.Parameters.yield_stress = tau_y
    s.bodyforce = sympy.Matrix([[0.0, -2.0 * sympy.cos(sympy.pi * x)]])
    s.add_essential_bc((sympy.oo, 0.0), "Top")
    s.add_essential_bc((sympy.oo, 0.0), "Bottom")
    s.add_essential_bc((0.0, sympy.oo), "Left")
    s.add_essential_bc((0.0, sympy.oo), "Right")
    s.petsc_use_pressure_nullspace = True
    s.petsc_options.delValue("ksp_monitor")
    s.consistent_jacobian = tangent
    s.tolerance = 1.0e-6
    # The frozen tangent converges LINEARLY: at 1e-8 it exhausted the default 50-iteration
    # cap on this fixture. Give it room, so a converged baseline exists to compare against.
    s.petsc_options["snes_max_it"] = 300
    return s, v, p


def test_frozen_tangent_picard_changes_nothing():
    """Under consistent_jacobian=False the whole solve already takes Picard steps, so
    picard=3 must reproduce picard=0 exactly — same iterations, same answer. The old
    nrichardson sweeps moved the starting state and so changed the path."""
    s0, v0, _ = _yielding_box("f0", False)
    s0.solve(zero_init_guess=True, picard=0)
    r0 = s0.solve_report
    ref = np.array(v0.data, copy=True)

    s3, v3, _ = _yielding_box("f3", False)
    s3.solve(zero_init_guess=True, picard=3)
    r3 = s3.solve_report

    assert str(r0.reason_str).startswith("CONVERGED"), r0.reason_str
    assert r3.nl_its == r0.nl_its, (
        f"picard=3 took {r3.nl_its} Newton iterations vs {r0.nl_its} for picard=0 under the "
        "frozen tangent — something other than frozen-tangent iterations ran first "
        "(the #791 nrichardson sweep)")
    # tight tolerance rather than bit-equality: the two solvers carry different variable
    # names, and bit-identity across builds would depend on JIT term ordering (#752)
    assert np.linalg.norm(np.array(v3.data) - ref) <= 1.0e-12 * np.linalg.norm(ref), (
        "picard=3 changed the frozen-tangent solution: the warm-up is not a no-op")


def test_continuation_picard_gives_max_of_n_and_the_natural_stage():
    """picard=N makes the alpha=0 (frozen-tangent) stage run max(N, natural) iterations,
    both measured against the ORIGINAL residual — the rotated path's rule. Hard baselines:

      * N below the natural count -> exactly the natural count (picard adds nothing);
      * N above it                -> exactly N.

    Guards the restart trap: PETSc's rtol is relative to EACH snes.solve call's own
    starting residual, so re-running a relative stage 1 after the N-iteration block
    asked for a further reduction and gave N + (another whole stage 1)."""
    s0, v0, _ = _yielding_box("c0", "continuation")
    s0.solve(zero_init_guess=True)
    natural = s0._continuation_stages["frozen_iterations"]
    ref = np.array(v0.data, copy=True)
    assert str(s0.solve_report.reason_str).startswith("CONVERGED")
    assert natural >= 3, f"fixture too easy to discriminate: natural stage = {natural}"

    low = max(1, natural // 3)
    s1, v1, _ = _yielding_box("cl", "continuation")
    s1.solve(zero_init_guess=True, picard=low)
    assert s1._continuation_stages["frozen_iterations"] == natural, (
        f"picard={low} (< natural {natural}) gave "
        f"{s1._continuation_stages['frozen_iterations']} frozen iterations; expected "
        f"exactly {natural}. More means stage 1 restarted its relative clock.")

    high = natural + 5
    s2, v2, _ = _yielding_box("ch", "continuation")
    s2.solve(zero_init_guess=True, picard=high)
    assert s2._continuation_stages["frozen_iterations"] == high, (
        f"picard={high} (> natural {natural}) gave "
        f"{s2._continuation_stages['frozen_iterations']} frozen iterations; expected "
        f"exactly {high}.")
    for s, v in ((s1, v1), (s2, v2)):
        assert str(s.solve_report.reason_str).startswith("CONVERGED"), s.solve_report.reason_str
        assert np.linalg.norm(np.array(v.data) - ref) / np.linalg.norm(ref) < 1.0e-5


def test_continuation_stages_do_not_go_stale():
    """A non-continuation solve after a continuation one must not report the old stages."""
    s, _, _ = _yielding_box("cs", "continuation")
    s.solve(zero_init_guess=True)
    assert s._continuation_stages is not None
    s.consistent_jacobian = False
    s.solve(zero_init_guess=True)
    assert s._continuation_stages is None


def test_pure_newton_picard_raises_on_a_nonlinear_residual():
    """The pure-Newton compile carries no frozen tangent; silently running something else
    (as the nrichardson sweep did) is replaced by a clear refusal, matching the rotated path."""
    s, _, _ = _yielding_box("n", True)
    with pytest.raises(NotImplementedError, match="continuation"):
        s.solve(zero_init_guess=True, picard=2)


def test_pure_newton_picard_is_ignored_on_a_linear_residual():
    """On a linear residual the frozen tangent IS the tangent, so picard is meaningless:
    it must neither raise nor change anything (same iterations as picard=0)."""
    runs = {}
    for n in (0, 2):
        s, v, _ = _yielding_box("l%d" % n, True, tau_y=1.0e6)     # never yields
        s.solve(zero_init_guess=True, picard=n)
        assert str(s.solve_report.reason_str).startswith("CONVERGED")
        runs[n] = (s.solve_report.nl_its, np.array(v.data, copy=True))
    assert runs[2][0] == runs[0][0]
    assert np.linalg.norm(runs[2][1] - runs[0][1]) <= 1.0e-12 * np.linalg.norm(runs[0][1])


def test_newton_first_step_from_rest_is_the_picard_step():
    """The claim that replaced the automatic warm-up. From a state of rest the strain rate is
    zero, the yield branch is inactive, and the consistent tangent coincides with the frozen
    one — so ONE Newton iteration from rest equals ONE Picard iteration.

    ⚠️ SCOPE: this holds when the rest state is exactly zero — this fixture is body-force
    driven with homogeneous essential BCs. It does NOT hold for a boundary-driven problem:
    the Dirichlet values yield the driven layer at once (measured: first steps 45% apart on
    a sheared box), nor for a stress-history model. There, a Picard entry needs
    consistent_jacobian="continuation" + picard=N."""
    states = {}
    for tangent in (False, True):
        s, v, _ = _yielding_box("s%s" % tangent, tangent)
        s.petsc_options["snes_max_it"] = 1
        s.solve(zero_init_guess=True)
        assert s.solve_report.nl_its == 1
        states[tangent] = np.array(v.data, copy=True)
    rel = (np.linalg.norm(states[True] - states[False])
           / np.linalg.norm(states[False]))
    assert rel < 1.0e-6, (
        f"first Newton step from rest differs from the Picard step by {rel:.3e} — the "
        "consistent tangent at rest is NOT the frozen one, and a real warm-up is needed")
