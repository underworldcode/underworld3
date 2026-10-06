"""A Picard step is a Newton iteration with the FROZEN tangent — not a residual sweep (#791).

The recipe for hard viscoplastic solves is "a few Picard steps, then Newton", and that must be one
flag: ``solve(picard=N)``. A Picard step is successive substitution, A(u_k) u_{k+1} = f — a linear
Stokes solve (velocity block and Schur complement) with the viscosity frozen at the current state.
The standard path used to run SNES ``nrichardson`` with no nonlinear preconditioner,
``x <- x - lambda F(x)``: a residual step with no linear solve, nearly inert, and not a Picard step.

The contract, per tangent mode:

  * ``consistent_jacobian=True`` — ``picard=N`` gives exactly N Picard steps, then Newton; a cold
    start takes ONE automatically (``picard=-1`` switches it off). Held against the definition:
    after ``picard=1`` with the Newton stage capped at zero iterations, the state must equal ONE
    frozen-tangent iteration — on a BOUNDARY-DRIVEN problem, where Newton's first step from rest is
    NOT a Picard step (measured 45% apart), so the test cannot pass by coincidence.
  * ``"continuation"`` — ONE solve with alpha keyed on the residual relative to F_ref = ||F(u=0)||;
    ``picard=N`` holds alpha = 0 for the first N iterations, so the frozen-tangent count is
    max(N, natural). A warm start enters the ramp at the alpha its residual calls for.
  * ``False`` — every iteration is already a Picard step, so ``picard`` changes nothing.
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
    # These fixtures pin the frozen-tangent semantics on the EXACT hard Min, the law whose
    # kink makes the warm-up necessary. Since 2026-10-05 the model default is the smooth
    # law (softmin, sqrt, anchor "onset", delta 0.1), under which pure Newton converges here.
    cm.yield_mode = "min"
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


def _driven_box(tag, tangent, tau_y=0.30):
    """Sheared by a moving top wall: boundary-driven, so the rest state is NOT zero — the
    Dirichlet values yield the driven layer at once, and the consistent tangent at rest
    differs from the frozen one."""
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25)
    v = uw.discretisation.MeshVariable("Vdb" + tag, mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("Pdb" + tag, mesh, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    s.constitutive_model.Parameters.yield_stress = tau_y
    s.constitutive_model.yield_mode = "min"      # exact hard Min, see _yielding_box
    s.add_essential_bc((1.0, 0.0), "Top")
    s.add_essential_bc((0.0, 0.0), "Bottom")
    s.petsc_use_pressure_nullspace = True
    s.petsc_options.delValue("ksp_monitor")
    s.consistent_jacobian = tangent
    s.tolerance = 1.0e-6
    s.petsc_options["snes_max_it"] = 300
    return s, v


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
    """picard=N holds alpha = 0 for the first N iterations of the ramp, so the frozen-tangent
    count is max(N, natural). Hard baselines:

      * N below the natural count -> exactly the natural count (picard adds nothing);
      * N above it                -> exactly N.

    And all three land on the same solution."""
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


def test_newton_warmup_step_is_exactly_a_picard_step():
    """THE DEFINITION. Under consistent_jacobian=True, picard=1 followed by a Newton stage
    capped at zero iterations must leave exactly the state of ONE frozen-tangent iteration.
    Boundary-driven, so an nrichardson sweep (old code) or no warm-up (first fix) both differ."""
    sN, vN = _driven_box("n", True)
    sN.petsc_options["snes_max_it"] = 0          # Newton stage does nothing
    sN.solve(zero_init_guess=True, picard=1)
    assert sN._picard_stages == dict(picard_iterations=1), sN._picard_stages

    sP, vP = _driven_box("p", False)
    sP.petsc_options["snes_max_it"] = 1          # one frozen-tangent iteration
    sP.solve(zero_init_guess=True)
    assert sP.solve_report.nl_its == 1

    a, b = np.array(vN.data), np.array(vP.data)
    rel = np.linalg.norm(a - b) / np.linalg.norm(b)
    assert rel < 1.0e-8, (
        f"the Newton-path Picard step differs from a frozen-tangent step by {rel:.3e}")


FNORM = ("CONVERGED_FNORM_ABS", "CONVERGED_FNORM_RELATIVE")


def test_newton_warmup_takes_n_steps_then_converges_to_the_same_answer():
    """picard=N gives exactly N Picard steps, then Newton converges on the RESIDUAL, to the
    same solution as a different warm-up count. ⚠️ Require FNORM convergence: without a
    warm-up, pure Newton on this boundary-driven box stops on step stagnation —
    CONVERGED_SNORM_RELATIVE at ||F|| = 4.5e-5 whatever the tolerance — which PETSc reports
    as "converged". That run is exactly what the warm-up exists to prevent, and it must not
    be used as a reference."""
    runs = {}
    for n in (1, 3):
        s, v = _driven_box("r%d" % n, True)
        s.solve(zero_init_guess=True, picard=n)
        assert s._picard_stages == dict(picard_iterations=n), s._picard_stages
        assert str(s.solve_report.reason_str) in FNORM, (
            f"picard={n}: {s.solve_report.reason_str} at ||F|| = {s.solve_report.fnorm:.2e}")
        runs[n] = np.array(v.data, copy=True)
    assert np.linalg.norm(runs[3] - runs[1]) / np.linalg.norm(runs[1]) < 1.0e-6


def test_pure_newton_without_warmup_stagnates_on_a_driven_box():
    """Why the automatic warm-up exists, pinned: switch it off (picard=-1) and pure Newton
    on the boundary-driven box stalls on step size instead of converging on the residual.
    If this ever starts converging on FNORM, the fixture no longer demonstrates the need
    and the warm-up tests above lose their contrast — re-tune tau_y."""
    s, _ = _driven_box("nw", True)
    s.solve(zero_init_guess=True, picard=-1)
    assert s._picard_stages is None
    assert str(s.solve_report.reason_str) not in FNORM, (
        f"pure Newton converged on the residual ({s.solve_report.reason_str}); the fixture "
        "no longer shows why the warm-up is needed")


def test_cold_newton_takes_one_automatic_picard_step_and_warm_takes_none():
    """Layer 1: a cold start under the consistent tangent takes ONE Picard step
    automatically; a warm re-solve takes none."""
    s, v = _driven_box("a", True)
    s.solve(zero_init_guess=True)
    assert s._picard_stages == dict(picard_iterations=1), s._picard_stages
    assert str(s.solve_report.reason_str).startswith("CONVERGED")
    s.solve(zero_init_guess=False)
    assert s._picard_stages is None, "a warm re-solve took a Picard warm-up"


def test_continuation_alpha_is_linear_in_log_residual():
    """The ramp itself: alpha = 0 at F = start * F_ref, 1 at F = newton_at * F_ref, linear in
    log F between, clipped outside."""
    s, _, _ = _yielding_box("ra", "continuation")
    a0, a1 = s.continuation_alpha_start, s.continuation_newton_at
    assert (a0, a1) == (1.0e-1, 5.0e-4)
    assert s._continuation_alpha(1.0, 1.0) == 0.0
    assert s._continuation_alpha(a0, 1.0) == 0.0
    assert abs(s._continuation_alpha(a1, 1.0) - 1.0) < 1.0e-12
    assert abs(s._continuation_alpha(np.sqrt(a0 * a1), 1.0) - 0.5) < 1.0e-12
    assert s._continuation_alpha(1.0e-9, 1.0) == 1.0
    assert s._continuation_alpha(0.0, 1.0) == 1.0


def test_continuation_warm_start_enters_the_ramp_at_its_residual():
    """#791: alpha is keyed on F_ref = ||F(u=0)||, not on the solve's own starting residual.
    Keyed on its own F0, a warm start re-ran the whole Picard phase (measured on the notch:
    763 vs 10 iterations). So a warm start near the solution must take NO frozen-tangent
    iteration, converge to the cold answer, and a converged state must take none at all."""
    s, v, _ = _yielding_box("wc", "continuation")
    s.solve(zero_init_guess=True)
    cold = s.solve_report
    ref = np.array(v.data, copy=True)
    assert str(cold.reason_str).startswith("CONVERGED"), cold.reason_str
    st = s._continuation_stages
    assert st["frozen_iterations"] >= 1 and st["final_alpha"] == 1.0, st
    F_ref = st["F_ref"]

    s.solve(zero_init_guess=False)                   # re-solve the converged state
    assert s.solve_report.nl_its == 0, s.solve_report.nl_its

    with uw.synchronised_array_update():
        v.data[...] = 1.02 * ref                     # 2% off the solution
    s.solve(zero_init_guess=False)
    st = s._continuation_stages
    assert st["F_ref"] == pytest.approx(F_ref, rel=1.0e-12), "F_ref must not depend on the start"
    assert st["frozen_iterations"] == 0, (
        f"a warm start 2% from the solution took {st['frozen_iterations']} frozen-tangent "
        "iterations: the ramp is keyed on the solve's own residual again")
    assert str(s.solve_report.reason_str).startswith("CONVERGED"), s.solve_report.reason_str
    assert s.solve_report.nl_its < cold.nl_its
    assert np.linalg.norm(np.array(v.data) - ref) / np.linalg.norm(ref) < 1.0e-5


def test_continuation_restores_line_search_and_user_update_hook():
    """The ramp borrows the SNES update hook and the line search for its own solve. Both must
    be handed back: user callbacks still fire every iteration inside the ramp, and the line
    search is the configured one again afterwards."""
    s, _, _ = _yielding_box("hk", "continuation")
    calls = []
    s.add_update_callback(lambda solver, it: calls.append(it))
    s.solve(zero_init_guess=True)
    n = s.solve_report.nl_its
    assert str(s.solve_report.reason_str).startswith("CONVERGED")
    inside = [c for c in calls if c >= 0]
    assert len(inside) == n, f"user callback fired {len(inside)} times in {n} iterations"
    # nothing configures the line search on this fixture, so it is PETSc's newtonls default
    assert s.snes.getLineSearch().getType() == "bt", (
        f"line search after the ramp is {s.snes.getLineSearch().getType()!r}: the "
        "continuation solve's l2 leaked out")


def test_solve_report_config_is_what_the_solve_ran_with():
    """#806: the report carries the configuration IN FORCE during the solve, snapshotted at
    the call. The continuation ramp switches the line search to l2 and the atol to
    tolerance * F_ref for its own solve and restores both after, so the report must say l2
    while the live SNES says bt again. And the snapshot must not change the solver: reading
    the nonlinear preconditioner with SNESGetNPC creates one (measured: DIVERGED_INNER)."""
    s, _, _ = _yielding_box("cf", "continuation")
    s.solve(zero_init_guess=True)
    r = s.solve_report
    assert str(r.reason_str).startswith("CONVERGED"), r.reason_str
    cfg = r.config
    assert cfg["linesearch"]["type"] == "l2", cfg["linesearch"]
    assert s.snes.getLineSearch().getType() == "bt"
    F_ref = s._continuation_stages["F_ref"]
    assert cfg["snes"]["atol"] == pytest.approx(s.tolerance * F_ref, rel=1.0e-12)
    assert cfg["tangent"]["consistent_jacobian"] == "continuation"
    assert cfg["ksp"]["pc"] == "fieldsplit" and len(cfg["ksp"]["splits"]) == 2
    assert "npc" not in cfg["snes"] and not s.snes.hasNPC(), (
        "the config snapshot attached a nonlinear preconditioner")

    s2, _, _ = _yielding_box("cf2", False)
    s2.solve(zero_init_guess=True)
    assert s2.solve_report.config["tangent"]["consistent_jacobian"] is False
    assert s2.solve_report.config["linesearch"]["type"] == "bt"


def _pressure_yield_box(tag, pressure_coupling):
    """Lid-driven, open sides (no pressure nullspace), yield stress that depends on PRESSURE
    so the consistent tangent has a d(eta)/dp term in the velocity-pressure block."""
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25)
    v = uw.discretisation.MeshVariable("Vpy" + tag, mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("Ppy" + tag, mesh, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    # positive for every p, with a non-zero derivative in p (a smoothed |p|)
    s.constitutive_model.Parameters.yield_stress = 0.3 + 0.3 * sympy.sqrt(p.sym[0] ** 2 + 1.0e-2)
    s.add_essential_bc((1.0, 0.0), "Top")
    s.add_essential_bc((0.0, 0.0), "Bottom")
    s.petsc_options.delValue("ksp_monitor")
    s.consistent_jacobian = True
    s.newton_pressure_coupling = pressure_coupling
    s.tolerance = 1.0e-6
    s.petsc_options["snes_max_it"] = 300
    return s, v, p


def _blocks(s):
    """(A, J_vp, J_pv) of the assembled Jacobian at the solver's current state."""
    snes = s.snes
    J, P, _ = snes.getJacobian()
    snes.computeJacobian(snes.getSolution(), J, P)
    keys = list(s._subdict)
    isv, isp = s._subdict[keys[0]][0], s._subdict[keys[1]][0]
    return J.createSubMatrix(isv, isv), J.createSubMatrix(isv, isp), J.createSubMatrix(isp, isv)


def test_partial_newton_keeps_the_velocity_block_and_restores_the_pressure_column():
    """newton_pressure_coupling=False: uu block identical to Newton's, J_vp = -B^T exactly
    (the Picard column), where the full Newton tangent's J_vp is NOT -B^T."""
    sN, vN, pN = _pressure_yield_box("N", True)
    sP, vP, pP = _pressure_yield_box("P", False)
    # the same state in both: one Newton step from rest, copied across
    sN.petsc_options["snes_max_it"] = 1
    sN.solve(zero_init_guess=True, picard=-1)
    with uw.synchronised_array_update():
        vP.data[...] = vN.data; pP.data[...] = pN.data
    sP.petsc_options["snes_max_it"] = 0
    sP.solve(zero_init_guess=False, picard=-1)
    A_N, Jvp_N, Jpv_N = _blocks(sN)
    A_P, Jvp_P, Jpv_P = _blocks(sP)

    dA = A_N.copy(); dA.axpy(-1.0, A_P)
    assert dA.norm() / A_N.norm() < 1.0e-12, "partial Newton changed the velocity block"
    BT = Jpv_P.copy(); BT.transpose()
    dP = Jvp_P.copy(); dP.axpy(1.0, BT)
    assert dP.norm() / Jpv_P.norm() < 1.0e-12, "partial Newton's pressure column is not -B^T"
    BT_N = Jpv_N.copy(); BT_N.transpose()
    dN = Jvp_N.copy(); dN.axpy(1.0, BT_N)
    assert dN.norm() / Jpv_N.norm() > 1.0e-3, \
        "fixture has no pressure coupling in the Newton tangent; the test proves nothing"
    assert sP.solve_report.config["tangent"]["newton_pressure_coupling"] is False
    assert sN.solve_report.config["tangent"]["newton_pressure_coupling"] is True


def test_partial_newton_converges_to_the_frozen_tangent_answer():
    """The residual is untouched, so the partial tangent reaches the same solution as the
    frozen (Picard) one."""
    sF, vF, pF = _pressure_yield_box("F", True)
    sF.consistent_jacobian = False
    sF.solve(zero_init_guess=True)
    assert sF.solve_report.converged, str(sF.solve_report)
    sP, vP, pP = _pressure_yield_box("Q", False)
    sP.solve(zero_init_guess=True)
    assert sP.solve_report.converged, str(sP.solve_report)
    dv = np.abs(np.asarray(vP.data) - np.asarray(vF.data)).max()
    assert dv < 1.0e-4 * np.abs(np.asarray(vF.data)).max(), dv


# ---- the warm-up inside the solve's contracts (review of PR #794) ---------------------

def test_the_newton_stage_is_anchored_to_the_cold_residual():
    """After the warm-up, Newton stops where an uninterrupted solve would: at
    tolerance x ||F(u_cold)||, not a further tolerance below the post-warm-up residual.
    Without the anchor a secretly-linear yielding box (tau_y far above any stress),
    which converges in 2 Newton iterations without the warm-up, stagnated on step size
    (SNORM_RELATIVE) after it. The report counts the warm-up and measures the
    reduction from the cold residual."""
    s, _ = _driven_box("anc", True, tau_y=1.0e4)
    s.solve(zero_init_guess=True)
    assert s._picard_stages == dict(picard_iterations=1), s._picard_stages
    r = s.solve_report
    assert str(r.reason_str) in FNORM, r.reason_str
    assert r.config["tangent"]["picard_warmup_its"] == 1
    # The one Picard step solves this secretly-linear problem, so the anchored Newton
    # stage has nothing to do: the solve is that ONE step, counted in the report (the
    # report used to show only the Newton solve's iterations). Before the anchor, Newton
    # took 5 more iterations and stopped on SNORM.
    assert r.nl_its == 1, r.nl_its
    assert r.reduction <= s.tolerance * (1.0 + 1.0e-9), r.reduction


def test_force_setup_is_not_read_as_picard():
    """`picard` follows every other argument of the Scalar/Vector/MultiComponent solve():
    five subclasses pass `_force_setup` positionally, and with `picard` second it landed
    there — a forced Picard warm-up and a blended-kernel compile on a LINEAR
    advection-diffusion solve."""
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0), cellSize=0.5)
    x, y = mesh.X
    T = uw.discretisation.MeshVariable("T_fs", mesh, 1, degree=2)
    adv = uw.systems.AdvDiffusion(mesh, T, sympy.Matrix([[-y, x]]))
    for b in ("Left", "Right", "Top", "Bottom"):
        adv.add_dirichlet_bc(0.0, b)
    adv.solve(timestep=0.01, _force_setup=True)
    assert adv._picard_stages is None, adv._picard_stages
    assert adv._picard_blend is False


def test_changing_the_tangent_recompiles_it():
    """The tangent is compiled into the Jacobian, so changing consistent_jacobian or
    newton_pressure_coupling after a solve marks a rewire; without it the next solve ran
    the old tangent while the report recorded the new one. alpha is per solver: a shared
    container let one solver's warm-up move another solver's tangent."""
    s, _ = _driven_box("tg", True)
    s.solve(zero_init_guess=True)
    assert s._needs_function_rewire is False
    s.consistent_jacobian = False
    assert s._needs_function_rewire is True
    s.solve(zero_init_guess=False)
    assert s._needs_function_rewire is False
    s.newton_pressure_coupling = False
    assert s._needs_function_rewire is True
    s2, _ = _driven_box("tg2", True)
    assert s._get_newton_alpha() is not s2._get_newton_alpha()
