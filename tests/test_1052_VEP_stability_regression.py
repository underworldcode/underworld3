"""VEP stability regressions — yield-surface lock under fixed and variable dt.

These tests exercise code paths that historically had subtle bugs and
would catch their re-introduction. They are slower than the level_1
suite (~30s each) so are tagged level_2; run with::

    pytest -m "level_2 and tier_a" tests/test_1052_VEP_stability_regression.py

Bugs each test would have caught
--------------------------------

1. ``test_vep_yield_lock_fixed_dt`` — under sustained loading at constant
   dt with ε̇ above yield, σ should hold at τ_y. Catches: any failure of
   Min-mode to clip stress correctly (e.g. softmin under-clip masquerading
   as Min, JIT mis-compilation of Min(η_ve, η_pl)).

2. ``test_vep_yield_lock_variable_dt`` — halving and doubling dt at yield
   should not knock σ off the yield surface. Catches: the implicit-
   projection drift fixed by the psi_snapshot machinery (see commits
   8f2b0dd, 31abad1, 380ab4b).  Pre-fix this test would have shown peak
   |σ| ≈ 0.65 against τ_y = 0.5.

3. ``test_vep_snes_no_divergence_loading_through_yield`` — loading from
   σ=0 through the yield onset should not produce SNES divergences (with
   the divergence_retries safety net). Catches: regressions in the
   Picard-retry mechanism or in the BDF-2 stability at the kink (the
   reason ``bdf_blend`` was retired).

4. ``test_pure_ve_variable_dt_accuracy`` — pure-VE accuracy under variable
   dt against the analytical Maxwell square-wave solution. Catches:
   regressions in variable-dt BDF-2 coefficients or in the snapshot
   refresh ordering. Threshold loose enough (max_err < 0.10) that it
   doesn't false-trip on minor BDF-coefficient tuning.
"""

import pytest
import numpy as np
import sympy
import underworld3 as uw
from underworld3.function import expression


pytestmark = [pytest.mark.level_2, pytest.mark.tier_a]


# ---------------------------------------------------------------------------
# Common setup: shear box at unit Maxwell time, τ_y = 0.5, ε̇ = 0.5 (above yield)
# ---------------------------------------------------------------------------

ETA = 1.0
MU = 1.0
TAU_Y = 0.5
V0 = 0.5  # gives ε̇_xy = 0.5 in the symmetric strain rate convention
T_R = ETA / MU
HALF_PERIOD = 2.0 * T_R


def _build_stokes(label, yield_mode="min", yield_stress=TAU_Y):
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(16, 8), minCoords=(-1.0, -0.5), maxCoords=(1.0, 0.5),
    )
    v = uw.discretisation.MeshVariable(f"U_{label}", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable(f"P_{label}", mesh, 1, degree=1)
    stokes = uw.systems.VE_Stokes(mesh, velocityField=v, pressureField=p, order=2)
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = ETA
    stokes.constitutive_model.Parameters.shear_modulus = MU
    stokes.constitutive_model.Parameters.yield_stress = yield_stress
    stokes.constitutive_model.Parameters.strainrate_inv_II_min = 1.0e-6
    stokes.constitutive_model._yield_mode = yield_mode
    V_top = expression(rf"V_{{{label}}}", sympy.Float(V0), "Top V")
    stokes.add_dirichlet_bc((V_top, 0.0), "Top")
    stokes.add_dirichlet_bc((-V_top, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    stokes.tolerance = 1.0e-6
    stokes.petsc_options["snes_force_iteration"] = True
    return mesh, stokes, V_top


def _step(stokes, V_top, dt, V_sign=1.0):
    V_top.sym = sympy.Float(V_sign * V0)
    stokes.constitutive_model.Parameters.dt_elastic = dt
    stokes.solve(zero_init_guess=False, timestep=dt, divergence_retries=2)
    return int(stokes.snes.getConvergedReason())


def _sigma(stokes, c=np.array([[0.0, 0.0]])):
    return float(uw.function.evaluate(stokes.tau.sym[0, 1], c).flatten()[0])


# ---------------------------------------------------------------------------
# Test 1: yield surface lock under fixed dt
# ---------------------------------------------------------------------------

def test_vep_yield_lock_fixed_dt():
    """Under sustained constant-ε̇ loading above yield, σ should clamp at τ_y.

    Catches: Min-mode mis-compilation, viscosity-formula drift, projection
    failure. Pre-fix the bug surfaced as σ slowly drifting upward each step
    once at yield.
    """
    _, stokes, V_top = _build_stokes("yield_lock_fixed")
    dt = 0.20 * T_R
    sigmas = []
    for _ in range(20):
        _step(stokes, V_top, dt)
        sigmas.append(_sigma(stokes))
    sigmas = np.array(sigmas)
    # After ~4 steps σ should be at yield. Take the second half as the
    # plateau and verify it sits exactly on τ_y.
    plateau = sigmas[10:]
    assert np.abs(plateau).max() <= TAU_Y * 1.001, (
        f"yield surface violation under fixed dt: peak={np.abs(plateau).max():.4f} "
        f"vs τ_y={TAU_Y}"
    )
    assert np.abs(plateau).min() >= TAU_Y * 0.999, (
        f"yield surface under-clip: min={np.abs(plateau).min():.4f} "
        f"vs τ_y={TAU_Y}"
    )


# ---------------------------------------------------------------------------
# Test 2: yield surface lock under variable dt (the snapshot fix)
# ---------------------------------------------------------------------------

def test_vep_yield_lock_variable_dt():
    """Halving and doubling dt at yield must not push σ off the yield surface.

    Catches: implicit-projection drift fixed by psi_snapshot. Pre-fix this
    test would have shown peak |σ| ≈ 0.65 (30% violation).
    """
    _, stokes, V_top = _build_stokes("yield_lock_var")
    dt_max, dt_min = 0.20 * T_R, 0.10 * T_R
    # Phase 1: warm up to yield at dt_max
    for _ in range(5):
        _step(stokes, V_top, dt_max)
    assert abs(_sigma(stokes) - TAU_Y) < TAU_Y * 0.001, "warm-up did not reach yield"
    # Phase 2: halve dt, take 4 steps — these are where the bug surfaced
    sigmas_after_halve = []
    for _ in range(4):
        _step(stokes, V_top, dt_min)
        sigmas_after_halve.append(_sigma(stokes))
    # Phase 3: double dt back, take 4 steps
    sigmas_after_double = []
    for _ in range(4):
        _step(stokes, V_top, dt_max)
        sigmas_after_double.append(_sigma(stokes))
    sigmas = np.array(sigmas_after_halve + sigmas_after_double)
    assert np.abs(sigmas).max() <= TAU_Y * 1.01, (
        f"variable-dt yield violation: peak={np.abs(sigmas).max():.4f} "
        f"vs τ_y={TAU_Y}. Pre-fix this would have been ~0.65."
    )


# ---------------------------------------------------------------------------
# Test 3: SNES convergence through yield onset
# ---------------------------------------------------------------------------

def test_vep_snes_no_divergence_loading_through_yield():
    """Loading from σ=0 through yield onset must not produce SNES divergences.

    Catches: regressions in divergence_retries, Picard machinery, or BDF-2
    stability at the Min kink (the reason ``bdf_blend`` existed). Allows
    a small number of soft divergences that the retry machinery rescues.
    """
    _, stokes, V_top = _build_stokes("snes_loading")
    dt = 0.20 * T_R
    final_reasons = []
    for _ in range(15):
        reason = _step(stokes, V_top, dt)
        final_reasons.append(reason)
    final_reasons = np.array(final_reasons)
    # All steps should converge once retries are accounted for
    n_diverged = int((final_reasons < 0).sum())
    assert n_diverged == 0, (
        f"SNES diverged on {n_diverged}/{len(final_reasons)} steps loading through yield. "
        f"Reasons: {final_reasons.tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 4: pure VE accuracy under variable dt (analytical comparison)
# ---------------------------------------------------------------------------

def _step_square_analytical(t, eta, mu, gamma_dot, half_period):
    t_r = eta / mu
    sigma_ss = eta * gamma_dot
    out = np.zeros_like(t, dtype=float)
    for i, ti in enumerate(t):
        n = int(ti / half_period)
        t_local = ti - n * half_period
        sigma_start = 0.0
        for j in range(n):
            sign = 1.0 if j % 2 == 0 else -1.0
            target = sign * sigma_ss
            sigma_start = target + (sigma_start - target) * np.exp(-half_period / t_r)
        sign = 1.0 if n % 2 == 0 else -1.0
        target = sign * sigma_ss
        out[i] = target + (sigma_start - target) * np.exp(-t_local / t_r)
    return out


def test_pure_ve_variable_dt_accuracy():
    """Pure VE under variable dt should match the analytical square-wave
    solution within a loose tolerance.

    Catches: regressions in variable-dt BDF-2 coefficients or in the
    snapshot refresh ordering.  The threshold (max_err < 0.10) is loose
    enough that minor BDF-coefficient tuning won't false-trip it but
    tight enough to catch large drifts. Current head produces ~0.06.
    """
    # Pure VE: yield_stress -> infinity
    _, stokes, V_top = _build_stokes("pure_ve_acc", yield_stress=1.0e6)
    flip_times = (HALF_PERIOD * 1, HALF_PERIOD * 2, HALF_PERIOD * 3)
    window = 0.40 * T_R
    dt_max, dt_min = 0.20 * T_R, 0.10 * T_R

    def schedule_dt(t):
        for f in flip_times:
            if abs(t - f) <= window or 0 <= f - t <= window:
                return dt_min
        return dt_max

    times, taus = [], []
    t_cur = 0.0
    t_end = 4 * HALF_PERIOD  # 8 t_r
    while t_cur < t_end - 1e-9:
        dt = min(schedule_dt(t_cur), t_end - t_cur)
        n_half = int((t_cur + 0.5 * dt) / HALF_PERIOD)
        V_sign = 1.0 if n_half % 2 == 0 else -1.0
        _step(stokes, V_top, dt, V_sign=V_sign)
        t_cur += dt
        times.append(t_cur)
        taus.append(_sigma(stokes))
    times = np.array(times)
    taus = np.array(taus)

    sigma_analytical = _step_square_analytical(
        times, ETA, MU, 2.0 * V0, HALF_PERIOD
    )
    max_err = float(np.max(np.abs(taus - sigma_analytical)))
    assert max_err < 0.10, (
        f"pure-VE accuracy regressed under variable dt: "
        f"max|err|={max_err:.4f} (threshold 0.10; current head ~0.06)"
    )
