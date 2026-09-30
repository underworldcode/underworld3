"""Locks the uw.meshing.follow_metric() public API.

Two-knob, cell-size-envelope adapter for the anisotropic node mover.
The user passes:

    refinement  : factor by which the finest cells are smaller than h0
    coarsening  : factor by which the coarsest cells are bigger than h0
                  ("auto" = refinement**(1/d), the budget-conserving
                  minimum)
    metric      : "front-following" or "gradient-uniform"

These tests pin the metric construction (refinement=1 is a no-op,
ρ_min/ρ_max envelope is correct, geomean(ρ) ≈ 1 by construction so
the mover's G normalisation is bypassed) and end-to-end that the
mover actually moves the mesh and lands an approximate cell-size
envelope.

The mover's achieved cell-size envelope is APPROXIMATE — anisotropic
cells + iterative deformation map mean the eigenvalue clamp doesn't
literally bound the achieved h_max on a per-cell basis (since the
2026-07 mover retirement the backing mover is the variational MMPDE
mover with theta=0.5, pure equidistribution weighting). Validation
on a sharp-tanh annulus shows:

* refinement side: achieved h_min within ~30-40% of h0/refinement
  (platform-dependent: ~27% on macOS, ~40% on Linux CI at refinement=2)
* coarsening side: achieved h_max within ~2x of h0*coarsening

The tests here use loose tolerances reflecting that empirical reality.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.meshing import smoothing as _sm


def _build_annulus_with_field():
    """Return a fresh annulus + a sharp tanh T field that mimics a
    thermal boundary layer near r=0.7."""
    m = uw.meshing.Annulus(radiusInner=0.5, radiusOuter=1.0,
                           cellSize=0.08, qdegree=3)
    T = uw.discretisation.MeshVariable(
        "T", m, vtype=uw.VarType.SCALAR, degree=2, continuous=True)
    proj = uw.systems.Projection(m, T)
    proj.smoothing = 0.0
    x, y = m.X
    proj.uw_function = 0.5 * (
        1.0 + sympy.tanh(40.0 * (0.7 - sympy.sqrt(x*x + y*y))))
    proj.solve()
    return m, T


def _cell_h_stats(mesh):
    """Per-cell mean-edge h, returns (min, max, mean)."""
    tris = _sm._tri_cells(mesh.dm)
    p = np.asarray(mesh.X.coords)[tris]
    h = (np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
         + np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
         + np.linalg.norm(p[:, 0] - p[:, 2], axis=1)) / 3.0
    return float(h.min()), float(h.max()), float(h.mean())


@pytest.mark.tier_a
@pytest.mark.level_1
def test_refinement_one_is_noop_metric():
    """refinement=1, coarsening=1: metric is uniformly 1 (no adapt)."""
    m, T = _build_annulus_with_field()
    rho = uw.meshing.metric_density_from_gradient(
        m, T, refinement=1.0, coarsening=1.0, name="check_ref1")
    rho_vals = np.asarray(
        uw.function.evaluate(rho, m.X.coords)).reshape(-1)
    assert np.allclose(rho_vals, 1.0)


@pytest.mark.tier_a
@pytest.mark.level_1
def test_envelope_in_rho_space_d_independent():
    """For any (refinement, coarsening), the metric ρ has the
    envelope [1/coarsening², refinement²] independent of d."""
    m, T = _build_annulus_with_field()
    for ref, coar in [(2.0, "auto"), (2.0, 1.5), (3.0, 2.0)]:
        rho = uw.meshing.metric_density_from_gradient(
            m, T, refinement=ref, coarsening=coar,
            name=f"env_{ref}_{coar}")
        rho_vals = np.asarray(
            uw.function.evaluate(rho, m.X.coords)).reshape(-1)
        coar_val = (ref ** 0.5) if coar == "auto" else float(coar)
        expected_max = ref ** 2
        expected_min = 1.0 / coar_val ** 2
        # Tolerance: percentile clipping + FE interpolation noise
        assert rho_vals.max() == pytest.approx(expected_max, rel=0.05), (
            f"({ref}, {coar}): ρ_max = {rho_vals.max()}, "
            f"expected ≈ {expected_max}")
        assert rho_vals.min() == pytest.approx(expected_min, rel=0.05), (
            f"({ref}, {coar}): ρ_min = {rho_vals.min()}, "
            f"expected ≈ {expected_min}")


@pytest.mark.tier_a
@pytest.mark.level_1
def test_geomean_rho_is_unity_front_following():
    """The piecewise-log-linear ansatz has geomean(ρ) = 1 by
    construction so the mover's G normalisation passes through."""
    m, T = _build_annulus_with_field()
    rho = uw.meshing.metric_density_from_gradient(
        m, T, refinement=2.0, coarsening="auto",
        metric_choice="front-following", name="g1")
    rho_vals = np.asarray(
        uw.function.evaluate(rho, m.X.coords)).reshape(-1)
    log_geomean = float(np.mean(np.log(rho_vals)))
    # Tight: the analytic break-point makes this exactly 0 in the
    # absence of FE interpolation noise.
    assert abs(log_geomean) < 0.05, (
        f"geomean(ρ) log = {log_geomean}, expected ≈ 0")


@pytest.mark.tier_a
@pytest.mark.level_1
def test_auto_coarsening_is_d_th_root_of_refinement():
    """coarsening='auto' uses the budget-conserving minimum
    refinement**(1/d) — verified by setting the equivalent
    explicit coarsening and checking ρ is identical."""
    m, T = _build_annulus_with_field()
    d = m.cdim
    ref = 3.0
    rho_auto = uw.meshing.metric_density_from_gradient(
        m, T, refinement=ref, coarsening="auto", name="auto")
    rho_explicit = uw.meshing.metric_density_from_gradient(
        m, T, refinement=ref, coarsening=ref ** (1.0/d),
        name="explicit")
    v_a = np.asarray(uw.function.evaluate(rho_auto, m.X.coords)).reshape(-1)
    v_e = np.asarray(uw.function.evaluate(rho_explicit, m.X.coords)).reshape(-1)
    assert np.allclose(v_a, v_e)


@pytest.mark.tier_a
@pytest.mark.level_1
def test_invalid_args():
    """refinement<1 and coarsening<1 must raise."""
    m, T = _build_annulus_with_field()
    with pytest.raises(ValueError, match="refinement must be >= 1.0"):
        uw.meshing.metric_density_from_gradient(
            m, T, refinement=0.5, name="badref")
    with pytest.raises(ValueError, match="coarsening must be >= 1.0"):
        uw.meshing.metric_density_from_gradient(
            m, T, refinement=2.0, coarsening=0.5, name="badcoar")
    with pytest.raises(ValueError, match="metric_choice must be"):
        uw.meshing.metric_density_from_gradient(
            m, T, refinement=2.0, metric_choice="unknown",
            name="badmc")


@pytest.mark.tier_a
@pytest.mark.level_1
def test_follow_metric_moves_mesh():
    """follow_metric returns True when the mesh is actually moved."""
    m, T = _build_annulus_with_field()
    old_X = np.asarray(m.X.coords).copy()
    moved = uw.meshing.follow_metric(
        m, T, refinement=2.0, skip_threshold=None)
    new_X = np.asarray(m.X.coords)
    assert moved is True
    assert not np.allclose(new_X, old_X)


@pytest.mark.tier_a
@pytest.mark.level_1
def test_follow_metric_refinement_envelope_approximate():
    """The per-cell mean-edge minimum stays at or above the
    requested h_min = h0/refinement.

    The min-edge spring caps the SHORTEST EDGE of any cell at
    h0/refinement; that forces the cell's MEAN edge ≥ h0/refinement
    (since mean ≥ min). On most cells this means a slight
    over-spec (mean-edge slightly larger than asked), which is
    the right side of the trade — the user asked for caps on
    *extreme* refinement, accepting mild under-refinement.

    Occasional slivers can still occur on cells with very strong
    gradients (the spring can't perfectly counteract the metric
    pull at every cell); the *mean-edge* guarantee is the robust
    one.
    """
    m0, _ = _build_annulus_with_field()
    _, _, h0 = _cell_h_stats(m0)

    for ref in [1.5, 2.0]:
        m, T = _build_annulus_with_field()
        uw.meshing.follow_metric(
            m, T, refinement=ref, skip_threshold=None)
        h_min_cell, _, _ = _cell_h_stats(m)
        target_h_min = h0 / ref
        # The MMPDE mover honours the envelope approximately (it
        # equidistributes toward the metric, it does not clamp cells).
        # Measured spread at refinement=2: 1.27 (macOS) - 1.40 (Linux
        # CI) over-spec; the bounds catch over-refinement (unsafe) and
        # gross under-refinement, not the platform jitter.
        assert h_min_cell / target_h_min > 0.85, (
            f"refinement={ref}: mean-edge h_min/target = "
            f"{h_min_cell/target_h_min:.3f}, want > 0.85 "
            f"(should not over-refine past spec)")
        assert h_min_cell / target_h_min < 1.5, (
            f"refinement={ref}: mean-edge h_min/target = "
            f"{h_min_cell/target_h_min:.3f}, want < 1.5 "
            f"(should not under-refine far past spec)")


@pytest.mark.tier_a
@pytest.mark.level_1
def test_follow_metric_coarsening_envelope_approximate():
    """Achieved h_max is within ~30% of h0·coarsening on the
    sharp-tanh annulus problem.

    Before the per-cell rest-size spring was added to the mover,
    h_max overshot the spec by 50-130%. The spring restores
    over-coarsened cells by pulling their nodes back toward the
    rest positions, tightening the cap. The remaining ~15-30%
    overshoot reflects the unavoidable trade-off between metric
    fidelity (which the SNES wants) and cap enforcement (which
    the spring wants); 30% is a wide enough tolerance that it's
    not flaky on this fixture problem.
    """
    m0, _ = _build_annulus_with_field()
    _, _, h0 = _cell_h_stats(m0)
    for ref, coar in [(1.5, "auto"), (2.0, "auto"), (2.0, 2.0),
                       (3.0, "auto")]:
        m, T = _build_annulus_with_field()
        uw.meshing.follow_metric(
            m, T, refinement=ref, coarsening=coar,
            skip_threshold=None)
        _, h_max, _ = _cell_h_stats(m)
        coar_val = (ref**0.5) if coar == "auto" else float(coar)
        target_h_max = h0 * coar_val
        # Allow up to 30% over spec
        assert h_max / target_h_max < 1.30, (
            f"ref={ref}, coar={coar}: achieved h_max/target = "
            f"{h_max/target_h_max:.3f}, want < 1.30")


@pytest.mark.tier_a
@pytest.mark.level_1
def test_follow_metric_no_slivers_after_adaptive_polish():
    """The adaptive Jacobi polish loop should drive the worst
    cell-shape quality above `polish_quality_target` (default 0.3)
    on the sharp-tanh annulus problem. Cells with q < 0.3 are
    visible slivers; q=1 is equilateral. This locks the contract
    that follow_metric's built-in polish eliminates the slivers
    its mover would otherwise leave behind."""
    import numpy as np
    for ref in [2.0, 3.0]:
        m, T = _build_annulus_with_field()
        uw.meshing.follow_metric(
            m, T, refinement=ref, skip_threshold=None)
        tris = _sm._tri_cells(m.dm)
        p = np.asarray(m.X.coords)[tris]
        e0 = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
        e1 = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
        e2 = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
        A = np.abs(_sm._signed_areas(np.asarray(m.X.coords), tris))
        q = 4.0 * np.sqrt(3.0) * A / (e0**2 + e1**2 + e2**2 + 1e-30)
        # No slivers after polish (well below the equilateral q=1
        # but well above the visible-sliver threshold q≈0.3).
        assert q.min() >= 0.3, (
            f"refinement={ref}: q_min={q.min():.3f} "
            f"(want ≥ 0.3 — adaptive polish should have run "
            f"more iterations)")


@pytest.mark.tier_a
@pytest.mark.level_1
def test_follow_metric_skip_threshold_skips_aligned_mesh():
    # (An xfail marker for a #202-era field-transfer nudge was removed
    # 2026-07: the MMPDE-backed follow_metric passes this cleanly.)
    """A mesh that's already aligned (here: any uniform mesh with
    the field still in the natural shape) gets the skip-on-align
    short-circuit when skip_threshold is permissive."""
    m, T = _build_annulus_with_field()
    # First adapt — mesh moves
    moved1 = uw.meshing.follow_metric(
        m, T, refinement=2.0, skip_threshold=None)
    assert moved1 is True
    # Second adapt with a relatively tight threshold — should skip
    # because the mesh is already well-aligned to T.
    moved2 = uw.meshing.follow_metric(
        m, T, refinement=2.0, skip_threshold=0.9)
    assert moved2 is False


# ---------------------------------------------------------------------------
# arc-length metric_choice + Monge–Ampère mover (the clean default candidate)
# ---------------------------------------------------------------------------
def _inverted_count(mesh):
    """Cells whose signed area flipped sign vs the dominant orientation —
    a true tangling check (|A|-based quality is blind to inversion)."""
    tris = _sm._tri_cells(mesh.dm)
    A = _sm._signed_areas(np.asarray(mesh.X.coords), tris)
    orient = np.sign(np.median(A)) or 1.0
    return int((A * orient < 0).sum())


@pytest.mark.tier_a
@pytest.mark.level_1
def test_metric_choice_arc_length_builds_envelope():
    # Smooth ρ = √(1+(A·ĝ)²): ≥ 1 everywhere, capped at the refinement
    # envelope ρ_max = refinement² (the MA-stable contrast).
    m, T = _build_annulus_with_field()
    rho = _sm.metric_density_from_gradient(
        m, T, refinement=3.0, metric_choice="arc-length", name="al")
    v = np.asarray(uw.function.evaluate(
        rho, np.asarray(m.X.coords))).reshape(-1)
    assert np.all(np.isfinite(v))
    assert v.min() >= 1.0 - 1.0e-9
    assert v.max() <= 3.0 ** 2 + 1.0e-6


# NOTE (LE-20 / WA-22): two xfail tests were deleted here. They asserted
# elliptic-ma MA-mover capabilities (strong metric capture; boundary slip in
# follow_metric) that the development merge deliberately dropped. Surviving
# coverage: arc-length capture via the OT mover (test_0760); tangential
# boundary slip via smooth_mesh_interior(method='mmpde', slip_surfaces=...)
# (test_0855). See git history if the MA mover returns.


@pytest.mark.tier_a
@pytest.mark.level_1
def test_follow_metric_invalid_metric_raises():
    m, T = _build_annulus_with_field()
    with pytest.raises(ValueError):
        uw.meshing.follow_metric(m, T, refinement=3.0, metric="bogus")
