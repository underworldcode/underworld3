"""Unified δ-parameterised yield law + yield homotopy.

Covers the PR-2 contract:

1. ``test_delta0_is_exact_min`` — the unified soft-min law at δ=0 reproduces
   ``Min(η_ve, η_pl)`` to machine precision (g(f) = max(1, f)).
2. ``test_yield_softness_runtime_no_recompile`` — δ lives in ``constants[]``;
   changing it and repacking does NOT change the solver's JIT cache key.
3. ``test_yield_mode_softmin_deprecated`` — ``yield_mode='softmin'`` warns and
   maps to ``'min'``; ``'smooth'`` is a hard error.
4. ``test_yield_homotopy_loading_converges`` — with the homotopy enabled,
   loading a shear box through yield gives no SNES divergences AND locks σ at
   τ_y (cold hard-Min diverges on this problem).
"""

import warnings

import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.function import expression
from underworld3.function.expressions import unwrap_expression

ETA = 1.0
MU = 1.0
TAU_Y = 0.5
V0 = 0.5
T_R = ETA / MU


def _build_vep(label, order=2):
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(16, 8), minCoords=(-1.0, -0.5), maxCoords=(1.0, 0.5),
    )
    v = uw.discretisation.MeshVariable(f"U_{label}", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable(f"P_{label}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    cm = uw.constitutive_models.ViscoElasticPlasticFlowModel(stokes.Unknowns, order=order)
    stokes.constitutive_model = cm
    cm.Parameters.shear_viscosity_0 = ETA
    cm.Parameters.shear_modulus = MU
    cm.Parameters.yield_stress = TAU_Y
    cm.Parameters.strainrate_inv_II_min = 1.0e-6
    V_top = expression(rf"V_{{{label}}}", sympy.Float(V0), "Top V")
    stokes.add_dirichlet_bc((V_top, 0.0), "Top")
    stokes.add_dirichlet_bc((-V_top, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    stokes.tolerance = 1.0e-6
    stokes.petsc_options["snes_force_iteration"] = True
    return mesh, stokes, cm, V_top


@pytest.mark.level_1
@pytest.mark.tier_a
def test_delta0_is_exact_min():
    """δ=0 soft-min law equals Min(η_ve, η_pl) to machine precision."""
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    v = uw.discretisation.MeshVariable("Um", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("Pm", mesh, 1, degree=1)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    cm = uw.constitutive_models.ViscoElasticPlasticFlowModel(s.Unknowns, order=2)
    s.constitutive_model = cm
    cm._yield_mode = "min"
    cm._yield_softness = 0.0
    for eta_ve, eta_pl in [(1.0, 2.0), (2.0, 1.0), (1.0, 1.0), (0.3, 5.0), (5.0, 0.3)]:
        comb = cm._combine_yield(sympy.Float(eta_ve), sympy.Float(eta_pl))
        val = float(unwrap_expression(comb, mode="nondimensional"))
        assert abs(val - min(eta_ve, eta_pl)) < 1.0e-12


@pytest.mark.level_1
@pytest.mark.tier_a
def test_yield_mode_softmin_deprecated():
    """'softmin' warns and maps to 'min'; 'smooth' is a hard error."""
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    v = uw.discretisation.MeshVariable("Ud", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("Pd", mesh, 1, degree=1)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    cm = uw.constitutive_models.ViscoElasticPlasticFlowModel(s.Unknowns, order=2)
    s.constitutive_model = cm

    with pytest.warns(DeprecationWarning):
        cm.yield_mode = "softmin"
    assert cm.yield_mode == "min"

    with pytest.raises(ValueError):
        cm.yield_mode = "smooth"

    with pytest.raises(ValueError):
        cm.yield_mode = "not_a_mode"


@pytest.mark.level_2
@pytest.mark.tier_a
def test_yield_softness_runtime_no_recompile():
    """Ramping δ through its constants[] atom must not trigger a JIT recompile."""
    _, stokes, cm, V_top = _build_vep("norecompile")
    cm._yield_softness = 0.3  # δ>0 so the soft-min atom is exercised
    dt = 0.20 * T_R
    cm.Parameters.dt_elastic = dt
    stokes.solve(zero_init_guess=False, timestep=dt, divergence_retries=2)
    key0 = stokes._current_jit_cache_key
    assert key0 is not None

    cm._get_yield_softness()
    for d in (0.2, 0.1, 0.0):
        cm._yield_softness_expr.sym = sympy.Float(d)
        stokes._update_constants()
        stokes.solve(zero_init_guess=False, timestep=dt, divergence_retries=2)
    assert stokes._current_jit_cache_key == key0, "δ ramp forced a JIT recompile"


@pytest.mark.level_2
@pytest.mark.tier_a
def test_yield_homotopy_loading_converges():
    """Yield homotopy: no SNES divergence loading through yield, σ locked at τ_y."""
    _, stokes, cm, V_top = _build_vep("homotopy")
    cm.enable_yield_homotopy(delta_start=0.5)  # residual schedule, basic linesearch
    dt = 0.20 * T_R
    reasons, sigmas = [], []
    centre = np.array([[0.0, 0.0]])
    for _ in range(15):
        V_top.sym = sympy.Float(V0)
        cm.Parameters.dt_elastic = dt
        stokes.solve(zero_init_guess=False, timestep=dt, divergence_retries=2)
        reasons.append(int(stokes.snes.getConvergedReason()))
        sigmas.append(float(uw.function.evaluate(stokes.tau.sym[0, 1], centre).flatten()[0]))
    reasons = np.array(reasons)
    plateau = np.abs(np.array(sigmas)[10:])
    assert int((reasons < 0).sum()) == 0, f"SNES diverged: {reasons.tolist()}"
    # δ ends at 0 each step → exact yield surface: σ locks at τ_y.
    assert plateau.max() <= TAU_Y * 1.02, f"yield over-shoot: {plateau.max():.4f}"
    assert plateau.min() >= TAU_Y * 0.97, f"yield under-clip: {plateau.min():.4f}"


@pytest.mark.level_1
@pytest.mark.tier_a
def test_powermean_smoother_undershoots_min():
    """Power-mean smooth-min: ≤ Min everywhere (no over-yield), overflow-safe on
    geodynamic ranges, and → Min as δ→0 (s=1/δ→∞)."""
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    v = uw.discretisation.MeshVariable("Upm", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("Ppm", mesh, 1, degree=1)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    cm = uw.constitutive_models.ViscoElasticPlasticFlowModel(s.Unknowns, order=2)
    s.constitutive_model = cm

    cm.yield_mode = "min"
    cm.yield_smoother = "powermean"        # bumps δ 0 → 1 (s=1, harmonic mean)
    assert cm.yield_smoother == "powermean"
    assert cm.yield_softness == 1.0

    # Includes a geodynamic-range pair (1e25, 1e21) that overflows the naive
    # η^(−s) form above s≈40 but is finite in the harmonic-normalised form.
    pairs = [(1.0, 2.0), (2.0, 1.0), (1.0, 1.0), (0.3, 5.0), (5.0, 0.3), (1e25, 1e21)]
    for delta in (1.0, 0.5, 0.1, 0.02):
        cm.yield_softness = delta
        for eta_ve, eta_pl in pairs:
            comb = cm._combine_yield(sympy.Float(eta_ve), sympy.Float(eta_pl))
            val = float(unwrap_expression(comb, mode="nondimensional"))
            assert np.isfinite(val), f"overflow δ={delta} ({eta_ve},{eta_pl})"
            assert val <= min(eta_ve, eta_pl) * (1 + 1e-9), \
                f"power-mean over-yields δ={delta}: {val} > {min(eta_ve, eta_pl)}"

    # smallest δ (largest s) is close to exact Min.
    cm.yield_softness = 0.02
    for eta_ve, eta_pl in pairs:
        comb = cm._combine_yield(sympy.Float(eta_ve), sympy.Float(eta_pl))
        val = float(unwrap_expression(comb, mode="nondimensional"))
        m = min(eta_ve, eta_pl)
        assert abs(val - m) <= 0.05 * m, f"not near Min at δ=0.02: {val} vs {m}"


@pytest.mark.level_1
@pytest.mark.tier_a
def test_yield_smoother_validation():
    """yield_smoother only accepts the two known families."""
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    v = uw.discretisation.MeshVariable("Usm", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("Psm", mesh, 1, degree=1)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    cm = uw.constitutive_models.ViscoElasticPlasticFlowModel(s.Unknowns, order=2)
    s.constitutive_model = cm
    assert cm.yield_smoother == "sqrt"           # default family unchanged
    with pytest.raises(ValueError):
        cm.yield_smoother = "not_a_family"
