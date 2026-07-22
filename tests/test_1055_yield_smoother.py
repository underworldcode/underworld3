"""δ-parameterised yield soft-min smoother (sqrt + power-mean families).

Pins the smoother contract on development (the ``_combine_yield`` helper + the
runtime-rampable δ ``constants[]`` atom, ported from the yield-homotopy work, with
development's tri-modal ``yield_mode`` semantics preserved):

1. ``test_delta0_is_exact_min`` — the sqrt soft-min at δ=0 equals ``Min(η_ve, η_pl)``
   to machine precision (g(f) = max(1, f)).
2. ``test_powermean_smoother_undershoots_min`` — the power-mean family is ≤ Min
   everywhere (never over-yields), overflow-safe on geodynamic ranges, and → Min as δ→0.
3. ``test_yield_smoother_validation`` — only the two known families; default "sqrt".
4. ``test_yield_softness_runtime_no_recompile`` — ramping δ through its constants[] atom
   does NOT change the solver's JIT cache key.
5. ``test_dp_model_smoother_optin`` — the non-elastic Drucker–Prager model defaults to
   exact hard Min and can opt into the softmin/power-mean homotopy.

NOTE (vs the branch): development keeps ``yield_mode`` tri-modal — ``"min"`` is the exact
hard ``Min`` (not the soft-min), so the smoother tests select ``yield_mode="softmin"``.
The ``enable_yield_homotopy`` in-solve ramp driver is intentionally NOT part of this PR.
"""

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
    """δ=0 sqrt soft-min law equals Min(η_ve, η_pl) to machine precision."""
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    v = uw.discretisation.MeshVariable("Um", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("Pm", mesh, 1, degree=1)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    cm = uw.constitutive_models.ViscoElasticPlasticFlowModel(s.Unknowns, order=2)
    s.constitutive_model = cm
    cm.yield_mode = "softmin"          # the δ-family path (dev "min" = hard Min)
    cm.yield_softness = 0.0            # δ=0 ⇒ g(f)=max(1,f) ⇒ exact Min
    for eta_ve, eta_pl in [(1.0, 2.0), (2.0, 1.0), (1.0, 1.0), (0.3, 5.0), (5.0, 0.3)]:
        comb = cm._combine_yield(sympy.Float(eta_ve), sympy.Float(eta_pl))
        val = float(unwrap_expression(comb, mode="nondimensional"))
        assert abs(val - min(eta_ve, eta_pl)) < 1.0e-12


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

    cm.yield_mode = "softmin"              # dev: soft-min family lives under "softmin"
    cm.yield_softness = 0.0                # start from δ=0 so the powermean select bumps it
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
    """yield_smoother only accepts the two known families; default is 'sqrt'."""
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    v = uw.discretisation.MeshVariable("Usm", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("Psm", mesh, 1, degree=1)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    cm = uw.constitutive_models.ViscoElasticPlasticFlowModel(s.Unknowns, order=2)
    s.constitutive_model = cm
    assert cm.yield_smoother == "sqrt"           # default family unchanged
    with pytest.raises(ValueError):
        cm.yield_smoother = "not_a_family"


@pytest.mark.level_2
@pytest.mark.tier_a
def test_yield_softness_runtime_no_recompile():
    """Ramping δ through its constants[] atom must not trigger a JIT recompile."""
    _, stokes, cm, V_top = _build_vep("norecompile")
    cm.yield_softness = 0.3  # δ>0 so the soft-min atom is exercised
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


@pytest.mark.level_1
@pytest.mark.tier_a
def test_dp_model_smoother_optin():
    """The non-elastic Drucker–Prager model defaults to exact hard Min and can opt
    into the δ soft-min / power-mean homotopy (its new capability)."""
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    v = uw.discretisation.MeshVariable("Udp", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("Pdp", mesh, 1, degree=1)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    cm = uw.constitutive_models.ViscoPlasticFlowModel(s.Unknowns)
    s.constitutive_model = cm

    # default: exact hard Min
    assert cm.yield_mode == "min"
    assert cm.yield_smoother == "sqrt"
    for eta_ve, eta_pl in [(1.0, 2.0), (2.0, 1.0), (0.3, 5.0)]:
        comb = cm._combine_yield(sympy.Float(eta_ve), sympy.Float(eta_pl))
        val = float(unwrap_expression(comb, mode="nondimensional"))
        assert abs(val - min(eta_ve, eta_pl)) < 1.0e-12

    # opt into the power-mean homotopy: undershoots Min
    cm.yield_mode = "softmin"
    cm.yield_smoother = "powermean"           # bumps δ→1
    assert cm.yield_softness == 1.0
    for eta_ve, eta_pl in [(1.0, 2.0), (5.0, 0.3), (1e25, 1e21)]:
        comb = cm._combine_yield(sympy.Float(eta_ve), sympy.Float(eta_pl))
        val = float(unwrap_expression(comb, mode="nondimensional"))
        assert np.isfinite(val)
        assert val <= min(eta_ve, eta_pl) * (1 + 1e-9)

    # bad mode rejected
    with pytest.raises(ValueError):
        cm.yield_mode = "not_a_mode"
