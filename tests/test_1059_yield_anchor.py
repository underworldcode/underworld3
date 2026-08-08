"""Which point of the soft-min yield law is pinned to exact Min (`yield_anchor`).

A soft-min rounds the yield corner by delta, and rounding a corner moves the whole
curve -- so the family has one free constant, fixed by choosing which point the
smoothed law must reproduce exactly. That choice decides which SIDE of the exact law
the curve sits on, and for a homotopy the side is the property that matters: an entry
problem that is WEAKER than the sharp problem is not an easier version of it.

The historical anchor pins the unyielded limit f -> 0. Both families then sit BELOW
exact Min at and above the yield point -- the sqrt family undershoots for f < 2 (a
third under-stressed at nominal yield), and the power mean degenerates to the p-norm,
whose viscosity falls away towards zero as delta grows.

Anchoring at the yield point f = 1 instead puts both families on or above exact Min
everywhere. The sqrt family gets there with the offset delta/2; the power mean needs no
offset, because averaging its two terms makes it a generalised mean and a generalised
mean returns the common value when its arguments are equal.

The tests that pin the new behaviour also assert the OLD anchor FAILS it, so they
cannot pass against a no-op implementation.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.function.expressions import unwrap


def _model(smoother="sqrt"):
    m = uw.meshing.UnstructuredSimplexBox(minCoords=(0,0), maxCoords=(1,1), cellSize=0.5, qdegree=2)
    v = uw.discretisation.MeshVariable('U', m, m.dim, degree=2)
    p = uw.discretisation.MeshVariable('P', m, 1, degree=1)
    s = uw.systems.Stokes(m, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    c = s.constitutive_model; c.yield_mode='softmin'; c.yield_smoother=smoother
    return c

def eta(c, f, d):
    """eta_eff/eta_ve at overstress f (eta_ve=1 => eta_pl=1/f)."""
    c.yield_softness = d
    e = c._combine_yield(sympy.Float(1.0), sympy.Float(1.0/f) if f>0 else sympy.oo)
    return float(sympy.N(unwrap(e, keep_constants=False)))

FS = (0.25, 0.5, 1.0, 2.0, 4.0, 10.0)

# delta is NOT the same parameter in the two families: for the power mean
# s = 1/(delta + 0.001), so delta = 1 is already the harmonic mean and the law sits a
# factor 2^delta from Min. Each family is therefore probed over its own useful range.
DELTAS = {"sqrt": (1.0, 16.0, 64.0), "powermean": (0.1, 0.25, 1.0)}

@pytest.mark.level_1
@pytest.mark.tier_a
def test_delta_zero_is_exact_min_for_both_anchors():
    """Only the sqrt family is exactly Min at delta=0; the power mean's sharpness
    saturates at s=1000, so it is checked against its own 2^0.001 bound."""
    c = _model("sqrt")
    for anchor in ("onset", "yield"):
        c.yield_anchor = anchor
        for f in FS:
            assert abs(eta(c, f, 0.0) - 1.0/max(1.0, f)) < 1e-12, (anchor, f)
    c = _model("powermean")
    for anchor in ("onset", "yield"):
        c.yield_anchor = anchor
        for f in FS:
            # The bound is attained, not approached: 2^0.001 - 1 = 6.93e-4 exactly.
            assert abs(eta(c, f, 0.0)/(1.0/max(1.0, f)) - 1.0) < 1e-3, (anchor, f)

@pytest.mark.level_1
@pytest.mark.tier_a
@pytest.mark.parametrize("smoother", ("sqrt", "powermean"))
def test_yield_anchor_keeps_onset_stress_exact(smoother):
    """tau/tau_y = f*eta/eta_ve = 1 at f=1 for every delta. The onset anchor does NOT."""
    c = _model(smoother); c.yield_anchor = "yield"
    for d in DELTAS[smoother]:
        assert abs(eta(c, 1.0, d) - 1.0) < 1e-10, d
    # The defect being fixed: sqrt is a third under-stressed at nominal yield, and the
    # p-norm the power mean falls back to is HALF the correct viscosity at delta=1.
    c.yield_anchor = "onset"
    assert abs(eta(c, 1.0, DELTAS[smoother][-1]) - 1.0) > 0.3

@pytest.mark.level_1
@pytest.mark.tier_a
@pytest.mark.parametrize("smoother", ("sqrt", "powermean"))
def test_yield_anchor_never_falls_below_exact_min(smoother):
    """A genuine approach from ABOVE. The onset anchor undershoots."""
    c = _model(smoother); c.yield_anchor = "yield"
    for d in DELTAS[smoother]:
        for f in FS:
            assert eta(c, f, d) >= 1.0/max(1.0, f) - 1e-12, (d, f)
    c.yield_anchor = "onset"
    assert eta(c, 0.5, DELTAS[smoother][-1]) < 1.0 - 1e-3

@pytest.mark.level_1
@pytest.mark.tier_a
def test_yield_anchor_removes_the_power_mean_degenerate_basin():
    """The p-norm's viscosity collapses towards zero as delta grows -- the 'spurious
    fully-yielded fixed point'. It is an artefact of the missing 1/2: the power mean is
    bounded below by exact Min for every delta, and tends to the geometric mean."""
    c = _model("powermean")
    c.yield_anchor = "onset"
    collapse = [eta(c, 10.0, d) for d in (1.0, 4.0, 16.0)]
    assert collapse[-1] < 0.01 * 0.1                    # 0.1 is exact Min at f=10
    assert collapse[0] > collapse[1] > collapse[2]      # monotone collapse
    c.yield_anchor = "yield"
    bounded = [eta(c, 10.0, d) for d in (1.0, 4.0, 16.0)]
    assert all(v >= 0.1 for v in bounded)
    assert max(bounded) < np.sqrt(1.0 * 0.1) * 1.001    # geometric mean is the ceiling

@pytest.mark.level_1
@pytest.mark.tier_a
def test_anchor_rejects_nonsense_and_defaults_to_onset():
    c = _model()
    assert c.yield_anchor == "onset"
    with pytest.raises(ValueError):
        c.yield_anchor = "middle"
