"""Guards on UWexpression construction and constitutive Parameter assignment.

1. ``sympy.Max/Min(UWexpression, number)`` must construct symbolically instead of
   crashing on Float internals (issue #415 — ``is_comparable`` advertised the wrapped
   contents' comparability, so sympy attempted an immediate numeric comparison).
2. Reading a constitutive Parameter and assigning it straight back must be a no-op,
   not a self-nesting that recurses to stack death on the next ``.value`` access
   (issue #447); the composite form ``P.x = P.x * 2`` must snapshot, not cycle.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw


@pytest.mark.level_1
@pytest.mark.tier_a
def test_max_min_of_expression_and_number_construct():
    """Issue #415: Max/Min(UWexpression, number) stays symbolic, resolves on unwrap."""
    k = uw.expression(r"k_{415}", 1.0, "constant")
    clamped = sympy.Max(k, 0.5)              # crashed with AttributeError '_prec'
    floored = sympy.Min(k, 2.0)
    assert clamped.has(k) and floored.has(k)
    # The lazy contract: comparison resolves once the expression is unwrapped.
    from underworld3.function.expressions import unwrap_expression
    assert float(unwrap_expression(clamped)) == 1.0    # max(1.0, 0.5)
    assert float(unwrap_expression(floored)) == 1.0    # min(1.0, 2.0)
    k.sym = 0.1
    assert float(unwrap_expression(clamped)) == 0.5    # max(0.1, 0.5) — still live


def _viscoplastic_stokes():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.5
    )
    v = uw.discretisation.MeshVariable("v447", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p447", mesh, 1, degree=1, continuous=True)
    st = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    st.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    st.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    st.constitutive_model.Parameters.yield_stress = 10.0
    return st


@pytest.mark.level_1
@pytest.mark.tier_a
def test_parameter_readback_is_noop():
    """Issue #447: p = Params.x; Params.x = p must not nest the container in itself."""
    st = _viscoplastic_stokes()
    params = st.constitutive_model.Parameters
    saved = params.yield_stress
    before = params.yield_stress.sym
    params.yield_stress = saved              # recursed to stack death before the fix
    assert params.yield_stress.sym == before
    # .value walks the contents — this is where the recursion previously died.
    float(params.yield_stress.value)
    # The flux still builds (smooth_max over the parameter was the reported site).
    st.constitutive_model.flux


@pytest.mark.level_1
@pytest.mark.tier_a
def test_parameter_composite_self_reference_snapshots():
    """Issue #447 (composite): Params.x = Params.x * 2 snapshots the current value."""
    st = _viscoplastic_stokes()
    params = st.constitutive_model.Parameters
    base = float(params.yield_stress.value)
    params.yield_stress = params.yield_stress * 2
    assert np.isclose(float(params.yield_stress.value), 2 * base)
    float(params.yield_stress.value)         # no recursion
