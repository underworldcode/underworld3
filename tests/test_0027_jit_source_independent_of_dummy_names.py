"""The generated kernel source does not depend on SymPy's Dummy counter (#917).

SymPy names an unnamed ``Dummy`` ``Dummy_<count>``, with ``<count>`` the number of
dummies the process has made so far, and orders dummies by that name as a string.
The JIT graph made unnamed dummies to hide a body's leaves from ``cse``
(``KernelGraph._split_shared``). A build whose dummies straddled a power of ten
ordered them differently from one that did not ("Dummy_1000" sorts before
"Dummy_998"), so ``cse`` shared different subexpressions: the same kernel
mathematically, but a different source and JIT key. (``slot_derivative``'s dummies are
named the same way for uniformity; they are substituted back before anything sorts
them, and no fixture shows an effect there.) How many dummies a process had made
before a build depended on what ran before it. So test_0023's softness test, which
asserts that changing a constants[] value keeps the key, failed in some CI batch
orders (#917).

This rebuilds test_0023's yielding box (the power-mean soft-min with the consistent
tangent) after each softness change. Before each rebuild the Dummy counter is set so
the rebuild crosses a new power of ten partway through: after 5, 10, ... 275 of the
280 or so dummies a rebuild makes, and the test checks that it did cross. Every
rebuild must give the first build's key. On the previous code the keys that differ
come in windows about five dummies wide, about 56 apart, and this grid lands in every
one: crossings after 70, 75, 125, 130, 180 and 235 dummies gave four other keys.
"""
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _yielding_box():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25)
    v = uw.discretisation.MeshVariable("V0027", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P0027", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    cm = stokes.constitutive_model
    cm.Parameters.shear_viscosity_0 = 1.0
    cm.Parameters.yield_stress = 0.8
    cm.yield_mode = "softmin"
    cm.yield_smoother = "powermean"
    cm.yield_softness = 1.0
    stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
    stokes.add_dirichlet_bc((1.0, 0.0), "Top")
    stokes.tolerance = 1.0e-10
    stokes.consistent_jacobian = True
    return stokes


def test_a_rebuild_does_not_depend_on_the_dummy_counter():
    uw.reset_default_model()
    stokes = _yielding_box()
    stokes.solve()
    first = stokes._current_jit_cache_key
    exponent = max(6, len(str(sympy.Dummy._count)) + 1)
    changed = {}
    for k, offset in enumerate(range(5, 280, 5)):
        stokes.constitutive_model.yield_softness = (0.25, 1.0)[k % 2]
        # the counter only rises: each rebuild crosses a power of ten not yet reached
        boundary = 10 ** (exponent + k)
        sympy.Dummy._count = boundary - offset
        # the build solve() makes first after a change (the solver's fast path 1)
        stokes._setup_pointwise_functions()
        assert sympy.Dummy._count > boundary, (
            f"the rebuild made {sympy.Dummy._count - boundary + offset} dummies, fewer "
            f"than {offset}: it no longer crosses the power of ten, so this test cannot "
            f"see #917; lower the offsets")
        if stokes._current_jit_cache_key != first:
            changed[offset] = stokes._current_jit_cache_key
    assert not changed, (
        f"rebuilding the same kernels changed the JIT key from {first} when the build "
        f"crossed a power of ten in SymPy's Dummy counter after these many dummies: "
        f"{changed} (#917)")
