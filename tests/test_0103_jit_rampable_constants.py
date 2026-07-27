"""Live-rampable JIT constants — issue #302.

A constant UWexpression listed in ``solver.constants_manifest`` must be
readable from its ``constants[]`` slot in the compiled kernels, never baked
as a C literal. The old two-phase lowering substituted constants[] slots
with a TOP-LEVEL xreplace before unwrapping, so any constant nested inside
another UWexpression — which includes EVERY ``Parameters.*`` value, since
the setter template-wraps it — was silently folded to a literal: ramping
its ``.sym`` between solves had no effect until a full rebuild.
"""

import numpy as np
import pytest
import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _build():
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(8, 8), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0))
    T = uw.discretisation.MeshVariable("T302", mesh, 1, degree=2)

    k_direct = uw.expression(r"k_d", 1.0, "constant at parameter top level")
    k_nested = uw.expression(r"k_n302", 1.0, "constant nested in a wrapper")
    # Non-constant wrapper: the collection recurses into it and manifests
    # k_nested, but a top-level substitution cannot see inside it — the
    # baked-constant topology from issue #302.
    wrapper = uw.expression(
        r"\eta_{w302}", k_nested * 2.0 + 0.05 * T.sym[0] ** 2, "wrapper")

    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = (
        k_direct * (1.0 + 0.1 * T.sym[0] ** 2) + wrapper)
    poisson.f = 1.0
    poisson.add_dirichlet_bc(0.0, "Top")
    poisson.add_dirichlet_bc(0.0, "Bottom")
    return poisson, T, k_direct, k_nested


def _mean(poisson, T):
    # COLD every time. The diffusivity depends on T, so this is a nonlinear solve and
    # its answer is only pinned to the SNES tolerance — a warm start stops at a
    # different point in that same tolerance ball. This test compares solutions at
    # rtol 1e-12 to check that a constant was read from its slot rather than baked as a
    # literal, so it must hold the initial guess fixed or it measures the warm-start
    # policy instead of the thing it is about.
    poisson.solve(zero_init_guess=True)
    return float(np.asarray(T.data)[:, 0].mean())


def test_manifested_constants_ramp_without_rebuild():
    poisson, T, k_direct, k_nested = _build()
    base = _mean(poisson, T)

    names = [e.name for _i, e in poisson.constants_manifest]
    assert "k_d" in names and "k_n302" in names

    # Ramp each constant via .sym alone — REAL solves, no rebuild between.
    k_direct.sym = 10.0
    direct_ramped = _mean(poisson, T)
    k_direct.sym = 1.0

    k_nested.sym = 10.0
    nested_ramped = _mean(poisson, T)
    k_nested.sym = 1.0

    assert abs(direct_ramped - base) > 1e-8, (
        "top-level constant did not ramp without a rebuild")
    assert abs(nested_ramped - base) > 1e-8, (
        "NESTED constant did not ramp without a rebuild (issue #302)")

    # Ramping back restores the base solution exactly — no hidden state.
    assert np.isclose(_mean(poisson, T), base, rtol=1e-12)


def test_ramped_solution_matches_rebuilt_solution():
    # The live-ramped answer must equal the answer from a full rebuild at
    # the same parameter value — the manifest and the C source agree.
    poisson, T, k_direct, k_nested = _build()
    _mean(poisson, T)
    k_nested.sym = 10.0
    ramped = _mean(poisson, T)

    poisson2, T2, _kd2, k_nested2 = _build()
    k_nested2.sym = 10.0
    rebuilt = _mean(poisson2, T2)

    assert np.isclose(ramped, rebuilt, rtol=1e-10), (
        f"ramped {ramped} != rebuilt {rebuilt}")
