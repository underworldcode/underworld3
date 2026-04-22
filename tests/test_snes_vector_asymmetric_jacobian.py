"""SNES_Vector Jacobian regression test — asymmetric F1.

Guards against the `[fc, df, gc, dg]` vs `[fc, gc, df, dg]` layout bug
documented in `docs/developer/subsystems/petsc-jacobian-layout.md`.

Every current production consumer of `SNES_Vector` uses an F1 that is
symmetric under the trial-index swap `(gc, dg) ↔ (dg, gc)` — e.g.
strain-rate-based smoothing `Unknowns.E`, or deviatoric Stokes stress —
which hides a Jacobian-layout bug. This test constructs a consumer
whose F1 is *asymmetric* (raw gradient `Unknowns.L`, not symmetrised)
so the bug, if present, shows up immediately.

With the explicit-index construction in place, components given
identical targets should produce identical solutions.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.utilities._api_tools import Template


class _RawLSmoothingProjection(uw.systems.Vector_Projection):
    """SNES_Vector_Projection subclass with non-symmetric L-based smoothing.

    Standard ``Vector_Projection.F1`` uses ``self.Unknowns.E`` (symmetric
    strain rate) which hides Jacobian-layout bugs. Here we use the raw
    Jacobian ``L`` directly — no symmetrisation — so the Jacobian
    ``∂F1[i,j]/∂L[k,l] = smoothing·δ_ik·δ_jl`` has no `(k,l)↔(l,k)`
    symmetry, and an incorrect `[fc, df, gc, dg]` layout shows up as
    spurious cross-component coupling.
    """

    F1 = Template(
        r"F1_{\mathrm{raw}-L}",
        lambda self: self.smoothing * self.Unknowns.L,
        "Raw gradient smoothing, not symmetrised.",
    )


@pytest.mark.level_1
@pytest.mark.tier_a
@pytest.mark.parametrize("smoothing", [1e-4, 1e-2, 1e-1])
def test_snes_vector_asymmetric_f1_components_decouple(smoothing):
    """Identical targets in each component must give identical solutions.

    Cross-component coupling introduced by an incorrect Jacobian layout
    would show up as different component values at non-zero smoothing.
    """
    mesh = uw.meshing.StructuredQuadBox(elementRes=(8, 8))
    x, y = mesh.X
    same_target = sympy.sin(x) * sympy.cos(y)

    vp = _RawLSmoothingProjection(mesh, degree=2)
    vp.tolerance = 1e-10
    vp.petsc_options["snes_type"] = "ksponly"
    vp.petsc_options["ksp_type"] = "preonly"
    vp.petsc_options["pc_type"] = "lu"
    vp.uw_function = sympy.Matrix([[same_target, same_target]])
    vp.smoothing = smoothing
    vp.solve()

    arr = np.asarray(vp.u.array)
    u0 = arr[:, 0, 0]
    u1 = arr[:, 0, 1]
    rel = np.linalg.norm(u0 - u1) / max(np.linalg.norm(u0), 1e-30)
    assert rel < 1e-8, (
        f"Asymmetric-F1 SNES_Vector at smoothing={smoothing}: components "
        f"with identical targets differ by rel-L2 {rel:.3e} — suggests a "
        f"Jacobian layout bug. See docs/developer/subsystems/"
        f"petsc-jacobian-layout.md."
    )


@pytest.mark.level_1
@pytest.mark.tier_a
@pytest.mark.parametrize("smoothing", [1e-4, 1e-2, 1e-1])
def test_snes_vector_asymmetric_f1_matches_multicomponent(smoothing):
    """SNES_Vector with raw-L F1 must agree with SNES_MultiComponent_Projection.

    The multi-component projector builds its Jacobians by explicit
    per-entry construction and is independently tested. If SNES_Vector's
    layout is wrong, the two will disagree.
    """
    mesh = uw.meshing.StructuredQuadBox(elementRes=(8, 8))
    x, y = mesh.X

    # Two *different* targets — more sensitive than identical targets.
    target_0 = sympy.sin(x) * sympy.cos(y)
    target_1 = sympy.cos(x) - 2 * y

    vp = _RawLSmoothingProjection(mesh, degree=2)
    vp.tolerance = 1e-10
    vp.petsc_options["snes_type"] = "ksponly"
    vp.petsc_options["ksp_type"] = "preonly"
    vp.petsc_options["pc_type"] = "lu"
    vp.uw_function = sympy.Matrix([[target_0, target_1]])
    vp.smoothing = smoothing
    vp.solve()
    vp_arr = np.asarray(vp.u.array)

    mc = uw.systems.MultiComponent_Projection(mesh, n_components=2, degree=2)
    mc.tolerance = 1e-10
    mc.petsc_options["snes_type"] = "ksponly"
    mc.petsc_options["ksp_type"] = "preonly"
    mc.petsc_options["pc_type"] = "lu"
    mc.uw_function = sympy.Matrix([[target_0, target_1]])
    mc.smoothing = smoothing
    mc.solve()
    mc_arr = np.asarray(mc.u.array)

    for k in range(2):
        rel = np.linalg.norm(vp_arr[:, 0, k] - mc_arr[:, 0, k]) / max(
            np.linalg.norm(mc_arr[:, 0, k]), 1e-30
        )
        assert rel < 1e-8, (
            f"smoothing={smoothing}, component {k}: SNES_Vector differs "
            f"from SNES_MultiComponent_Projection by rel-L2 {rel:.3e}."
        )
