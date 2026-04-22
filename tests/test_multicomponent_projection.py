"""Validation tests for SNES_MultiComponent_Projection.

The multi-component projector solves all N components in a single SNES
solve sharing one DM, replacing the per-component cycling of
SNES_Tensor_Projection. These tests confirm equivalence with the legacy
solvers and check the DM-rebuild economics.
"""

import math

import numpy as np
import pytest
import sympy

import underworld3 as uw


# --- helpers ----------------------------------------------------------------


def _structured_box(res=(8, 8)):
    return uw.meshing.StructuredQuadBox(elementRes=res)


# --- tests ------------------------------------------------------------------


@pytest.mark.level_1
@pytest.mark.tier_a
def test_multicomponent_matches_scalar_at_nc_one():
    """Nc=1 multi-component projection agrees with SNES_Projection."""
    mesh = _structured_box()
    x, y = mesh.X

    # Scalar reference.
    scalar_var = uw.discretisation.MeshVariable(
        "scalar_ref", mesh, 1, vtype=uw.VarType.SCALAR, degree=2, continuous=True
    )
    ref = uw.systems.Projection(mesh, u_Field=scalar_var, degree=2)
    ref.tolerance = 1e-10
    ref.uw_function = sympy.sin(x) * sympy.cos(y)
    ref.smoothing = 0.0
    ref.solve()

    # N=1 multi-component.
    mc = uw.systems.MultiComponent_Projection(mesh, n_components=1, degree=2)
    mc.tolerance = 1e-10
    mc.uw_function = sympy.Matrix([[sympy.sin(x) * sympy.cos(y)]])
    mc.smoothing = 0.0
    mc.solve()

    ref_vals = ref.u.array[:, 0, 0]
    mc_vals = mc.u.array[:, 0, 0]

    assert ref_vals.shape == mc_vals.shape
    diff = np.linalg.norm(ref_vals - mc_vals) / np.linalg.norm(ref_vals)
    assert diff < 1e-7, (
        f"Nc=1 multi-component differs from SNES_Projection by rel-L2 {diff:.3e}"
    )


@pytest.mark.level_1
@pytest.mark.tier_a
def test_multicomponent_matches_tensor_projection_sym2d():
    """Nc=3 multi-component projection agrees with SNES_Tensor_Projection (2D sym)."""
    mesh = _structured_box()
    x, y = mesh.X

    # Legacy: SYM_TENSOR + cycling tensor projector.
    tensor_var = uw.discretisation.MeshVariable(
        "tensor_ref",
        mesh,
        (mesh.dim, mesh.dim),
        vtype=uw.VarType.SYM_TENSOR,
        degree=2,
        continuous=True,
    )
    work_var = uw.discretisation.MeshVariable(
        "tensor_work", mesh, 1, degree=2, continuous=True
    )
    flux_xx = x * y + 1
    flux_xy = x - y
    flux_yy = sympy.sin(x) * sympy.cos(y)
    flux_full = sympy.Matrix([[flux_xx, flux_xy], [flux_xy, flux_yy]])

    legacy = uw.systems.Tensor_Projection(
        mesh, tensor_Field=tensor_var, scalar_Field=work_var, degree=2
    )
    legacy.tolerance = 1e-10
    legacy.uw_function = flux_full
    legacy.smoothing = 0.0
    legacy.solve()

    # New: 3-component multi-component projection (unpacked sym-tensor).
    mc = uw.systems.MultiComponent_Projection(mesh, n_components=3, degree=2)
    mc.tolerance = 1e-10
    mc.uw_function = sympy.Matrix([[flux_xx, flux_xy, flux_yy]])
    mc.smoothing = 0.0
    mc.solve()

    # Compare legacy [i,j] against mc[0,k] for the three indep indices.
    indep = [(0, 0), (0, 1), (1, 1)]
    for k, (i, j) in enumerate(indep):
        legacy_vals = tensor_var.array[:, i, j]
        mc_vals = mc.u.array[:, 0, k]
        rel = np.linalg.norm(legacy_vals - mc_vals) / max(
            np.linalg.norm(legacy_vals), 1e-30
        )
        assert rel < 1e-7, (
            f"Component ({i},{j}) differs rel-L2 {rel:.3e} between "
            f"Tensor_Projection and MultiComponent_Projection"
        )


@pytest.mark.level_1
@pytest.mark.tier_a
def test_multicomponent_dm_builds_scale_with_outer_solves():
    """DM build count scales with outer solves, not Nc × outer solves.

    The legacy SNES_Tensor_Projection performs Nc DM rebuilds per outer
    call (one per scalar component). SNES_MultiComponent_Projection should
    rebuild the DM *at most once* per outer call.
    """
    mesh = _structured_box()
    x, y = mesh.X

    mc = uw.systems.MultiComponent_Projection(mesh, n_components=3, degree=2)

    n_solves = 5
    for k in range(n_solves):
        mc.uw_function = sympy.Matrix(
            [[k * x, k * y, sympy.sin(x + k)]]
        )
        mc.smoothing = 0.0
        mc.solve()

    # Exactly n_solves rebuilds is the current behaviour (is_setup=False on
    # uw_function change). The important invariant is: not Nc × n_solves.
    assert mc._dm_build_count <= n_solves, (
        f"Expected at most {n_solves} DM builds across {n_solves} solves "
        f"(n_components=3), got {mc._dm_build_count}."
    )


@pytest.mark.level_1
@pytest.mark.tier_a
def test_multicomponent_matches_tensor_projection_full2d():
    """Nc=4 multi-component projection agrees with Tensor_Projection (2D non-sym).

    The legacy SNES_Tensor_Projection will cycle all four (i,j) entries —
    the new solver projects them together in a single DM.
    """
    mesh = _structured_box()
    x, y = mesh.X

    tensor_var = uw.discretisation.MeshVariable(
        "tensor_full",
        mesh,
        (mesh.dim, mesh.dim),
        vtype=uw.VarType.TENSOR,
        degree=2,
        continuous=True,
    )
    work_var = uw.discretisation.MeshVariable(
        "tensor_full_work", mesh, 1, degree=2, continuous=True
    )
    flux_full = sympy.Matrix(
        [[x * y, x - y], [sympy.sin(x), sympy.cos(y)]]
    )

    legacy = uw.systems.Tensor_Projection(
        mesh, tensor_Field=tensor_var, scalar_Field=work_var, degree=2
    )
    legacy.tolerance = 1e-10
    legacy.uw_function = flux_full
    legacy.smoothing = 0.0
    legacy.solve()

    mc = uw.systems.MultiComponent_Projection(mesh, n_components=4, degree=2)
    mc.tolerance = 1e-10
    mc.uw_function = sympy.Matrix(
        [[flux_full[0, 0], flux_full[0, 1], flux_full[1, 0], flux_full[1, 1]]]
    )
    mc.smoothing = 0.0
    mc.solve()

    index_map = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for k, (i, j) in enumerate(index_map):
        legacy_vals = tensor_var.array[:, i, j]
        mc_vals = mc.u.array[:, 0, k]
        rel = np.linalg.norm(legacy_vals - mc_vals) / max(
            np.linalg.norm(legacy_vals), 1e-30
        )
        assert rel < 1e-7, (
            f"Full-tensor component ({i},{j}) differs rel-L2 {rel:.3e}"
        )


@pytest.mark.level_1
@pytest.mark.tier_a
@pytest.mark.parametrize("smoothing", [1e-4, 1e-2, 1.0])
def test_multicomponent_matches_scalar_with_smoothing(smoothing):
    """Non-zero smoothing: Nc=1 multi-component matches SNES_Projection.

    Exercises the F1 (Laplacian) pathway and the G2/G3 Jacobian blocks
    that `smoothing=0` skips entirely.
    """
    mesh = _structured_box()
    x, y = mesh.X
    target = sympy.sin(2 * x) * sympy.cos(2 * y) + x * y

    scalar_var = uw.discretisation.MeshVariable(
        f"scalar_ref_sm_{int(smoothing*1e6)}", mesh, 1,
        vtype=uw.VarType.SCALAR, degree=2, continuous=True,
    )
    ref = uw.systems.Projection(mesh, u_Field=scalar_var, degree=2)
    ref.tolerance = 1e-10
    ref.uw_function = target
    ref.smoothing = smoothing
    ref.solve()

    mc = uw.systems.MultiComponent_Projection(mesh, n_components=1, degree=2)
    mc.tolerance = 1e-10
    mc.uw_function = sympy.Matrix([[target]])
    mc.smoothing = smoothing
    mc.solve()

    ref_vals = ref.u.array[:, 0, 0]
    mc_vals = mc.u.array[:, 0, 0]
    rel = np.linalg.norm(ref_vals - mc_vals) / np.linalg.norm(ref_vals)
    assert rel < 1e-6, (
        f"Nc=1 differs from SNES_Projection at smoothing={smoothing}: "
        f"rel-L2 {rel:.3e}"
    )


@pytest.mark.level_1
@pytest.mark.tier_a
@pytest.mark.parametrize("smoothing", [1e-4, 1e-2])
def test_multicomponent_matches_tensor_sym2d_with_smoothing(smoothing):
    """Non-zero smoothing: Nc=3 sym-tensor matches SNES_Tensor_Projection.

    SNES_Tensor_Projection also applies per-component scalar smoothing,
    so this is a direct apples-to-apples check of the Laplacian blocks
    for multiple components.
    """
    mesh = _structured_box()
    x, y = mesh.X
    flux_xx = x * y + 1
    flux_xy = sympy.sin(x) - y
    flux_yy = sympy.cos(x) + x * y
    flux_full = sympy.Matrix([[flux_xx, flux_xy], [flux_xy, flux_yy]])

    tag = int(smoothing * 1e6)
    tensor_var = uw.discretisation.MeshVariable(
        f"tensor_ref_sm_{tag}", mesh, (mesh.dim, mesh.dim),
        vtype=uw.VarType.SYM_TENSOR, degree=2, continuous=True,
    )
    work_var = uw.discretisation.MeshVariable(
        f"tensor_ref_sm_work_{tag}", mesh, 1, degree=2, continuous=True,
    )

    legacy = uw.systems.Tensor_Projection(
        mesh, tensor_Field=tensor_var, scalar_Field=work_var, degree=2,
    )
    legacy.tolerance = 1e-10
    legacy.uw_function = flux_full
    legacy.smoothing = smoothing
    legacy.solve()

    mc = uw.systems.MultiComponent_Projection(mesh, n_components=3, degree=2)
    mc.tolerance = 1e-10
    mc.uw_function = sympy.Matrix([[flux_xx, flux_xy, flux_yy]])
    mc.smoothing = smoothing
    mc.solve()

    indep = [(0, 0), (0, 1), (1, 1)]
    for k, (i, j) in enumerate(indep):
        legacy_vals = tensor_var.array[:, i, j]
        mc_vals = mc.u.array[:, 0, k]
        rel = np.linalg.norm(legacy_vals - mc_vals) / max(
            np.linalg.norm(legacy_vals), 1e-30
        )
        assert rel < 1e-6, (
            f"smoothing={smoothing}, ({i},{j}): rel-L2 {rel:.3e}"
        )


@pytest.mark.level_1
@pytest.mark.tier_a
def test_multicomponent_smoothing_reduces_high_frequencies():
    """Smoking test: large smoothing damps high-frequency content, as expected.

    Projects a high-wavenumber target; the smoothed projection should
    have strictly smaller L2 norm than the unsmoothed one. This is a
    qualitative sanity check that the smoothing term is active (not
    silently dropped).
    """
    mesh = _structured_box(res=(16, 16))
    x, y = mesh.X
    high_freq = sympy.sin(10 * x) * sympy.cos(10 * y)
    target = sympy.Matrix([[high_freq, 2 * high_freq, 0.5 * high_freq]])

    mc_smooth = uw.systems.MultiComponent_Projection(mesh, n_components=3, degree=2)
    mc_smooth.tolerance = 1e-10
    mc_smooth.uw_function = target
    mc_smooth.smoothing = 1e-1
    mc_smooth.solve()

    mc_raw = uw.systems.MultiComponent_Projection(mesh, n_components=3, degree=2)
    mc_raw.tolerance = 1e-10
    mc_raw.uw_function = target
    mc_raw.smoothing = 0.0
    mc_raw.solve()

    for k in range(3):
        norm_smooth = np.linalg.norm(mc_smooth.u.array[:, 0, k])
        norm_raw = np.linalg.norm(mc_raw.u.array[:, 0, k])
        assert norm_smooth < norm_raw, (
            f"Component {k}: smoothed norm {norm_smooth:.4e} is not less "
            f"than unsmoothed {norm_raw:.4e} — smoothing appears inactive."
        )
