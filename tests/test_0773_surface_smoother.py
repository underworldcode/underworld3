"""Tests for ``smooth_surface_field`` — graph-Laplacian / Taubin low-pass of a
scalar field over a (surface) mesh's vertex-edge graph.

The free-surface height-field smoother: it must attenuate high-wavenumber
(sawtooth) content while preserving the smooth, long-wavelength field and its
mean (volume). Exercised on a codim-1 1-manifold surface (annulus ``Upper``
loop), where an FE solve is not yet available — the graph operator needs only
edge connectivity.
"""

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.tier_a, pytest.mark.level_1]


def _surface_with_modes(cell_size=0.04, low_k=2, high_k=20, high_amp=0.3):
    ann = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=cell_size)
    surf = ann.extract_surface("Upper")
    h = uw.discretisation.MeshVariable("h", surf, 1, degree=1)
    c = np.asarray(h.coords)
    th = np.arctan2(c[:, 1], c[:, 0])
    h.data[:, 0] = np.cos(low_k * th) + high_amp * np.cos(high_k * th)
    return surf, h, th


def _amp(x, th, k):
    return float((x * np.cos(k * th)).mean() * 2.0)


def test_taubin_preserves_low_attenuates_high():
    surf, h, th = _surface_with_modes()
    b_low, b_high = _amp(h.data[:, 0], th, 2), _amp(h.data[:, 0], th, 20)

    uw.meshing.smooth_surface_field(h, n_iters=80, alpha=0.6, taubin=True)

    a_low, a_high = _amp(h.data[:, 0], th, 2), _amp(h.data[:, 0], th, 20)

    # low (signal) mode preserved; high (sawtooth) mode strongly attenuated
    assert a_low / b_low > 0.95, f"low mode not preserved: {a_low/b_low:.3f}"
    assert abs(a_high / b_high) < 0.3, f"high mode not killed: {a_high/b_high:.3f}"


def test_constant_field_preserved_exactly():
    # The graph-Laplacian constant mode (eigenvalue 0) is invariant under any
    # blend: a constant field must come back unchanged (no spurious DC shift).
    ann = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.06)
    surf = ann.extract_surface("Upper")
    h = uw.discretisation.MeshVariable("hc", surf, 1, degree=1)
    h.data[:, 0] = 0.37
    uw.meshing.smooth_surface_field(h, n_iters=50, alpha=0.6, taubin=True)
    assert np.abs(h.data[:, 0] - 0.37).max() < 1.0e-12


def test_taubin_beats_plain_laplacian_on_signal_preservation():
    # Plain Laplacian damps everything (shrinks the signal); Taubin preserves
    # the passband. Same iterations/alpha — Taubin must keep the low mode better.
    surf_t, h_t, th = _surface_with_modes()
    b_low = _amp(h_t.data[:, 0], th, 2)
    uw.meshing.smooth_surface_field(h_t, n_iters=40, alpha=0.6, taubin=True)
    taubin_low_kept = _amp(h_t.data[:, 0], th, 2) / b_low

    surf_l, h_l, _ = _surface_with_modes()
    uw.meshing.smooth_surface_field(h_l, n_iters=40, alpha=0.6, taubin=False)
    laplacian_low_kept = _amp(h_l.data[:, 0], th, 2) / b_low

    assert taubin_low_kept > laplacian_low_kept + 0.03
    assert taubin_low_kept > 0.97


def test_smoother_works_on_2d_surface():
    # The graph operator is dimension-agnostic: it works on a 2-manifold
    # (3D shell -> sphere surface) exactly as on the 1-manifold annulus loop.
    shell = uw.meshing.SphericalShell(
        radiusOuter=1.0, radiusInner=0.5, cellSize=0.15)
    surf = shell.extract_surface("Upper")          # dim=2, cdim=3
    h = uw.discretisation.MeshVariable("h2d", surf, 1, degree=1)
    c = np.asarray(h.coords)
    z = c[:, 2]
    rough = 0.3 * np.cos(40 * np.arctan2(c[:, 1], c[:, 0])) \
        * np.sin(30 * np.arccos(np.clip(z, -1, 1)))
    h.data[:, 0] = z + rough

    low0 = (h.data[:, 0] * z).sum() / (z * z).sum()
    res0 = np.std(h.data[:, 0] - low0 * z)

    uw.meshing.smooth_surface_field(h, n_iters=60, alpha=0.6, taubin=True)

    low1 = (h.data[:, 0] * z).sum() / (z * z).sum()
    res1 = np.std(h.data[:, 0] - low1 * z)

    assert abs(low1 / low0 - 1.0) < 0.05           # smooth (l=1) mode preserved
    assert res1 / res0 < 0.5                        # rough content attenuated


def test_more_iterations_attenuate_more():
    surf, h12, th = _surface_with_modes()
    b_high = _amp(h12.data[:, 0], th, 20)
    uw.meshing.smooth_surface_field(h12, n_iters=12, alpha=0.6, taubin=True)
    killed_12 = 1.0 - abs(_amp(h12.data[:, 0], th, 20) / b_high)

    _, h60, _ = _surface_with_modes()
    uw.meshing.smooth_surface_field(h60, n_iters=60, alpha=0.6, taubin=True)
    killed_60 = 1.0 - abs(_amp(h60.data[:, 0], th, 20) / b_high)

    assert killed_60 > killed_12
