"""Contract tests for ``Mesh.extract_surface`` (codim-1 surface submesh).

The third submesh flavour (alongside ``extract_region``): extract a parent
mesh's boundary stratum as a real ``uw.Mesh`` of one lower topological
dimension, sharing exact vertex positions with the parent.

Covered:
  - loud-fail contract (unknown label / empty stratum raise, no degenerate mesh)
  - geometry (dim = parent.dim-1, cdim preserved, vertices on the parent surface)
  - lineage (parent, registration)
  - bit-exact restrict -> prolongate roundtrip at the shared DOFs
  - on-surface symbolic evaluate

Both a 3D shell (-> 2-manifold in 3-space) and a 2D annulus (-> 1-manifold
loop in 2-space; the case the free-surface mover work consumes) are exercised.
"""

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.tier_a, pytest.mark.level_1]


def _shell():
    return uw.meshing.SphericalShell(
        radiusOuter=1.0, radiusInner=0.5, cellSize=0.25)


def _annulus():
    return uw.meshing.Annulus(
        radiusOuter=1.0, radiusInner=0.5, cellSize=0.2)


# ---------------------------------------------------------------------------
# loud-fail contract
# ---------------------------------------------------------------------------
def test_extract_surface_unknown_label_raises():
    m = _annulus()
    with pytest.raises(ValueError):
        m.extract_surface("Bogus")


def test_extract_surface_empty_stratum_raises():
    m = _annulus()
    # A value that exists as a name but not as a live stratum value.
    with pytest.raises(ValueError):
        m.extract_surface("Upper", label_value=99999)


# ---------------------------------------------------------------------------
# geometry + lineage
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("builder,radius", [(_shell, 1.0), (_annulus, 1.0)])
def test_extract_surface_geometry(builder, radius):
    m = builder()
    surf = m.extract_surface("Upper")

    assert surf.parent is m
    assert surf in m._registered_submeshes
    assert surf.dim == m.dim - 1
    assert surf.cdim == m.cdim

    r = np.linalg.norm(np.asarray(surf.X.coords), axis=1)
    assert np.abs(r - radius).max() < 1.0e-10


# ---------------------------------------------------------------------------
# bit-exact restrict -> prolongate roundtrip
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("builder", [_shell, _annulus])
def test_extract_surface_roundtrip_bit_exact(builder):
    m = builder()
    surf = m.extract_surface("Upper")

    Tp = uw.discretisation.MeshVariable("rt_Tp", m, 1, degree=1)
    Ts = uw.discretisation.MeshVariable("rt_Ts", surf, 1, degree=1)
    Tb = uw.discretisation.MeshVariable("rt_Tb", m, 1, degree=1)

    pc = np.asarray(Tp.coords)
    planted = np.zeros_like(Tp.data)
    planted[:, 0] = pc[:, 0] + 2.0 * pc[:, 1] - (pc[:, 2] if m.cdim == 3 else 0.0)
    Tp.pack_raw_data_to_petsc(planted, sync=True)

    surf.restrict(Tp, Ts)
    surf.prolongate(Ts, Tb)

    nz = np.where(np.abs(Tb.data[:, 0]) > 0)[0]
    assert nz.size >= Ts.data.shape[0]          # touched every surface DOF
    err = np.abs(Tp.data[nz, 0] - Tb.data[nz, 0]).max() if nz.size else 0.0
    assert err < 1.0e-12


# ---------------------------------------------------------------------------
# on-surface symbolic evaluate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("builder", [
    _shell,
    pytest.param(_annulus, marks=pytest.mark.xfail(
        reason="Point-location evaluate on a 1-manifold surface (dim=1, "
        "cdim=2) is not yet supported: the in-cell test's FE element-entity "
        "handling for edge cells is incomplete. Full dim=1 manifold support "
        "(point-location + FE solve) is a tracked follow-up; the free-surface "
        "smoother uses restrict/prolongate + surface-graph adjacency, which "
        "do not need 1D point-location.",
        strict=False)),
])
def test_extract_surface_evaluate(builder):
    m = builder()
    surf = m.extract_surface("Upper")

    Te = uw.discretisation.MeshVariable("ev_Te", surf, 1, degree=1)
    sc = np.asarray(Te.coords)
    planted = np.zeros_like(Te.data)
    planted[:, 0] = sc[:, 0] + 2.0 * sc[:, 1] - (sc[:, 2] if m.cdim == 3 else 0.0)
    Te.pack_raw_data_to_petsc(planted, sync=True)

    q = sc[::7]
    vals = np.asarray(uw.function.evaluate(Te.sym[0], q)).reshape(-1)
    expected = q[:, 0] + 2.0 * q[:, 1] - (q[:, 2] if m.cdim == 3 else 0.0)
    assert np.abs(vals - expected).max() < 1.0e-10
