"""Locks uw.meshing.fault_metric_tensor + the supplied-tensor r-adapt path.

The builder produces the analytic, Eulerian normal-aligned anisotropic
metric tensor M(x) = base[I + (R^2-1) sum_i exp(-(d_i/W)^2) n_i n_i^T] for
refining a thin band ACROSS one or more faults. These tests pin:

* tensor structure — at a fault the across-fault eigenvalue is base*R^2 and
  the along-fault eigenvalue is base; far away M = base*I;
* the supplied-tensor mover (smooth_mesh_interior method="anisotropic",
  metric=M) centres TWO close faults on their lines (|offset| < one refined
  cell) with the topology preserved (r-adapt, not h-adapt);
* the builder accepts Surface objects equivalently to raw segments;
* 3D raises NotImplementedError (the mover is 2D-only).
"""
import numpy as np
import sympy
import pytest

import underworld3 as uw
from underworld3.meshing import smoothing as _sm
from underworld3.meshing.smoothing import _tri_cells, _signed_areas

_C = np.array([0.5, 0.5])
_TH = np.radians(40.0)
_U = np.array([np.cos(_TH), np.sin(_TH)])
_N = np.array([-np.sin(_TH), np.cos(_TH)])
_L, _GAP = 0.40, 0.06
_SEG = [(_C + s * (_GAP / 2) * _N - (_L / 2) * _U,
         _C + s * (_GAP / 2) * _N + (_L / 2) * _U) for s in (+1.0, -1.0)]
_SEG3 = [np.array([list(a) + [0.0], list(b) + [0.0]]) for (a, b) in _SEG]


def _box(cs=1.0 / 40):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=cs, qdegree=3)


def _eval_M(M, pt):
    out = np.empty((2, 2))
    for i in range(2):
        for j in range(2):
            e = M[i, j]
            if getattr(e, "free_symbols", None):
                out[i, j] = float(np.asarray(
                    uw.function.evaluate(e, np.array([pt]))).reshape(-1)[0])
            else:
                out[i, j] = float(e)
    return out


@pytest.mark.tier_a
@pytest.mark.level_1
def test_fault_metric_tensor_structure():
    m = _box()
    R, W = 3.0, 0.01
    M = uw.meshing.fault_metric_tensor(m, [_SEG3[0]], refinement=R, width=W)
    assert M.shape == (2, 2)
    # ON the fault (segment midpoint): across eigenvalue base*R^2, along base
    on = _eval_M(M, _SEG[0][0] * 0.5 + _SEG[0][1] * 0.5)
    w = np.sort(np.linalg.eigvalsh(on))
    assert abs(w[0] - 1.0) < 1e-6                  # along-fault = base
    assert abs(w[1] - R ** 2) < 1e-3               # across-fault = base*R^2
    # the large-eigenvalue direction is the fault normal
    _, V = np.linalg.eigh(on)
    assert abs(abs(np.dot(V[:, 1], _N)) - 1.0) < 1e-3
    # FAR from the fault: M -> base * I
    far = _eval_M(M, np.array([0.1, 0.9]))
    assert np.allclose(far, np.eye(2), atol=1e-6)


@pytest.mark.tier_a
@pytest.mark.level_1
def test_fault_metric_tensor_from_surface_matches_segments():
    m = _box()
    surfs = [uw.meshing.Surface(f"flt{i}", m, _SEG3[i]) for i in range(2)]
    Ms = uw.meshing.fault_metric_tensor(m, surfs, refinement=3.0, width=0.01)
    Mr = uw.meshing.fault_metric_tensor(m, _SEG3, refinement=3.0, width=0.01)
    for pt in (np.array([0.5, 0.5]), np.array([0.55, 0.52]), _SEG[0][0]):
        assert np.allclose(_eval_M(Ms, pt), _eval_M(Mr, pt), atol=1e-9)


@pytest.mark.tier_a
@pytest.mark.level_1
def test_fault_metric_tensor_centres_two_close_faults():
    m = _box(cs=1.0 / 40)
    n0 = len(np.asarray(m.X.coords))
    nc0 = len(_tri_cells(m.dm))
    M = uw.meshing.fault_metric_tensor(m, _SEG3, refinement=3.0, width=0.002)
    _sm.smooth_mesh_interior(
        m, metric=M, method="anisotropic", boundary_slip=False,
        method_kwargs=dict(n_outer=14, relax=0.4))
    Xa = np.asarray(m.X.coords)
    tris = _tri_cells(m.dm)
    # topology preserved (r-adapt): same vertex / cell count, no inversion
    a = _signed_areas(Xa, tris)
    assert len(Xa) == n0 and len(tris) == nc0
    assert int((np.sign(a) != np.sign(np.median(a))).sum()) == 0
    # both bands centred on their lines (|offset| < one refined cell h0/R)
    tc = (Xa - _C) @ _N
    al = (Xa - _C) @ _U
    refined_cell = (1.0 / 40) / 3.0
    for f in (+0.03, -0.03):
        band = (np.abs(tc - f) < 0.012) & (np.abs(al) < _L / 2)
        assert band.sum() > 20
        assert abs(float(tc[band].mean()) - f) < refined_cell


@pytest.mark.tier_a
@pytest.mark.level_1
def test_fault_metric_tensor_non2d_raises():
    # The builder is 2D-only and checks mesh.cdim first, before touching any
    # mesh data — so the contract (non-2D -> NotImplementedError) is locked
    # deterministically with a minimal cdim stand-in. (A real 3D box is
    # avoided here: constructing one is fragile under prior-test PETSc
    # coordinate-space state, an environment issue unrelated to this guard.)
    class _Mesh3D:
        cdim = 3
    seg = np.array([[0.2, 0.5, 0.5], [0.8, 0.5, 0.5]])
    with pytest.raises(NotImplementedError):
        uw.meshing.fault_metric_tensor(_Mesh3D(), [seg],
                                       refinement=3.0, width=0.05)


# ---------------------------------------------------------------------------
# fault_comb_metric — scalar comb for a uniform-ish band on the MA mover
# ---------------------------------------------------------------------------
def _evalrho(rho, pts):
    return np.asarray(uw.function.evaluate(rho, pts)).reshape(-1)


@pytest.mark.tier_a
@pytest.mark.level_1
def test_fault_comb_metric_teeth_structure():
    m = _box()
    dx = 0.01
    rho = uw.meshing.fault_comb_metric(m, [_SEG3[0]], cell_size=dx, n_across=4)
    # sample along the across-fault normal through the segment midpoint
    mid = 0.5 * (_SEG[0][0] + _SEG[0][1])
    at_teeth = _evalrho(rho, np.array([mid + k * dx * _N for k in (0, 1, 2)]))
    at_valleys = _evalrho(rho, np.array([mid + (k + 0.5) * dx * _N
                                         for k in (0, 1)]))
    far = _evalrho(rho, np.array([mid + 6 * dx * _N]))
    assert np.all(at_teeth > 1.5)             # teeth refine
    assert np.all(at_teeth[:2] > at_valleys * 1.3)   # teeth > valleys between
    assert far[0] == pytest.approx(1.0, abs=1e-6)    # base far away


@pytest.mark.tier_a
@pytest.mark.level_1
def test_fault_comb_metric_two_faults_band():
    m = _box(cs=1.0 / 60)
    n0 = len(np.asarray(m.X.coords)); nc0 = len(_tri_cells(m.dm))
    dx = 0.006
    rho = uw.meshing.fault_comb_metric(m, _SEG3, cell_size=dx, n_across=4)
    _sm.smooth_mesh_interior(m, metric=rho, method="ma", boundary_slip=False,
                             method_kwargs=dict(n_outer=1, n_picard=25))
    Xa = np.asarray(m.X.coords); tris = _tri_cells(m.dm)
    a = _signed_areas(Xa, tris)
    assert len(Xa) == n0 and len(tris) == nc0        # topology preserved
    assert int((np.sign(a) != np.sign(np.median(a))).sum()) == 0
    cc = Xa[tris].mean(axis=1)
    tc = (cc - _C) @ _N; al = (cc - _C) @ _U
    p = Xa[tris]
    edges = np.stack([np.linalg.norm(p[:, 1] - p[:, 0], axis=1),
                      np.linalg.norm(p[:, 2] - p[:, 1], axis=1),
                      np.linalg.norm(p[:, 0] - p[:, 2], axis=1)], axis=1)
    short = edges.min(axis=1)
    D = 2 * dx
    for f in (+0.03, -0.03):
        band = (np.abs(tc - f) < D) & (np.abs(al) < _L / 2)
        assert band.sum() > 15
        # band is refined (cells well below h0) and centred on the fault
        assert np.median(short[band]) < (1.0 / 60) * 0.8
        assert abs(float(tc[band & (np.abs(tc - f) < dx * 0.6)].mean()) - f) < dx


@pytest.mark.tier_a
@pytest.mark.level_1
def test_fault_comb_metric_curved():
    # a short circular arc (polyline) — bands must follow the curve
    m = _box(cs=1.0 / 50)
    n0 = len(np.asarray(m.X.coords)); nc0 = len(_tri_cells(m.dm))
    phis = np.linspace(np.radians(25), np.radians(65), 6)
    arc = np.array([0.2, 0.2]) + 0.42 * np.column_stack(
        [np.cos(phis), np.sin(phis)])
    rho = uw.meshing.fault_comb_metric(m, [arc], cell_size=0.008, n_across=4)
    _sm.smooth_mesh_interior(m, metric=rho, method="ma", boundary_slip=False,
                             method_kwargs=dict(n_outer=1, n_picard=25))
    Xa = np.asarray(m.X.coords); tris = _tri_cells(m.dm)
    a = _signed_areas(Xa, tris)
    assert len(Xa) == n0 and len(tris) == nc0
    assert int((np.sign(a) != np.sign(np.median(a))).sum()) == 0
    # cells near the arc are refined below h0

    def adist(P):
        d = np.full(P.shape[0], np.inf)
        for k in range(len(arc) - 1):
            ab = arc[k + 1] - arc[k]
            t = np.clip(((P - arc[k]) @ ab) / (ab @ ab), 0, 1)
            d = np.minimum(d, np.linalg.norm(P - (arc[k] + np.outer(t, ab)), axis=1))
        return d
    cc = Xa[tris].mean(axis=1)
    p = Xa[tris]
    short = np.stack([np.linalg.norm(p[:, 1] - p[:, 0], axis=1),
                      np.linalg.norm(p[:, 2] - p[:, 1], axis=1),
                      np.linalg.norm(p[:, 0] - p[:, 2], axis=1)], axis=1).min(axis=1)
    neararc = adist(cc) < 0.008
    assert neararc.sum() > 20
    assert np.median(short[neararc]) < (1.0 / 50) * 0.85   # refined along curve


@pytest.mark.tier_a
@pytest.mark.level_1
def test_fault_comb_metric_non2d_raises():
    class _Mesh3D:
        cdim = 3
    seg = np.array([[0.2, 0.5, 0.5], [0.8, 0.5, 0.5]])
    with pytest.raises(NotImplementedError):
        uw.meshing.fault_comb_metric(_Mesh3D(), [seg], cell_size=0.01)


# ---------------------------------------------------------------------------
# fault_metric facade — one intent (cell_size + n_across), per-method object
# ---------------------------------------------------------------------------
@pytest.mark.tier_a
@pytest.mark.level_1
def test_fault_metric_facade_ma_matches_comb():
    m = _box()
    rho_f = uw.meshing.fault_metric(m, _SEG3, method="ma",
                                    cell_size=0.006, n_across=4)
    rho_d = uw.meshing.fault_comb_metric(m, _SEG3, cell_size=0.006, n_across=4)
    assert not isinstance(rho_f, sympy.MatrixBase)        # scalar density
    pts = np.array([[0.5, 0.5], [0.55, 0.52], [0.1, 0.9]])
    assert np.allclose(_evalrho(rho_f, pts), _evalrho(rho_d, pts), atol=1e-9)


@pytest.mark.tier_a
@pytest.mark.level_1
def test_fault_metric_facade_anisotropic_is_tensor():
    m = _box()
    M = uw.meshing.fault_metric(m, _SEG3, method="anisotropic",
                                cell_size=0.006, n_across=4)
    assert isinstance(M, sympy.MatrixBase) and M.shape == (2, 2)
    # refines (not the bare identity) near a fault
    on = _eval_M(M, _SEG[0][0] * 0.5 + _SEG[0][1] * 0.5)
    assert np.max(np.linalg.eigvalsh(on)) > 1.5


@pytest.mark.tier_a
@pytest.mark.level_1
def test_fault_metric_facade_adapt_h2_field():
    m = _box(cs=1.0 / 40)
    dx, hf = 0.006, 1.0 / 40
    metric = uw.meshing.fault_metric(m, _SEG3, method="adapt",
                                     cell_size=dx, n_across=4, h_far=hf,
                                     name="fm_adapt")
    assert isinstance(metric, uw.discretisation.MeshVariable)
    P = np.asarray(metric.coords)
    # nearest node to a fault midpoint -> metric ~ 1/dx^2; far corner -> ~1/hf^2
    mid = 0.5 * (_SEG[0][0] + _SEG[0][1])
    i_near = np.argmin(np.linalg.norm(P - mid, axis=1))
    i_far = np.argmin(np.linalg.norm(P - np.array([0.05, 0.95]), axis=1))
    h_near = 1.0 / np.sqrt(metric.data[i_near, 0])
    h_farv = 1.0 / np.sqrt(metric.data[i_far, 0])
    assert h_near < 1.5 * dx                  # band node ~ cell_size
    assert h_farv > 0.7 * hf                  # far node ~ h_far


@pytest.mark.tier_a
@pytest.mark.level_1
def test_fault_metric_facade_unknown_method_raises():
    m = _box()
    with pytest.raises(ValueError):
        uw.meshing.fault_metric(m, _SEG3, method="bogus", cell_size=0.006)


# ---------------------------------------------------------------------------
# compose_metrics + smooth_mesh_interior accepts a list of metrics
# ---------------------------------------------------------------------------
@pytest.mark.tier_a
@pytest.mark.level_1
def test_compose_metrics_single_passthrough():
    m = _box()
    rho = uw.meshing.fault_comb_metric(m, [_SEG3[0]], cell_size=0.01)
    out = uw.meshing.compose_metrics([rho])
    pts = np.array([[0.5, 0.5], [0.6, 0.4]])
    assert np.allclose(_evalrho(out, pts), _evalrho(rho, pts), atol=1e-9)


@pytest.mark.tier_a
@pytest.mark.level_1
def test_compose_metrics_max_equal_weights():
    m = _box()
    r1 = uw.meshing.fault_comb_metric(m, [_SEG3[0]], cell_size=0.01)
    r2 = uw.meshing.fault_comb_metric(m, [_SEG3[1]], cell_size=0.01)
    composed = uw.meshing.compose_metrics([r1, r2])
    plain_max = 1 + sympy.Max(r1 - 1, r2 - 1)
    pts = np.array([[0.5, 0.5], _SEG[0][0], _SEG[1][1], [0.1, 0.9]])
    assert np.allclose(_evalrho(composed, pts), _evalrho(plain_max, pts),
                       atol=1e-9)


@pytest.mark.tier_a
@pytest.mark.level_1
def test_compose_metrics_weighted_excess():
    m = _box()
    r = uw.meshing.fault_comb_metric(m, [_SEG3[0]], cell_size=0.01)
    # weight 3 should scale the excess (rho-1) by 3 in this single-item case
    w3 = uw.meshing.compose_metrics([(r, 3.0)])
    pts = np.array([0.5 * (_SEG[0][0] + _SEG[0][1])])     # ON the fault
    rho_at = _evalrho(r, pts)[0]
    w3_at = _evalrho(w3, pts)[0]
    assert abs((w3_at - 1.0) - 3.0 * (rho_at - 1.0)) < 1e-9


@pytest.mark.tier_a
@pytest.mark.level_1
def test_compose_metrics_rejects_tensor():
    m = _box()
    M = uw.meshing.fault_metric_tensor(m, [_SEG3[0]], refinement=3.0, width=0.01)
    with pytest.raises(ValueError):
        uw.meshing.compose_metrics([M])


@pytest.mark.tier_a
@pytest.mark.level_1
def test_smooth_mesh_interior_list_of_metrics():
    # smooth_mesh_interior accepts a list and composes internally
    m = _box(cs=1.0 / 50)
    n0 = len(np.asarray(m.X.coords)); nc0 = len(_tri_cells(m.dm))
    r1 = uw.meshing.fault_comb_metric(m, [_SEG3[0]], cell_size=0.008)
    r2 = uw.meshing.fault_comb_metric(m, [_SEG3[1]], cell_size=0.008)
    _sm.smooth_mesh_interior(m, metric=[(r1, 1.0), (r2, 1.0)], method="ma",
                             boundary_slip=False,
                             method_kwargs=dict(n_outer=1, n_picard=25))
    Xa = np.asarray(m.X.coords); tris = _tri_cells(m.dm)
    a = _signed_areas(Xa, tris)
    assert len(Xa) == n0 and len(tris) == nc0
    assert int((np.sign(a) != np.sign(np.median(a))).sum()) == 0
