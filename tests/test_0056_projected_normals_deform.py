"""Per-boundary normals (mesh.boundary_normal) must track a DEFORMED surface
and must NOT be contaminated by a neighbouring boundary at a corner.

Regression for the deformed-normal bug: the global mesh.Gamma_P1 point-evaluates
the mesh.Gamma base scalars, whose C-code maps to the PETSc facet normal
petsc_n[] — defined ONLY inside surface-integral kernels. A general point
evaluation fell back to the coordinate value, so the "normal" came out
≈ (x,y)/r = RADIAL (or the box coordinate) no matter how the mesh deformed.
Every Nitsche/penalty free-slip BC then constrained v·(stale normal) on a
deformed surface, leaking throughflow ∝ surface tilt.

The fix is mesh.boundary_normal(label): assemble the EXACT PETSc facet normals
(dm.computeCellGeometryFVM) area-weighted onto the vertices of ONLY that
boundary's faces. Smooth boundary → smooth deformed normal; a corner shared
with another boundary keeps each boundary's own (one-sided) normal instead of
averaging across the discontinuity.

Covers (a) a curved free surface (annulus Upper), and (b) a CARTESIAN free
surface (box Top deformed, vertical side walls) — the corner case.

Run: pixi run python -m pytest tests/test_0056_projected_normals_deform.py -v
"""

import pytest
import numpy as np
import underworld3 as uw

pytestmark = [
    pytest.mark.level_1,
    pytest.mark.tier_a,
    # Geometric assembly is verified against a rank-local reference; run serial.
    pytest.mark.skipif(uw.mpi.size > 1, reason="serial geometric-assembly check"),
]


def _facet_vertex_normals(mesh, label_name, label_value):
    """Ground-truth: area-weighted outward vertex normals from the EXACT PETSc
    facet normals of one boundary's faces, computed independently of the code
    under test."""
    dm = mesh.dm
    cdim = mesh.cdim
    coords = np.asarray(mesh.X.coords)
    accum = np.zeros_like(coords)
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    vS, vE = dm.getDepthStratum(0)
    xc = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, cdim)
    fS, fE = dm.getHeightStratum(1)
    label = dm.getLabel(label_name)
    pis = label.getStratumIS(label_value)
    for f in pis.getIndices():
        if not (fS <= int(f) < fE) or dm.getSupportSize(int(f)) != 1:
            continue
        area, cent, nrm = dm.computeCellGeometryFVM(int(f))
        nrm = np.asarray(nrm)[:cdim]
        cell = dm.getSupport(int(f))[0]
        _, ccent, _ = dm.computeCellGeometryFVM(cell)
        if np.dot(nrm, np.asarray(cent)[:cdim] - np.asarray(ccent)[:cdim]) < 0:
            nrm = -nrm
        for v in dm.getCone(int(f)):
            if vS <= v < vE:
                _, idx = tree.query(xc[v - vS], k=1)
                accum[idx] += area * nrm
    mag = np.sqrt((accum ** 2).sum(1))
    good = mag > 1e-30
    accum[good] /= mag[good, None]
    return accum, good


def _angle_deg(a, b):
    return np.degrees(np.arccos(np.clip(np.abs((a * b).sum(1)), -1.0, 1.0)))


def _bvalue(mesh, name):
    for b in mesh.boundaries:
        if b.name == name:
            return b.value
    raise KeyError(name)


def _eval_boundary_normal(mesh, name, pts):
    bn = mesh.boundary_normal(name)
    gx = np.asarray(uw.function.evaluate(bn[0], pts)).flatten()
    gy = np.asarray(uw.function.evaluate(bn[1], pts)).flatten()
    return np.column_stack([gx, gy])


def test_boundary_normal_tracks_deformed_annulus():
    r_i, r_o, cs = 0.5, 1.0, 0.2
    mesh = uw.meshing.Annulus(radiusOuter=r_o, radiusInner=r_i, cellSize=cs, qdegree=3)
    X = np.asarray(mesh.X.coords).copy()
    R = np.sqrt((X ** 2).sum(1)); TH = np.arctan2(X[:, 1], X[:, 0])
    surf = np.abs(R - r_o) < 0.5 * cs
    s_idx = np.where(surf)[0]

    # undeformed: boundary_normal(Upper) ≈ radial
    n0 = _eval_boundary_normal(mesh, "Upper", X[s_idx])
    rhat0 = X[s_idx] / R[s_idx, None]
    assert _angle_deg(n0, rhat0).max() < 3.0

    # deform outer surface in mode-3 (12%)
    Xd = X.copy()
    Xd[surf] *= (1.0 + 0.12 * np.cos(3 * TH[surf]))[:, None]
    mesh.deform(Xd, dt=1.0)
    Xn = np.asarray(mesh.X.coords); Rn = np.sqrt((Xn ** 2).sum(1))
    rhat_d = Xn[s_idx] / Rn[s_idx, None]

    ref, good = _facet_vertex_normals(mesh, "Upper", _bvalue(mesh, "Upper"))
    ref_s, good_s = ref[s_idx], good[s_idx]
    tilt = _angle_deg(ref_s[good_s], rhat_d[good_s])
    assert tilt.max() > 8.0, "test setup: deformation should tilt the surface"

    nd = _eval_boundary_normal(mesh, "Upper", Xn[s_idx])
    err = _angle_deg(nd[good_s], ref_s[good_s])
    assert err.max() < 5.0, (
        f"boundary_normal(Upper) must track the deformed facet normal; "
        f"max error {err.max():.1f}° (radial bug ~ {tilt.max():.1f}°)")


def test_boundary_normal_cartesian_free_surface_corner():
    """Cartesian free surface (box Top deformed, vertical side walls). The
    Top normal must follow the deformed top and must NOT be averaged with the
    side-wall normals at the corners."""
    cs = 1.0 / 8
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=cs, qdegree=3)
    X = np.asarray(mesh.X.coords).copy()
    top = np.abs(X[:, 1] - 1.0) < 1e-9
    t_idx = np.where(top)[0]

    # deform the top as a smooth bump h(x) = 0.1 sin(pi x); sides stay vertical
    Xd = X.copy()
    Xd[top, 1] += 0.1 * np.sin(np.pi * X[top, 0])
    mesh.deform(Xd, dt=1.0)
    Xn = np.asarray(mesh.X.coords)

    ref, good = _facet_vertex_normals(mesh, "Top", _bvalue(mesh, "Top"))
    ref_t, good_t = ref[t_idx], good[t_idx]

    nd = _eval_boundary_normal(mesh, "Top", Xn[t_idx])
    err = _angle_deg(nd[good_t], ref_t[good_t])
    assert err.max() < 5.0, (
        f"boundary_normal(Top) must track the deformed top; max {err.max():.1f}°")

    # the deformed top genuinely tilts (∂h/∂x ≠ 0 away from the crest)
    yhat = np.tile([0.0, 1.0], (good_t.sum(), 1))
    assert _angle_deg(ref_t[good_t], yhat).max() > 8.0

    # CORNER CHECK: the top-corner vertices (x=0, x=1) must carry the TOP
    # normal, NOT a 45° average with the vertical walls. ∂h/∂x at x=0,1 is
    # ±0.1π, so the true top normal there is tilted ~17° from vertical; it must
    # equal the one-sided top facet normal (the reference), which `err` already
    # bounds. Assert the corner normal is not the 45° wall-average:
    xc = Xn[t_idx, 0]
    corner = (np.abs(xc) < 1e-9) | (np.abs(xc - 1.0) < 1e-9)
    if corner.any():
        wall45 = np.tile([np.sqrt(0.5), np.sqrt(0.5)], (corner.sum(), 1))
        # corner top normal must be FAR from the (±0.707,0.707) wall-blend
        assert _angle_deg(nd[corner], wall45).min() > 10.0, (
            "top corner normal looks averaged with the side wall (45°)")
