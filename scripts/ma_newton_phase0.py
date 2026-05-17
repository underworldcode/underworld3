"""Phase 0 (run 2) — Newton/cofactor vs BFO-Picard, measured on the
TRANSPORT MAP (the right yardstick).

Run 1 showed `det(I+H_rec)-g` has a large irreducible floor (the
recovered-Hessian under-estimation = the settled grading-cap root
cause); it is NOT a contraction metric. The efficiency question is:
does Newton reach the SAME fixed-node map in far fewer iterations
than BFO's ~20-25?  Metrics, geometry held FIXED:

  d_k   = max nodal |grad phi_k|              (size of the map)
  Dlt_k = max |grad phi_k - grad phi_{k-1}|   (map increment -> 0)

and, after ONE signed-area-backtracked node move at the end, the
honest deep/near grading (must match between schemes == same fixed
point == regression guard).

  * BFO    : Delta phi = sqrt(...)-2, omega=0.4   (shipped)
  * Newton : div(C_k grad dphi) = g-det(I+H_k), C_k=cof(I+H_k),
             phi <- phi + lam dphi, fixed lam, det(I+H)>0 backtrack
             (the legitimate convexity safeguard only)

No src/ changes. Per-iter print. Writes the design-doc summary.
"""
from __future__ import annotations
import time
import numpy as np
import sympy
import underworld3 as uw
from underworld3.constitutive_models import DiffusionModel
from underworld3.meshing.smoothing import (
    _hessian_recovery_class, _use_direct_solver, _auto_pinned_labels,
    _edge_pairs, _pinned_mask, _tri_cells, _signed_areas)

R_O, R_I, WIDTH, RES, AMP = 1.0, 0.5, 0.12, 16, 8.0
N_BFO, N_NEWT, NEWT_LAM = 40, 15, 0.5

m = uw.meshing.Annulus(radiusOuter=R_O, radiusInner=R_I,
                        cellSize=1.0 / RES, qdegree=3)
r0 = uw.discretisation.MeshVariable("r0n", m, vtype=uw.VarType.SCALAR,
                                    degree=1, continuous=True)
coords0 = np.asarray(m.X.coords).copy()
r0.data[:, 0] = np.sqrt((coords0 ** 2).sum(axis=1))
metric = 1.0 + AMP * sympy.exp(-(((r0.sym[0]) - R_O) / WIDTH) ** 2)
rho_t = np.asarray(uw.function.evaluate(metric, coords0)).reshape(-1)
c = 1.0 / (float(np.mean(1.0 / np.sqrt(rho_t))) ** 2)
g_expr = c / metric
cdim = m.cdim
pin = _auto_pinned_labels(m)
edges = _edge_pairs(m.dm)
is_pin = _pinned_mask(m.dm, tuple(pin))
tris = _tri_cells(m.dm)


def honest_dn_after_move(disp):
    free = ~is_pin
    a0 = _signed_areas(coords0, tris)
    orient = np.sign(np.median(a0)) or 1.0
    scale, new = 1.0, coords0.copy()
    for _ in range(12):
        trial = coords0.copy()
        trial[free] += scale * disp[free]
        if (_signed_areas(trial, tris) * orient).min() > 0.0:
            new = trial
            break
        scale *= 0.5
    v0, v1 = edges[:, 0], edges[:, 1]
    Le = np.linalg.norm(new[v1] - new[v0], axis=1)
    nv = new.shape[0]
    s = np.zeros(nv); cnt = np.zeros(nv)
    for a in (v0, v1):
        np.add.at(s, a, Le); np.add.at(cnt, a, 1.0)
    nl = s / np.maximum(cnt, 1.0)
    rr = np.sqrt((new ** 2).sum(axis=1))
    deep = (rr >= R_I) & (rr < R_I + 0.20)
    near = (rr > R_O - 0.05)
    return float(nl[deep].mean() / nl[near].mean()), scale


def make_gproj(phi):
    gv = uw.discretisation.MeshVariable(
        f"gp_{id(phi)}", m, vtype=uw.VarType.VECTOR, degree=2,
        continuous=True)
    gp = uw.systems.Vector_Projection(m, gv)
    gp.smoothing = 0.0
    _use_direct_solver(gp)
    X = m.CoordinateSystem.X
    gp.uw_function = sympy.Matrix(
        [phi.sym[0].diff(X[i]) for i in range(cdim)]).T
    return gp, gv


def disp_of(gp, gv):
    gp.solve()
    return np.asarray(uw.function.evaluate(
        gv.sym, coords0)).reshape(coords0.shape)


def report(tag, hist_d, hist_dl, dn, sc):
    a = np.array(hist_dl)
    tol = 1e-3 * hist_d[0] if hist_d[0] else 1e-4
    w = np.nonzero(a < tol)[0]
    nconv = int(w[0]) if w.size else -1
    print(f"  {tag}: final max|grad phi|={hist_d[-1]:.4e}  "
          f"iters(Dlt<1e-3·d0)={nconv}  d/n={dn:.4f} (scale={sc:.3f})")
    return nconv


def run_bfo():
    phi = uw.discretisation.MeshVariable(
        "phi_b", m, vtype=uw.VarType.SCALAR, degree=3, continuous=True)
    ps = uw.systems.Poisson(m, phi)
    ps.constitutive_model = uw.constitutive_models.DiffusionModel
    ps.constitutive_model.Parameters.diffusivity = 1.0
    ps.constant_nullspace = True
    _use_direct_solver(ps, singular=True)
    hs = _hessian_recovery_class()(m, phi, degree=2, verbose=False)
    hs.tolerance = 1.0e-6
    _use_direct_solver(hs)
    Hf = hs.u.sym
    Hxx, Hyy = Hf[0], Hf[3]
    Hxy = (Hf[1] + Hf[2]) / 2
    f_src = sympy.sqrt((Hxx - Hyy) ** 2 + 4 * Hxy ** 2 + 4 * g_expr) - 2
    ps.f = sympy.Matrix([[-1.0 * f_src]])
    hs.u.array[...] = 0.0
    gp, gv = make_gproj(phi)
    omega = 0.4
    prev = None
    hd, hl = [], []
    print("  BFO-Picard:")
    for it in range(N_BFO):
        pp = np.asarray(phi.array).copy()
        ps.solve(zero_init_guess=True)
        phi.array[...] = (1 - omega) * pp + omega * np.asarray(phi.array)
        hs.solve()
        d = disp_of(gp, gv)
        dmax = float(np.linalg.norm(d, axis=1).max())
        dl = float(np.abs(d - prev).max()) if prev is not None else dmax
        prev = d
        hd.append(dmax); hl.append(dl)
        if it < 5 or it % 5 == 0 or it == N_BFO - 1:
            print(f"    it {it:2d}  max|gφ|={dmax:.4e}  Δ={dl:.3e}",
                  flush=True)
    dn, sc = honest_dn_after_move(prev)
    return hd, hl, dn, sc


_CK = [None]
PD_EPS = 0.05


class _CofDiff(DiffusionModel):
    def _build_c_tensor(self):
        self._c = _CK[0]


def project_pd(hs):
    """Projection onto convex Hessians: clip eigenvalues of the
    symmetrised (I+H) to >= PD_EPS, write H back. The MA convexity
    safeguard that BFO's +sqrt branch supplies structurally."""
    A = np.asarray(hs.u.array).reshape(-1, 4)
    Hxx, Hyy = A[:, 0], A[:, 3]
    Hxy = 0.5 * (A[:, 1] + A[:, 2])
    a = 1.0 + Hxx
    d = 1.0 + Hyy
    b = Hxy
    tr = a + d
    dsc = np.sqrt(np.maximum((a - d) ** 2 + 4 * b ** 2, 0.0))
    l1 = 0.5 * (tr - dsc)
    l2 = 0.5 * (tr + dsc)
    c1 = np.clip(l1, PD_EPS, None)
    c2 = np.clip(l2, PD_EPS, None)
    # eigenvectors of [[a,b],[b,d]]: rebuild M' = sum ci vi vi^T
    # v for l1: (b, l1-a) normalised (fallback to axis if b~0)
    vx = b.copy()
    vy = l1 - a
    nrm = np.sqrt(vx ** 2 + vy ** 2)
    small = nrm < 1e-14
    vx = np.where(small, 1.0, vx / np.where(small, 1.0, nrm))
    vy = np.where(small, 0.0, vy / np.where(small, 1.0, nrm))
    wx, wy = -vy, vx                       # orthogonal eigenvector
    Mxx = c1 * vx * vx + c2 * wx * wx
    Mxy = c1 * vx * vy + c2 * wx * wy
    Myy = c1 * vy * vy + c2 * wy * wy
    A[:, 0] = Mxx - 1.0
    A[:, 1] = Mxy
    A[:, 2] = Mxy
    A[:, 3] = Myy - 1.0
    hs.u.array[...] = A.reshape(hs.u.array.shape)


def run_newton():
    phi = uw.discretisation.MeshVariable(
        "phi_n", m, vtype=uw.VarType.SCALAR, degree=3, continuous=True)
    dphi = uw.discretisation.MeshVariable(
        "dphi_n", m, vtype=uw.VarType.SCALAR, degree=3, continuous=True)
    hs = _hessian_recovery_class()(m, phi, degree=2, verbose=False)
    hs.tolerance = 1.0e-6
    _use_direct_solver(hs)
    Hf = hs.u.sym
    Hxx, Hyy = Hf[0], Hf[3]
    Hxy = (Hf[1] + Hf[2]) / 2
    M = sympy.Matrix([[1 + Hxx, Hxy], [Hxy, 1 + Hyy]])
    Ck = sympy.Matrix([[1 + Hyy, -Hxy], [-Hxy, 1 + Hxx]])
    detIH = M.det()
    _CK[0] = Ck
    ps = uw.systems.Poisson(m, dphi)
    ps.constitutive_model = _CofDiff
    ps.constant_nullspace = True
    _use_direct_solver(ps, singular=True)
    ps.f = sympy.Matrix([[detIH - g_expr]])      # div(Ck∇dphi)=g-det
    phi.array[...] = 0.0
    hs.u.array[...] = 0.0
    gp, gv = make_gproj(phi)
    prev = None
    hd, hl = [], []
    print(f"  Newton/cofactor (lam={NEWT_LAM}, PD-projected Hessian, "
          f"eps={PD_EPS}):")
    for it in range(N_NEWT):
        hs.solve()
        project_pd(hs)                       # convexity safeguard
        ps.solve(zero_init_guess=True)
        step = np.asarray(dphi.array).copy()
        base = np.asarray(phi.array).copy()
        lam = NEWT_LAM
        phi.array[...] = base + lam * step
        hs.solve()
        project_pd(hs)
        d = disp_of(gp, gv)
        dmax = float(np.linalg.norm(d, axis=1).max())
        dl = float(np.abs(d - prev).max()) if prev is not None else dmax
        prev = d
        hd.append(dmax); hl.append(dl)
        print(f"    it {it:2d}  max|gφ|={dmax:.4e}  Δ={dl:.3e}  "
              f"lam={lam:.3g}", flush=True)
        if dl < 1e-4 and it > 0:
            break
    dn, sc = honest_dn_after_move(prev)
    return hd, hl, dn, sc


t = time.perf_counter()
bd, bl, bdn, bsc = run_bfo()
nd, nl, ndn, nsc = run_newton()
dt = time.perf_counter() - t
print(f"\n=== Phase 0 run2  AMP={AMP} RES={RES}  ({dt:.1f}s) ===")
nb = report("BFO   ", bd, bl, bdn, bsc)
nn = report("NEWT  ", nd, nl, ndn, nsc)
print(f"map agreement: |d/n_BFO - d/n_NEWT| = {abs(bdn-ndn):.4f}  "
      f"(same fixed point if ~0; baseline d/n≈1.71)")
np.savez("/tmp/metric_mesh/ma_newton_phase0.npz",
         bfo_d=bd, bfo_dl=bl, nwt_d=nd, nwt_dl=nl,
         bfo_dn=bdn, nwt_dn=ndn, amp=AMP, res=RES)
print("saved /tmp/metric_mesh/ma_newton_phase0.npz")
