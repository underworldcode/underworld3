"""(3) anisotropic tensor mover — radial-feature validation.

Canonical comparison point (res-16 Annulus, radial Gaussian
peaked at r=R_O, AMP=8) — the SAME problem as
scripts/ma_radial_anisotropy.py so the numbers line up with the
settled MA results. Diagnostics are ANISOTROPY-AWARE (radial /
tangential edge split vs radius + minA/meanA), NOT the
anisotropy-blind d/n, and overlay the exact 1-D radial OT.

Success criterion (per the kickoff brief): (3) does NOT beat the
fixed node-count grading cap. It earns its keep by IMPROVING cell
alignment / quality — i.e. a LESS degenerate rim layer (higher
minA/meanA) than the isotropic MA, with comparable radial grading
and the tangential edges not frozen/blown-out. Grids are rendered
(the project norm is to SHOW, not just quote scalars).
"""
from __future__ import annotations
import sys
import time
import numpy as np
import sympy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import underworld3 as uw
from underworld3.meshing.smoothing import (
    _winslow_elliptic, _winslow_anisotropic, _winslow_spring,
    _edge_pairs, _auto_pinned_labels, _tri_cells, _signed_areas)

R_O, R_I, WIDTH, RES, AMP = 1.0, 0.5, 0.12, 16, 8.0
# Peak radius: R_O = the boundary-peaked pathology (documented:
# every method over-collapses the pinned rim; the SEPARABLE case
# where the explicit 1-D OT is the right tool). An interior peak
# (e.g. R_O-2.5*WIDTH) gives the feature room on both sides — the
# honest place to judge (3)'s alignment/quality on a radial
# feature.  `python aniso_validate_radial.py 0.70`
PEAK = float(sys.argv[1]) if len(sys.argv) > 1 else R_O
TAG = f"peak{PEAK:.2f}".replace(".", "p")


def case(tag):
    m = uw.meshing.Annulus(radiusOuter=R_O, radiusInner=R_I,
                           cellSize=1.0 / RES, qdegree=3)
    r0 = uw.discretisation.MeshVariable(
        f"r0_{tag}_{TAG}", m, vtype=uw.VarType.SCALAR, degree=1,
        continuous=True)
    X0 = np.asarray(m.X.coords)
    r0.data[:, 0] = np.sqrt((X0 ** 2).sum(axis=1))
    f = 1.0 + AMP * sympy.exp(-(((r0.sym[0]) - PEAK) / WIDTH) ** 2)
    return m, f


def split_edges(coords, edges):
    p0, p1 = coords[edges[:, 0]], coords[edges[:, 1]]
    mid = 0.5 * (p0 + p1)
    rmid = np.linalg.norm(mid, axis=1)
    rhat = mid / np.maximum(rmid, 1e-30)[:, None]
    ev = p1 - p0
    L = np.linalg.norm(ev, axis=1)
    rad_frac = np.abs((ev * rhat).sum(axis=1)) / np.maximum(L, 1e-30)
    return rmid, L, rad_frac > np.cos(np.pi / 4)


m0, _ = case("u")
edges = _edge_pairs(m0.dm)
tris = _tri_cells(m0.dm)
X0 = np.asarray(m0.X.coords).copy()
dr0 = (R_O - R_I) / RES

# exact 1-D radial OT profile (ground truth, radial direction)
s = np.linspace(R_I, R_O, 200_000)
rho = 1.0 + AMP * np.exp(-(((s - PEAK) / WIDTH) ** 2))
dens = rho * s
mcum = np.concatenate(
    [[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * (s[1] - s[0]))])
mcum /= mcum[-1]
rn = np.interp(np.linspace(0, 1, RES + 1), mcum, s)
ot_rmid = 0.5 * (rn[1:] + rn[:-1])
ot_dr = np.diff(rn)

bins = np.linspace(R_I, R_O, 13)
bc = 0.5 * (bins[1:] + bins[:-1])


def radial_profile(coords):
    rmid, L, is_rad = split_edges(coords, edges)
    out_r, out_t = [], []
    for i in range(len(bins) - 1):
        mb = (rmid >= bins[i]) & (rmid < bins[i + 1])
        out_r.append(L[mb & is_rad].mean() if (mb & is_rad).any()
                     else np.nan)
        out_t.append(L[mb & ~is_rad].mean() if (mb & ~is_rad).any()
                     else np.nan)
    return np.array(out_r), np.array(out_t)


def run(name, fn):
    m, f = case(name)
    pin = _auto_pinned_labels(m)
    t = time.perf_counter()
    fn(m, f, pin)
    dt = time.perf_counter() - t
    return np.asarray(m.X.coords).copy(), dt


runs = [("undeformed", X0, 0.0)]
X, dt = run("ma", lambda m, f, p: _winslow_elliptic(
    m, f, p, False, phi_degree=2))
runs.append(("MA (isotropic)", X, dt))
X, dt = run("aniso", lambda m, f, p: _winslow_anisotropic(
    m, f, p, True, phi_degree=2))   # robust defaults (cap 2)
runs.append(("anisotropic (3)", X, dt))
X, dt = run("spring", lambda m, f, p: _winslow_spring(
    m, f, p, False))
runs.append(("spring", X, dt))

print(f"\nAnnulus res-{RES}, radial Gaussian peak r={PEAK:.2f}, "
      f"AMP={AMP:g}")
print(f"{'mesh':>16} | rim radial Δ | vs undef | MIN radial Δ "
      f"| vs undef | minA/meanA | time")
print("-" * 92)
und_rr = und_minr = None
prof = {}
for name, Xc, dt in runs:
    pr, pt = radial_profile(Xc)
    prof[name] = (pr, pt)
    rr = pr[np.isfinite(pr)][-1]
    rmid, L, is_rad = split_edges(Xc, edges)
    minr = L[is_rad].min()
    A = np.abs(_signed_areas(Xc, tris))
    minA = A.min() / A.mean()
    if name == "undeformed":
        und_rr, und_minr = rr, minr
    print(f"{name:>16} | {rr:12.4f} | {rr/und_rr:8.3f} | "
          f"{minr:12.5f} | {minr/und_minr:8.3f} | {minA:10.4f} "
          f"| {dt:5.2f}s")
print(f"exact 1-D radial OT  rim/dr0 = {ot_dr.min()/dr0:.3f} "
      f"(deep/near = {ot_dr.max()/ot_dr.min():.2f})")

# ---- figure: profiles + grids -------------------------------------
fig = plt.figure(figsize=(19, 9.5))
gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.15])
axr = fig.add_subplot(gs[0, 0:2])
axt = fig.add_subplot(gs[0, 2:4])
cols = {"undeformed": "k", "MA (isotropic)": "#c0392b",
        "anisotropic (3)": "#1f4e8c", "spring": "#2a9d8f"}
for name, _, _ in runs:
    pr, pt = prof[name]
    axr.plot(bc, pr, "o-", lw=1.6, ms=4, color=cols[name], label=name)
    axt.plot(bc, pt, "o-", lw=1.6, ms=4, color=cols[name], label=name)
axr.plot(ot_rmid, ot_dr, "k--", lw=1.4, label="exact 1-D radial OT")
axr.set_title("RADIAL edge length vs radius")
axt.set_title("TANGENTIAL edge length vs radius")
for a in (axr, axt):
    a.set_xlabel("radius"); a.set_ylabel("edge length")
    a.axvline(R_O, color="grey", ls=":", lw=0.8)
    a.legend(fontsize=8); a.grid(alpha=0.3)
for k, (name, Xc, _) in enumerate(runs):
    a = fig.add_subplot(gs[1, k])
    a.triplot(mtri.Triangulation(Xc[:, 0], Xc[:, 1], tris),
              lw=0.35, color=cols[name])
    th = np.linspace(0, 2 * np.pi, 300)
    a.plot(R_O * np.cos(th), R_O * np.sin(th), "grey", lw=0.6,
           ls=":")
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.set_title(name, fontsize=11)
fig.suptitle(f"(3) anisotropic mover vs isotropic MA — radial "
             f"Gaussian peak r={PEAK:.2f}, res-{RES} AMP={AMP:g}  "
             f"(success = fewer slivers / better alignment, "
             f"NOT a bigger d/n)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
_out = f"/tmp/metric_mesh/aniso_radial_{TAG}.png"
fig.savefig(_out, dpi=125)
print(f"\nsaved {_out}")
