"""Anisotropy-aware truth: measure RADIAL vs TANGENTIAL edge length
vs radius for the MA-redistributed Annulus (res-16, AMP=8), P1/P2/P3
+ undeformed. The reported deep/near ≈1.71 is a per-node mean of ALL
incident edges — it averages the collapsed radial edges with the
≈-unchanged tangential ones and so HIDES the true radial
compression the user sees in the grid. Overlay the exact 1D radial
OT spacing for reference. Saves /tmp/metric_mesh/ma_radial_profile.png.
"""
from __future__ import annotations
import numpy as np
import sympy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import underworld3 as uw
from underworld3.meshing.smoothing import (
    _winslow_elliptic, _edge_pairs, _auto_pinned_labels)

R_O, R_I, WIDTH, RES, AMP = 1.0, 0.5, 0.12, 16, 8.0


def case(tag):
    m = uw.meshing.Annulus(radiusOuter=R_O, radiusInner=R_I,
                           cellSize=1.0 / RES, qdegree=3)
    r0 = uw.discretisation.MeshVariable(
        f"r0_{tag}", m, vtype=uw.VarType.SCALAR,
        degree=1, continuous=True)
    X0 = np.asarray(m.X.coords)
    r0.data[:, 0] = np.sqrt((X0 ** 2).sum(axis=1))
    f = 1.0 + AMP * sympy.exp(-(((r0.sym[0]) - R_O) / WIDTH) ** 2)
    return m, f


def split_edges(coords, edges):
    """Return (r_mid, Lrad, Ltan) for each edge: classify by the
    angle of the edge vector to the local radial direction."""
    p0, p1 = coords[edges[:, 0]], coords[edges[:, 1]]
    mid = 0.5 * (p0 + p1)
    rmid = np.linalg.norm(mid, axis=1)
    rhat = mid / np.maximum(rmid, 1e-30)[:, None]
    ev = p1 - p0
    L = np.linalg.norm(ev, axis=1)
    radial_frac = np.abs((ev * rhat).sum(axis=1)) / np.maximum(L, 1e-30)
    is_rad = radial_frac > np.cos(np.pi / 4)      # within 45° of r̂
    return rmid, L, is_rad


m0, _ = case("u")
edges = _edge_pairs(m0.dm)
X0 = np.asarray(m0.X.coords).copy()
dr0 = (R_O - R_I) / RES                            # nominal radial Δ

# exact 1-D radial OT profile (the ground truth for the radial dir)
s = np.linspace(R_I, R_O, 200_000)
rho = 1.0 + AMP * np.exp(-(((s - R_O) / WIDTH) ** 2))
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
        m = (rmid >= bins[i]) & (rmid < bins[i + 1])
        out_r.append(L[m & is_rad].mean() if (m & is_rad).any()
                     else np.nan)
        out_t.append(L[m & ~is_rad].mean() if (m & ~is_rad).any()
                     else np.nan)
    return np.array(out_r), np.array(out_t)


fig, (axr, axt) = plt.subplots(1, 2, figsize=(15, 5.6))
runs = [("undeformed", X0)]
for p in (1, 2, 3):
    m, f = case(f"p{p}")
    pin = _auto_pinned_labels(m)
    _winslow_elliptic(m, f, pin, False, phi_degree=p)
    runs.append((f"P{p}", np.asarray(m.X.coords).copy()))

cols = {"undeformed": "k", "P1": "#2a9d8f",
        "P2": "#1f4e8c", "P3": "#c0392b"}
from underworld3.meshing.smoothing import _tri_cells, _signed_areas
tris = _tri_cells(m0.dm)
und_rad = None
und_minrad = None
print(f"{'mesh':>10} | band-mean radial Δ (rim) | vs undef | "
      f"MIN radial Δ | vs undef | minA/meanA")
print("-" * 82)
for name, X in runs:
    pr, pt = radial_profile(X)
    axr.plot(bc, pr, "o-", lw=1.6, ms=4, color=cols[name], label=name)
    axt.plot(bc, pt, "o-", lw=1.6, ms=4, color=cols[name], label=name)
    rr = pr[np.isfinite(pr)][-1]
    # global MIN radial edge length (the thinnest sliver anywhere)
    rmid, L, is_rad = split_edges(X, edges)
    minrad = L[is_rad].min()
    A = np.abs(_signed_areas(X, tris))
    minA = A.min() / A.mean()
    if name == "undeformed":
        und_rad, und_minrad = rr, minrad
    print(f"{name:>10} | {rr:24.4f} | {rr/und_rad:8.3f} | "
          f"{minrad:12.5f} | {minrad/und_minrad:8.3f} | {minA:8.4f}")

axr.plot(ot_rmid, ot_dr, "k--", lw=1.4,
         label="exact 1-D radial OT")
axr.set_title("RADIAL edge length vs radius")
axt.set_title("TANGENTIAL edge length vs radius")
for a in (axr, axt):
    a.set_xlabel("radius"); a.set_ylabel("edge length")
    a.axvline(R_O, color="grey", ls=":", lw=0.8)
    a.legend(fontsize=9); a.grid(alpha=0.3)
fig.suptitle(f"Annulus res-{RES}, AMP={AMP:g}: the d/n≈1.71 metric "
             f"averages these two — radial collapses, tangential ~frozen",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("/tmp/metric_mesh/ma_radial_profile.png", dpi=130)
print("\nsaved /tmp/metric_mesh/ma_radial_profile.png")
print(f"exact 1-D radial OT near/dr0 (AMP={AMP:g}) = "
      f"{ot_dr.min()/dr0:.3f}")
