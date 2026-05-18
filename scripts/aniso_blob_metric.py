"""What does the (3) TARGET metric look like for the non-separable
blob, and does the realised mesh match it?

The metric is GRADIENT-based: M = (1/h0²)[I + β ĝĝᵀ(|∇ρ|/gref)²],
eigen-clamped (shipped default aniso_cap=2, β=200). For a Gaussian
blob ρ=1+AMP·exp(-|X-P|²/W²):
  * centre  (∇ρ=0)  → isotropic, coarsest (clamp floor)
  * flank   (|∇ρ| max @ |X-P|≈W/√2) → finest, anisotropic, short
                                       axis pointing at P
  * far     (∇ρ→0)  → isotropic, coarsest
So it resolves the blob EDGE, not the CORE.

Panels:
  A  ρ contours + the desired-cell ellipses (the TARGET metric,
     EXACTLY the construction the shipped mover uses).
  B  the realised method="anisotropic" mesh (zoom), with the
     |∇ρ|-max ring drawn — do the small/flat cells sit on it?
  C  mean edge length & cell-aspect vs distance d=|X-P|: expect a
     DIP at d≈W (flank) with the centre (d→0) and far field NOT
     refined — the quantitative form of the question.
"""
from __future__ import annotations
import numpy as np
import sympy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
import underworld3 as uw
from underworld3.meshing.smoothing import (
    _winslow_anisotropic, _auto_pinned_labels, _tri_cells,
    _edge_pairs, _signed_areas)

R_O, R_I, RES, AMP = 1.0, 0.5, 24, 8.0
PX, PY, W = 0.78, 0.0, 0.10
BETA, ACAP = 200.0, 2.0          # shipped defaults


def mk(tag):
    m = uw.meshing.Annulus(radiusOuter=R_O, radiusInner=R_I,
                           cellSize=1.0 / RES, qdegree=3)
    Xv = uw.discretisation.MeshVariable(
        f"X0_{tag}", m, vtype=uw.VarType.VECTOR, degree=1,
        continuous=True)
    X0 = np.asarray(m.X.coords)
    Xv.data[:, 0] = X0[:, 0]
    Xv.data[:, 1] = X0[:, 1]
    x, y = Xv.sym[0], Xv.sym[1]
    f = 1.0 + AMP * sympy.exp(
        -(((x - PX) ** 2 + (y - PY) ** 2) / W ** 2))
    return m, f, X0.copy()


def grad_blob(x, y):
    d2 = (x - PX) ** 2 + (y - PY) ** 2
    e = np.exp(-d2 / W ** 2)
    rho = 1.0 + AMP * e
    gx = AMP * e * (-2.0 * (x - PX) / W ** 2)
    gy = AMP * e * (-2.0 * (y - PY) / W ** 2)
    return rho, np.stack([gx, gy], axis=1)


m0, _, X0u = mk("u")
tris = _tri_cells(m0.dm)
edges = _edge_pairs(m0.dm)
ep = _edge_pairs(m0.dm)
h0 = float(np.linalg.norm(
    X0u[ep[:, 1]] - X0u[ep[:, 0]], axis=1).mean())   # mover's h0


def build_metric(grad):
    """EXACTLY the shipped mover's D construction (β, aniso_cap,
    h0, g_eps) — returns desired-cell semi-axes + eigenvectors."""
    n = grad.shape[0]
    gn = np.linalg.norm(grad, axis=1)
    g_eps = 1.0e-9
    gmax = gn.max()
    gref = gmax if gmax > g_eps else 1.0
    base = 1.0 / h0 ** 2
    lam_lo = 1.0 / h0 ** 2
    lam_hi = 1.0 / (h0 / np.sqrt(ACAP)) ** 2
    H = np.empty((n, 2))
    V = np.zeros((n, 2, 2))
    for i in range(n):
        g, gni = grad[i], gn[i]
        if gni > g_eps and gmax > g_eps:
            gh = g / gni
            M = base * (np.eye(2) + BETA * (gni / gref) ** 2
                        * np.outer(gh, gh))
        else:
            M = base * np.eye(2)
        w, Vec = np.linalg.eigh(M)
        w = np.clip(w, lam_lo, lam_hi)
        H[i] = 1.0 / np.sqrt(w)        # desired spacing per axis
        V[i] = Vec
    return H, V, gn


# --- A: target metric ellipse field on a clean sample grid -------
gx = np.linspace(PX - 0.34, PX + 0.34, 26)
gy = np.linspace(PY - 0.34, PY + 0.34, 26)
GX, GY = np.meshgrid(gx, gy)
inside = (np.hypot(GX, GY) > R_I + 0.02) & (np.hypot(GX, GY)
                                            < R_O - 0.02)
Xs = np.stack([GX[inside], GY[inside]], axis=1)
rho_s, grad_s = grad_blob(Xs[:, 0], Xs[:, 1])
H, V, gn = build_metric(grad_s)

# --- run the shipped mover ---------------------------------------
m, f, X0 = mk("an")
_winslow_anisotropic(m, f, _auto_pinned_labels(m), True)
Xan = np.asarray(m.X.coords).copy()

# --- C: profiles vs d = |X-P| ------------------------------------
def edge_profile(X):
    p0, p1 = X[edges[:, 0]], X[edges[:, 1]]
    mid = 0.5 * (p0 + p1)
    d = np.hypot(mid[:, 0] - PX, mid[:, 1] - PY)
    L = np.linalg.norm(p1 - p0, axis=1)
    bins = np.linspace(0.0, 6 * W, 16)
    bc = 0.5 * (bins[1:] + bins[:-1])
    out = [L[(d >= bins[i]) & (d < bins[i + 1])].mean()
           if ((d >= bins[i]) & (d < bins[i + 1])).any() else np.nan
           for i in range(len(bins) - 1)]
    return bc, np.array(out)

bc, La = edge_profile(Xan)
_, Lu = edge_profile(X0u)

# per-cell aspect ratio (longest/shortest edge) vs d, realised
ca = Xan[tris[:, 0]]
cb = Xan[tris[:, 1]]
cc = Xan[tris[:, 2]]
cen = (ca + cb + cc) / 3.0
dc = np.hypot(cen[:, 0] - PX, cen[:, 1] - PY)
e01 = np.linalg.norm(cb - ca, axis=1)
e12 = np.linalg.norm(cc - cb, axis=1)
e20 = np.linalg.norm(ca - cc, axis=1)
asp = np.maximum.reduce([e01, e12, e20]) / np.maximum(
    np.minimum.reduce([e01, e12, e20]), 1e-30)
abins = np.linspace(0.0, 6 * W, 16)
abc = 0.5 * (abins[1:] + abins[:-1])
aspm = np.array([asp[(dc >= abins[i]) & (dc < abins[i + 1])].mean()
                 if ((dc >= abins[i]) & (dc < abins[i + 1])).any()
                 else np.nan for i in range(len(abins) - 1)])

r_gmax = W / np.sqrt(2.0)          # radius of max |∇ρ|

fig = plt.figure(figsize=(20, 6.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05])

# Panel A — target metric
aA = fig.add_subplot(gs[0, 0])
rr = np.hypot(X0u[:, 0], X0u[:, 1])
aA.tricontourf(mtri.Triangulation(X0u[:, 0], X0u[:, 1], tris),
               1.0 + AMP * np.exp(
                   -(((X0u[:, 0] - PX) ** 2
                      + (X0u[:, 1] - PY) ** 2) / W ** 2)),
               levels=18, cmap="Blues", alpha=0.45)
pats, cv = [], []
disp_sc = 0.9
for i in range(Xs.shape[0]):
    ang = np.degrees(np.arctan2(V[i, 1, 1], V[i, 0, 1]))
    pats.append(Ellipse((Xs[i, 0], Xs[i, 1]),
                         width=2 * disp_sc * H[i, 1],
                         height=2 * disp_sc * H[i, 0], angle=ang))
    cv.append(H[i].max() / max(H[i].min(), 1e-30))
pc = PatchCollection(pats, facecolor="none", lw=0.8,
                     edgecolor="#c0392b")
aA.add_collection(pc)
th = np.linspace(0, 2 * np.pi, 200)
aA.plot(PX + r_gmax * np.cos(th), PY + r_gmax * np.sin(th),
        "k--", lw=1.2, label="|∇ρ| max ring")
aA.plot(PX, PY, "k+", ms=12, mew=2, label="blob centre (∇ρ=0)")
aA.set_xlim(PX - 0.34, PX + 0.34)
aA.set_ylim(PY - 0.34, PY + 0.34)
aA.set_aspect("equal")
aA.set_xticks([])
aA.set_yticks([])
aA.legend(fontsize=8, loc="upper right")
aA.set_title("A  TARGET metric — desired cells\n(small/flat on the "
             "edge ring; circular+coarse at centre & far)",
             fontsize=10)

# Panel B — realised mesh (zoom)
aB = fig.add_subplot(gs[0, 1])
aB.triplot(mtri.Triangulation(Xan[:, 0], Xan[:, 1], tris),
           lw=0.5, color="#1f4e8c")
aB.plot(PX + r_gmax * np.cos(th), PY + r_gmax * np.sin(th),
        "k--", lw=1.2)
aB.plot(PX, PY, "k+", ms=12, mew=2)
aB.set_xlim(PX - 0.34, PX + 0.34)
aB.set_ylim(PY - 0.34, PY + 0.34)
aB.set_aspect("equal")
aB.set_xticks([])
aB.set_yticks([])
aB.set_title("B  realised method=\"anisotropic\" mesh\n(do the "
             "small/aligned cells sit on the ring?)", fontsize=10)

# Panel C — profiles vs d
aC = fig.add_subplot(gs[0, 2])
aC.plot(bc, Lu, "o-", color="0.6", lw=1.6, ms=4,
        label="undeformed edge len")
aC.plot(bc, La, "o-", color="#1f4e8c", lw=1.8, ms=4,
        label="(3) edge len")
aC.axvline(r_gmax, color="k", ls="--", lw=1.1,
           label="|∇ρ| max (d=W/√2)")
aC.axvline(0.0, color="#c0392b", ls=":", lw=1.0)
aC.set_xlabel("distance from blob centre  d = |X-P|")
aC.set_ylabel("mean edge length")
aC.legend(fontsize=8, loc="lower right")
aC.grid(alpha=0.3)
aC2 = aC.twinx()
aC2.plot(abc, aspm, "s--", color="#e07b00", lw=1.4, ms=3,
         label="(3) cell aspect")
aC2.set_ylabel("cell aspect ratio", color="#e07b00")
aC2.tick_params(axis="y", colors="#e07b00")
aC.set_title("C  edge length & cell aspect vs d\n(min at the edge "
             "ring; centre & far NOT refined)", fontsize=10)

fig.suptitle("(3) is a GRADIENT metric: it resolves the blob EDGE "
             "(|∇ρ| ring), not the CORE — by design", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("/tmp/metric_mesh/aniso_blob_metric.png", dpi=130)
print("saved /tmp/metric_mesh/aniso_blob_metric.png")
print(f"h0={h0:.4f}  |∇ρ|-max ring radius d=W/√2={r_gmax:.4f}")
emin = np.nanargmin(La)
print(f"realised (3) min mean-edge at d≈{bc[emin]:.3f} "
      f"(ring is d≈{r_gmax:.3f}); centre d≈0 edge="
      f"{La[0]:.4f} vs undef {Lu[0]:.4f}; "
      f"far edge={La[-1]:.4f} vs undef {Lu[-1]:.4f}")
