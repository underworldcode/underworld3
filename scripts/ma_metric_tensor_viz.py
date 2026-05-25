"""(3) metric-tensor machinery — visual sanity check.

Scalar density rho(x) in  ->  anisotropic metric tensor derived from
its GRADIENT, with NO (r,theta) frame ever specified. The point: the
eigenframe of M = I + beta (grad rho (x) grad rho) auto-aligns to
the feature (r-hat for a radial feature, theta-hat for an angular
one). We draw the desired-cell ellipse (semi-axes h_i = lambda_i^-1/2
of M, eigen-clamped) at sampled nodes for two features and check the
orientation/flattening is correct BEFORE building the anisotropic
mover.

grad rho is computed analytically in CARTESIAN (chain rule through
r,theta only to get the closed form) — the construction itself only
ever sees grad rho in (x,y); the (r,theta) alignment is emergent.
In production grad rho would be a Vector_Projection of rho
(first-derivative, UW3-clean) — same machinery, same result.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
import underworld3 as uw
from underworld3.meshing.smoothing import _tri_cells

R_O, R_I, RES, AMP = 1.0, 0.5, 16, 8.0
H0 = (R_O - R_I) / RES                 # nominal spacing
H_MIN, H_MAX = 0.12 * H0, 1.0 * H0     # eigen-clamp band (≤8:1)
BETA = 200.0                           # anisotropy strength


def grad_radial(x, y):
    """rho = 1 + AMP exp(-((r-r0)/Wr)^2); grad is purely radial."""
    r0, Wr = 0.85, 0.12
    r = np.hypot(x, y)
    rho = 1.0 + AMP * np.exp(-((r - r0) / Wr) ** 2)
    drdr = AMP * np.exp(-((r - r0) / Wr) ** 2) * (-2 * (r - r0) / Wr ** 2)
    gx = drdr * x / r
    gy = drdr * y / r
    return rho, np.stack([gx, gy], axis=1)


def grad_angular(x, y):
    """rho = 1 + AMP/(1+(Δθ/Wθ)^2); grad is purely tangential."""
    th0, Wth = 0.6, 0.5
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    dth = np.arctan2(np.sin(th - th0), np.cos(th - th0))
    u = dth / Wth
    rho = 1.0 + AMP / (1.0 + u ** 2)
    drdth = AMP * (-1.0) / (1.0 + u ** 2) ** 2 * (2 * u / Wth)
    that = np.stack([-y / r, x / r], axis=1)          # θ̂
    g = (drdth / r)[:, None] * that
    return rho, g


def build_metric(grad):
    """THE machinery: M = (1/h0^2)[ I + beta ĝĝᵀ (|g|/gref)^2 ],
    eigen-clamped so spacing ∈ [H_MIN, H_MAX]. Returns desired-cell
    semi-axes (h1,h2) and eigenvectors per node."""
    n = grad.shape[0]
    gnorm = np.linalg.norm(grad, axis=1)
    gref = gnorm.max() if gnorm.max() > 0 else 1.0
    lam_hi = 1.0 / H_MIN ** 2
    lam_lo = 1.0 / H_MAX ** 2
    H1 = np.empty(n); H2 = np.empty(n)
    V1 = np.zeros((n, 2)); V2 = np.zeros((n, 2))
    base = 1.0 / H0 ** 2
    for i in range(n):
        g = grad[i]
        gn = gnorm[i]
        M = base * np.eye(2)
        if gn > 1e-30:
            gh = g / gn
            M = base * (np.eye(2)
                        + BETA * (gn / gref) ** 2 * np.outer(gh, gh))
        w, Vec = np.linalg.eigh(M)              # ascending
        w = np.clip(w, lam_lo, lam_hi)
        # desired spacing along each eigenvector = w^-1/2
        H2[i], H1[i] = 1.0 / np.sqrt(w[1]), 1.0 / np.sqrt(w[0])
        V2[i], V1[i] = Vec[:, 1], Vec[:, 0]
    return H1, V1, H2, V2, gnorm


m = uw.meshing.Annulus(radiusOuter=R_O, radiusInner=R_I,
                        cellSize=1.0 / RES, qdegree=3)
Xn = np.asarray(m.X.coords)
tris = _tri_cells(m.dm)
# clean polar SAMPLE grid (decouple the ellipse field from mesh-node
# density — the construction is identical, only the sample points
# are a regular grid for legibility)
rr = np.linspace(R_I + 0.02, R_O - 0.02, 11)
tt = np.linspace(0, 2 * np.pi, 49)[:-1]
RG, TG = np.meshgrid(rr, tt)
Xs = np.stack([(RG * np.cos(TG)).ravel(),
               (RG * np.sin(TG)).ravel()], axis=1)

fig, ax = plt.subplots(1, 2, figsize=(15, 7.2))
for a, (name, gfn, mark) in zip(
        ax,
        [("radial feature  ρ(r)", grad_radial, ("ring", 0.85)),
         ("angular feature  ρ(θ)", grad_angular, ("spoke", 0.6))]):
    rho_n, _ = gfn(Xn[:, 0], Xn[:, 1])
    _, grad = gfn(Xs[:, 0], Xs[:, 1])
    H1, V1, H2, V2, gn = build_metric(grad)
    a.tricontourf(mtri.Triangulation(Xn[:, 0], Xn[:, 1], tris),
                  rho_n, levels=20, cmap="Blues", alpha=0.40)
    aniso = H1 / np.maximum(H2, 1e-30)          # ≥1, =1 isotropic
    sc = 1.1                                    # display scale
    pats, cv = [], []
    for i in range(Xs.shape[0]):
        ang = np.degrees(np.arctan2(V2[i, 1], V2[i, 0]))
        pats.append(Ellipse((Xs[i, 0], Xs[i, 1]),
                            width=2 * sc * H2[i],     # along ∇ρ
                            height=2 * sc * H1[i],    # along feature
                            angle=ang))
        cv.append(aniso[i])
    pc = PatchCollection(pats, facecolor="none", lw=0.9,
                         cmap="autumn_r")
    pc.set_array(np.array(cv))
    pc.set_clim(1.0, aniso.max())
    a.add_collection(pc)
    if mark[0] == "ring":
        th = np.linspace(0, 2 * np.pi, 300)
        a.plot(mark[1] * np.cos(th), mark[1] * np.sin(th),
                "k--", lw=1.0, alpha=0.5)
    else:
        a.plot([R_I * np.cos(mark[1]), R_O * np.cos(mark[1])],
                [R_I * np.sin(mark[1]), R_O * np.sin(mark[1])],
                "k--", lw=1.0, alpha=0.5)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.set_xlim(-1.08, 1.08); a.set_ylim(-1.08, 1.08)
    a.set_title(f"{name}\nellipse = desired cell (short ⟂ to the "
                f"feature); max anisotropy {aniso.max():.1f}:1",
                fontsize=12)
    fig.colorbar(pc, ax=a, fraction=0.046, pad=0.02,
                 label="anisotropy ratio")

fig.suptitle("(3) gradient-derived metric tensor — eigenframe "
             "auto-aligns to the feature (no r,θ specified)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("/tmp/metric_mesh/ma_metric_tensor.png", dpi=135)
print("saved /tmp/metric_mesh/ma_metric_tensor.png")
print(f"H0={H0:.4f}  clamp=[{H_MIN:.4f},{H_MAX:.4f}]  beta={BETA}")
