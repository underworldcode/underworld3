"""Look at the meshes: triangles shaded by shape-quality q
(4√3·A/Σℓ², 1=equilateral, red→sliver). 2 rows (step 20 = the
violent overshoot where equidist is much worse; step 70 = where
Stokes actually DIVERGED) × cols [a16c2 | a16e | a16ed]. Bad-cell
counts annotated. Writes one PNG for Preview."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import underworld3 as uw
from underworld3.meshing.smoothing import _tri_cells
D = "/tmp/metric_mesh/sat"
TAGS = ["a16c2", "a16e", "a16ed"]
STEPS = [20, 70]
fig, ax = plt.subplots(len(STEPS), len(TAGS),
                       figsize=(5.2 * len(TAGS), 5.0 * len(STEPS)))
for r, step in enumerate(STEPS):
    for c, tag in enumerate(TAGS):
        a = ax[r, c]
        try:
            m = uw.discretisation.Mesh(
                f"{D}/sat_{tag}.mesh.{step:05}.h5")
        except Exception:
            a.set_title(f"{tag} step {step}: no ckpt")
            a.set_axis_off()
            continue
        tri = _tri_cells(m.dm)
        X = np.asarray(m.X.coords)[:, :2]
        v0, v1, v2 = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
        aa = np.linalg.norm(v1 - v2, axis=1)
        bb = np.linalg.norm(v2 - v0, axis=1)
        cc = np.linalg.norm(v0 - v1, axis=1)
        Ar = 0.5 * np.abs(np.cross(v1 - v0, v2 - v0))
        q = (4.0 * np.sqrt(3.0) * np.maximum(Ar, 1e-300)
             / (aa * aa + bb * bb + cc * cc))
        T = mtri.Triangulation(X[:, 0], X[:, 1], tri)
        tpc = a.tripcolor(T, facecolors=q, cmap="RdYlGn",
                          vmin=0.15, vmax=0.7, edgecolors="k",
                          linewidth=0.15)
        nbad = int((q < 0.3).sum())
        nvbad = int((q < 0.2).sum())
        a.set_aspect("equal")
        a.set_axis_off()
        a.set_title(f"{tag}  step {step}\n"
                    f"q<0.3: {nbad}   q<0.2: {nvbad}   "
                    f"q_min={q.min():.3f}", fontsize=11)
cb = fig.colorbar(tpc, ax=ax, shrink=0.6,
                  label="cell shape quality q (red = poor)")
fig.suptitle("Poor-cell map — equidist (a16e/a16ed) vs hand-tuned "
             "cc=2 (a16c2). Row1 step20 (overshoot, equidist much "
             "worse). Row2 step70 (Stokes DIVERGED — meshes "
             "comparable).", fontsize=12)
out = f"{D}/cellquality_a16c2_vs_equidist.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved {out}")
