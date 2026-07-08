"""Why does the inner (Lower) thermal BL NOT gather nodes the way
the outer (Upper) region does, even though the metric ρ∝|∇T| is
BRIGHTEST at the inner BL?

Decisive radial profiles on the cached res-32 warm state:
  * ρ(r)      — the metric density (∝|∇T|): does it peak at r=R_I?
  * |∇ρ|(r)   — what the mover ACTUALLY refines on (the
                gradient-metric clusters where |∇ρ| is large, NOT
                where ρ is large — the blob-core lesson).
  * node radial distribution BEFORE vs AFTER the mover — where
    did points actually snuggle?

Three things differ between the two boundaries:
  (1) velocity BC: inner = no-slip (add_essential_bc), outer =
      free-slip (add_natural_bc) → different BL dynamics;
  (2) the inner |∇T| peak sits ON the pinned wall, where ∇ρ≈0
      (the blob-CORE de-refinement) AND pinned;
  (3) annulus geometry: inner circumference πD=π vs outer 2π —
      half the tangential room at r=R_I.
This script isolates (2): show ρ peaks at R_I but |∇ρ| (the real
driver) does NOT — it's ~0 at the on-wall peak.
"""
from __future__ import annotations
import numpy as np
import sympy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import underworld3 as uw
from underworld3.meshing import smooth_mesh_interior
from underworld3.meshing.smoothing import _tri_cells

RA, AMP, RES = 1.0e5, 8.0, 32
N_WARM = 5
r_inner, r_o = 0.5, 1.0
C32 = f"/tmp/metric_mesh/conv_ra{RA:.0e}_res{RES}_warm{N_WARM}.npz"


def build():
    mesh = uw.meshing.Annulus(
        radiusOuter=r_o, radiusInner=r_inner,
        cellSize=1.0 / RES, qdegree=3)
    v = uw.discretisation.MeshVariable(
        "V32", mesh, vtype=uw.VarType.VECTOR, degree=2,
        continuous=True)
    T = uw.discretisation.MeshVariable(
        "T32", mesh, vtype=uw.VarType.SCALAR, degree=3,
        continuous=True)
    return mesh, v, T


mesh, v, T = build()
z = np.load(C32)
T.data[...] = z["T"].reshape(T.data.shape)
v.data[...] = z["V"].reshape(v.data.shape)
X0 = np.asarray(mesh.X.coords).copy()
r0 = np.hypot(X0[:, 0], X0[:, 1])
tris = _tri_cells(mesh.dm)

# --- ρ field and ∇ρ field -----------------------------------------
Xs = mesh.CoordinateSystem.X
gradT = uw.discretisation.MeshVariable(
    "gT", mesh, vtype=uw.VarType.VECTOR, degree=1, continuous=True)
gp = uw.systems.Vector_Projection(mesh, gradT)
gp.smoothing = 0.0
gp.uw_function = sympy.Matrix(
    [T.sym[0].diff(Xs[i]) for i in range(2)]).T
gp.solve()
rho0 = uw.discretisation.MeshVariable(
    "r0f", mesh, vtype=uw.VarType.SCALAR, degree=1, continuous=True)
gmag = np.linalg.norm(np.asarray(uw.function.evaluate(
    gradT.sym, rho0.coords)).reshape(-1, 2), axis=1)
g_lo, g_hi = np.percentile(gmag, 50.0), np.percentile(gmag, 97.0)
rho0.data[:, 0] = np.clip(
    (gmag - g_lo) / max(g_hi - g_lo, 1e-30), 0.0, 1.0)
metric = 1.0 + AMP * rho0.sym[0]

# |∇ρ| = AMP·|∇(normalised|∇T|)| — exactly what M is built from.
gradR = uw.discretisation.MeshVariable(
    "gR", mesh, vtype=uw.VarType.VECTOR, degree=1, continuous=True)
gpr = uw.systems.Vector_Projection(mesh, gradR)
gpr.smoothing = 0.0
gpr.uw_function = sympy.Matrix(
    [metric.diff(Xs[i]) for i in range(2)]).T
gpr.solve()

bins = np.linspace(r_inner, r_o, 26)
bc = 0.5 * (bins[1:] + bins[:-1])
rho_v = np.asarray(uw.function.evaluate(metric, X0)).reshape(-1)
gradrho_v = np.linalg.norm(np.asarray(uw.function.evaluate(
    gradR.sym, X0)).reshape(-1, 2), axis=1)


def prof(val, rr):
    return np.array([val[(rr >= bins[i]) & (rr < bins[i + 1])].mean()
                     if ((rr >= bins[i]) & (rr < bins[i + 1])).any()
                     else np.nan for i in range(len(bins) - 1)])


rho_p = prof(rho_v, r0)
grho_p = prof(gradrho_v, r0)
cnt_before = np.array([
    ((r0 >= bins[i]) & (r0 < bins[i + 1])).sum()
    for i in range(len(bins) - 1)], dtype=float)

# --- run the mover, recount nodes by radius -----------------------
smooth_mesh_interior(mesh, metric=metric, method="anisotropic",
                     verbose=False)
Xr = np.asarray(mesh.X.coords).copy()
rr = np.hypot(Xr[:, 0], Xr[:, 1])
cnt_after = np.array([
    ((rr >= bins[i]) & (rr < bins[i + 1])).sum()
    for i in range(len(bins) - 1)], dtype=float)

print(f"{'r':>6} {'rho':>7} {'|grad rho|':>10} {'n_before':>9} "
      f"{'n_after':>8} {'Δn':>6}")
for i in range(len(bc)):
    print(f"{bc[i]:6.3f} {rho_p[i]:7.2f} {grho_p[i]:10.3f} "
          f"{cnt_before[i]:9.0f} {cnt_after[i]:8.0f} "
          f"{cnt_after[i]-cnt_before[i]:+6.0f}")

fig, ax = plt.subplots(1, 2, figsize=(15, 5.6))
a = ax[0]
a.plot(bc, rho_p, "o-", color="#1f4e8c", label=r"$\rho=1+8\,\hat{|\nabla T|}$ (metric)")
a.set_xlabel("radius"); a.set_ylabel(r"$\rho$", color="#1f4e8c")
a.tick_params(axis="y", colors="#1f4e8c")
a.axvline(r_inner, color="#c0392b", ls="--", lw=1.2,
          label="inner wall (Lower, no-slip)")
a.axvline(r_o, color="#e07b00", ls="--", lw=1.2,
          label="outer wall (Upper, free-slip)")
a2 = a.twinx()
a2.plot(bc, grho_p, "s--", color="#c0392b",
        label=r"$|\nabla\rho|$ — what the mover refines on")
a2.set_ylabel(r"$|\nabla\rho|$", color="#c0392b")
a2.tick_params(axis="y", colors="#c0392b")
a.set_title("ρ peaks AT the inner wall, but |∇ρ| (the real\n"
            "driver) is ≈0 there — the blob-CORE de-refinement")
h1, l1 = a.get_legend_handles_labels()
h2, l2 = a2.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center")
a.grid(alpha=0.3)

a = ax[1]
w = (bins[1] - bins[0]) * 0.4
a.bar(bc - w / 2, cnt_before, w, color="0.6", label="before")
a.bar(bc + w / 2, cnt_after, w, color="#1f4e8c", label="after")
a.axvline(r_inner, color="#c0392b", ls="--", lw=1.2)
a.axvline(r_o, color="#e07b00", ls="--", lw=1.2)
a.set_xlabel("radius"); a.set_ylabel("node count in radial bin")
a.set_title("nodes gather toward the OUTER half / plume region,\n"
            "NOT the inner BL (its ρ-peak is pinned + ∇ρ≈0)")
a.legend(fontsize=9)
a.grid(alpha=0.3)
fig.suptitle("Inner vs outer BL: why the gradient metric + pinned "
             "wall refines one and not the other", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("/tmp/metric_mesh/aniso_bl_asymmetry.png", dpi=130)
print("saved /tmp/metric_mesh/aniso_bl_asymmetry.png")
