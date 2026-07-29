"""Ra=1e5 annulus convection (20 steps, FIXED mesh) → then refine
the mesh on the temperature gradient with the (3) anisotropic
mover.  "First not dynamically adaptive — run for 20 steps, then
refine."

The metric is the mover-side analogue of UW3's adaptation metric
`adaptivity.metric_from_gradient`: that maps a *normalised*
gradient magnitude |∇T| (clipped between a low/high window) to a
target edge length h ∈ [h_min, h_max] (isotropic M = h⁻²I, because
MMG can change the node count). Our mover has a FIXED node budget,
so it does the *relative* analogue: a target *density*
ρ = 1 + amp · t,  t = clip((|∇T|-g_lo)/(g_hi-g_lo), 0, 1),
fed as `metric=` (larger ρ ⇒ finer cells). `amp` is the
"bunching intensity"; the [g_lo,g_hi] window is the same
percentile idea as metric_from_gradient.

|∇T| is pre-projected to a *scalar* Lagrangian field (rho0) so the
mover's internal `metric.diff(X)` is a FIRST derivative of a field
(UW3-legal) — and the metric rides material points (the mover
requires a Lagrangian metric).

20-step Ra=1e5 state is cached to npz (memory feedback: save
results, never re-run) — delete it to force a fresh solve.
"""
from __future__ import annotations
import os
import numpy as np
import sympy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import underworld3 as uw


DT_SAFETY = 0.1


def scalar_dt(value):
    """Convert estimate_dt() output to a plain scalar timestep."""
    try:
        dt = float(value)
    except TypeError:
        arr = np.asarray(value, dtype=float)
        dt = float(np.nanmin(arr))

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"Bad timestep from estimate_dt(): {value!r}")

    return DT_SAFETY * dt
from underworld3.meshing import smooth_mesh_interior
from underworld3.meshing.smoothing import _tri_cells, _signed_areas

RES, N_STEPS, RA = 16, 20, 1.0e5
AMP = 8.0                       # bunching intensity (ρ = 1+AMP·t)
G_LO_PCT, G_HI_PCT = 50.0, 97.0   # |∇T| normalisation window
CACHE = f"/tmp/metric_mesh/conv_ra{RA:.0e}_res{RES}_n{N_STEPS}.npz"
r_inner, r_o = 0.5, 1.0


def build():
    cellsize = 1.0 / RES
    mesh = uw.meshing.Annulus(
        radiusOuter=r_o, radiusInner=r_inner,
        cellSize=cellsize, qdegree=3)
    r, th = mesh.CoordinateSystem.R
    v = uw.discretisation.MeshVariable(
        "V", mesh, vtype=uw.VarType.VECTOR, degree=2,
        continuous=True, varsymbol=r"\mathbf{v}")
    P = uw.discretisation.MeshVariable(
        "P", mesh, vtype=uw.VarType.SCALAR, degree=1,
        continuous=True, varsymbol="p")
    t_soln = uw.discretisation.MeshVariable(
        "T", mesh, vtype=uw.VarType.SCALAR, degree=3,
        continuous=True, varsymbol="T")
    return mesh, r, th, v, P, t_soln, cellsize


def run_convection(mesh, r, th, v, P, t_soln, cellsize):
    stokes = uw.systems.Stokes(mesh, velocityField=v,
                               pressureField=P)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.tolerance = 1.0e-5
    stokes.penalty = 0.0
    unit_r = mesh.CoordinateSystem.unit_e_0
    stokes.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
    stokes.add_essential_bc((0.0, 0.0), mesh.boundaries.Upper.name)
    T_cond = (r_o - r) / (r_o - r_inner)
    stokes.bodyforce = RA * (t_soln.sym[0] - T_cond) * unit_r

    adv_diff = uw.systems.AdvDiffusionSLCN(
        mesh, u_Field=t_soln, V_fn=v.sym, verbose=False,
        theta=0.5, monotone_mode="clamp")
    adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
    adv_diff.constitutive_model.Parameters.diffusivity = 1.0
    adv_diff.tolerance = 1.0e-4
    adv_diff.add_dirichlet_bc(1.0, mesh.boundaries.Lower.name)
    adv_diff.add_dirichlet_bc(0.0, mesh.boundaries.Upper.name)

    init_t = (0.01 * sympy.sin(5.0 * th)
              * sympy.sin(np.pi * (r - r_inner) / (r_o - r_inner))
              + (r_o - r) / (r_o - r_inner))
    t_soln.data[...] = np.asarray(uw.function.evaluate(
        init_t, t_soln.coords)).reshape(-1, 1)
    stokes.solve(zero_init_guess=True)

    t_sim = 0.0
    for s in range(N_STEPS):
        dt = scalar_dt(adv_diff.estimate_dt())
        adv_diff.solve(timestep=dt, zero_init_guess=False)
        stokes.solve(zero_init_guess=True)
        t_sim += dt
        tt = t_soln.data[:, 0]
        print(f"  step {s+1:2d}: t={t_sim:.4f} Δt={dt:.2e} "
              f"T=[{tt.min():+.3f},{tt.max():+.3f}]", flush=True)
    return t_sim


# --- 1. developed convection (cached) ------------------------------
mesh, r, th, v, P, t_soln, cellsize = build()
if os.path.exists(CACHE):
    print(f"loading cached 20-step state {CACHE}")
    z = np.load(CACHE)
    t_soln.data[...] = z["T"].reshape(t_soln.data.shape)
    v.data[...] = z["V"].reshape(v.data.shape)
else:
    print(f"=== Ra={RA:.0e} annulus convection, {N_STEPS} steps, "
          f"res-{RES} (FIXED mesh) ===")
    run_convection(mesh, r, th, v, P, t_soln, cellsize)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, T=np.asarray(t_soln.data),
             V=np.asarray(v.data),
             Xc=np.asarray(mesh.X.coords))
    print(f"cached → {CACHE}")

X_orig = np.asarray(mesh.X.coords).copy()
tris = _tri_cells(mesh.dm)

# --- 2. metric ρ ∝ normalised |∇T|  (Lagrangian scalar field) ------
Xs = mesh.CoordinateSystem.X
gradT = uw.discretisation.MeshVariable(
    "gradT", mesh, vtype=uw.VarType.VECTOR, degree=1,
    continuous=True)
gp = uw.systems.Vector_Projection(mesh, gradT)
gp.smoothing = 0.0
gp.uw_function = sympy.Matrix(
    [t_soln.sym[0].diff(Xs[i]) for i in range(2)]).T
gp.solve()

rho0 = uw.discretisation.MeshVariable(   # frozen Lagrangian density
    "rho0", mesh, vtype=uw.VarType.SCALAR, degree=1,
    continuous=True)
gmag = np.linalg.norm(
    np.asarray(uw.function.evaluate(gradT.sym, rho0.coords)
               ).reshape(-1, 2), axis=1)
g_lo = np.percentile(gmag, G_LO_PCT)
g_hi = np.percentile(gmag, G_HI_PCT)
t_norm = np.clip((gmag - g_lo) / max(g_hi - g_lo, 1e-30), 0.0, 1.0)
rho0.data[:, 0] = t_norm                       # store normalised t
metric = 1.0 + AMP * rho0.sym[0]               # ρ = 1 + AMP·t
rho_field = np.asarray(uw.function.evaluate(
    metric, X_orig)).reshape(-1)
print(f"|∇T|: g_lo(p{G_LO_PCT:.0f})={g_lo:.3f} "
      f"g_hi(p{G_HI_PCT:.0f})={g_hi:.3f}  ρ∈"
      f"[{rho_field.min():.2f},{rho_field.max():.2f}]")

# --- 3. refine the mesh on the T-gradient metric -------------------
A0 = np.abs(_signed_areas(X_orig, tris))
print("=== refine: node_redistribution on ρ∝|∇T| ===")
uw.meshing.node_redistribution(mesh, metric)
X_ref = np.asarray(mesh.X.coords).copy()
A1 = np.abs(_signed_areas(X_ref, tris))
print(f"minA/meanA  before={A0.min()/A0.mean():.4f}  "
      f"after={A1.min()/A1.mean():.4f}  "
      f"max|Δx|={np.linalg.norm(X_ref-X_orig,axis=1).max():.3e}")

# --- 4. figure -----------------------------------------------------
# T at the *vertex* coords (t_soln is P3 → its .data has far more
# DOFs than vertices; the triangulation is on vertices). T rides
# DOFs Lagrangianly through the move, so the per-vertex value is
# the same on the original and refined meshes — only the vertex
# *positions* differ.
Tn = np.asarray(uw.function.evaluate(
    t_soln.sym[0], X_orig)).reshape(-1)
tro = mtri.Triangulation(X_orig[:, 0], X_orig[:, 1], tris)
trr = mtri.Triangulation(X_ref[:, 0], X_ref[:, 1], tris)
fig, ax = plt.subplots(2, 2, figsize=(13.5, 13))

a = ax[0, 0]
a.tricontourf(tro, Tn, levels=24, cmap="inferno")
a.triplot(tro, lw=0.18, color="white", alpha=0.35)
a.set_title(f"developed T  (Ra={RA:.0e}, {N_STEPS} steps, "
            f"FIXED mesh res-{RES})", fontsize=11)

a = ax[0, 1]
cf = a.tricontourf(tro, rho_field, levels=24, cmap="viridis")
a.set_title(r"target metric  $\rho = 1 + %g\,\hat{|\nabla T|}$  "
            "(what we refine on)" % AMP, fontsize=11)
fig.colorbar(cf, ax=a, fraction=0.046, pad=0.02)

a = ax[1, 0]
a.triplot(trr, lw=0.4, color="#1f4e8c")
a.set_title(f"refined mesh  (anisotropic mover; minA/meanA "
            f"{A0.min()/A0.mean():.2f}→{A1.min()/A1.mean():.2f})",
            fontsize=11)

a = ax[1, 1]
a.tricontourf(trr, Tn, levels=24, cmap="inferno")
a.triplot(trr, lw=0.3, color="white", alpha=0.4)
a.set_title("refined mesh + T  (cells gather along the thermal "
            "BLs / plume edges)", fontsize=11)

for a in ax.ravel():
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])
fig.suptitle("Ra=1e5 annulus convection → refine on ∇T with the "
             "(3) anisotropic mover (fixed node budget)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = "/tmp/metric_mesh/aniso_convection.png"
fig.savefig(out, dpi=130)
print(f"saved {out}")
