"""res-32 annulus convection, WARM-STARTED from the cached res-16
20-step state, run a few steps, then refine on ∇T with the (3)
anisotropic mover ("point snuggling").

Warm start: rebuild the res-16 P3 T field from the cached npz,
interpolate it onto the res-32 T nodes (uw.function.evaluate
across meshes — same annulus geometry), re-solve Stokes for a
consistent velocity, then run N_WARM adv-diff + Stokes steps so
the field settles on the finer mesh. res-32 post-warm state is
itself cached (save results, never re-run).
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
from underworld3.meshing import smooth_mesh_interior
from underworld3.meshing.smoothing import _tri_cells, _signed_areas

RA, AMP = 1.0e5, 8.0
RES16, RES32 = 16, 32
N_WARM = 5
G_LO_PCT, G_HI_PCT = 50.0, 97.0
r_inner, r_o = 0.5, 1.0
C16 = f"/tmp/metric_mesh/conv_ra{RA:.0e}_res{RES16}_n20.npz"
C32 = f"/tmp/metric_mesh/conv_ra{RA:.0e}_res{RES32}_warm{N_WARM}.npz"


def build(res, tag):
    mesh = uw.meshing.Annulus(
        radiusOuter=r_o, radiusInner=r_inner,
        cellSize=1.0 / res, qdegree=3)
    r, th = mesh.CoordinateSystem.R
    v = uw.discretisation.MeshVariable(
        f"V{tag}", mesh, vtype=uw.VarType.VECTOR, degree=2,
        continuous=True, varsymbol=r"\mathbf{v}")
    P = uw.discretisation.MeshVariable(
        f"P{tag}", mesh, vtype=uw.VarType.SCALAR, degree=1,
        continuous=True, varsymbol="p")
    T = uw.discretisation.MeshVariable(
        f"T{tag}", mesh, vtype=uw.VarType.SCALAR, degree=3,
        continuous=True, varsymbol="T")
    return mesh, r, th, v, P, T


def make_solvers(mesh, r, v, P, T):
    stokes = uw.systems.Stokes(mesh, velocityField=v,
                               pressureField=P)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.tolerance = 1.0e-5
    stokes.penalty = 0.0
    unit_r = mesh.CoordinateSystem.unit_e_0
    stokes.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
    stokes.add_natural_bc(1.0e6 * v.sym.dot(unit_r) * unit_r,
                          mesh.boundaries.Upper.name)
    T_cond = (r_o - r) / (r_o - r_inner)
    stokes.bodyforce = RA * (T.sym[0] - T_cond) * unit_r
    adv = uw.systems.AdvDiffusionSLCN(
        mesh, u_Field=T, V_fn=v.sym, verbose=False,
        theta=0.5, monotone_mode="clamp")
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = 1.0
    adv.tolerance = 1.0e-4
    adv.add_dirichlet_bc(1.0, mesh.boundaries.Lower.name)
    adv.add_dirichlet_bc(0.0, mesh.boundaries.Upper.name)
    return stokes, adv


# --- res-32 warm-started state (cached) ----------------------------
mesh, r, th, v, P, T = build(RES32, "32")
if os.path.exists(C32):
    print(f"loading cached res-32 warm state {C32}")
    z = np.load(C32)
    T.data[...] = z["T"].reshape(T.data.shape)
    v.data[...] = z["V"].reshape(v.data.shape)
    stokes, adv = make_solvers(mesh, r, v, P, T)
else:
    if not os.path.exists(C16):
        raise SystemExit(f"missing res-16 cache {C16} — run "
                         f"aniso_convection_demo.py first")
    print(f"warm-starting res-{RES32} from res-{RES16} ({C16})")
    m16, r16, th16, v16, P16, T16 = build(RES16, "16")
    z16 = np.load(C16)
    T16.data[...] = z16["T"].reshape(T16.data.shape)
    # interpolate the res-16 T onto the res-32 T nodes
    T.data[:, 0] = np.asarray(uw.function.evaluate(
        T16.sym[0], T.coords)).reshape(-1)
    stokes, adv = make_solvers(mesh, r, v, P, T)
    stokes.solve(zero_init_guess=True)         # consistent V
    t_sim = 0.0
    print(f"=== res-{RES32} warm run, {N_WARM} steps ===")
    for s in range(N_WARM):
        dt = adv.estimate_dt()
        adv.solve(timestep=dt, zero_init_guess=False)
        stokes.solve(zero_init_guess=False)
        t_sim += dt
        tt = T.data[:, 0]
        print(f"  step {s+1}: Δt={dt:.2e} "
              f"T=[{tt.min():+.3f},{tt.max():+.3f}]", flush=True)
    np.savez(C32, T=np.asarray(T.data), V=np.asarray(v.data))
    print(f"cached → {C32}")

X_orig = np.asarray(mesh.X.coords).copy()
tris = _tri_cells(mesh.dm)

# --- metric ρ ∝ normalised |∇T| (Lagrangian scalar field) ----------
Xs = mesh.CoordinateSystem.X
gradT = uw.discretisation.MeshVariable(
    "gradT32", mesh, vtype=uw.VarType.VECTOR, degree=1,
    continuous=True)
gp = uw.systems.Vector_Projection(mesh, gradT)
gp.smoothing = 0.0
gp.uw_function = sympy.Matrix(
    [T.sym[0].diff(Xs[i]) for i in range(2)]).T
gp.solve()
rho0 = uw.discretisation.MeshVariable(
    "rho032", mesh, vtype=uw.VarType.SCALAR, degree=1,
    continuous=True)
gmag = np.linalg.norm(np.asarray(uw.function.evaluate(
    gradT.sym, rho0.coords)).reshape(-1, 2), axis=1)
g_lo = np.percentile(gmag, G_LO_PCT)
g_hi = np.percentile(gmag, G_HI_PCT)
rho0.data[:, 0] = np.clip(
    (gmag - g_lo) / max(g_hi - g_lo, 1e-30), 0.0, 1.0)
metric = 1.0 + AMP * rho0.sym[0]
rho_field = np.asarray(uw.function.evaluate(
    metric, X_orig)).reshape(-1)

# --- refine ("point snuggling") ------------------------------------
A0 = np.abs(_signed_areas(X_orig, tris))
print("=== refine res-32: method='anisotropic' on ρ∝|∇T| ===")
smooth_mesh_interior(mesh, metric=metric, method="anisotropic",
                     verbose=True)
X_ref = np.asarray(mesh.X.coords).copy()
A1 = np.abs(_signed_areas(X_ref, tris))
print(f"minA/meanA  before={A0.min()/A0.mean():.4f}  "
      f"after={A1.min()/A1.mean():.4f}  "
      f"max|Δx|={np.linalg.norm(X_ref-X_orig,axis=1).max():.3e}")

# --- figure --------------------------------------------------------
Tn = np.asarray(uw.function.evaluate(
    T.sym[0], X_orig)).reshape(-1)
tro = mtri.Triangulation(X_orig[:, 0], X_orig[:, 1], tris)
trr = mtri.Triangulation(X_ref[:, 0], X_ref[:, 1], tris)
fig, ax = plt.subplots(2, 2, figsize=(13.5, 13))
a = ax[0, 0]
a.tricontourf(tro, Tn, levels=24, cmap="inferno")
a.triplot(tro, lw=0.15, color="white", alpha=0.3)
a.set_title(f"T  (Ra={RA:.0e}, res-{RES32} warm-started from "
            f"res-{RES16}, +{N_WARM} steps)", fontsize=11)
a = ax[0, 1]
cf = a.tricontourf(tro, rho_field, levels=24, cmap="viridis")
a.set_title(r"target metric  $\rho = 1+%g\,\hat{|\nabla T|}$"
            % AMP, fontsize=11)
fig.colorbar(cf, ax=a, fraction=0.046, pad=0.02)
a = ax[1, 0]
a.triplot(trr, lw=0.35, color="#1f4e8c")
a.set_title(f"refined res-{RES32} mesh  (minA/meanA "
            f"{A0.min()/A0.mean():.2f}→{A1.min()/A1.mean():.2f})",
            fontsize=11)
a = ax[1, 1]
a.tricontourf(trr, Tn, levels=24, cmap="inferno")
a.triplot(trr, lw=0.25, color="white", alpha=0.35)
a.set_title("refined mesh + T  (points snuggle into the BLs / "
            "plume edges)", fontsize=11)
for a in ax.ravel():
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])
fig.suptitle(f"res-{RES32} convection (warm from res-{RES16}) → "
             f"refine on ∇T with the (3) anisotropic mover",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = "/tmp/metric_mesh/aniso_convection_res32.png"
fig.savefig(out, dpi=130)
print(f"saved {out}")
