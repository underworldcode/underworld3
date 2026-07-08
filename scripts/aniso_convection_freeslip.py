"""Controlled test: SYMMETRIC velocity BC — free-slip on BOTH
boundaries (release the inner no-slip), nullspaces lit up
(constant pressure + the annulus rigid-rotation mode (-y,x)).
Warm-start T from the res-16 run, let convection develop for a
while, then refine on ∇T with the (3) anisotropic mover.

If the inner/outer BL gathering becomes symmetric (cf.
aniso_bl_asymmetry.png with no-slip inner) → the velocity BC was
the dominant cause; any residual asymmetry is the annulus
geometry (inner circumference π vs outer 2π) + the gradient-metric
/ pinned-wall effect.
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

RA, AMP, RES16, RES = 1.0e5, 8.0, 16, 32
N_RUN = 25
G_LO_PCT, G_HI_PCT = 50.0, 97.0
r_inner, r_o = 0.5, 1.0
C16 = f"/tmp/metric_mesh/conv_ra{RA:.0e}_res{RES16}_n20.npz"
CFS = f"/tmp/metric_mesh/conv_ra{RA:.0e}_res{RES}_freeslip{N_RUN}.npz"


def build(res, tag):
    mesh = uw.meshing.Annulus(
        radiusOuter=r_o, radiusInner=r_inner,
        cellSize=1.0 / res, qdegree=3)
    r, th = mesh.CoordinateSystem.R
    v = uw.discretisation.MeshVariable(
        f"V{tag}", mesh, vtype=uw.VarType.VECTOR, degree=2,
        continuous=True)
    P = uw.discretisation.MeshVariable(
        f"P{tag}", mesh, vtype=uw.VarType.SCALAR, degree=1,
        continuous=True)
    T = uw.discretisation.MeshVariable(
        f"T{tag}", mesh, vtype=uw.VarType.SCALAR, degree=3,
        continuous=True)
    return mesh, r, th, v, P, T


def make_solvers(mesh, r, v, P, T):
    stokes = uw.systems.Stokes(mesh, velocityField=v,
                               pressureField=P)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.tolerance = 1.0e-5
    stokes.penalty = 0.0
    unit_r = mesh.CoordinateSystem.unit_e_0
    # FREE-SLIP on BOTH boundaries: no-penetration penalty on the
    # NORMAL component only (tangential / rotation free) — releases
    # the inner no-slip.
    stokes.add_natural_bc(1.0e6 * v.sym.dot(unit_r) * unit_r,
                          mesh.boundaries.Lower.name)
    stokes.add_natural_bc(1.0e6 * v.sym.dot(unit_r) * unit_r,
                          mesh.boundaries.Upper.name)
    # Nullspaces lit up EXPLICITLY: constant pressure + the annulus
    # rigid-rotation mode (-y,x) (= r·e_θ; an exact null mode since
    # the BC penalises only the normal velocity and the radial
    # buoyancy has zero torque).
    x, y = mesh.CoordinateSystem.X
    stokes.petsc_use_pressure_nullspace = True
    stokes.petsc_velocity_nullspace_basis = [sympy.Matrix([-y, x])]
    # Linear Stokes ⇒ the Newton line search is unnecessary and
    # spuriously fails (DIVERGED_LINE_SEARCH @0 iters) on the
    # correctly-singular nullspace-bearing operator. Take the full
    # step.
    stokes.petsc_options["snes_linesearch_type"] = "basic"
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


mesh, r, th, v, P, T = build(RES, "fs")
if os.path.exists(CFS):
    print(f"loading cached free-slip state {CFS}")
    z = np.load(CFS)
    T.data[...] = z["T"].reshape(T.data.shape)
    v.data[...] = z["V"].reshape(v.data.shape)
    stokes, adv = make_solvers(mesh, r, v, P, T)
else:
    if not os.path.exists(C16):
        raise SystemExit(f"missing res-16 cache {C16}")
    print(f"warm-start res-{RES} from res-{RES16}; free-slip BOTH "
          f"+ nullspaces; {N_RUN} steps")
    m16, r16, t16, v16, P16, T16 = build(RES16, "16")
    z16 = np.load(C16)
    T16.data[...] = z16["T"].reshape(T16.data.shape)
    T.data[:, 0] = np.asarray(uw.function.evaluate(
        T16.sym[0], T.coords)).reshape(-1)
    stokes, adv = make_solvers(mesh, r, v, P, T)
    stokes.solve(zero_init_guess=True)
    t_sim = 0.0
    for s in range(N_RUN):
        dt = adv.estimate_dt()
        adv.solve(timestep=dt, zero_init_guess=False)
        stokes.solve(zero_init_guess=False)
        t_sim += dt
        tt = T.data[:, 0]
        if (s + 1) % 5 == 0 or s == 0:
            print(f"  step {s+1:2d}: Δt={dt:.2e} "
                  f"T=[{tt.min():+.3f},{tt.max():+.3f}]", flush=True)
    np.savez(CFS, T=np.asarray(T.data), V=np.asarray(v.data))
    print(f"cached → {CFS}")

X0 = np.asarray(mesh.X.coords).copy()
r0 = np.hypot(X0[:, 0], X0[:, 1])
tris = _tri_cells(mesh.dm)

# metric ρ ∝ normalised |∇T|
Xs = mesh.CoordinateSystem.X
gradT = uw.discretisation.MeshVariable(
    "gTfs", mesh, vtype=uw.VarType.VECTOR, degree=1,
    continuous=True)
gp = uw.systems.Vector_Projection(mesh, gradT)
gp.smoothing = 0.0
gp.uw_function = sympy.Matrix(
    [T.sym[0].diff(Xs[i]) for i in range(2)]).T
gp.solve()
rho0 = uw.discretisation.MeshVariable(
    "r0fs", mesh, vtype=uw.VarType.SCALAR, degree=1,
    continuous=True)
gmag = np.linalg.norm(np.asarray(uw.function.evaluate(
    gradT.sym, rho0.coords)).reshape(-1, 2), axis=1)
g_lo, g_hi = np.percentile(gmag, G_LO_PCT), np.percentile(
    gmag, G_HI_PCT)
rho0.data[:, 0] = np.clip(
    (gmag - g_lo) / max(g_hi - g_lo, 1e-30), 0.0, 1.0)
metric = 1.0 + AMP * rho0.sym[0]
rho_field = np.asarray(uw.function.evaluate(
    metric, X0)).reshape(-1)

A0 = np.abs(_signed_areas(X0, tris))
bins = np.linspace(r_inner, r_o, 26)
cnt_before = np.array([((r0 >= bins[i]) & (r0 < bins[i + 1])).sum()
                       for i in range(len(bins) - 1)], dtype=float)
print("=== refine: free-slip-both, method='anisotropic' ===")
smooth_mesh_interior(mesh, metric=metric, method="anisotropic",
                     verbose=False)
Xr = np.asarray(mesh.X.coords).copy()
rr = np.hypot(Xr[:, 0], Xr[:, 1])
A1 = np.abs(_signed_areas(Xr, tris))
cnt_after = np.array([((rr >= bins[i]) & (rr < bins[i + 1])).sum()
                      for i in range(len(bins) - 1)], dtype=float)
bc = 0.5 * (bins[1:] + bins[:-1])
print(f"minA/meanA {A0.min()/A0.mean():.3f}→{A1.min()/A1.mean():.3f}"
      f"  inner-BL Δn(r<0.6)={cnt_after[bc<0.6].sum()-cnt_before[bc<0.6].sum():+.0f}"
      f"  outer-BL Δn(r>0.9)={cnt_after[bc>0.9].sum()-cnt_before[bc>0.9].sum():+.0f}")

# figure
Tn = np.asarray(uw.function.evaluate(T.sym[0], X0)).reshape(-1)
tro = mtri.Triangulation(X0[:, 0], X0[:, 1], tris)
trr = mtri.Triangulation(Xr[:, 0], Xr[:, 1], tris)
fig, ax = plt.subplots(2, 2, figsize=(13.5, 13))
a = ax[0, 0]
a.tricontourf(tro, Tn, levels=24, cmap="inferno")
a.triplot(tro, lw=0.15, color="white", alpha=0.3)
a.set_title(f"T  (free-slip BOTH, nullspaces lit; res-{RES}, "
            f"{N_RUN} steps from res-{RES16})", fontsize=11)
a = ax[0, 1]
cf = a.tricontourf(tro, rho_field, levels=24, cmap="viridis")
a.set_title(r"metric  $\rho=1+%g\,\hat{|\nabla T|}$" % AMP,
            fontsize=11)
fig.colorbar(cf, ax=a, fraction=0.046, pad=0.02)
a = ax[1, 0]
a.triplot(trr, lw=0.35, color="#1f4e8c")
a.set_title(f"refined mesh  (minA/meanA "
            f"{A0.min()/A0.mean():.2f}→{A1.min()/A1.mean():.2f})",
            fontsize=11)
a = ax[1, 1]
w = (bins[1] - bins[0]) * 0.4
a.bar(bc - w / 2, cnt_before, w, color="0.6", label="before")
a.bar(bc + w / 2, cnt_after, w, color="#1f4e8c", label="after")
a.axvline(r_inner, color="#c0392b", ls="--", lw=1.2,
          label="inner (free-slip now)")
a.axvline(r_o, color="#e07b00", ls="--", lw=1.2,
          label="outer (free-slip)")
a.set_xlabel("radius")
a.set_ylabel("node count")
a.set_title("radial node gather — symmetric now?")
a.legend(fontsize=8)
a.grid(alpha=0.3)
for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])
fig.suptitle("Free-slip BOTH boundaries (inner no-slip released, "
             "nullspaces lit) → convection → refine on ∇T",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = "/tmp/metric_mesh/aniso_convection_freeslip.png"
fig.savefig(out, dpi=130)
print(f"saved {out}")
