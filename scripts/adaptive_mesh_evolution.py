"""Watch the mesh re-snuggle at each adaptation event during an
Ra=1e5 adaptive convection run (res-16, harness setup: no-slip
inner / free-slip outer — BCs not the point here).

Captures (coords, T@vertices) just BEFORE and just AFTER every
adaptation, then renders one row per event:
  col 0  T + mesh just before the mover fires
  col 1  T + mesh just after  (the "snuggle" move)
so the mesh visibly chases the growing plumes / boundary layers.
Snapshots cached → replot is free.
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
from underworld3.meshing import (
    smooth_mesh_interior, metric_density_from_gradient)
from underworld3.meshing.smoothing import _tri_cells, _signed_areas

RA, RES, N_STEPS, ADAPT_EVERY, AMP = 1.0e5, 16, 15, 5, 8.0
r_inner, r_o = 0.5, 1.0
CACHE = f"/tmp/metric_mesh/mesh_evo_Ra{RA:.0e}_res{RES}_n{N_STEPS}.npz"


def build():
    mesh = uw.meshing.Annulus(
        radiusOuter=r_o, radiusInner=r_inner,
        cellSize=1.0 / RES, qdegree=3)
    r, th = mesh.CoordinateSystem.R
    v = uw.discretisation.MeshVariable(
        "V", mesh, vtype=uw.VarType.VECTOR, degree=2,
        continuous=True)
    P = uw.discretisation.MeshVariable(
        "P", mesh, vtype=uw.VarType.SCALAR, degree=1,
        continuous=True)
    T = uw.discretisation.MeshVariable(
        "T", mesh, vtype=uw.VarType.SCALAR, degree=3,
        continuous=True)
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
    init_t = (0.01 * sympy.sin(5.0 * th)
              * sympy.sin(np.pi * (r - r_inner) / (r_o - r_inner))
              + (r_o - r) / (r_o - r_inner))
    T.data[...] = np.asarray(uw.function.evaluate(
        init_t, T.coords)).reshape(-1, 1)
    return mesh, v, P, T, stokes, adv


mesh, v, P, T, stokes, adv = build()
tris = _tri_cells(mesh.dm)


def Tvert(X):
    return np.asarray(uw.function.evaluate(
        T.sym[0], X)).reshape(-1)


if os.path.exists(CACHE):
    print(f"loading cached snapshots {CACHE}")
    z = np.load(CACHE, allow_pickle=True)
    snaps = list(z["snaps"])
else:
    stokes.solve(zero_init_guess=True)
    snaps = []
    t_sim = 0.0
    for s in range(N_STEPS):
        dt = adv.estimate_dt()
        adv.solve(timestep=dt, zero_init_guess=False)
        stokes.solve(zero_init_guess=False)
        t_sim += dt
        if (s + 1) % ADAPT_EVERY == 0:
            Xb = np.asarray(mesh.X.coords).copy()
            Tb = Tvert(Xb)
            Ab = np.abs(_signed_areas(Xb, tris))
            rho = metric_density_from_gradient(
                mesh, T, amp=AMP, name="evo")
            smooth_mesh_interior(
                mesh, metric=rho, method="anisotropic",
                method_kwargs=dict(aniso_cap=2.0, relax=0.2,
                                   n_outer=8))
            Xa = np.asarray(mesh.X.coords).copy()
            Ta = Tvert(Xa)
            Aa = np.abs(_signed_areas(Xa, tris))
            dmax = float(np.linalg.norm(Xa - Xb, axis=1).max())
            snaps.append(dict(
                step=s + 1, t=t_sim, Xb=Xb, Tb=Tb, Xa=Xa, Ta=Ta,
                qb=Ab.min() / Ab.mean(), qa=Aa.min() / Aa.mean(),
                dmax=dmax))
            print(f"  adapt @ step {s+1:2d} t={t_sim:.4f}  "
                  f"max|Δx|={dmax:.3e}  minA/meanA "
                  f"{Ab.min()/Ab.mean():.3f}→{Aa.min()/Aa.mean():.3f}",
                  flush=True)
    np.savez(CACHE, snaps=np.array(snaps, dtype=object))
    print(f"cached → {CACHE}")

n = len(snaps)
fig, ax = plt.subplots(n, 2, figsize=(11, 5.2 * n))
if n == 1:
    ax = ax[None, :]
for i, sn in enumerate(snaps):
    for j, (X, Tn, tag, q) in enumerate([
            (sn["Xb"], sn["Tb"], "before", sn["qb"]),
            (sn["Xa"], sn["Ta"], "after", sn["qa"])]):
        a = ax[i, j]
        tr = mtri.Triangulation(X[:, 0], X[:, 1], tris)
        a.tricontourf(tr, Tn, levels=22, cmap="inferno")
        a.triplot(tr, lw=0.35,
                  color=("white" if j == 0 else "#7fdbff"),
                  alpha=0.55)
        a.set_aspect("equal")
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(
            f"adapt #{i+1} (step {sn['step']}, t={sn['t']:.4f}) "
            f"— {tag}  minA/meanA={q:.3f}"
            + (f"  max|Δx|={sn['dmax']:.2e}" if j == 1 else ""),
            fontsize=10)
fig.suptitle(f"Mesh re-snuggling through adaptive convection "
             f"(Ra={RA:.0e}, res-{RES}, adapt every "
             f"{ADAPT_EVERY} steps) — T + mesh, before → after "
             f"each update", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.985])
out = "/tmp/metric_mesh/adaptive_mesh_evolution.png"
fig.savefig(out, dpi=125)
print(f"saved {out}")
