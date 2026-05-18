"""Adaptive-convection TEST HARNESS (scaffold for the next phase).

Goal: does Ra=1e5 annulus convection on a coarse res-16 mesh that
is *adaptively snuggled* on |∇T| reproduce the diagnostics of a
uniform res-24 reference (the "truth") — at far fewer nodes?

Structure
  * reference  : uniform res-24, run N steps, record Nu/vrms(t).
  * adaptive   : res-16, every ADAPT_EVERY steps call the locked-in
                 API (metric_density_from_gradient +
                 smooth_mesh_interior method="anisotropic"); record
                 Nu/vrms(t).
  * compare    : Nu(t), vrms(t) reference vs adaptive (matched
                 physical time) + an error summary + a figure.

THE OPEN PIECE (next phase): when the mover moves nodes mid-run,
the mesh has a velocity ``v_mesh = Δx_adapt / Δt`` that the
advection–diffusion system must see (ALE: effective transport
velocity ``v_fluid − v_mesh``), or T must be conservatively
remapped onto the moved nodes. Without it the adaptation injects a
spurious advection. ``apply_adaptation_correction`` is the explicit
hook: ``--correction none`` runs the *uncorrected* baseline (this
is expected to drift — it is the thing the next phase fixes);
``--correction ale`` raises with the precise spec to implement.

Both runs' histories are cached to npz (never re-run).
"""
from __future__ import annotations
import os
import argparse
import numpy as np
import sympy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import underworld3 as uw
from underworld3.meshing import (
    smooth_mesh_interior, metric_density_from_gradient)

p = argparse.ArgumentParser()
p.add_argument("--ref-res", type=int, default=24)
p.add_argument("--adapt-res", type=int, default=16)
p.add_argument("--n-steps", type=int, default=20)
p.add_argument("--adapt-every", type=int, default=5)
p.add_argument("--Ra", type=float, default=1.0e5)
p.add_argument("--amp", type=float, default=8.0)
p.add_argument("--correction", choices=["none", "ale"],
               default="none")
args = p.parse_args()
r_inner, r_o = 0.5, 1.0
CDIR = "/tmp/metric_mesh"
os.makedirs(CDIR, exist_ok=True)


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
    stokes = uw.systems.Stokes(mesh, velocityField=v,
                               pressureField=P)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.tolerance = 1.0e-5
    stokes.penalty = 0.0
    unit_r = mesh.CoordinateSystem.unit_e_0
    # The validated benchmark config (trusted): no-slip inner,
    # free-slip outer — no Stokes nullspace.
    stokes.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
    stokes.add_natural_bc(1.0e6 * v.sym.dot(unit_r) * unit_r,
                          mesh.boundaries.Upper.name)
    T_cond = (r_o - r) / (r_o - r_inner)
    stokes.bodyforce = args.Ra * (T.sym[0] - T_cond) * unit_r
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
    return mesh, v, P, T, stokes, adv, unit_r


def nusselt(mesh, T, cellsize):
    th = np.linspace(0, 2 * np.pi, 401, endpoint=False)
    p1 = np.column_stack([(r_o - 1.5 * cellsize) * np.cos(th),
                          (r_o - 1.5 * cellsize) * np.sin(th)])
    p2 = np.column_stack([(r_o - 0.5 * cellsize) * np.cos(th),
                          (r_o - 0.5 * cellsize) * np.sin(th)])
    T1 = np.asarray(uw.function.evaluate(T.sym[0], p1))
    T2 = np.asarray(uw.function.evaluate(T.sym[0], p2))
    dTdr = float(np.mean(T2 - T1) / cellsize)
    return -dTdr / (-1.0 / (r_o - r_inner))


def vrms(mesh, v):
    a = np.asarray(uw.function.evaluate(
        v.sym.dot(v.sym), mesh.X.coords))
    return float(np.sqrt(np.mean(a)))


def apply_adaptation_correction(adv, v, v_mesh_arr, mode):
    r"""HOOK — the open next-phase piece.

    After the mover displaces nodes by ``Δx`` over the step
    interval ``Δt``, the mesh has velocity ``v_mesh = Δx/Δt``.
    The SLCN advection–diffusion must transport along the
    *material* velocity relative to the moving mesh
    (ALE): ``V_fn = v_fluid − v_mesh`` for the step(s) following an
    adaptation — otherwise the pure coordinate move is read by the
    solver as a spurious advection of T (cf. the free-surface ALE
    finding: a Lagrangian mesh move needs ``V_fn = v − v_mesh`` or
    convection is non-physically damped). Alternatively: a
    conservative remap of T onto the moved nodes.

    ``none`` → uncorrected baseline (expected to drift; the thing
    the next phase fixes). ``ale`` → not yet implemented.
    """
    if mode == "none":
        return
    raise NotImplementedError(
        "ALE adaptation correction is the NEXT-PHASE deliverable. "
        "Implement: v_mesh field from the adaptation displacement / "
        "Δt, then drive AdvDiffusionSLCN with V_fn = v - v_mesh for "
        "the post-adapt step (or a conservative T remap). See the "
        "design-doc kickoff brief + the free-surface ALE memory.")


def run(mesh, v, P, T, stokes, adv, cellsize, adaptive, tag):
    cache = f"{CDIR}/harness_{tag}_Ra{args.Ra:.0e}_n{args.n_steps}.npz"
    if os.path.exists(cache):
        print(f"  [{tag}] loading cached history {cache}")
        z = np.load(cache)
        return z["t"], z["Nu"], z["vrms"]
    stokes.solve(zero_init_guess=True)
    t_sim = 0.0
    hist_t, hist_Nu, hist_v = [], [], []
    for s in range(args.n_steps):
        dt = adv.estimate_dt()
        adv.solve(timestep=dt, zero_init_guess=False)
        stokes.solve(zero_init_guess=False)
        t_sim += dt
        if adaptive and (s + 1) % args.adapt_every == 0:
            X_before = np.asarray(mesh.X.coords).copy()
            rho = metric_density_from_gradient(
                mesh, T, amp=args.amp, name="harness")
            smooth_mesh_interior(
                mesh, metric=rho, method="anisotropic",
                method_kwargs=dict(aniso_cap=2.0, relax=0.2,
                                   n_outer=8))
            v_mesh = (np.asarray(mesh.X.coords) - X_before) / dt
            apply_adaptation_correction(adv, v, v_mesh,
                                        args.correction)
        Nu = nusselt(mesh, T, cellsize)
        vr = vrms(mesh, v)
        hist_t.append(t_sim)
        hist_Nu.append(Nu)
        hist_v.append(vr)
        if (s + 1) % 5 == 0 or s == 0:
            tt = T.data[:, 0]
            print(f"  [{tag}] step {s+1:2d} t={t_sim:.4f} "
                  f"Nu={Nu:+.3f} vrms={vr:.3e} "
                  f"T=[{tt.min():+.2f},{tt.max():+.2f}]", flush=True)
    t = np.array(hist_t)
    Nu = np.array(hist_Nu)
    vr = np.array(hist_v)
    np.savez(cache, t=t, Nu=Nu, vrms=vr)
    print(f"  [{tag}] cached → {cache}")
    return t, Nu, vr


print(f"=== adaptive-convection harness  Ra={args.Ra:.0e}  "
      f"ref res-{args.ref_res} (uniform) vs adapt res-"
      f"{args.adapt_res} (every {args.adapt_every}, "
      f"correction={args.correction}) ===")
print(f"reference (uniform res-{args.ref_res}):")
m, v, P, T, st, ad, ur = build(args.ref_res, "ref")
tR, NuR, vR = run(m, v, P, T, st, ad, 1.0 / args.ref_res,
                  False, f"ref{args.ref_res}")
print(f"adaptive (res-{args.adapt_res}, correction="
      f"{args.correction}):")
m, v, P, T, st, ad, ur = build(args.adapt_res, "ad")
tA, NuA, vA = run(m, v, P, T, st, ad, 1.0 / args.adapt_res,
                  True, f"adapt{args.adapt_res}_{args.correction}")

# compare on the overlapping physical-time window
tmax = min(tR.max(), tA.max())
tg = np.linspace(min(tR.min(), tA.min()), tmax, 60)
NuR_i = np.interp(tg, tR, NuR)
NuA_i = np.interp(tg, tA, NuA)
vR_i = np.interp(tg, tR, vR)
vA_i = np.interp(tg, tA, vA)
nu_err = float(np.sqrt(np.mean((NuA_i - NuR_i) ** 2)))
v_err = float(np.sqrt(np.mean((vA_i - vR_i) ** 2))
              / max(np.mean(np.abs(vR_i)), 1e-30))
print(f"\nrms ΔNu(adaptive-ref) = {nu_err:.4f}   "
      f"rel rms Δvrms = {v_err:.4f}   "
      f"(adaptive res-{args.adapt_res} vs ref res-{args.ref_res}; "
      f"correction={args.correction})")

fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
ax[0].plot(tR, NuR, "o-", color="k", lw=1.6, ms=3,
           label=f"ref uniform res-{args.ref_res}")
ax[0].plot(tA, NuA, "s--", color="#1f4e8c", lw=1.6, ms=3,
           label=f"adapt res-{args.adapt_res} ({args.correction})")
ax[0].set_xlabel("sim time")
ax[0].set_ylabel("Nu")
ax[0].set_title("Nusselt(t)")
ax[1].plot(tR, vR, "o-", color="k", lw=1.6, ms=3,
           label=f"ref uniform res-{args.ref_res}")
ax[1].plot(tA, vA, "s--", color="#1f4e8c", lw=1.6, ms=3,
           label=f"adapt res-{args.adapt_res} ({args.correction})")
ax[1].set_xlabel("sim time")
ax[1].set_ylabel("vrms")
ax[1].set_title("vrms(t)")
for a in ax:
    a.legend(fontsize=9)
    a.grid(alpha=0.3)
fig.suptitle(f"Adaptive-convection harness — Ra={args.Ra:.0e}  "
             f"(rms ΔNu={nu_err:.3f}; correction="
             f"{args.correction} — ALE correction is next-phase)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = f"{CDIR}/adaptive_convection_harness.png"
fig.savefig(out, dpi=130)
print(f"saved {out}")
