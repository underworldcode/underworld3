"""Fixed-mesh free-slip thermal convection on the same Annulus
geometry, with the same SemiLagrangian advection-diffusion + Ra
+ monotone clamp + P3 T as the free-surface runs. NO mesh
deformation, NO surface load BC, NO FSSA, NO Nitsche.

Diagnostic: if the T field stays clean here (no spotty BL), then
the regression is in the mesh-movement path. If it's also spotty,
the regression is in the adv-diff system itself.
"""
from __future__ import annotations
import os
import sys
import argparse
import time
import numpy as np
import sympy

import underworld3 as uw

p = argparse.ArgumentParser()
p.add_argument('--res', type=int, default=20)
p.add_argument('--n-steps', type=int, default=35)
p.add_argument('--Ra', type=float, default=1.0e5)
p.add_argument('--diffusion-theta', type=float, default=0.5)
p.add_argument('--monotone-mode', type=str, default='clamp')
p.add_argument('--t-degree', type=int, default=3)
p.add_argument('--v-degree', type=int, default=2)
p.add_argument('--p-degree', type=int, default=1)
p.add_argument('--stokes-tol', type=float, default=1.0e-5)
p.add_argument('--capture-every', type=int, default=5)
p.add_argument('--snap-dir', type=str,
               default='output/convection_zoo_snapshots_fixed_mesh')
args = p.parse_args()

os.makedirs(args.snap_dir, exist_ok=True)

# Build mesh: same Annulus as the convection zoo
r_inner, r_o = 0.5, 1.0
cellsize = 1.0 / args.res
qdeg = max(3, args.v_degree + 1)
mesh = uw.meshing.Annulus(
    radiusOuter=r_o, radiusInner=r_inner,
    cellSize=cellsize, qdegree=qdeg,
)
r, th = mesh.CoordinateSystem.R

pair_tag = f"v{args.v_degree}p{args.p_degree}"
v = uw.discretisation.MeshVariable(
    f"V_conv_{pair_tag}", mesh, vtype=uw.VarType.VECTOR,
    degree=args.v_degree, continuous=True,
    varsymbol=r"\mathbf{v}")
P = uw.discretisation.MeshVariable(
    f"P_conv_{pair_tag}", mesh, vtype=uw.VarType.SCALAR,
    degree=args.p_degree, continuous=True, varsymbol="p")
t_soln = uw.discretisation.MeshVariable(
    f"T_conv_{pair_tag}", mesh, vtype=uw.VarType.SCALAR,
    degree=args.t_degree, continuous=True, varsymbol="T")

# Stokes solver: free-slip on both boundaries (radial velocity = 0)
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=P)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes.tolerance = args.stokes_tol
stokes.penalty = 0.0
unit_r = mesh.CoordinateSystem.unit_e_0
# Free-slip = no penetration: v · r̂ = 0 on both boundaries.
# Use natural BCs with a large normal penalty for simplicity, or
# Dirichlet on radial component. Easiest: full-no-slip on inner +
# free-slip on outer. Wait — for the test, full no-slip inner +
# free-slip outer matches the free-surface case's inner BC.
stokes.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
# Free-slip on outer: radial velocity = 0. Approximate via natural
# BC with penalty on v·r̂.
KFS = 1.0e6
fs_term = (KFS * v.sym.dot(unit_r) * unit_r)
stokes.add_natural_bc(fs_term, mesh.boundaries.Upper.name)

# Buoyancy: body force = Ra · (T - T_cond) · r̂ (same as zoo, but
# without -rho_g·r̂ since no free-surface restoring needed)
T_cond = (r_o - r) / (r_o - r_inner)
stokes.bodyforce = args.Ra * (t_soln.sym[0] - T_cond) * unit_r

# AdvDiffusion: same as zoo — SLCN with monotone clamp + theta
adv_diff = uw.systems.AdvDiffusionSLCN(
    mesh, u_Field=t_soln, V_fn=v.sym, verbose=False,
    theta=args.diffusion_theta,
    monotone_mode=(None if args.monotone_mode in ('None', 'none')
                   else args.monotone_mode),
)
adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = 1.0
adv_diff.tolerance = 1.0e-4
adv_diff.add_dirichlet_bc(1.0, mesh.boundaries.Lower.name)
adv_diff.add_dirichlet_bc(0.0, mesh.boundaries.Upper.name)

# Initial T: conductive + mode-5 perturbation (same as zoo)
init_t = (
    0.01 * sympy.sin(5.0 * th)
    * sympy.sin(np.pi * (r - r_inner) / (r_o - r_inner))
    + (r_o - r) / (r_o - r_inner)
)
t_soln.data[...] = np.asarray(uw.function.evaluate(
    init_t, t_soln.coords)).reshape(-1, 1)

# Initial Stokes solve
stokes.solve(zero_init_guess=True)


def _diagnostics(scheme_label, step, t_sim):
    """Match the zoo's diagnostic output style."""
    v_arr = np.asarray(uw.function.evaluate(
        v.sym.dot(v.sym), mesh.X.coords))
    vrms = float(np.sqrt(np.mean(v_arr)))
    T_arr = t_soln.data[:, 0]
    # Nu = -(dT/dr at r=r_o) / (dT_cond/dr at r=r_o); approximate
    # by sampling T near the upper boundary
    upper_th = np.linspace(0, 2 * np.pi, 401, endpoint=False)
    upper_r1 = np.full_like(upper_th, r_o - 1.5 * cellsize)
    upper_r2 = np.full_like(upper_th, r_o - 0.5 * cellsize)
    pts1 = np.column_stack([upper_r1 * np.cos(upper_th),
                            upper_r1 * np.sin(upper_th)])
    pts2 = np.column_stack([upper_r2 * np.cos(upper_th),
                            upper_r2 * np.sin(upper_th)])
    T1 = np.asarray(uw.function.evaluate(t_soln.sym[0], pts1))
    T2 = np.asarray(uw.function.evaluate(t_soln.sym[0], pts2))
    dTdr = float(np.mean(T2 - T1) / cellsize)
    Nu = -dTdr / (-1.0 / (r_o - r_inner))   # / conductive gradient
    return vrms, Nu, T_arr.min(), T_arr.max()


def _capture(scheme_label, step, t_sim):
    label = f'step{step:04d}' if step > 0 else 'final'
    fname = f"uw_{scheme_label}_{label}"
    mesh.write_timestep(
        filename=fname, index=0, outputPath=args.snap_dir,
        meshVars=[t_soln, v], meshUpdates=True,
        create_xdmf=True)


# Time loop: pure explicit advection-diffusion. Use adv_diff's
# estimate_dt for stability.
t_sim = 0.0
print(f"=== fixed_mesh_rk4_sl ===", flush=True)
print(f"diffusion AM θ = {args.diffusion_theta} "
      f"({'CN' if args.diffusion_theta == 0.5 else 'BE' if args.diffusion_theta == 1.0 else 'mixed'})",
      flush=True)

for s in range(args.n_steps):
    dt = adv_diff.estimate_dt()
    t_pre = t_soln.data[:, 0].copy()

    adv_diff.solve(timestep=dt, zero_init_guess=False)
    t_post = t_soln.data[:, 0].copy()

    # Re-solve Stokes with new T
    stokes.solve(zero_init_guess=False)

    t_sim += dt
    dT_max = float(np.abs(t_post - t_pre).max())
    print(f"  step {s+1}: t={t_sim:.4f} Δt={dt:.3e} "
          f"T=[{t_post.min():+.4f},{t_post.max():+.4f}] "
          f"|ΔT|max={dT_max:.3e}", flush=True)

    if args.capture_every > 0 and (s + 1) % args.capture_every == 0:
        _capture('fixed_mesh_rk4_sl', s + 1, t_sim)

    if (s + 1) % args.capture_every == 0:
        vrms, Nu, Tmin, Tmax = _diagnostics(
            'fixed_mesh_rk4_sl', s + 1, t_sim)
        print(f"  [step{s+1:04d}] vrms={vrms:.3e} Nu={Nu:+.3f} "
              f"T=[{Tmin:+.4f},{Tmax:+.4f}]", flush=True)

_capture('fixed_mesh_rk4_sl', 0, t_sim)
print(f"Done. Saved checkpoints to {args.snap_dir}", flush=True)
