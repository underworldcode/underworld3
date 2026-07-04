"""Isolate _deform_mesh via a pure Winslow node move.

smooth_mesh_interior (PR #190) calls mesh._deform_mesh() — the
SAME path deform_by_inc uses — but with NO diffuser and NO physics
solve: just a graph-Laplacian interior-node move, boundary pinned.

Sequence (on the broken run's step-25 deformed state):
  1. load mesh + T + V, set up Stokes + adv-diff (zoo config)
  2. Stokes solve  -> Nu_before, vrms_before; snapshot T DOFs
  3. smooth_mesh_interior(n_iters=5)  [interior nodes move,
     Lower/Upper pinned, no solve]
  4. assert T DOF values UNCHANGED (Lagrangian carry — no
     re-interpolation in _deform_mesh)
  5. Stokes solve again -> Nu_after, vrms_after
  6. 5 frozen steps (adv-diff + Stokes, NO further deform)

Reads:
  - T_before == T_after  -> DOFs carried (expected)
  - Nu_after ~ Nu_before AND heals over frozen steps
        -> _deform_mesh path clean; damage is diffuser-specific
  - Nu_after garbage / no heal despite unchanged DOFs
        -> DS / solver tooling failed to rebuild after node move
"""
from __future__ import annotations
import os
import numpy as np
import sympy

import underworld3 as uw
from underworld3.meshing import smooth_mesh_interior

SNAP = "output/convection_zoo_snapshots_rk4_bothfixes"
ROOT = "uw_rk4_step0025"
Ra, rho_g = 1.0e5, 1.0e5
r_inner, r_o = 0.5, 1.0
pt = "v2p1"

mesh = uw.discretisation.Mesh(f"{SNAP}/{ROOT}.mesh.00000.h5")
r, th = mesh.CoordinateSystem.R
v = uw.discretisation.MeshVariable(
    f"V_conv_{pt}", mesh, vtype=uw.VarType.VECTOR,
    degree=2, continuous=True, varsymbol=r"\mathbf{v}")
P = uw.discretisation.MeshVariable(
    f"P_conv_{pt}", mesh, vtype=uw.VarType.SCALAR,
    degree=1, continuous=True, varsymbol="p")
t_soln = uw.discretisation.MeshVariable(
    f"T_conv_{pt}", mesh, vtype=uw.VarType.SCALAR,
    degree=3, continuous=True, varsymbol="T")
t_soln.read_timestep(ROOT, f"T_conv_{pt}", 0, outputPath=SNAP)
v.read_timestep(ROOT, f"V_conv_{pt}", 0, outputPath=SNAP)

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=P)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes.penalty = 0.0
stokes.tolerance = 1.0e-5
unit_r = mesh.CoordinateSystem.unit_e_0
T_cond = (r_o - r) / (r_o - r_inner)
stokes.bodyforce = (-rho_g * unit_r
                    + Ra * (t_soln.sym[0] - T_cond) * unit_r)
stokes.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
stokes.add_natural_bc(sympy.Matrix([0.0, 0.0]),
                      mesh.boundaries.Upper.name)

adv_diff = uw.systems.AdvDiffusionSLCN(
    mesh, u_Field=t_soln, V_fn=v.sym, verbose=False)
adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = 1.0
adv_diff.tolerance = 1.0e-4
adv_diff.add_dirichlet_bc(1.0, mesh.boundaries.Lower.name)
adv_diff.add_dirichlet_bc(0.0, mesh.boundaries.Upper.name)
adv_diff.DuDt.theta = 0.5
adv_diff.DFDt.theta = 0.5
adv_diff.DuDt.monotone_mode = "clamp"
adv_diff.DFDt.monotone_mode = "clamp"


def _diag():
    gTn = (t_soln.sym[0].diff(mesh.X[0]) * unit_r[0]
           + t_soln.sym[0].diff(mesh.X[1]) * unit_r[1])
    Nu = -float(uw.maths.BdIntegral(
        mesh, gTn, mesh.boundaries.Upper.name).evaluate())
    vol = abs(float(uw.maths.Integral(mesh, 1.0).evaluate()))
    vrms = float(np.sqrt(abs(float(
        uw.maths.Integral(mesh, v.sym.dot(v.sym)).evaluate()))
        / vol))
    return Nu, vrms


stokes.solve(zero_init_guess=False)
Nu0, vrms0 = _diag()
T_before = np.asarray(t_soln.data).copy()
X_before = np.asarray(mesh.X.coords).copy()
print(f"[before Winslow]  Nu={Nu0:+8.3f} vrms={vrms0:8.3f}",
      flush=True)

# --- the pure node move: Winslow, boundary pinned, NO solve ---
smooth_mesh_interior(mesh, n_iters=5, alpha=0.5)

T_after = np.asarray(t_soln.data)
X_after = np.asarray(mesh.X.coords)
dT = float(np.abs(T_after - T_before).max())
dX = float(np.linalg.norm(X_after - X_before, axis=1).max())
# boundary-vertex movement check
rr = np.sqrt(X_after[:, 0] ** 2 + X_after[:, 1] ** 2)
rr0 = np.sqrt(X_before[:, 0] ** 2 + X_before[:, 1] ** 2)
is_b = (np.abs(rr0 - r_inner) < 1e-6) | (np.abs(rr0 - r_o) < 1e-6)
bnd_move = (float(np.abs(rr - rr0)[is_b].max())
            if is_b.any() else 0.0)
print(f"[Winslow move]    max|dT_dof|={dT:.3e}  "
      f"max node move={dX:.3e}  max boundary move={bnd_move:.3e}",
      flush=True)
print(f"  -> T DOFs carried unchanged: {dT < 1e-12}", flush=True)

stokes.solve(zero_init_guess=False)
Nu1, vrms1 = _diag()
print(f"[after Winslow]   Nu={Nu1:+8.3f} vrms={vrms1:8.3f}  "
      f"(ΔNu vs before = {Nu1-Nu0:+.3f})", flush=True)

print("--- 5 frozen steps after the Winslow move "
      "(no further deform) ---", flush=True)
for s in range(5):
    dt = adv_diff.estimate_dt()
    adv_diff.solve(timestep=dt, zero_init_guess=False)
    stokes.solve(zero_init_guess=False)
    Nu, vrms = _diag()
    print(f"  frozen step {s+1}: Nu={Nu:+8.3f} vrms={vrms:8.3f} "
          f"T=[{t_soln.data.min():+.4f},{t_soln.data.max():+.4f}]",
          flush=True)
print("Done.", flush=True)
