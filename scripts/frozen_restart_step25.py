"""Frozen-mesh restart diagnostic.

Load the BROKEN run's step-25 state (deformed mesh + its T + V),
replicate the zoo's exact Stokes/adv-diff configuration, then run
10 steps with the mesh FROZEN (no deform_by_inc, no _deform_mesh).

Interpretation:
  - If Nu RECOVERS (climbs back toward the strong regime) → the
    per-step mesh-deformation machinery is corrupting state; the
    step-25 geometry itself is fine. (mesh-DEFORMATION problem)
  - If Nu STAYS LOW / degrades further → the step-25 deformed
    geometry is itself the problem (bad cells). (mesh-DISTORTION
    problem)

Fixed-mesh adv-diff was already shown clean on a fresh Annulus,
so a frozen run that stays broken would implicate the deformed
geometry specifically.
"""
from __future__ import annotations
import os
import numpy as np
import sympy

import underworld3 as uw

SNAP = "output/convection_zoo_snapshots_rk4_bothfixes"
ROOT = "uw_rk4_step0025"
OUT = "output/convection_zoo_snapshots_frozen25"
N_STEPS = 10
Ra = 1.0e5
rho_g = 1.0e5
r_inner, r_o = 0.5, 1.0
pair_tag = "v2p1"

os.makedirs(OUT, exist_ok=True)

# 1. Load the step-25 deformed mesh + its T, V from the broken run
mesh = uw.discretisation.Mesh(
    f"{SNAP}/{ROOT}.mesh.00000.h5")
r, th = mesh.CoordinateSystem.R

v = uw.discretisation.MeshVariable(
    f"V_conv_{pair_tag}", mesh, vtype=uw.VarType.VECTOR,
    degree=2, continuous=True, varsymbol=r"\mathbf{v}")
P = uw.discretisation.MeshVariable(
    f"P_conv_{pair_tag}", mesh, vtype=uw.VarType.SCALAR,
    degree=1, continuous=True, varsymbol="p")
t_soln = uw.discretisation.MeshVariable(
    f"T_conv_{pair_tag}", mesh, vtype=uw.VarType.SCALAR,
    degree=3, continuous=True, varsymbol="T")

t_soln.read_timestep(ROOT, f"T_conv_{pair_tag}", 0,
                     outputPath=SNAP)
v.read_timestep(ROOT, f"V_conv_{pair_tag}", 0,
                outputPath=SNAP)

print(f"Loaded step-25 state: "
      f"T=[{t_soln.data.min():+.4f},{t_soln.data.max():+.4f}] "
      f"|V|max={np.abs(np.asarray(v.data)).max():.3e}",
      flush=True)

# 2. Stokes — EXACT zoo config (bodyforce, BCs)
stokes = uw.systems.Stokes(mesh, velocityField=v,
                           pressureField=P)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes.penalty = 0.0
stokes.tolerance = 1.0e-5
unit_r = mesh.CoordinateSystem.unit_e_0
T_cond = (r_o - r) / (r_o - r_inner)
# zoo bodyforce: lithostatic + buoyancy perturbation
stokes.bodyforce = (
    -rho_g * unit_r
    + Ra * (t_soln.sym[0] - T_cond) * unit_r
)
stokes.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
# zoo Upper natural BC with all stabilisation terms = 0
# (delta_h_load=0, fssa_theta=0, nitsche_penalty=0) → traction-free
stokes.add_natural_bc(sympy.Matrix([0.0, 0.0]),
                      mesh.boundaries.Upper.name)

# 3. AdvDiffusionSLCN — zoo config; theta=0.5 (CN),
#    monotone_mode set post-construction (baseline-lineage style)
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

# Re-solve Stokes once to get v consistent with loaded T on this
# (fixed) geometry
stokes.solve(zero_init_guess=False)


def _nu():
    """Zoo-convention Nu: -∮_Upper ∇T·n̂ dS (unnormalised)."""
    gTn = (t_soln.sym[0].diff(mesh.X[0])
           * mesh.CoordinateSystem.unit_e_0[0]
           + t_soln.sym[0].diff(mesh.X[1])
           * mesh.CoordinateSystem.unit_e_0[1])
    Nu = -float(uw.maths.BdIntegral(
        mesh, gTn, mesh.boundaries.Upper.name).evaluate())
    vol = abs(float(uw.maths.Integral(mesh, 1.0).evaluate()))
    vrms = float(np.sqrt(abs(float(
        uw.maths.Integral(mesh, v.sym.dot(v.sym)).evaluate()))
        / vol))
    return Nu, vrms


Nu0, vrms0 = _nu()
print(f"=== FROZEN restart from step 25 "
      f"(mesh NOT deformed) ===", flush=True)
print(f"  step  0 (loaded): Nu={Nu0:+8.3f} vrms={vrms0:8.3f} "
      f"T=[{t_soln.data.min():+.4f},{t_soln.data.max():+.4f}]",
      flush=True)

for s in range(N_STEPS):
    dt = adv_diff.estimate_dt()
    adv_diff.solve(timestep=dt, zero_init_guess=False)
    stokes.solve(zero_init_guess=False)   # NO mesh deform
    Nu, vrms = _nu()
    print(f"  step {s+1:2d}: dt={dt:.3e} Nu={Nu:+8.3f} "
          f"vrms={vrms:8.3f} "
          f"T=[{t_soln.data.min():+.4f},"
          f"{t_soln.data.max():+.4f}]", flush=True)
    mesh.write_timestep(
        filename=f"uw_frozen25_step{s+1:04d}", index=0,
        outputPath=OUT, meshVars=[t_soln, v],
        meshUpdates=True, create_xdmf=True)

print("Done.", flush=True)
