"""Sanity check: does our convection setup blow up *without* the free surface?

Same Ra=1e4, same annulus geometry, same initial T, same operator-split
time loop as `_phase_i_fs_convection_zoo.py` — but with a Nitsche-penalty
free-slip Upper BC at fixed radius instead of the deforming free surface.

If vrms stays in the O(1-10) range out to t~0.02: the convection setup
is fine; the runaway in the zoo run is caused by the free-surface
coupling.

If vrms blows up around t=0.005-0.008 (where the zoo run lost stability):
the convection setup itself has a bug — body-force scaling, BC, IC,
or solver tolerance.
"""

import time as _time
import numpy as np
import sympy

import underworld3 as uw

Ra = 1.0e4
n_steps = 25
res = 20
cellsize = 1.0 / res

mesh = uw.meshing.Annulus(
    radiusInner=0.5, radiusOuter=1.0, cellSize=cellsize, qdegree=3,
)

v = uw.discretisation.MeshVariable(
    "V_chk", mesh, vtype=uw.VarType.VECTOR, degree=2, continuous=True)
p = uw.discretisation.MeshVariable(
    "P_chk", mesh, vtype=uw.VarType.SCALAR, degree=1, continuous=True)
t_soln = uw.discretisation.MeshVariable(
    "T_chk", mesh, vtype=uw.VarType.SCALAR, degree=3, continuous=True)

unit_e_0 = mesh.CoordinateSystem.unit_e_0
r_sym, th_sym = mesh.CoordinateSystem.R
r_i, r_o = 0.5, 1.0

# Stokes — same body force as the zoo benchmark
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes.bodyforce = Ra * t_soln.sym[0] * unit_e_0
stokes.tolerance = 1.0e-5
stokes.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
# Nitsche penalty: enforces v.r̂ ≈ 0 on Upper (free-slip at fixed radius).
# This is what Ex_Convection_Cylinder-FS.py does. NOT a free surface.
stokes.add_natural_bc(
    10000.0 * unit_e_0.dot(v.sym) * unit_e_0.T,
    mesh.boundaries.Upper.name,
)

# AdvDiff
adv_diff = uw.systems.AdvDiffusionSLCN(
    mesh, u_Field=t_soln, V_fn=v.sym, verbose=False)
adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = 1.0
adv_diff.tolerance = 1.0e-4
adv_diff.add_dirichlet_bc(1.0, mesh.boundaries.Lower.name)
adv_diff.add_dirichlet_bc(0.0, mesh.boundaries.Upper.name)

# Initial T (same as zoo benchmark)
init_t = (
    0.01 * sympy.sin(5.0 * th_sym)
         * sympy.sin(np.pi * (r_sym - r_i) / (r_o - r_i))
    + (r_o - r_sym) / (r_o - r_i)
)
t_soln.data[...] = np.asarray(
    uw.function.evaluate(init_t, t_soln.coords)).reshape(-1, 1)

# Initial Stokes solve
stokes.solve(zero_init_guess=True)

vol_initial = abs(float(uw.maths.Integral(mesh, 1.0).evaluate()))

print(f"Ra={Ra}, n_steps={n_steps}, NO free surface (Nitsche penalty Upper)",
      flush=True)
print(f"vol_initial = {vol_initial:.6e}", flush=True)
print("# step  t           dt          vrms        T_avg       wall_s",
      flush=True)

t_sim = 0.0
for s in range(n_steps):
    t0 = _time.time()
    stokes.solve(zero_init_guess=False)
    dt = float(adv_diff.estimate_dt())
    adv_diff.solve(timestep=dt, zero_init_guess=False)
    vol = abs(float(uw.maths.Integral(mesh, 1.0).evaluate()))
    vrms = float(np.sqrt(
        abs(float(uw.maths.Integral(mesh, v.sym.dot(v.sym)).evaluate())) / vol))
    T_avg = abs(float(
        uw.maths.Integral(mesh, t_soln.sym[0]).evaluate())) / vol
    wall = _time.time() - t0
    t_sim += dt
    print(f"  {s+1:3d}    {t_sim:.3e}   {dt:.3e}   {vrms:.3e}   "
          f"{T_avg:.3e}   {wall:.1f}", flush=True)
