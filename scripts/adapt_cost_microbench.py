"""Micro-benchmark: how many (adv+stokes) solve-steps does ONE
pristine adaptation cost? Isolated single process (no contention),
res-16 Ra=1e5, same setup as the saturation runner. Breaks the
adaptation into metric / mover / remap+restokes.
"""
from __future__ import annotations
import time
import numpy as np
import sympy
import underworld3 as uw
from underworld3.meshing import (
    smooth_mesh_interior, metric_density_from_gradient)

RA, RES, r_inner, r_o = 1.0e5, 16, 0.5, 1.0

m = uw.meshing.Annulus(radiusOuter=r_o, radiusInner=r_inner,
                        cellSize=1.0 / RES, qdegree=3)
r, th = m.CoordinateSystem.R
v = uw.discretisation.MeshVariable("V", m, vtype=uw.VarType.VECTOR,
                                   degree=2, continuous=True)
P = uw.discretisation.MeshVariable("P", m, vtype=uw.VarType.SCALAR,
                                   degree=1, continuous=True)
T = uw.discretisation.MeshVariable("T", m, vtype=uw.VarType.SCALAR,
                                   degree=3, continuous=True)
stokes = uw.systems.Stokes(m, velocityField=v, pressureField=P)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes.tolerance = 1.0e-5
stokes.penalty = 0.0
ur = m.CoordinateSystem.unit_e_0
stokes.add_essential_bc((0.0, 0.0), m.boundaries.Lower.name)
stokes.add_natural_bc(1.0e6 * v.sym.dot(ur) * ur,
                      m.boundaries.Upper.name)
stokes.bodyforce = RA * (T.sym[0] - (r_o - r) / (r_o - r_inner)) * ur
adv = uw.systems.AdvDiffusionSLCN(m, u_Field=T, V_fn=v.sym,
                                  verbose=False, theta=0.5,
                                  monotone_mode="clamp")
adv.constitutive_model = uw.constitutive_models.DiffusionModel
adv.constitutive_model.Parameters.diffusivity = 1.0
adv.tolerance = 1.0e-4
adv.add_dirichlet_bc(1.0, m.boundaries.Lower.name)
adv.add_dirichlet_bc(0.0, m.boundaries.Upper.name)
init_t = (0.01 * sympy.sin(5.0 * th)
          * sympy.sin(np.pi * (r - r_inner) / (r_o - r_inner))
          + (r_o - r) / (r_o - r_inner))
T.data[...] = np.asarray(uw.function.evaluate(
    init_t, T.coords)).reshape(-1, 1)
X0 = np.asarray(m.X.coords).copy()
X0_Tx = np.asarray(T.coords).copy()
stokes.solve(zero_init_guess=True)

# warm a few steps so the field is representative (plumes growing)
for _ in range(8):
    dt = adv.estimate_dt()
    adv.solve(timestep=dt, zero_init_guess=False)
    stokes.solve(zero_init_guess=False)

# (1) time plain (adv+stokes) steps
N = 8
t0 = time.perf_counter()
for _ in range(N):
    dt = adv.estimate_dt()
    adv.solve(timestep=dt, zero_init_guess=False)
    stokes.solve(zero_init_guess=False)
t_step = (time.perf_counter() - t0) / N
print(f"plain (adv+stokes) step      : {t_step:6.3f} s  (mean of {N})")

# (2) time ONE pristine adaptation, broken down
X_prev = np.asarray(m.X.coords).copy()
T_prev = np.asarray(T.data).copy()

ta = time.perf_counter()
vals0 = np.asarray(uw.function.evaluate(T.sym[0], X0_Tx)).reshape(-1)
m._deform_mesh(X0); T.data[:, 0] = vals0
t_remap_in = time.perf_counter() - ta

tb = time.perf_counter()
rho = metric_density_from_gradient(m, T, amp=8.0, name="mb")
t_metric = time.perf_counter() - tb

X0c = np.asarray(m.X.coords).copy(); T0 = np.asarray(T.data).copy()
tc = time.perf_counter()
smooth_mesh_interior(m, metric=rho, method="anisotropic",
                     method_kwargs=dict(aniso_cap=2.0, relax=0.2,
                                        n_outer=8))
t_mover = time.perf_counter() - tc

new_X = np.asarray(m.X.coords).copy()
new_Tx = np.asarray(T.coords).copy()
td = time.perf_counter()
m._deform_mesh(X0c); T.data[...] = T0
valsN = np.asarray(uw.function.evaluate(T.sym[0], new_Tx)).reshape(-1)
m._deform_mesh(new_X); T.data[:, 0] = valsN
t_remap_out = time.perf_counter() - td

te = time.perf_counter()
stokes.solve(zero_init_guess=False)
t_restokes = time.perf_counter() - te

t_adapt = (t_remap_in + t_metric + t_mover + t_remap_out
           + t_restokes)
print(f"  remap-in  (eval+deform)    : {t_remap_in:6.3f} s")
print(f"  metric    (grad projection): {t_metric:6.3f} s")
print(f"  MOVER     (anisotropic)    : {t_mover:6.3f} s")
print(f"  remap-out (eval+deform x2) : {t_remap_out:6.3f} s")
print(f"  re-stokes                  : {t_restokes:6.3f} s")
print(f"ONE pristine adaptation TOTAL: {t_adapt:6.3f} s")
print(f"\nratio  adaptation / (adv+stokes step) = "
      f"{t_adapt / t_step:5.1f}x")
print(f"amortised over adapt-every-5: +{t_adapt/(5*t_step):.1f}x "
      f"work vs a non-adaptive res-16 run "
      f"(5 steps cost {5*t_step:.2f}s + 1 adapt {t_adapt:.2f}s)")
