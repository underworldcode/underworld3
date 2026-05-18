"""Wall-clock per-step cost: res-24 plain step vs res-16 plain
step vs one a16s aggressive pristine-adaptation overhead (with
the REORDERED loop — so the adaptation adds only metric+mover+
remap; the loop's single Stokes is already counted in the step).
Contends with live runs ⇒ absolutes inflated, RATIO valid.
"""
from __future__ import annotations
import time
import numpy as np
import sympy
import underworld3 as uw
from underworld3.meshing import (
    smooth_mesh_interior, metric_density_from_gradient)

RA, r_inner, r_o = 1.0e5, 0.5, 1.0


def build(res):
    m = uw.meshing.Annulus(radiusOuter=r_o, radiusInner=r_inner,
                           cellSize=1.0 / res, qdegree=3)
    r, th = m.CoordinateSystem.R
    v = uw.discretisation.MeshVariable(f"V{res}", m,
        vtype=uw.VarType.VECTOR, degree=2, continuous=True)
    P = uw.discretisation.MeshVariable(f"P{res}", m,
        vtype=uw.VarType.SCALAR, degree=1, continuous=True)
    T = uw.discretisation.MeshVariable(f"T{res}", m,
        vtype=uw.VarType.SCALAR, degree=3, continuous=True)
    st = uw.systems.Stokes(m, velocityField=v, pressureField=P)
    st.constitutive_model = uw.constitutive_models.ViscousFlowModel
    st.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    st.tolerance = 1.0e-5
    st.penalty = 0.0
    ur = m.CoordinateSystem.unit_e_0
    st.add_essential_bc((0.0, 0.0), m.boundaries.Lower.name)
    st.add_natural_bc(1.0e6 * v.sym.dot(ur) * ur,
                      m.boundaries.Upper.name)
    st.bodyforce = RA * (T.sym[0]
                         - (r_o - r) / (r_o - r_inner)) * ur
    ad = uw.systems.AdvDiffusionSLCN(m, u_Field=T, V_fn=v.sym,
        verbose=False, theta=0.5, monotone_mode="clamp")
    ad.constitutive_model = uw.constitutive_models.DiffusionModel
    ad.constitutive_model.Parameters.diffusivity = 1.0
    ad.tolerance = 1.0e-4
    ad.add_dirichlet_bc(1.0, m.boundaries.Lower.name)
    ad.add_dirichlet_bc(0.0, m.boundaries.Upper.name)
    T.data[...] = np.asarray(uw.function.evaluate(
        0.01 * sympy.sin(5.0 * th) * sympy.sin(
            np.pi * (r - r_inner) / (r_o - r_inner))
        + (r_o - r) / (r_o - r_inner), T.coords)).reshape(-1, 1)
    return m, v, P, T, st, ad


def time_steps(res, n=4):
    m, v, P, T, st, ad = build(res)
    st.solve(zero_init_guess=True)
    for _ in range(4):                       # warm (grow plumes)
        ad.solve(timestep=ad.estimate_dt(), zero_init_guess=False)
        st.solve(zero_init_guess=False)
    t0 = time.perf_counter()
    for _ in range(n):
        ad.solve(timestep=ad.estimate_dt(), zero_init_guess=False)
        st.solve(zero_init_guess=False)
    ts = (time.perf_counter() - t0) / n
    return ts, m, v, P, T, st, ad


t24, *_ = time_steps(24)
print(f"res-24 plain (adv+stokes) step : {t24:7.3f} s")
t16, m, v, P, T, st, ad = time_steps(16)
print(f"res-16 plain (adv+stokes) step : {t16:7.3f} s")

# one a16s aggressive pristine adaptation OVERHEAD (no stokes —
# the reordered loop's single stokes is counted in t16)
X0 = np.asarray(m.X.coords).copy()
X0_Tx = np.asarray(T.coords).copy()
X_prev = np.asarray(m.X.coords).copy()
ta = time.perf_counter()
vals0 = np.asarray(uw.function.evaluate(T.sym[0], X0_Tx)).reshape(-1)
m._deform_mesh(X0); T.data[:, 0] = vals0
rho = metric_density_from_gradient(m, T, amp=16.0, name="mb")
X0c = np.asarray(m.X.coords).copy(); T0 = np.asarray(T.data).copy()
smooth_mesh_interior(m, metric=rho, method="anisotropic",
                     method_kwargs=dict(aniso_cap=4.0, relax=0.05,
                                        n_outer=25))
new_X = np.asarray(m.X.coords).copy()
new_Tx = np.asarray(T.coords).copy()
m._deform_mesh(X0c); T.data[...] = T0
valsN = np.asarray(uw.function.evaluate(T.sym[0], new_Tx)).reshape(-1)
m._deform_mesh(new_X); T.data[:, 0] = valsN
t_ov = time.perf_counter() - ta
print(f"a16s adaptation OVERHEAD (no stokes): {t_ov:7.3f} s")

eff = t16 + t_ov / 5.0
r_step = eff / t24
print(f"\na16s effective per-step (5 steps + 1 adapt)/5 = "
      f"{eff:7.3f} s")
print(f"per-STEP slowdown   a16s / ref24 = {r_step:5.2f}x")
print(f"dt penalty (ref24/a16s, from histories) = 1.07x")
print(f"per-SIM-TIME slowdown a16s / ref24 ≈ {r_step*1.07:5.2f}x")
print(f"(context: a16p adapt is cheaper [n_outer=8] & dt 0.87x; "
      f"uniform res-16 is ~{t16/t24:.2f}x/step & 0.61x steps "
      f"⇒ ~{(t16/t24)*0.61:.2f}x/sim-time vs ref24)")
