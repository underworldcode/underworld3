"""Probe: which stress quantity at the internal interface predicts the
isostatic-equilibrium topography?

Background: the stress-equilibrium integrator needs h_eq = (stress) / (Δρ g)
measured at the flat internal interface. Using the total radial normal stress
σ_rr = τ_rr - p underestimates the kinematic (rk4) equilibrium (h_pole ≈ 0.024)
by ~2.5×. This probe decomposes the stress at the pole node (above the blob)
into τ_rr (deviatoric) and p, and prints candidate topographies, to find the
correct quantity / reference.

Internal boundary ⇒ no held-lid BC is possible (Nitsche/penalty/constraint are
one-sided domain-boundary operators). h_eq must come from the volumetric field.
"""

import numpy as np
import sympy
import underworld3 as uw

import nest_asyncio
nest_asyncio.apply()

res = 16
blob_amp = 0.5
r_inner, r_o, r_outer = 0.5, 1.0, 1.5
cellsize = 1.0 / res
x_b, y_b, sigma_b = 0.85, 0.0, 0.06

mesh = uw.meshing.AnnulusInternalBoundary(
    radiusOuter=r_outer, radiusInternal=r_o, radiusInner=r_inner,
    cellSize_Outer=3.0 * cellsize, cellSize=cellsize, qdegree=3)

r, th = mesh.CoordinateSystem.R
unit_r = mesh.CoordinateSystem.unit_e_0

blob_fn = sympy.exp(-((mesh.X[0] - x_b) ** 2 + (mesh.X[1] - y_b) ** 2)
                    / (2.0 * sigma_b ** 2))

v = uw.discretisation.MeshVariable("V", mesh, vtype=uw.VarType.VECTOR,
                                   degree=2, continuous=True)
p = uw.discretisation.MeshVariable("P", mesh, vtype=uw.VarType.SCALAR,
                                   degree=1, continuous=True)
M = uw.discretisation.MeshVariable("M", mesh, vtype=uw.VarType.SCALAR,
                                   degree=0, continuous=False)

layer_fn = sympy.Piecewise((1.0, r <= r_o), (0.0, True))
M.data[:, 0] = np.asarray(uw.function.evaluate(layer_fn, M.coords)).flatten()

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 0.1 + 0.9 * M.sym[0]
stokes.penalty = 0.0
stokes.bodyforce = -(M.sym[0] - blob_amp * blob_fn) * unit_r
stokes.tolerance = 1.0e-6
stokes.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
stokes.add_essential_bc((0.0, 0.0), mesh.boundaries.Upper.name)
stokes.solve()

# Internal-boundary node identification (flat IC)
X0 = mesh.X.coords.copy()
R0 = np.sqrt(X0[:, 0] ** 2 + X0[:, 1] ** 2)
TH0 = np.arctan2(X0[:, 1], X0[:, 0])
is_int = np.abs(R0 - r_o) < 0.5 * cellsize / r_o
int_idx = np.where(is_int)[0]
int_idx = int_idx[np.argsort(TH0[int_idx])]
int_th = TH0[int_idx]
ip = int_idx[int(np.argmin(np.abs(int_th)))]      # pole node (θ≈0)
pole_xy = mesh.X.coords[ip:ip + 1]

# Stress decomposition at the pole
sigma = stokes.stress
tau = stokes.stress_deviator
sig_rr = (unit_r * sigma * unit_r.T)[0, 0]
tau_rr = (unit_r * tau * unit_r.T)[0, 0]


def ev(expr):
    return float(np.asarray(uw.function.evaluate(expr, pole_xy)).flatten()[0])


sig_rr_v = ev(sig_rr)
tau_rr_v = ev(tau_rr)
p_v = ev(p.sym[0])
u_n_v = ev(v.sym.dot(unit_r))

print("\n=== Stress decomposition at the pole (flat internal interface) ===")
print(f"  rk4 kinematic equilibrium h_pole  ≈ +0.024  (target)")
print(f"  u_n_pole          = {u_n_v:+.5e}")
print(f"  p   (pressure)    = {p_v:+.5e}")
print(f"  τ_rr (deviatoric) = {tau_rr_v:+.5e}")
print(f"  σ_rr = τ_rr - p   = {sig_rr_v:+.5e}")
print("\n  candidate h_eq = (quantity)/(Δρ g),  Δρ g = 1:")
print(f"    σ_rr            -> {sig_rr_v:+.5e}")
print(f"    -p              -> {-p_v:+.5e}")
print(f"    τ_rr            -> {tau_rr_v:+.5e}")
print(f"    -p + τ_rr (=σ)  -> {sig_rr_v:+.5e}")
print(f"    +p              -> {p_v:+.5e}")
print(f"    -σ_rr           -> {-sig_rr_v:+.5e}")

# Also report the surface-averaged pressure as a candidate hydrostatic datum:
p_int = np.asarray(uw.function.evaluate(p.sym[0],
                                        mesh.X.coords[int_idx])).flatten()
sig_int = np.asarray(uw.function.evaluate(sig_rr,
                                          mesh.X.coords[int_idx])).flatten()
print(f"\n  mean p over interface          = {p_int.mean():+.5e}")
print(f"  σ_rr at pole minus mean σ_rr   = {sig_int[int(np.argmin(np.abs(int_th)))] - sig_int.mean():+.5e}")
print(f"  (-p at pole) minus (-mean p)   = {-(p_v) - (-p_int.mean()):+.5e}")


# ----------------------------------------------------------------------
# HELD-LID measurement: impose u_n=0 on the internal interface with the
# DIRECT penalty (natural BC) — well-posed on an internal facet because
# (v·n)² is normal-sign-independent (unlike Nitsche). The resulting
# n·σ·n is the rigid-lid dynamic topography = infinite-time relaxation.
# Stress taken from a PROJECTION (not the penalty reaction / multiplier).
# ----------------------------------------------------------------------
print("\n=== HELD-LID (penalty free-slip) dynamic topography ===")
v_h = uw.discretisation.MeshVariable("Vh", mesh, vtype=uw.VarType.VECTOR,
                                     degree=2, continuous=True)
p_h = uw.discretisation.MeshVariable("Ph", mesh, vtype=uw.VarType.SCALAR,
                                     degree=1, continuous=True)
topo = uw.discretisation.MeshVariable("topo", mesh,
                                      vtype=uw.VarType.SCALAR, degree=1,
                                      continuous=True)

for pen in (1.0e3, 1.0e4, 1.0e6):
    stokes_h = uw.systems.Stokes(mesh, velocityField=v_h, pressureField=p_h)
    stokes_h.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes_h.constitutive_model.Parameters.shear_viscosity_0 = 0.1 + 0.9 * M.sym[0]
    stokes_h.penalty = 0.0
    stokes_h.bodyforce = -(M.sym[0] - blob_amp * blob_fn) * unit_r
    stokes_h.tolerance = 1.0e-6
    stokes_h.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
    stokes_h.add_essential_bc((0.0, 0.0), mesh.boundaries.Upper.name)
    # Direct penalty free-slip on the internal interface
    stokes_h.add_natural_bc(pen * unit_r.dot(v_h.sym) * unit_r,
                            mesh.boundaries.Internal.name)
    stokes_h.solve()

    # u_n on the held interface (should be small if penalty binds)
    un_h = np.asarray(uw.function.evaluate(
        v_h.sym.dot(unit_r), mesh.X.coords[int_idx])).flatten()

    # Project the held-lid radial normal stress n·σ·n onto a continuous field
    sig_rr_h = (unit_r * stokes_h.stress * unit_r.T)[0, 0]
    proj = uw.systems.Projection(mesh, topo)
    proj.uw_function = sig_rr_h
    proj.smoothing = 0.0
    proj.solve()
    heq = np.asarray(uw.function.evaluate(
        topo.sym[0], mesh.X.coords[int_idx])).flatten()
    ipL = int(np.argmin(np.abs(int_th)))

    # direct (un-projected) n·σ·n at the held interface, pole
    sig_direct = np.asarray(uw.function.evaluate(
        sig_rr_h, mesh.X.coords[int_idx])).flatten()
    # penalty reaction (multiplier-like) for cross-check, pole
    reac = pen * un_h[ipL]
    print(f"  pen={pen:.0e}: max|u_n_held|={np.abs(un_h).max():.3e}  "
          f"proj n·σ·n={heq[ipL]:+.5e}  "
          f"direct n·σ·n={sig_direct[ipL]:+.5e}  "
          f"reaction={reac:+.4e}  (target ≈ +0.024)")

    stokes_h._reset()
