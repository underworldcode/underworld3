"""Probe: dynamic topography from the held-lid stress on the EXTERNAL
upper free surface (the well-posed case).

Geometry matches _phase_i_fs_etd_annulus.py --buoyancy:
  - Annulus r_i=0.5 (no-slip inner), r_o=1.0 (free upper surface)
  - whole annulus is heavy fluid ρ=1, free surface to vacuum at r_o
  - buoyant blob at (0.7, 0), σ=0.08, mag 0.6, η=1

On the EXTERNAL boundary a free-slip hold IS consistent:
  - Nitsche (consistency term present) ⇒ projected n·σ·n should be the
    proper, penalty-independent dynamic topography.
  - Pure penalty (no consistency term) ⇒ projected n·σ·n ≈ 0; only the
    penalty-dependent reaction carries the load (as seen on the internal
    interface).
We compare Nitsche-projected σ_rr, penalty-projected σ_rr, penalty
reaction, and the free-surface σ_rr (≈0), against the kinematic
equilibrium from the curvS reference run.
"""

import numpy as np
import sympy
import underworld3 as uw

import nest_asyncio
nest_asyncio.apply()

res = 16
r_i, r_o = 0.5, 1.0
cellsize = 1.0 / res
x0, y0, sig, blob_mag0 = 0.7, 0.0, 0.08, 0.6

mesh = uw.meshing.Annulus(radiusOuter=r_o, radiusInner=r_i,
                          cellSize=cellsize, qdegree=3)
r, th = mesh.CoordinateSystem.R
unit_r = mesh.CoordinateSystem.unit_e_0

v = uw.discretisation.MeshVariable("V", mesh, vtype=uw.VarType.VECTOR,
                                   degree=2, continuous=True)
p = uw.discretisation.MeshVariable("P", mesh, vtype=uw.VarType.SCALAR,
                                   degree=1, continuous=True)
rho_var = uw.discretisation.MeshVariable("rho", mesh,
                                         vtype=uw.VarType.SCALAR, degree=0,
                                         continuous=False)
topo = uw.discretisation.MeshVariable("topo", mesh,
                                      vtype=uw.VarType.SCALAR, degree=1,
                                      continuous=True)

# Buoyant blob (per-element P0), ρ_ref sharp step at r_o
cen = mesh._centroids
blob = blob_mag0 * np.exp(-((cen[:, 0] - x0) ** 2 + (cen[:, 1] - y0) ** 2)
                          / (2.0 * sig ** 2))
rho_var.data[:, 0] = 1.0 - blob
r_node = sympy.sqrt(mesh.X[0] ** 2 + mesh.X[1] ** 2)
rho_ref = sympy.Piecewise((1.0, r_node < r_o), (0.0, True))
anomaly = rho_var.sym[0] - rho_ref


def make_stokes():
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    s.penalty = 1.0
    s.bodyforce = -anomaly * unit_r
    s.tolerance = 1.0e-6
    s.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
    return s


# Upper-boundary node identification
X0 = mesh.X.coords.copy()
R0 = np.sqrt(X0[:, 0] ** 2 + X0[:, 1] ** 2)
TH0 = np.arctan2(X0[:, 1], X0[:, 0])
is_up = R0 > (r_o - 0.5 * cellsize / r_o)
up_idx = np.where(is_up)[0]
up_idx = up_idx[np.argsort(TH0[up_idx])]
up_th = TH0[up_idx]
ipL = int(np.argmin(np.abs(up_th)))     # pole node θ≈0 (above blob)

sig_rr_expr = (unit_r * None * unit_r.T) if False else None  # set per-solve


def proj_sigrr(stokes):
    expr = (unit_r * stokes.stress * unit_r.T)[0, 0]
    pr = uw.systems.Projection(mesh, topo)
    pr.uw_function = expr
    pr.smoothing = 0.0
    pr.solve()
    return np.asarray(uw.function.evaluate(
        topo.sym[0], mesh.X.coords[up_idx])).flatten()


def un_at_upper():
    return np.asarray(uw.function.evaluate(
        v.sym.dot(unit_r), mesh.X.coords[up_idx])).flatten()


print("=== EXTERNAL upper free surface — dynamic topography probe ===")
print("  (reference: curvS equilibrium A_max from the annulus run)\n")

# (1) FREE upper surface
s_free = make_stokes()
s_free.solve()
sig_free = proj_sigrr(s_free)
un_free = un_at_upper()
print(f"  FREE surface:   proj σ_rr_pole={sig_free[ipL]:+.5e}  "
      f"u_n_pole={un_free[ipL]:+.5e}")
s_free._reset()

# (2) NITSCHE held-lid free-slip (consistency term present)
for g in (10.0, 30.0, 100.0):
    s_n = make_stokes()
    s_n.add_nitsche_bc(mesh.boundaries.Upper.name, gamma=g)
    s_n.solve()
    sig_n = proj_sigrr(s_n)
    un_n = un_at_upper()
    print(f"  NITSCHE γ={g:5.0f}: proj σ_rr_pole={sig_n[ipL]:+.5e}  "
          f"max|u_n|={np.abs(un_n).max():.2e}")
    s_n._reset()

# (3) PENALTY held-lid free-slip (no consistency term)
for pen in (1.0e3, 1.0e4):
    s_p = make_stokes()
    s_p.add_natural_bc(pen * unit_r.dot(v.sym) * unit_r,
                       mesh.boundaries.Upper.name)
    s_p.solve()
    sig_p = proj_sigrr(s_p)
    un_p = un_at_upper()
    print(f"  PENALTY pen={pen:.0e}: proj σ_rr_pole={sig_p[ipL]:+.5e}  "
          f"reaction={pen * un_p[ipL]:+.4e}  max|u_n|={np.abs(un_p).max():.2e}")
    s_p._reset()
