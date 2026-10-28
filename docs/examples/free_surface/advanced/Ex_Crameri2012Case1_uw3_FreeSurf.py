#!/usr/bin/env python
# coding: utf-8
"""
Crameri et al. (2012, GJI 189, 38-54) -- CASE 1: relaxation of a 2-D cosine
surface perturbation on a two-layer (lid + mantle) box.

    Ex_Crameri2012Case1_uw3_FreeSurf.py

Setup, from Fig. 1(a) and Section 2.4 of the paper
--------------------------------------------------
  box            2800 km wide x 700 km high
  lid            100 km thick, rho_L = 3300 kg/m3, eta_L = 1e23 Pa s
  mantle         rho_M = 3300 kg/m3, eta_M = 1e21 Pa s
  g              10 m/s2
  surface        cosine, initial maximum topography h_init = 7 km,
                 wavelength 2800 km (= the box width)
  lid base       FLAT at 600 km -- so the lid thickness varies laterally by
                 +/-7 per cent, as the paper states
  sides          symmetric / free slip
  bottom         no slip

Analytical solution (paper eq. 23)
----------------------------------
    h(t) = h_init * exp(gamma t),  gamma = -0.2139e-11 1/s,  tau = 14.825 ka

Two features of Case 1 make it the sharpest test available here:

1. rho_L = rho_M, so the DRIVING buoyancy is identically zero: the only force
   is the topographic self-load, the equilibrium surface is flat (h_inf = 0),
   and the relaxation is a SINGLE mode. That is precisely the regime in which
   FreeSurface's exponential update -- and the global least-squares gamma --
   are exact, so any error in the measured relaxation time is attributable to
   the spatial discretisation and to the material representation, not to the
   time integrator.

2. Density is uniform, so the material model affects ONLY the viscosity.
   Case 1 is therefore a clean, analytically-backed test of the viscosity
   mixing rule across the lid/mantle interface -- the participating codes in
   the paper do not agree on one (MILAMIN and SULEC use harmonic, FDCON and
   I2VIS arithmetic, STAGYY geometric; Table 1).
"""

import os
import sys
import numpy as np
import sympy
from mpi4py import MPI

import underworld3 as uw
from underworld3.cython.petsc_discretisation import petsc_dm_find_labeled_points_local


# =====================================================================
# switches
# =====================================================================
MATERIAL = "swarm"          # "swarm" (MaterialSwarm) or "cls" (MaterialCLS)
ETA_MIX  = "arithmetic"     # "arithmetic" | "harmonic" | "geometric"
res      = 128              # cell ~21.9 km; 256 -> ~10.9 km, the paper's Case 1 resolution
h_init_km = 7.0             # paper's Case 1; 0.7 is the small-amplitude check it also reports

if len(sys.argv) > 1:       # allow: python Ex_Crameri2012Case1_uw3_FreeSurf.py cls
    MATERIAL = sys.argv[1]
if len(sys.argv) > 2:
    ETA_MIX = sys.argv[2]


# =====================================================================
# dimensional setup and the non-dimensionalisation
# =====================================================================
# Same scaling as the Case 2 scripts:
#   length    L*    = 2800 km          (box width  -> 1.0)
#   viscosity eta*  = 1e21 Pa s        (mantle     -> 1.0)
#   bodyforce (rho g)* = 33 Pa/m       (rho_M g = 33000 Pa/m -> rho_g = 1000)
L_km, H_km, lid_km = 2800.0, 700.0, 100.0

L, H       = 1.0, H_km / L_km                 # 1.0, 0.25
H_mantle   = (H_km - lid_km) / L_km           # 0.21429 -- FLAT lid base
h_init     = h_init_km / L_km                 # 0.0025 for 7 km
lam        = 1.0                              # wavelength = box width = 2800 km

lid_eta, mantle_eta = 100.0, 1.0              # 1e23, 1e21 Pa s
lid_rho, mantle_rho = 3300.0, 3300.0          # equal: no driving buoyancy
rho_g = 1000.0                                # rho_M g in ND units

# time scale: u* = (rho g)* L*^2 / eta*,  t* = L* / u*
_L      = L_km * 1e3
_rhog   = mantle_rho * 10.0 / rho_g           # 33 Pa/m per ND unit
_u      = _rhog * _L**2 / 1.0e21
t_star  = _L / _u                             # 1.0823e13 s per ND time unit
_yr     = 365.0 * 24 * 3600
ka_per_nd = t_star / _yr / 1.0e3              # 343.18 ka per ND time unit

# paper eq. 23
gamma_dim = 0.2139e-11                        # 1/s
tau_ka    = 1.0 / gamma_dim / _yr / 1.0e3     # 14.825 ka
tau_nd    = tau_ka / ka_per_nd                # 0.043198

max_time  = 100.0 / ka_per_nd                 # run to 100 ka, as in Fig. 2
dt_set    = tau_nd / 20.0                     # 0.741 ka; the integrator is exact for
                                              # one mode, this bounds mesh distortion
save_every = 5

outputPath = "op_Ex_Crameri2012Case1_uw3_FreeSurf_%s_%s_res%d/" % (MATERIAL, ETA_MIX, res)
if uw.mpi.rank == 0:
    if os.path.exists(outputPath):
        for i in os.listdir(outputPath):
            os.remove(outputPath + i)
    else:
        os.makedirs(outputPath)
uw.mpi.barrier()

if uw.mpi.rank == 0:
    print("Crameri (2012) Case 1 -- %s materials, eta mixing = %s, res = %d"
          % (MATERIAL, ETA_MIX, res))
    print("  1 ND time unit = %.2f ka        tau = %.4f ka = %.6f ND"
          % (ka_per_nd, tau_ka, tau_nd))
    print("  h_init = %.1f km = %.5f ND      run to 100 ka = %.4f ND (%.2f tau)"
          % (h_init_km, h_init, max_time, max_time / tau_nd))
    print("  cell = %.2f km, lid = %.1f cells\n" % (L_km / res, lid_km / (L_km / res)))


# =====================================================================
# mesh and unknowns
# =====================================================================
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(L, H), cellSize=L / res, qdegree=3
)

vdegree, predegree, psidegree = 2, 1, 2
v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=vdegree, continuous=True)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=predegree, continuous=True)
timeField = uw.discretisation.MeshVariable("time", mesh, 1, degree=1)


# =====================================================================
# initial cosine topography
# =====================================================================
# -cos so the maximum sits at the box CENTRE rather than on the side walls.
# With reflective/free-slip sides both signs are the same eigenmode at the same
# rate, but reading the extremum at an interior node avoids the corner where
# the rotated free-slip wall meets the stress-free top.
deform_fn = -h_init * sympy.cos(2.0 * sympy.pi * mesh.X[0] / lam)

Dz = uw.discretisation.MeshVariable("Dz", mesh, 1, degree=1)
diffuser = uw.systems.Poisson(mesh, Dz)
diffuser.constitutive_model = uw.constitutive_models.DiffusionModel
diffuser.constitutive_model.Parameters.diffusivity = 1.0
diffuser.add_essential_bc((deform_fn,), "Top")
diffuser.add_essential_bc((0.0,), "Bottom")
diffuser.solve()

displacement = np.zeros((mesh.X.coords.shape[0], mesh.dim))
displacement[:, -1] = uw.function.evaluate(Dz.sym[0], mesh.X.coords)[:, 0, 0]
mesh.deform(mesh.X.coords + displacement)


# =====================================================================
# materials -- built AFTER the deform, and painted by PHYSICAL y
# =====================================================================
# Order matters. The Laplacian carrier above moves every interior node, not
# just the surface, so a lid painted before the deform would be carried down
# with it and its base would follow the cosine. Painting afterwards, against
# the deformed coordinates, puts the lid base FLAT at 600 km and leaves the
# lid thickness varying by +/-7 per cent -- which is the configuration the
# paper specifies.
x, y = mesh.X
yhat = sympy.Matrix([[0, 1]])

if MATERIAL == "swarm":
    params = uw.Params(
        uw_proxy_location="integration_points",
        uw_proxy_sampling="nearest",
        uw_cell_size=L / res,          # the ACTUAL cell size
        uw_fill_param=3,
    )
    materials = uw.swarm.MaterialSwarm(
        mesh,
        fill_param=params.uw_fill_param,
        proxy_location=params.uw_proxy_location,
        proxy_sampling=params.uw_proxy_sampling,
    )
    rho_swarm = materials.add_variable("rho", size=1, dtype=float)
    eta_swarm = materials.add_variable("eta", size=1, dtype=float)

    materials.add("mantle", rho=mantle_rho, eta=mantle_eta)
    materials.add("lid", rho=lid_rho, eta=lid_eta)
    # a swarm region is a BOOLEAN mask over particle coordinates
    materials["lid"] = y > H_mantle

    v_mat = None

elif MATERIAL == "cls":
    # A dedicated velocity MeshVariable, NOT v.sym. The level-set solvers are
    # compiled at construction and cannot be retargeted afterwards; given a
    # genuine MeshVariable, FreeSurface keeps its DATA in sync with the
    # consistent surface velocity (minus the mesh velocity) every step. Handed
    # a bare sympy expression it cannot, and the materials would silently keep
    # advecting with the free velocity and no ALE correction.
    v_mat = uw.discretisation.MeshVariable(
        "Vmat", mesh, mesh.dim, degree=vdegree, continuous=True
    )
    materials = uw.level_set.MaterialCLS(
        mesh, velocity=v_mat, degree=psidegree, epsilon_scale=1.75,
        advection="supg", reini_steps=1, reini_frequency=5,
    )
    materials.add("mantle", rho=mantle_rho, eta=mantle_eta)
    materials.add("lid", rho=lid_rho, eta=lid_eta)
    # a level-set region is a SIGNED DISTANCE; for a horizontal plane
    # (y - H_mantle) is the exact one, which is what reinitialisation wants
    materials["lid"] = y - H_mantle

else:
    raise ValueError("MATERIAL must be 'swarm' or 'cls', not %r" % MATERIAL)

materials.mixing(eta=ETA_MIX, rho="arithmetic")
rho = materials.rho
eta = materials.eta


# =====================================================================
# Stokes + free surface
# =====================================================================
bodyforce = (-1.0 * rho / 3300.0 * rho_g) * yhat

# rho_L == rho_M, so (-rho/3300*rho_g + rho_g) is identically zero: there is
# no driving buoyancy anywhere and the equilibrium surface is flat. Passing the
# zero vector explicitly says so, and spares the held lid a solve whose answer
# is known. background_buoyancy stays None because what we hand over here is
# already the driving-only part.
driving_buoyancy = sympy.Matrix([[0.0, 0.0]])

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = eta
stokes.bodyforce = bodyforce
stokes.materials = materials
stokes.add_essential_bc((0.0, 0.0), "Bottom")        # no slip     (paper, both cases)
stokes.add_rotated_freeslip_bc(0.0, "Left")          # symmetric / free slip
stokes.add_rotated_freeslip_bc(0.0, "Right")
stokes.petsc_options["snes_type"] = "ksponly"
stokes.tolerance = 1.0e-5

fs = uw.systems.FreeSurface(
    stokes, "Top",
    buoyancy_scale=rho_g,
    composition=materials,
    driving_buoyancy=driving_buoyancy,
)


# =====================================================================
# topography diagnostic
# =====================================================================
topwall = petsc_dm_find_labeled_points_local(mesh.dm, "Top")


def max_topography():
    """Global maximum surface elevation above the reference level H."""
    local = -1.0e300
    if topwall is not None and np.size(topwall) > 0:
        local = float((mesh.X.coords[topwall][:, -1] - H).max())
    return uw.mpi.comm.allreduce(local, op=MPI.MAX)


def surface_profile():
    """(x, h) along the top boundary on this rank, sorted by x."""
    if topwall is None or np.size(topwall) == 0:
        return np.empty(0), np.empty(0)
    c = mesh.X.coords[topwall]
    o = np.argsort(c[:, 0])
    return c[o, 0], c[o, -1] - H


# =====================================================================
# time loop
# =====================================================================
step, time = 0, 0.0
history = [(0.0, max_topography())]
if uw.mpi.rank == 0:
    print("  step      t [ka]   h_max [km]   analytic [km]    rel.err")
    print("  %4d  %10.3f  %11.4f  %14.4f  %9s"
          % (0, 0.0, history[0][1] * L_km, h_init_km, "-"))

while time < max_time:
    fs.solve()

    dt = min(fs.estimate_dt(), dt_set, max_time - time)
    fs.advance(dt)
    time += dt
    step += 1

    h = max_topography()
    history.append((time, h))

    if step % save_every == 0 or time >= max_time:
        t_ka = time * ka_per_nd
        ana = h_init * np.exp(-t_ka / tau_ka)
        if uw.mpi.rank == 0:
            print("  %4d  %10.3f  %11.4f  %14.4f  %8.2f%%"
                  % (step, t_ka, h * L_km, ana * L_km,
                     100.0 * (h - ana) / ana if ana != 0 else float("nan")))
        with mesh.access(timeField):
            timeField.data[:, 0] = time * ka_per_nd
        mesh.write_timestep(
            "mesh", meshUpdates=True, meshVars=[v, p, timeField],
            outputPath=outputPath, index=step,
        )

# =====================================================================
# output
# =====================================================================
if uw.mpi.rank == 0:
    hist = np.asarray(history)
    t_ka = hist[:, 0] * ka_per_nd
    h_km = hist[:, 1] * L_km
    ana = h_init_km * np.exp(-t_ka / tau_ka)
    np.savetxt(
        outputPath + "topography.csv",
        np.column_stack([t_ka, h_km, ana]),
        delimiter=",", header="t_ka,h_max_km,h_analytic_km", comments="",
    )

    # relaxation time by log-linear fit over the first 2 tau, where the signal
    # is well above the discretisation floor
    m = (t_ka > 0) & (t_ka < 2 * tau_ka) & (h_km > 0)
    slope = np.polyfit(t_ka[m], np.log(h_km[m]), 1)[0]
    tau_num = -1.0 / slope
    print("\n  fitted relaxation time = %.3f ka   (analytic %.3f ka, error %+.2f%%)"
          % (tau_num, tau_ka, 100.0 * (tau_num - tau_ka) / tau_ka))
    print("  wrote " + outputPath + "topography.csv")

    xs, hs = surface_profile()
    np.savetxt(outputPath + "surface_final.csv",
               np.column_stack([xs * L_km, hs * L_km]),
               delimiter=",", header="x_km,h_km", comments="")
