#!/usr/bin/env python
# coding: utf-8
"""
Crameri et al. (2012, GJI 189, 38-54) -- CASE 2: a buoyant plume rises and
loads the base of the lithosphere, deflecting the free surface.

    Ex_Crameri2012Case2_uw3_FreeSurf.py

Setup, from Section 2.3 and Fig. 1(b) of the paper
--------------------------------------------------
  box          2800 km wide x 700 km high
  lithosphere  100 km thick, rho_L = 3300 kg/m3, eta_L = 1e23 Pa s
  mantle       600 km thick, rho_M = 3300 kg/m3, eta_M = 1e21 Pa s
  plume        r_P = 50 km, rho_P = 3200 kg/m3, eta_P = 1e20 Pa s,
               centred 1400 km from the sides and 300 km above the bottom
               of the mantle layer
  g            10 m/s2
  sides        free slip;   bottom  no slip;   top  free surface

Reference results (true free-surface codes, Figs 6a and 7; DIGITISED off the
figures, so treat them as +/- 20 m)
-----------------------------------------------------------------------------
  ~0.2 Ma    ~250 m    end of the fast isostatic adjustment (Fig. 6b)
   4  Ma     ~460 m
   8  Ma     ~660 m
  16  Ma     ~810 m    maximum is reached around 16-17 Ma
  20  Ma     ~830 m
The profile (Fig. 7) is a bump ~700 km wide at x = 1400 km with NEGATIVE side
lobes near -100 m -- the flexural moat -- which is a useful shape check.

Timestep: the paper notes the isostatic relaxation time is ~15 ka and that
constant steps of 250-4000 yr were tested, with FDCON stable up to 16-32 ka.
Note that `fs.advance(1.0)` in ND units here would be 343 ka -- more than
twenty isostatic relaxation times in a single step.
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
MATERIAL    = "swarm"        # "swarm" (MaterialSwarm) or "cls" (MaterialCLS)
ETA_MIX     = "arithmetic"   # "arithmetic" | "harmonic" | "geometric" -- one switch,
                             # so both material models necessarily use the same rule
res         = 128            # cell 21.9 km. The paper's UNDERWORLD Case 2 free-surface
                             # run used ~10 km cells; see the plume-resolution line
                             # printed at startup before trusting any amplitude.
MAX_TIME_MA = 20.0           # Fig. 6 runs to 20 Ma; use 1.0 for a smoke test
DT_KA       = 20.0           # upper bound on the step; fs.estimate_dt() also applies
EPS_SCALE   = 1.75           # MaterialCLS interface half-width, in cells

# python Ex_Crameri2012Case2_uw3_FreeSurf.py [material] [eta_mix] [res] [max_time_Ma]
if len(sys.argv) > 1:
    MATERIAL = sys.argv[1]
if len(sys.argv) > 2:
    ETA_MIX = sys.argv[2]
if len(sys.argv) > 3:
    res = int(sys.argv[3])
if len(sys.argv) > 4:
    MAX_TIME_MA = float(sys.argv[4])


# =====================================================================
# dimensional setup and non-dimensionalisation
# =====================================================================
#   length    L*        = 2800 km       (box width -> 1.0)
#   viscosity eta*      = 1e21 Pa s     (mantle    -> 1.0)
#   bodyforce (rho g)*  = 33 Pa/m       (rho_M g = 33000 Pa/m -> rho_g = 1000)
L_km, H_km, lid_km = 2800.0, 700.0, 100.0

L, H     = 1.0, H_km / L_km                    # 1.0, 0.25
H_mantle = (H_km - lid_km) / L_km              # 0.21429

plume_r  = 50.0 / L_km                         # 0.017857
plume_x0 = 1400.0 / L_km                       # 0.5   -- 1400 km from the sides
plume_y0 = 300.0 / L_km                        # 0.10714 -- 300 km above the base

lid_eta, mantle_eta, plume_eta = 100.0, 1.0, 0.1        # 1e23, 1e21, 1e20 Pa s
lid_rho, mantle_rho, plume_rho = 3300.0, 3300.0, 3200.0
rho_g = 1000.0                                          # rho_M g in ND units

_L    = L_km * 1e3
_rhog = mantle_rho * 10.0 / rho_g                       # 33 Pa/m per ND unit
_u    = _rhog * _L**2 / 1.0e21
t_star = _L / _u
_yr   = 365.0 * 24 * 3600
ka_per_nd = t_star / _yr / 1.0e3                        # 343.18 ka per ND time unit

max_time = MAX_TIME_MA * 1.0e3 / ka_per_nd
dt_set   = DT_KA / ka_per_nd
SNAP_MA  = [4.0, 8.0, 12.0, 16.0, 20.0]                 # Figs 5 and 7 snapshots

# digitised from Figs 6(a)/7 -- see the docstring; +/- 20 m
REFERENCE_M = {4.0: 460.0, 8.0: 660.0, 12.0: 760.0, 16.0: 810.0, 20.0: 830.0}

outputPath = "op_Ex_Crameri2012Case2_uw3_FreeSurf_%s_%s_res%d/" % (MATERIAL, ETA_MIX, res)
if uw.mpi.rank == 0:
    if os.path.exists(outputPath):
        for i in os.listdir(outputPath):
            os.remove(outputPath + i)
    else:
        os.makedirs(outputPath)
uw.mpi.barrier()


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

x, y = mesh.X
yhat = sympy.Matrix([[0, 1]])

if uw.mpi.rank == 0:
    cell_km = L_km / res
    print("Crameri (2012) Case 2 -- %s materials, eta mixing = %s, res = %d"
          % (MATERIAL, ETA_MIX, res))
    print("  1 ND time unit = %.2f ka    run to %.1f Ma = %.3f ND, dt <= %.1f ka"
          % (ka_per_nd, MAX_TIME_MA, max_time, DT_KA))
    print("  cell = %.2f km : lid = %.1f cells, PLUME RADIUS = %.2f cells"
          % (cell_km, lid_km / cell_km, 50.0 / cell_km))
    if 50.0 / cell_km < 4.0:
        print("  ** the plume is under-resolved (the paper used ~10 km cells, ~5 cells")
        print("     per plume radius). Amplitudes here are resolution-limited. **")
    print("")


# =====================================================================
# materials  -- THE ONLY BLOCK THAT DIFFERS BETWEEN THE TWO TWINS
# =====================================================================
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
    materials.add("lid",    rho=lid_rho,    eta=lid_eta)
    materials.add("plume",  rho=plume_rho,  eta=plume_eta)

    # A swarm region is a BOOLEAN mask over particle coordinates. (A bare signed-
    # distance expression is refused here -- there is no convention telling the
    # swarm which sign is "inside".)
    materials["lid"]   = y > H_mantle
    materials["plume"] = sympy.sqrt((x - plume_x0)**2 + (y - plume_y0)**2) < plume_r


    def write_materials(index):
        """Particle fields, for the viscosity snapshots of Fig. 5."""
        rho_swarm.data[:, 0] = uw.function.evaluate(materials.rho, materials.coords)[:, 0, 0]
        eta_swarm.data[:, 0] = uw.function.evaluate(materials.eta, materials.coords)[:, 0, 0]
        materials.write_timestep(
            filename="swarm", swarmname="materials", outputPath=outputPath, index=index,
            swarmVars=[materials.index, rho_swarm, eta_swarm],
        )

elif MATERIAL == "cls":
    # A dedicated velocity MeshVariable, NOT v.sym. Each material's LevelSetSolver
    # compiles its velocity into its weak form at construction and cannot be
    # retargeted afterwards; given a genuine MeshVariable, FreeSurface keeps that
    # variable's DATA in sync with its consistent surface velocity (minus the mesh
    # velocity) every step. Handed a bare sympy expression it cannot, and the
    # materials would silently keep advecting with the free velocity and no ALE
    # correction -- on a mesh that deforms every step, with SUPG transport.
    v_mat = uw.discretisation.MeshVariable(
        "Vmat", mesh, mesh.dim, degree=vdegree, continuous=True
    )
    materials = uw.level_set.MaterialCLS(
        mesh, velocity=v_mat, degree=psidegree, epsilon_scale=EPS_SCALE,
        advection="supg", reini_steps=1, reini_frequency=5,
    )
    eta_mesh = uw.discretisation.MeshVariable(
        "etaM", mesh, vtype=uw.VarType.SCALAR, degree=psidegree, continuous=True)
    rho_mesh = uw.discretisation.MeshVariable(
        "rhoM", mesh, vtype=uw.VarType.SCALAR, degree=psidegree, continuous=True)

    materials.add("mantle", rho=mantle_rho, eta=mantle_eta)
    materials.add("lid",    rho=lid_rho,    eta=lid_eta)
    materials.add("plume",  rho=plume_rho,  eta=plume_eta)

    # A level-set region is a SIGNED DISTANCE, positive inside. (y - H_mantle) is
    # the exact distance to a horizontal plane, and (r_P - |x - x0|) the exact
    # distance to a circle -- which is what reinitialisation wants to see.
    materials["lid"]   = y - H_mantle
    materials["plume"] = plume_r - sympy.sqrt((x - plume_x0)**2 + (y - plume_y0)**2)


    def write_materials(index):
        """Blended fields on the mesh, for the viscosity snapshots of Fig. 5."""
        eta_mesh.data[:, 0] = uw.function.evaluate(materials.eta, eta_mesh.coords)[:, 0, 0]
        rho_mesh.data[:, 0] = uw.function.evaluate(materials.rho, rho_mesh.coords)[:, 0, 0]
        mesh.write_timestep(
            "materials", meshUpdates=True, meshVars=[eta_mesh, rho_mesh],
            outputPath=outputPath, index=index,
        )

else:
    raise ValueError("MATERIAL must be 'swarm' or 'cls', not %r" % MATERIAL)

materials.mixing(eta=ETA_MIX, rho="arithmetic")
rho = materials.rho
eta = materials.eta


# =====================================================================
# Stokes + free surface
# =====================================================================
# rho_P != rho_M, so the driving buoyancy is non-zero -- but ONLY inside the
# plume, since rho_L == rho_M makes the lithosphere neutrally buoyant. The
# held lid therefore recovers an h_inf driven purely by the plume load, and
# background_buoyancy stays None because what is handed over below is already
# the driving-only part of the body force.
bodyforce        = (-1.0 * rho / 3300.0 * rho_g) * yhat
driving_buoyancy = (-1.0 * rho / 3300.0 * rho_g + rho_g) * yhat

# Integrated driving buoyancy: the decisive number for comparing the two
# material models. It is non-zero only in the plume, so it measures how much
# buoyant material each representation actually carries. If the twins disagree
# here, they will disagree on topography amplitude for that reason alone, and
# no amount of surface-side debugging will explain it.
if uw.mpi.rank == 0:
    print("  integrated driving buoyancy = %.6e   <-- compare between the twins"
          % float(uw.maths.Integral(mesh, driving_buoyancy[1]).evaluate()))
    print("  (analytic for a sharp disc  = %.6e)"
          % (np.pi * plume_r**2 * (1.0 - plume_rho / 3300.0) * rho_g))
    print("")

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = eta
stokes.bodyforce = bodyforce
stokes.materials = materials
stokes.add_essential_bc((0.0, 0.0), "Bottom")     # no slip
stokes.add_rotated_freeslip_bc(0.0, "Left")       # free slip
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
# topography diagnostics
# =====================================================================
topwall = petsc_dm_find_labeled_points_local(mesh.dm, "Top")


def max_topography():
    """Global maximum surface elevation above the reference level H, in ND."""
    local = -1.0e300
    if topwall is not None and np.size(topwall) > 0:
        local = float((mesh.X.coords[topwall][:, -1] - H).max())
    return uw.mpi.comm.allreduce(local, op=MPI.MAX)


def save_profile(tag):
    """Surface profile in km/m, for comparison with Fig. 7."""
    if topwall is None or np.size(topwall) == 0:
        return
    c = mesh.X.coords[topwall]
    o = np.argsort(c[:, 0])
    np.savetxt(
        outputPath + "surface_%s.csv" % tag,
        np.column_stack([c[o, 0] * L_km, (c[o, -1] - H) * L_km * 1000.0]),
        delimiter=",", header="x_km,h_m", comments="",
    )


# =====================================================================
# time loop
# =====================================================================
step, time = 0, 0.0
history = [(0.0, 0.0)]
snaps = list(SNAP_MA)

if uw.mpi.rank == 0:
    print("  step     t [Ma]   h_max [m]    paper [m]")
    print("  %4d  %9.4f  %10.2f  %11s" % (0, 0.0, 0.0, "0"))

while time < max_time:
    fs.solve()

    dt = min(fs.estimate_dt(), dt_set, max_time - time)
    fs.advance(dt)
    time += dt
    step += 1

    t_Ma = time * ka_per_nd / 1.0e3
    h_m = max_topography() * L_km * 1000.0
    history.append((t_Ma, h_m))

    crossed = snaps and t_Ma >= snaps[0]
    if crossed or step % 25 == 0:
        ref = ""
        if crossed:
            key = snaps.pop(0)
            ref = "%.0f" % REFERENCE_M.get(key, float("nan"))
            save_profile("%02dMa" % int(round(key)))
            with mesh.access(timeField):
                timeField.data[:, 0] = t_Ma
            mesh.write_timestep(
                "mesh", meshUpdates=True, meshVars=[v, p, timeField],
                outputPath=outputPath, index=step,
            )
            write_materials(step)
        if uw.mpi.rank == 0:
            print("  %4d  %9.4f  %10.2f  %11s" % (step, t_Ma, h_m, ref))

# =====================================================================
# output
# =====================================================================
save_profile("final")
if uw.mpi.rank == 0:
    hist = np.asarray(history)
    np.savetxt(outputPath + "topography.csv", hist,
               delimiter=",", header="t_Ma,h_max_m", comments="")
    print("\n  %-10s %10s %10s %8s" % ("t [Ma]", "model [m]", "paper [m]", "diff"))
    for t_ref, h_ref in sorted(REFERENCE_M.items()):
        if t_ref <= hist[-1, 0]:
            h = float(np.interp(t_ref, hist[:, 0], hist[:, 1]))
            print("  %-10.1f %10.1f %10.1f %7.1f%%"
                  % (t_ref, h, h_ref, 100.0 * (h - h_ref) / h_ref))
    print("\n  wrote " + outputPath + "topography.csv")
