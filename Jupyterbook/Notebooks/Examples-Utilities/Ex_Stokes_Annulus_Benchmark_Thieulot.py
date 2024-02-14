# ## Annulus Benchmark: Isoviscous Incompressible Stokes
#
# ### Case: Infinitely thin density anomaly at $r = r'$
# #### [Benchmark paper](https://gmd.copernicus.org/articles/14/1899/2021/) 
#
# *Author: [Thyagarajulu Gollapalli](https://github.com/gthyagi)*

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# +
import underworld3 as uw
from underworld3.systems import Stokes

import numpy as np
import sympy
from sympy import lambdify
import os
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
import assess
# -

os.environ["SYMPY_USE_CACHE"] = "no"

if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

# mesh options
res = 1/32
res_int_fac = 1/2
r_o = 2.0
r_int = 1.8
r_i = 1.0

# visualize analytical solutions
plot_ana = True

# ### Analytical Solution

if plot_ana:
    n = 2 # wave number
    solution_above = assess.CylindricalStokesSolutionDeltaFreeSlip(n, +1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
    solution_below = assess.CylindricalStokesSolutionDeltaFreeSlip(n, -1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)

if plot_ana:
    mesh_ana = uw.meshing.AnnulusInternalBoundary(radiusOuter=r_o, 
                                                  radiusInternal=r_int, 
                                                  radiusInner=r_i, 
                                                  cellSize_Inner=res,
                                                  cellSize_Internal=res*res_int_fac,
                                                  cellSize_Outer=res,)

if plot_ana:
    v_ana = uw.discretisation.MeshVariable(r"\mathbf{u_a}", mesh_ana, 2, degree=2)
    p_ana = uw.discretisation.MeshVariable(r"p_a", mesh_ana, 1, degree=1)
    rho_ana = uw.discretisation.MeshVariable(r"rho_a", mesh_ana, 1, degree=1)

if uw.mpi.size == 1 and plot_ana:

    pvmesh = vis.mesh_to_pv_mesh(mesh_ana)
   
    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(pvmesh, edge_color="Grey", show_edges=True, use_transparency=False, opacity=1.0, )

    pl.show(cpos="xy")

if plot_ana:
    r_ana, th_ana = mesh_ana.CoordinateSystem.xR

if plot_ana:
    with mesh_ana.access(v_ana, p_ana, rho_ana):
        # velocities
        r = uw.function.evalf(r_ana, v_ana.coords)
        for i, coord in enumerate(v_ana.coords):
            if r[i]>r_int:
                v_ana.data[i] = solution_above.velocity_cartesian(coord)
            else:
                v_ana.data[i] = solution_below.velocity_cartesian(coord)
        
        
        # pressure 
        r = uw.function.evalf(r_ana, p_ana.coords)
        for i, coord in enumerate(p_ana.coords):
            if r[i]>r_int:
                p_ana.data[i] = solution_above.pressure_cartesian(coord)
            else:
                p_ana.data[i] = solution_below.pressure_cartesian(coord)
    
        # density
        r = uw.function.evalf(r_ana, rho_ana.coords)
        for i, coord in enumerate(rho_ana.coords):
            if r[i]>r_int:
                rho_ana.data[i] = solution_above.radial_stress_cartesian(coord)
            else:
                rho_ana.data[i] = solution_below.radial_stress_cartesian(coord)

if uw.mpi.size == 1 and plot_ana:
    pvmesh_ana = vis.mesh_to_pv_mesh(mesh_ana)
    pvmesh_ana.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh_ana, v_ana.sym)
    pvmesh_ana.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh_ana, 
                                                               sympy.sqrt(v_ana.sym.dot(v_ana.sym)))
    
    print(pvmesh_ana.point_data["Vmag"].min(), pvmesh_ana.point_data["Vmag"].max())
    
    velocity_points_ana = vis.meshVariable_to_pv_cloud(v_ana)
    velocity_points_ana.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points_ana, v_ana.sym)
    
    pl = pv.Plotter(window_size=(750, 750))
    pl.add_mesh(pvmesh_ana, cmap=cmc.lapaz.resampled(11), edge_color="Grey",
                scalars="Vmag", show_edges=False, use_transparency=False,
                opacity=0.7, clim=[0., 0.05] )
    pl.add_arrows(velocity_points_ana.points[::10], velocity_points_ana.point_data["V"][::10], 
                  mag=5, color='k')

    pl.show(cpos="xy")

if uw.mpi.size == 1 and plot_ana:
    pvmesh_ana = vis.mesh_to_pv_mesh(mesh_ana)
    pvmesh_ana.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh_ana, p_ana.sym)

    print(pvmesh_ana.point_data["P"].min(), pvmesh_ana.point_data["P"].max())
   
    pl = pv.Plotter(window_size=(750, 750))
    pl.add_mesh(pvmesh_ana, cmap=cmc.vik.resampled(41), edge_color="Grey",
                scalars="P", show_edges=False, use_transparency=False,
                opacity=1.0, clim=[-0.65, 0.65] )

    pl.show(cpos="xy")

if uw.mpi.size == 1 and plot_ana:
    pvmesh_ana = vis.mesh_to_pv_mesh(mesh_ana)
    pvmesh_ana.point_data["rho"] = vis.scalar_fn_to_pv_points(pvmesh_ana, rho_ana.sym)

    print(pvmesh_ana.point_data["rho"].min(), pvmesh_ana.point_data["rho"].max())
    
    pl = pv.Plotter(window_size=(750, 750))
    pl.add_mesh(pvmesh_ana, cmap=cmc.roma.resampled(41), edge_color="Grey",
                scalars="rho", show_edges=False, use_transparency=False,
                opacity=1.0, clim=[-0.8, 0.8] )

    pl.show(cpos="xy")

# ### Create Mesh for Numerical Solution

# mesh
mesh_uw = uw.meshing.AnnulusInternalBoundary(radiusOuter=r_o, 
                                             radiusInternal=r_int, 
                                             radiusInner=r_i, 
                                             cellSize_Inner=res,
                                             cellSize_Internal=res*res_int_fac,
                                             cellSize_Outer=res,)

# mesh variables
v_uw = uw.discretisation.MeshVariable(r"\mathbf{u}", mesh_uw, 2, degree=2)
p_uw = uw.discretisation.MeshVariable(r"p", mesh_uw, 1, degree=1)
v_err = uw.discretisation.MeshVariable(r"\mathbf{u_e}", mesh_uw, 2, degree=2)
p_err = uw.discretisation.MeshVariable(r"p_e", mesh_uw, 1, degree=1)

# Some useful coordinate stuff
unit_rvec = mesh_uw.CoordinateSystem.unit_e_0
r_uw, th_uw = mesh_uw.CoordinateSystem.xR

# +
# Create Stokes object

stokes = Stokes(mesh_uw, velocityField=v_uw, pressureField=p_uw, solver_name="stokes")

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0
stokes.saddle_preconditioner = 1.0

rho = sympy.cos(n*th_uw) * sympy.exp(-1e5 * ((r_uw - r_int) ** 2)) 

penalty = 1e7
Gamma = mesh_uw.Gamma
stokes.add_natural_bc(penalty * Gamma.dot(v_uw.sym) *  Gamma, "Upper")
stokes.add_natural_bc(penalty * Gamma.dot(v_uw.sym) *  Gamma, "Lower")
stokes.add_natural_bc(-rho * unit_rvec, "Internal")

stokes.bodyforce = sympy.Matrix([0,0])
# -

if uw.mpi.size == 1:
    pvmesh = vis.mesh_to_pv_mesh(mesh_uw)
    pvmesh.point_data["rho"] = vis.scalar_fn_to_pv_points(pvmesh, rho)

    print(pvmesh.point_data["rho"].min(), pvmesh.point_data["rho"].max())
    
    pl = pv.Plotter(window_size=(750, 750))
    pl.add_mesh(pvmesh, cmap=cmc.roma.resampled(31), edge_color="Grey",
                scalars="rho", show_edges=False, use_transparency=False,
                opacity=1.0, clim=[-1, 1] )

    pl.show(cpos="xy")

# +
# Stokes settings

stokes.tolerance = 1.0e-6
stokes.petsc_options["ksp_monitor"] = None

stokes.petsc_options["snes_type"] = "newtonls"
stokes.petsc_options["ksp_type"] = "fgmres"

# stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")

stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"

stokes.petsc_options[f"fieldsplit_velocity_ksp_type"] = "fcg"
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_max_it"] = 5
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

# gasm is super-fast ... but mg seems to be bulletproof
# gamg is toughest wrt viscosity

stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "additive")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# # # mg, multiplicative - very robust ... similar to gamg, additive

# stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")
# -

stokes.solve()

with mesh_uw.access(v_uw, p_uw, v_err, p_err):
        # velocities
        r = uw.function.evalf(r_uw, v_err.coords)
        for i, coord in enumerate(v_err.coords):
            if r[i]>r_int:
                v_err.data[i] = v_uw.data[i] - solution_above.velocity_cartesian(coord)
            else:
                v_err.data[i] = v_uw.data[i] - solution_below.velocity_cartesian(coord)
        
        
        # pressure 
        r = uw.function.evalf(r_uw, p_err.coords)
        for i, coord in enumerate(p_err.coords):
            if r[i]>r_int:
                p_err.data[i] = p_uw.data[i] - solution_above.pressure_cartesian(coord)
            else:
                p_err.data[i] = p_uw.data[i] - solution_below.pressure_cartesian(coord)

# plotting velocities from uw
if uw.mpi.size == 1:
    pvmesh = vis.mesh_to_pv_mesh(mesh_uw)
    pvmesh.point_data["V_uw"] = vis.vector_fn_to_pv_points(pvmesh, v_uw.sym)
    pvmesh.point_data["Vmag_uw"] = vis.scalar_fn_to_pv_points(pvmesh, sympy.sqrt(v_uw.sym.dot(v_uw.sym)))

    print(pvmesh.point_data["Vmag_uw"].min(), pvmesh.point_data["Vmag_uw"].max())
    
    velocity_points = vis.meshVariable_to_pv_cloud(v_uw)
    velocity_points.point_data["V_uw"] = vis.vector_fn_to_pv_points(velocity_points, v_uw.sym)

    pl = pv.Plotter(window_size=(750, 750))
    pl.add_mesh(pvmesh, cmap=cmc.lapaz.resampled(11), edge_color="Grey",
                scalars="Vmag_uw", show_edges=False, use_transparency=False,
                opacity=0.1, clim=[0., 0.05] )
    pl.add_arrows(velocity_points.points[::10], velocity_points.point_data["V_uw"][::10], mag=1e1, color='k')

    pl.show(cpos="xy")

# plotting errror in velocities
if uw.mpi.size == 1:
    pvmesh = vis.mesh_to_pv_mesh(mesh_uw)
    
    pvmesh.point_data["V_err"] = vis.vector_fn_to_pv_points(pvmesh, v_err.sym)
    pvmesh.point_data["Vmag_err"] = vis.scalar_fn_to_pv_points(pvmesh, sympy.sqrt(v_err.sym.dot(v_err.sym)))

    print(pvmesh.point_data["Vmag_err"].min(), pvmesh.point_data["Vmag_err"].max())
    
    velocity_points = vis.meshVariable_to_pv_cloud(v_uw)
    velocity_points.point_data["V_err"] = vis.vector_fn_to_pv_points(velocity_points, v_err.sym)

    pl = pv.Plotter(window_size=(750, 750))
    pl.add_mesh(pvmesh, cmap=cmc.lapaz.resampled(11), edge_color="Grey",
                scalars="Vmag_err", show_edges=False, use_transparency=False,
                opacity=0.7, clim=[0., 0.005] )
    pl.add_arrows(velocity_points.points[::50], velocity_points.point_data["V_err"][::50], mag=1e-1, color='k')

    pl.show(cpos="xy")

# plotting pressure from uw
if uw.mpi.size == 1:
    pvmesh = vis.mesh_to_pv_mesh(mesh_uw)
    pvmesh.point_data["P_uw"] = vis.scalar_fn_to_pv_points(pvmesh, p_uw.sym)

    print(pvmesh.point_data["P_uw"].min(), pvmesh.point_data["P_uw"].max())
   
    pl = pv.Plotter(window_size=(750, 750))
    pl.add_mesh(pvmesh, cmap=cmc.vik.resampled(41), edge_color="Grey",
                scalars="P_uw", show_edges=False, use_transparency=False,
                opacity=1.0, clim=[-0.85, 0.85] )

    pl.show(cpos="xy")

# plotting error in uw
if uw.mpi.size == 1:
    pvmesh = vis.mesh_to_pv_mesh(mesh_uw)
    pvmesh.point_data["P_err"] = vis.scalar_fn_to_pv_points(pvmesh, p_err.sym)

    print(pvmesh.point_data["P_err"].min(), pvmesh.point_data["P_err"].max())
   
    pl = pv.Plotter(window_size=(750, 750))
    pl.add_mesh(pvmesh, cmap=cmc.vik.resampled(41), edge_color="Grey",
                scalars="P_err", show_edges=False, use_transparency=False,
                opacity=1.0, clim=[-0.085, 0.085] )

    pl.show(cpos="xy")

# pressure error (L2 norm)
p_err.stats()[5]/p_ana.stats()[5]

mesh_uw.dm.view()




