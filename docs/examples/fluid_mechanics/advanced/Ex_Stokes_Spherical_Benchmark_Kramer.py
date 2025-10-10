# %% [markdown]
"""
# 🎓 Stokes Spherical Benchmark Kramer

**PHYSICS:** fluid_mechanics  
**DIFFICULTY:** advanced  
**MIGRATED:** From underworld3-documentation/Notebooks

## Description
This example has been migrated from the original UW3 documentation.
Additional documentation and parameter annotations will be added.

## Migration Notes
- Original complexity preserved
- Parameters to be extracted and annotated
- Claude hints to be added in future update
"""

# %% [markdown]
"""
## Original Code
The following is the migrated code with minimal modifications.
"""

# %%
# ## Spherical Benchmark: Isoviscous Incompressible Stokes
#
# #### [Benchmark paper](https://gmd.copernicus.org/articles/14/1899/2021/) 
#
# *Author: [Thyagarajulu Gollapalli](https://github.com/gthyagi)*
#
# ##### Case1: Freeslip boundaries and delta function density perturbation
# <!--
#     1. Works fine (i.e., bc produce results)
# -->
# ##### Case2: Freeslip boundaries and smooth density distribution
# <!--
#     1. Works fine (i.e., bc produce results)
#     2. Output contains null space (for normals = unit radial vector)
# -->
# ##### Case3: Noslip boundaries and delta function density perturbation
# <!--
#     1. Works fine (i.e., bc produce results)
# -->
# ##### Case4: Noslip boundaries and smooth density distribution 
# <!--
#     1. Works fine (i.e., bc produce results)
# -->

from mpi4py import MPI
import underworld3 as uw
from underworld3.systems import Stokes

import numpy as np
import sympy
import os
import assess
import h5py
import sys

if uw.mpi.size == 1:
    # to fix trame issue
    import nest_asyncio
    nest_asyncio.apply()
    
    import pyvista as pv
    import underworld3.visualisation as vis
    import matplotlib.pyplot as plt
    import cmcrameri.cm as cmc

os.environ["SYMPY_USE_CACHE"] = "no"
os.environ["UW_TIMING_ENABLE"] = "1"

# +
# mesh options
r_o = 2.22
r_int = 2.0
r_i = 1.22

res = uw.options.getInt("res", default=16) # 4, 8, 16, 32, 64, 96 
refine = None

cellsize = 1/res

# +
# specify the case 
case = uw.options.getString('case', default='case1')

# spherical harmonic fn degree (l) and order (m)
l = uw.options.getInt("l", default=2)
m = uw.options.getInt("m", default=1)
k = l+1 # power 

# +
# fem stuff
vdegree  = uw.options.getInt("vdegree", default=2)
pdegree = uw.options.getInt("pdegree", default=1)
pcont = uw.options.getBool("pcont", default=True)
pcont_str = str(pcont).lower()

vel_penalty = uw.options.getReal("vel_penalty", default=1e8)
stokes_tol = uw.options.getReal("stokes_tol", default=1e-5)
vel_penalty_str = str("{:.1e}".format(vel_penalty))
stokes_tol_str = str("{:.1e}".format(stokes_tol))
# -

# compute analytical solution
analytical = True
visualize = False
timing = True

# +
# choosing boundary condition and density perturbation type
freeslip, noslip, delta_fn, smooth = False, False, False, False

if case in ('case1'):
    freeslip, delta_fn = True, True
elif case in ('case2'):
    freeslip, smooth = True, True
elif case in ('case3'):
    noslip, delta_fn = True, True
elif case in ('case4'):
    noslip, smooth = True, True

# +
# output dir
output_dir = os.path.join(os.path.join("./output/Latex_Dir/"), 
                          f'{case}_l_{l}_m_{m}_k_{k}_res_{res}_refine_{refine}_vdeg_{vdegree}_pdeg_{pdegree}'\
                          f'_pcont_{pcont_str}_vel_penalty_{vel_penalty_str}_stokes_tol_{stokes_tol_str}/')

if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)
# -

# ### Analytical Solution

if analytical:
    if freeslip:
        if delta_fn:
            soln_above = assess.SphericalStokesSolutionDeltaFreeSlip(l, m, +1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
            soln_below = assess.SphericalStokesSolutionDeltaFreeSlip(l, m, -1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
        elif smooth:
            '''
            For smooth density distribution only single solution exists in the domain. 
            But for sake of code optimization I am creating two solution here.
            '''
            soln_above = assess.SphericalStokesSolutionSmoothFreeSlip(l, m, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)
            soln_below = assess.SphericalStokesSolutionSmoothFreeSlip(l, m, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)
    elif noslip:
        if delta_fn:
            soln_above = assess.SphericalStokesSolutionDeltaZeroSlip(l, m, +1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
            soln_below = assess.SphericalStokesSolutionDeltaZeroSlip(l, m, -1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
        elif smooth:
            '''
            For smooth density distribution only single solution exists in the domain. 
            But for sake of code optimization I am creating two solution here.
            '''
            soln_above = assess.SphericalStokesSolutionSmoothZeroSlip(l, m, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)
            soln_below = assess.SphericalStokesSolutionSmoothZeroSlip(l, m, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)

# ### Create Mesh

if timing:
    uw.timing.reset()
    uw.timing.start()

# +
# if case in ('case1', 'case3') and uw.mpi.size==1:
#     mesh = uw.meshing.SphericalShellInternalBoundary(radiusInner=r_i, radiusOuter=r_o, radiusInternal=r_int,
#                                                      cellSize=cellsize, qdegree=max(pdegree, vdegree), 
#                                                      filename=f'{output_dir}mesh.msh', refinement=refine)
# else:
#     mesh = uw.meshing.SphericalShell(radiusInner=r_i, radiusOuter=r_o, cellSize=cellsize, 
#                                      qdegree=max(pdegree, vdegree), 
#                                      filename=f'{output_dir}mesh.msh', refinement=refine)
# -

if case in ('case1', 'case3'):
    mesh = uw.meshing.SphericalShellInternalBoundary(radiusInner=r_i, radiusOuter=r_o, radiusInternal=r_int,
                                                     cellSize=cellsize, qdegree=max(pdegree, vdegree), 
                                                     filename=f'{output_dir}mesh.msh', refinement=refine)
else:
    mesh = uw.meshing.SphericalShell(radiusInner=r_i, radiusOuter=r_o, cellSize=cellsize, 
                                     qdegree=max(pdegree, vdegree), 
                                     filename=f'{output_dir}mesh.msh', refinement=refine)

if timing:
    uw.timing.stop()
    uw.timing.print_table(group_by='line_routine', output_file=f"{output_dir}mesh_create_time.txt",  display_fraction=1.00)

if uw.mpi.size == 1 and visualize:
    vis.plot_mesh(mesh, save_png=True, dir_fname=output_dir+'mesh.png', title='', clip_angle=135, cpos='yz')

# print mesh size in each cpu
uw.pprint('-------------------------------------------------------------------------------')
mesh.dm.view()
uw.pprint('-------------------------------------------------------------------------------')

# +
# mesh variables
v_uw = uw.discretisation.MeshVariable('V_u', mesh, mesh.data.shape[1], degree=vdegree)
p_uw = uw.discretisation.MeshVariable('P_u', mesh, 1, degree=pdegree, continuous=pcont)

if analytical:
    v_ana = uw.discretisation.MeshVariable('V_a', mesh, mesh.data.shape[1], degree=vdegree)
    p_ana = uw.discretisation.MeshVariable('P_a', mesh, 1, degree=pdegree, continuous=pcont)
    rho_ana = uw.discretisation.MeshVariable('RHO_a', mesh, 1, degree=pdegree, continuous=True)
    
    v_err = uw.discretisation.MeshVariable('V_e', mesh, mesh.data.shape[1], degree=vdegree)
    p_err = uw.discretisation.MeshVariable('P_e', mesh, 1, degree=pdegree, continuous=pcont)
# -

norm_v = uw.discretisation.MeshVariable("N", mesh, mesh.data.shape[1], degree=pdegree, varsymbol=r"{\hat{n}}")
with mesh.access(norm_v):
    norm_v.data[:,0] = uw.function.evaluate(mesh.CoordinateSystem.unit_e_0[0], norm_v.coords)
    norm_v.data[:,1] = uw.function.evaluate(mesh.CoordinateSystem.unit_e_0[1], norm_v.coords)
    norm_v.data[:,2] = uw.function.evaluate(mesh.CoordinateSystem.unit_e_0[2], norm_v.coords)

# +
# Some useful coordinate stuff
unit_rvec = mesh.CoordinateSystem.unit_e_0
r_uw, th_uw = mesh.CoordinateSystem.xR[0], mesh.CoordinateSystem.xR[1]
phi_uw =sympy.Piecewise((2*sympy.pi + mesh.CoordinateSystem.xR[2], mesh.CoordinateSystem.xR[2]<0), 
                        (mesh.CoordinateSystem.xR[2], True)
                       )

# Null space in velocity expressed in x,y,z coordinates
v_theta_phi_fn_xyz = sympy.Matrix(((0,1,1), (-1,0,1), (-1,-1,0))) * mesh.CoordinateSystem.N.T
# -

if analytical:
    with mesh.access(v_ana, p_ana):
        
        def get_ana_soln(_var, _r_int, _soln_above, _soln_below):
            # get analytical solution into mesh variables
            r = uw.function.evalf(r_uw, _var.coords)
            for i, coord in enumerate(_var.coords):
                if r[i]>_r_int:
                    _var.data[i] = _soln_above(coord)
                else:
                    _var.data[i] = _soln_below(coord)
                    
        # velocities
        get_ana_soln(v_ana, r_int, soln_above.velocity_cartesian, soln_below.velocity_cartesian)

        # pressure 
        get_ana_soln(p_ana, r_int, soln_above.pressure_cartesian, soln_below.pressure_cartesian)

# +
# plotting analytical velocities
if case in ('case1'):
    clim, vmag, vfreq = [0., 0.015], 5e0, 75
elif case in ('case2'):
    clim, vmag, vfreq = [0., 0.007], 1e1, 75
elif case in ('case3'):
    clim, vmag, vfreq = [0., 0.003], 2.5e1, 75
elif case in ('case4'):
    clim, vmag, vfreq = [0., 0.001], 5e2, 75
        
if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_vector(mesh, v_ana, vector_name='v_ana', cmap=cmc.lapaz.resampled(21), clim=clim, vmag=vmag, vfreq=vfreq,
                    save_png=True, dir_fname=output_dir+'vel_ana.png', clip_angle=135, show_arrows=False, cpos='yz')
    
    vis.save_colorbar(colormap=cmc.lapaz.resampled(21), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18, 
                      cb_orient='horizontal', cb_axis_label='Velocity', cb_label_xpos=0.5, cb_label_ypos=-2.05, fformat='pdf', 
                      output_path=output_dir, fname='v_ana')

# +
# plotting analytical pressure
if case in ('case1'):
    clim = [-0.25, 0.25]
elif case in ('case2'):
    clim = [-0.1, 0.1]
elif case in ('case3'):
    clim = [-0.3, 0.3]
elif case in ('case4'):
    clim = [-0.1, 0.1]

if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_scalar(mesh, p_ana.sym, 'p_ana', cmap=cmc.vik.resampled(41), clim=clim, save_png=True, clip_angle=135,
                    dir_fname=output_dir+'p_ana.png', cpos='yz')
    
    vis.save_colorbar(colormap=cmc.vik.resampled(41), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18, 
                      cb_orient='horizontal', cb_axis_label='Pressure', cb_label_xpos=0.5, cb_label_ypos=-2.0, fformat='pdf', 
                      output_path=output_dir, fname='p_ana')
# -

# Create Stokes object
stokes = Stokes(mesh, velocityField=v_uw, pressureField=p_uw)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0
stokes.saddle_preconditioner = 1.0

# +
# defining rho fn and bodyforce term
y_lm_real = sympy.sqrt((2*l + 1)/(4*sympy.pi) * sympy.factorial(l - m)/sympy.factorial(l + m)) * sympy.cos(m*phi_uw) * sympy.assoc_legendre(l, m, sympy.cos(th_uw))

gravity_fn = -1.0 * unit_rvec # gravity

# if delta_fn:
#     rho = sympy.exp(-1e3 * ((r_uw - r_int) ** 2))*y_lm_real
#     if case in ('case1', 'case3') and uw.mpi.size==1:
#         stokes.add_natural_bc(-rho * unit_rvec, mesh.boundaries.Internal.name)
#         stokes.bodyforce = sympy.Matrix([0., 0., 0.])
#     else:
#         stokes.bodyforce = rho*gravity_fn 

if delta_fn:
    rho = sympy.exp(-1e3 * ((r_uw - r_int) ** 2))*y_lm_real
    stokes.add_natural_bc(-rho * unit_rvec, mesh.boundaries.Internal.name)
    stokes.bodyforce = sympy.Matrix([0., 0., 0.])
    
if smooth:
    rho = ((r_uw/r_o)**k) * y_lm_real
    stokes.bodyforce = rho*gravity_fn 
# -

if analytical:
    with mesh.access(rho_ana):
        rho_ana.data[:] = np.c_[uw.function.evaluate(rho, rho_ana.coords)]

# +
# boundary conditions
v_diff =  v_uw.sym - v_ana.sym
stokes.add_natural_bc(vel_penalty*v_diff, mesh.boundaries.Upper.name)
stokes.add_natural_bc(vel_penalty*v_diff, mesh.boundaries.Lower.name)

# stokes.add_condition(v_uw.field_id, 'neumann', vel_penalty*v_diff, "Upper")
# stokes.add_condition(v_uw.field_id, 'neumann', vel_penalty*v_diff, "Lower")

# # pressure boundary condition (not required)
# stokes.add_condition(p_uw.field_id, "dirichlet", 
#                      sp.Matrix([0]), mesh.boundaries.Lower.name, 
#                      components = (0))

# stokes.add_condition(p_uw.field_id, "dirichlet", 
#                      sp.Matrix([0]), mesh.boundaries.Upper.name, 
#                      components = (0))

# +
# plotting analytical rho
clim = [-0.4, 0.4]
        
if uw.mpi.size == 1 and visualize:
    vis.plot_scalar(mesh, -rho_ana.sym, 'Rho', cmap=cmc.roma.resampled(31), clim=clim, save_png=True, 
                    dir_fname=output_dir+'rho_ana.png', clip_angle=135, cpos='yz')
    
    vis.save_colorbar(colormap=cmc.roma.resampled(31), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18, 
                      cb_orient='horizontal', cb_axis_label='Rho', cb_label_xpos=0.5, cb_label_ypos=-2.0, fformat='pdf', 
                      output_path=output_dir, fname='rho_ana')

# +
# Stokes settings
stokes.tolerance = stokes_tol
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

# # gasm is super-fast ... but mg seems to be bulletproof
# # gamg is toughest wrt viscosity
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "additive")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# mg, multiplicative - very robust ... similar to gamg, additive
stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# +
# stokes.petsc_options["fieldsplit_pressure_pc_gasm_type"] = "svd"
# stokes.petsc_options["fieldsplit_pressure_mg_coarse_pc_type"] = "svd"
# stokes.penalty = 10 # values 1 to 10 recommended
# -

if timing:
    uw.timing.reset()
    uw.timing.start()

stokes.solve(verbose=True, debug=False)

if timing:
    uw.timing.stop()
    uw.timing.print_table(group_by='line_routine', output_file=f"{output_dir}stokes_solve_time.txt", display_fraction=1.00)

# +
# Null space evaluation
I0 = uw.maths.Integral(mesh, v_theta_phi_fn_xyz.dot(v_uw.sym))
norm = I0.evaluate()

I0.fn = v_theta_phi_fn_xyz.dot(v_theta_phi_fn_xyz)
vnorm = I0.evaluate()
# print(norm/vnorm, vnorm)

with mesh.access(v_uw):
    dv = uw.function.evaluate(norm * v_theta_phi_fn_xyz, v_uw.coords) / vnorm
    v_uw.data[...] -= dv 
# -

# compute error
if analytical:
    with mesh.access(v_uw, p_uw, v_err, p_err):
    
        def get_error(_var_err, _var_uw, _r_int, _soln_above, _soln_below):
            # get error in numerical solution
            r = uw.function.evalf(r_uw, _var_err.coords)
            for i, coord in enumerate(_var_err.coords):
                if r[i]>_r_int:
                    _var_err.data[i] = _var_uw.data[i] - _soln_above(coord)
                else:
                    _var_err.data[i] = _var_uw.data[i] - _soln_below(coord)
                    
        # error in velocities
        get_error(v_err, v_uw, r_int, soln_above.velocity_cartesian, soln_below.velocity_cartesian)
        
        # error in pressure 
        get_error(p_err, p_uw, r_int, soln_above.pressure_cartesian, soln_below.pressure_cartesian)

# +
# plotting velocities from uw
if case in ('case1'):
    clim, vmag, vfreq = [0., 0.015], 5e0, 75
elif case in ('case2'):
    clim, vmag, vfreq = [0., 0.007], 1e1, 75
elif case in ('case3'):
    clim, vmag, vfreq = [0., 0.003], 2.5e1, 75
elif case in ('case4'):
    clim, vmag, vfreq = [0., 0.001], 5e2, 75
    
if uw.mpi.size == 1 and visualize:
    vis.plot_vector(mesh, v_uw, vector_name='v_uw', cmap=cmc.lapaz.resampled(21), clim=clim, vmag=vmag, vfreq=vfreq,
                    save_png=True, dir_fname=output_dir+'vel_uw.png', clip_angle=135, cpos='yz', show_arrows=False)

    vis.save_colorbar(colormap=cmc.lapaz.resampled(21), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18, 
                      cb_orient='horizontal', cb_axis_label='Velocity', cb_label_xpos=0.5, cb_label_ypos=-2.05, fformat='pdf', 
                      output_path=output_dir, fname='v_uw')

# +
# plotting relative errror in velocities
if case in ('case1'):
    clim, vmag, vfreq = [0., 0.005], 1e2, 75
elif case in ('case2'):
    clim, vmag, vfreq = [0., 1e-4], 1e2, 75
elif case in ('case3'):
    clim, vmag, vfreq = [0., 6e-3], 2e2, 75
elif case in ('case4'):
    clim, vmag, vfreq = [0., 1e-4], 1e5, 75
        
if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_vector(mesh, v_err, vector_name='v_err(relative)', cmap=cmc.lapaz.resampled(11), clim=clim, vmag=vmag, vfreq=vfreq,
                    save_png=True, dir_fname=output_dir+'vel_r_err.png', clip_angle=135, cpos='yz', show_arrows=False)

    vis.save_colorbar(colormap=cmc.lapaz.resampled(11), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18, 
                      cb_orient='horizontal', cb_axis_label='Velocity Error (relative)', cb_label_xpos=0.5, cb_label_ypos=-2.05, fformat='pdf', 
                      output_path=output_dir, fname='v_err_rel')

# +
# plotting magnitude error in percentage
clim = [0, 100]

if uw.mpi.size == 1 and analytical and visualize: 
    vmag_expr = (sympy.sqrt(v_err.sym.dot(v_err.sym))/sympy.sqrt(v_ana.sym.dot(v_ana.sym)))*100
    vis.plot_scalar(mesh, vmag_expr, 'vmag_err(%)', cmap=cmc.oslo_r.resampled(21), clim=clim, save_png=True, 
                    dir_fname=output_dir+'vel_p_err.png', clip_angle=135, cpos='yz')
    
    vis.save_colorbar(colormap=cmc.oslo_r.resampled(21), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18, 
                      cb_orient='horizontal', cb_axis_label='Velocity Error (%)', cb_label_xpos=0.5, cb_label_ypos=-2.05, fformat='pdf', 
                      output_path=output_dir, fname='v_err_perc')

# +
# plotting pressure from uw
if case in ('case1'):
    clim = [-0.25, 0.25]
elif case in ('case2'):
    clim = [-0.1, 0.1]
elif case in ('case3'):
    clim = [-0.3, 0.3]
elif case in ('case4'):
    clim = [-0.1, 0.1]
        
if uw.mpi.size == 1 and visualize:
    vis.plot_scalar(mesh, p_uw.sym, 'p_uw', cmap=cmc.vik.resampled(41), clim=clim, save_png=True, 
                    dir_fname=output_dir+'p_uw.png', clip_angle=135, cpos='yz')
    
    vis.save_colorbar(colormap=cmc.vik.resampled(41), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18, 
                      cb_orient='horizontal', cb_axis_label='Pressure', cb_label_xpos=0.5, cb_label_ypos=-2.0, fformat='pdf', 
                      output_path=output_dir, fname='p_uw')

# +
# plotting relative error in pressure
if case in ('case1', 'case3'):
    clim = [-0.065, 0.065]
elif case in ('case2', 'case4'):
    clim = [-0.01, 0.01]
        
if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_scalar(mesh, p_err.sym, 'p_err(relative)', cmap=cmc.vik.resampled(41), clim=clim, save_png=True, 
                    dir_fname=output_dir+'p_r_err.png', clip_angle=135, cpos='yz')

    vis.save_colorbar(colormap=cmc.vik.resampled(41), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18, 
                      cb_orient='horizontal', cb_axis_label='Pressure Error (relative)', cb_label_xpos=0.5, cb_label_ypos=-2.0, fformat='pdf', 
                      output_path=output_dir, fname='p_err_rel')

# +
# plotting percentage error in pressure
clim = [-1e2, 1e2]

if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_scalar(mesh, (p_err.sym[0]/p_ana.sym[0])*100, 'p_err(%)', cmap=cmc.vik.resampled(41), clim=clim, save_png=True, 
                    dir_fname=output_dir+'p_p_err.png', clip_angle=135, cpos='yz')
    
    vis.save_colorbar(colormap=cmc.vik.resampled(41), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18, 
                      cb_orient='horizontal', cb_axis_label='Pressure Error (%)', cb_label_xpos=0.5, cb_label_ypos=-2.0, fformat='pdf', 
                      output_path=output_dir, fname='p_err_perc')
# -

# computing L2 norm
if analytical:
    with mesh.access(v_err, p_err, p_ana, v_ana):    
        v_err_I = uw.maths.Integral(mesh, v_err.sym.dot(v_err.sym))
        v_ana_I = uw.maths.Integral(mesh, v_ana.sym.dot(v_ana.sym))
        v_err_l2 = np.sqrt(v_err_I.evaluate())/np.sqrt(v_ana_I.evaluate())
    
        p_err_I = uw.maths.Integral(mesh, p_err.sym.dot(p_err.sym))
        p_ana_I = uw.maths.Integral(mesh, p_ana.sym.dot(p_ana.sym))
        p_err_l2 = np.sqrt(p_err_I.evaluate())/np.sqrt(p_ana_I.evaluate())

        uw.pprint('Relative error in velocity in the L2 norm: ', v_err_l2)
            print('Relative error in pressure in the L2 norm: ', p_err_l2)

# +
# res = 8
# Relative error in velocity in the L2 norm:  0.06324715390545255
# Relative error in pressure in the L2 norm:  0.5028638359042983

# res = 16
# Relative error in velocity in the L2 norm:  0.02464714514385536
# Relative error in pressure in the L2 norm:  0.32251898971154225

# +
# writing l2 norms to h5 file
if uw.mpi.size == 1 and os.path.isfile(output_dir+'error_norm.h5'):
    os.remove(output_dir+'error_norm.h5')
    print('Old file removed')

uw.pprint('Creating new h5 file')
    with h5py.File(output_dir+'error_norm.h5', 'w') as f:
        f.create_dataset("k", data=k)
        f.create_dataset("res", data=res)
        f.create_dataset("cellsize", data=cellsize)
        f.create_dataset("v_l2_norm", data=v_err_l2)
        f.create_dataset("p_l2_norm", data=p_err_l2)
# -

# saving h5 and xdmf file
mesh.petsc_save_checkpoint(index=0, meshVars=[v_uw, p_uw, v_ana, p_ana, v_err, p_err, rho_ana], outputPath=os.path.relpath(output_dir)+'/output')


