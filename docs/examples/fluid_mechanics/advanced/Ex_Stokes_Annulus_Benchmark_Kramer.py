# %% [markdown]
"""
# 🎓 Stokes Annulus Benchmark Kramer

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
# ## Annulus Benchmark: Isoviscous Incompressible Stokes
#
# #### [Benchmark paper](https://gmd.copernicus.org/articles/14/1899/2021/) 
#
# *Author: [Thyagarajulu Gollapalli](https://github.com/gthyagi)*
#
#
# ##### Case1: Freeslip boundaries and delta function density perturbation
# <!--    1. Works fine (i.e., bc produce results) -->
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
from enum import Enum

if uw.mpi.size == 1:
    # to fix trame issue
    import nest_asyncio
    nest_asyncio.apply()
    
    import pyvista as pv
    import underworld3.visualisation as vis
    import matplotlib.pyplot as plt
    import cmcrameri.cm as cmc
    from scipy import integrate

os.environ["SYMPY_USE_CACHE"] = "no"
os.environ["UW_TIMING_ENABLE"] = "1"

# +
# mesh options
r_o = 2.22
r_int = 2.0
r_i = 1.22

res = uw.options.getInt("res", default=16) # 8, 16, 32, 64, 128
cellsize = 1/res
csize_int_fac = 1/2 # internal layer cellsize factor

vdegree  = uw.options.getInt("vdegree", default=2)
pdegree = uw.options.getInt("pdegree", default=1)
pcont = uw.options.getBool("pcont", default=True)
pcont_str = str(pcont).lower()

vel_penalty = uw.options.getReal("vel_penalty", default=2.5e8)
stokes_tol = uw.options.getReal("stokes_tol", default=1e-10)

vel_penalty_str = str("{:.1e}".format(vel_penalty))
stokes_tol_str = str("{:.1e}".format(stokes_tol))
# -

# which normals to use
ana_normal = not True # unit radial vector
petsc_normal = True # gamma function

# compute analytical solutions
analytical = True
timing = True
visualize= True

# +
# specify the case 
case = uw.options.getString('case', default='case2')

n = uw.options.getInt("n", default=2) # wave number
k = uw.options.getInt("k", default=3) # power (check the reference paper)

# +
# boundary condition and density perturbation
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
output_dir = os.path.join(os.path.join("./output/Annulus_Benchmark_Kramer/"), 
                          f'{case}_n_{n}_k_{k}_res_{res}_vdeg_{vdegree}_pdeg_{pdegree}'\
                          f'_pcont_{pcont_str}_vel_penalty_{vel_penalty_str}_stokes_tol_{stokes_tol_str}/')

if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)
# -

# ### Analytical Solution

if analytical:
    if freeslip:
        if delta_fn:
            soln_above = assess.CylindricalStokesSolutionDeltaFreeSlip(n, +1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
            soln_below = assess.CylindricalStokesSolutionDeltaFreeSlip(n, -1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
        elif smooth:
            '''
            For smooth density distribution only single solution exists in the domain. 
            But for sake of code optimization I am creating two solution here.
            '''
            soln_above = assess.CylindricalStokesSolutionSmoothFreeSlip(n, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)
            soln_below = assess.CylindricalStokesSolutionSmoothFreeSlip(n, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)
    elif noslip:
        if delta_fn:
            soln_above = assess.CylindricalStokesSolutionDeltaZeroSlip(n, +1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
            soln_below = assess.CylindricalStokesSolutionDeltaZeroSlip(n, -1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
        elif smooth:
            '''
            For smooth density distribution only single solution exists in the domain. 
            But for sake of code optimization I am creating two solution here.
            '''
            soln_above = assess.CylindricalStokesSolutionSmoothZeroSlip(n, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)
            soln_below = assess.CylindricalStokesSolutionSmoothZeroSlip(n, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)

# ### Create Mesh

if timing:
    uw.timing.reset()
    uw.timing.start()

# mesh
if delta_fn:
    mesh = uw.meshing.AnnulusInternalBoundary(radiusOuter=r_o, 
                                              radiusInternal=r_int, 
                                              radiusInner=r_i, 
                                              cellSize_Inner=cellsize,
                                              cellSize_Internal=cellsize*csize_int_fac,
                                              cellSize_Outer=cellsize,
                                              filename=f'{output_dir}/mesh.msh')
elif smooth:
    mesh = uw.meshing.Annulus(radiusOuter=r_o, radiusInner=r_i, cellSize=cellsize, 
                              qdegree=max(pdegree, vdegree), degree=1, 
                              filename=f'{output_dir}/mesh.msh', refinement=None)

if timing:
    uw.timing.stop()
    uw.timing.print_table(group_by='line_routine', output_file=f"{output_dir}/mesh_create_time.txt",  display_fraction=1.00)

if uw.mpi.size == 1 and visualize:
    # plot_mesh(mesh, _save_png=True, _dir_fname=output_dir+'mesh.png', _title=case)
    vis.plot_mesh(mesh, save_png=True, dir_fname=output_dir+'mesh.png', title='', clip_angle=0., cpos='xy')


# print mesh size in each cpu
uw.pprint(0, '-------------------------------------------------------------------------------')
mesh.dm.view()
uw.pprint(0, '-------------------------------------------------------------------------------')

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

norm_v = uw.discretisation.MeshVariable("N", mesh, 2, degree=1, varsymbol=r"{\hat{n}}")
with mesh.access(norm_v):
    norm_v.data[:,0] = uw.function.evaluate(mesh.CoordinateSystem.unit_e_0[0], norm_v.coords)
    norm_v.data[:,1] = uw.function.evaluate(mesh.CoordinateSystem.unit_e_0[1], norm_v.coords)

# +
# Some useful coordinate stuff
unit_rvec = mesh.CoordinateSystem.unit_e_0
r_uw, th_uw = mesh.CoordinateSystem.xR

# Null space in velocity (constant v_theta) expressed in x,y coordinates
v_theta_fn_xy = r_uw * mesh.CoordinateSystem.rRotN.T * sympy.Matrix((0,1))
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
    clim, vmag, vfreq = [0., 0.05], 5e0, 75
elif case in ('case2'):
    clim, vmag, vfreq = [0., 0.04], 6e0, 75
elif case in ('case3'):
    clim, vmag, vfreq = [0., 0.01], 2.5e1, 75
elif case in ('case4'):
    clim, vmag, vfreq = [0., 0.00925], 3e1, 75
        
if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_vector(mesh, v_ana, vector_name='v_ana', cmap=cmc.lapaz.resampled(11), clim=clim, vmag=vmag, vfreq=vfreq,
                    save_png=True, dir_fname=output_dir+'vel_ana.png', clip_angle=0., show_arrows=False, cpos='xy')
    
    vis.save_colorbar(colormap=cmc.lapaz.resampled(11), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18, 
                      cb_orient='horizontal', cb_axis_label='Velocity', cb_label_xpos=0.5, cb_label_ypos=-2.05, fformat='pdf', 
                      output_path=output_dir, fname='v_ana')

# +
# plotting analytical pressure
if case in ('case1'):
    clim = [-0.65, 0.65]
elif case in ('case2'):
    clim = [-0.5, 0.5]
elif case in ('case3'):
    clim = [-0.65, 0.65]
elif case in ('case4'):
    clim = [-0.5, 0.5]

if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_scalar(mesh, p_ana.sym, 'p_ana', cmap=cmc.vik.resampled(41), clim=clim, save_png=True, clip_angle=0.,
                    dir_fname=output_dir+'p_ana.png', cpos='xy')
    
    vis.save_colorbar(colormap=cmc.vik.resampled(41), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18, 
                      cb_orient='horizontal', cb_axis_label='Pressure', cb_label_xpos=0.5, cb_label_ypos=-2.0, fformat='pdf', 
                      output_path=output_dir, fname='p_ana')
# -

# Create Stokes object
stokes = Stokes(mesh, velocityField=v_uw, pressureField=p_uw)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0
stokes.saddle_preconditioner = 1.0

# defining rho fn
if delta_fn:
    rho = sympy.cos(n*th_uw) * sympy.exp(-1e5 * ((r_uw - r_int) ** 2))
    stokes.add_natural_bc(-rho * unit_rvec, "Internal")
    stokes.bodyforce = sympy.Matrix([0., 0.])
elif smooth:
    rho = ((r_uw/r_o)**k)*sympy.cos(n*th_uw)
    gravity_fn = -1.0 * unit_rvec # gravity
    stokes.bodyforce = rho*gravity_fn 

# rho
if analytical:
    with mesh.access(rho_ana):
        rho_ana.data[:] = np.c_[uw.function.evaluate(rho, rho_ana.coords)]

# +
# plotting analytical rho
clim = [-1, 1]
        
if uw.mpi.size == 1 and visualize:
    vis.plot_scalar(mesh, -rho_ana.sym, 'Rho', cmap=cmc.roma.resampled(31), clim=clim, save_png=True, 
                    dir_fname=output_dir+'rho_ana.png', clip_angle=0., cpos='xy')
    
    vis.save_colorbar(colormap=cmc.roma.resampled(31), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18, 
                      cb_orient='horizontal', cb_axis_label='Rho', cb_label_xpos=0.5, cb_label_ypos=-2.0, fformat='pdf', 
                      output_path=output_dir, fname='rho_ana')

# +
# boundary conditions
if freeslip:
    if ana_normal:
        Gamma = mesh.CoordinateSystem.unit_e_0
    elif petsc_normal:
        Gamma = mesh.Gamma

    # stokes.add_natural_bc(2.5e3 * Gamma.dot(v_uw.sym) *  Gamma, "Upper")
    # stokes.add_natural_bc(2.5e3 * Gamma.dot(v_uw.sym) *  Gamma, "Lower")
    
    v_diff =  v_uw.sym - v_ana.sym
    stokes.add_natural_bc(vel_penalty*v_diff, mesh.boundaries.Upper.name)
    stokes.add_natural_bc(vel_penalty*v_diff, mesh.boundaries.Lower.name)
    
elif noslip:
    # stokes.add_essential_bc(sympy.Matrix([0., 0.]), "Upper")
    # stokes.add_essential_bc(sympy.Matrix([0., 0.]), "Lower")

    v_diff =  v_uw.sym - v_ana.sym
    stokes.add_natural_bc(vel_penalty*v_diff, mesh.boundaries.Upper.name)
    stokes.add_natural_bc(vel_penalty*v_diff, mesh.boundaries.Lower.name)

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
# -

if timing:
    uw.timing.reset()
    uw.timing.start()

stokes.solve(verbose=True, debug=False)

if timing:
    uw.timing.stop()
    uw.timing.print_table(group_by='line_routine', output_file=f"{output_dir}/stokes_solve_time.txt", display_fraction=1.00)

# +
# Null space evaluation
I0 = uw.maths.Integral(mesh, v_theta_fn_xy.dot(v_uw.sym))
norm = I0.evaluate()
I0.fn = v_theta_fn_xy.dot(v_theta_fn_xy)
vnorm = I0.evaluate()

# print(norm/vnorm, vnorm)

with mesh.access(v_uw):
    dv = uw.function.evaluate(norm * v_theta_fn_xy, v_uw.coords) / vnorm
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
    clim, vmag, vfreq = [0., 0.05], 5e0, 75
elif case in ('case2'):
    clim, vmag, vfreq = [0., 0.04], 6e0, 75
elif case in ('case3'):
    clim, vmag, vfreq = [0., 0.01], 2.5e1, 75
elif case in ('case4'):
    clim, vmag, vfreq = [0., 0.00925], 3e1, 75
    
if uw.mpi.size == 1 and visualize:
    vis.plot_vector(mesh, v_uw, vector_name='v_uw', cmap=cmc.lapaz.resampled(11), clim=clim, vmag=vmag, vfreq=vfreq,
                    save_png=True, dir_fname=output_dir+'vel_uw.png', clip_angle=0., cpos='xy', show_arrows=False)

    vis.save_colorbar(colormap=cmc.lapaz.resampled(11), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18, 
                      cb_orient='horizontal', cb_axis_label='Velocity', cb_label_xpos=0.5, cb_label_ypos=-2.05, fformat='pdf', 
                      output_path=output_dir, fname='v_uw')

# +
# plotting relative errror in velocities
if case in ('case1'):
    clim, vmag, vfreq = [0., 0.005], 1e2, 75
elif case in ('case2'):
    clim, vmag, vfreq = [0., 7e-4], 1e2, 75
elif case in ('case3'):
    clim, vmag, vfreq = [0., 1e-4], 2e2, 75
elif case in ('case4'):
    clim, vmag, vfreq = [0., 1e-5], 5e4, 75
        
if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_vector(mesh, v_err, vector_name='v_err(relative)', cmap=cmc.lapaz.resampled(11), clim=clim, vmag=vmag, vfreq=vfreq,
                    save_png=True, dir_fname=output_dir+'vel_r_err.png', clip_angle=0., cpos='xy', show_arrows=False)

    vis.save_colorbar(colormap=cmc.lapaz.resampled(11), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18, 
                      cb_orient='horizontal', cb_axis_label='Velocity Error (relative)', cb_label_xpos=0.5, cb_label_ypos=-2.05, fformat='pdf', 
                      output_path=output_dir, fname='v_err_rel')

# +
# plotting magnitude error in percentage
if case in ('case1'):
    clim = [0, 20]
elif case in ('case2'):
    clim = [0, 20]
elif case in ('case3'):
    clim = [0, 5]
elif case in ('case4'):
    clim = [0, 1]

if uw.mpi.size == 1 and analytical and visualize:   
    vmag_expr = (sympy.sqrt(v_err.sym.dot(v_err.sym))/sympy.sqrt(v_ana.sym.dot(v_ana.sym)))*100
    vis.plot_scalar(mesh, vmag_expr, 'vmag_err(%)', cmap=cmc.oslo_r.resampled(21), clim=clim, save_png=True, 
                    dir_fname=output_dir+'vel_p_err.png', clip_angle=0., cpos='xy')
    
    vis.save_colorbar(colormap=cmc.oslo_r.resampled(21), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18, 
                      cb_orient='horizontal', cb_axis_label='Velocity Error (%)', cb_label_xpos=0.5, cb_label_ypos=-2.05, fformat='pdf', 
                      output_path=output_dir, fname='v_err_perc')

# +
# plotting pressure from uw
if case in ('case1'):
    clim = [-0.65, 0.65]
elif case in ('case2'):
    clim = [-0.5, 0.5]
elif case in ('case3'):
    clim = [-0.65, 0.65]
elif case in ('case4'):
    clim = [-0.5, 0.5]
        
if uw.mpi.size == 1 and visualize:
    vis.plot_scalar(mesh, p_uw.sym, 'p_uw', cmap=cmc.vik.resampled(41), clim=clim, save_png=True, 
                    dir_fname=output_dir+'p_uw.png', clip_angle=0., cpos='xy')
    
    vis.save_colorbar(colormap=cmc.vik.resampled(41), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18, 
                      cb_orient='horizontal', cb_axis_label='Pressure', cb_label_xpos=0.5, cb_label_ypos=-2.0, fformat='pdf', 
                      output_path=output_dir, fname='p_uw')

# +
# plotting relative error in uw
if case in ('case1'):
    clim = [-0.065, 0.065]
elif case in ('case2'):
    clim = [-0.003, 0.003]
elif case in ('case3'):
    clim = [-0.0065, 0.0065]
elif case in ('case4'):
    clim = [-0.0045, 0.0045]
        
if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_scalar(mesh, p_err.sym, 'p_err(relative)', cmap=cmc.vik.resampled(41), clim=clim, save_png=True, 
                    dir_fname=output_dir+'p_r_err.png', clip_angle=0., cpos='xy')

    vis.save_colorbar(colormap=cmc.vik.resampled(41), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18, 
                      cb_orient='horizontal', cb_axis_label='Pressure Error (relative)', cb_label_xpos=0.5, cb_label_ypos=-2.0, fformat='pdf', 
                      output_path=output_dir, fname='p_err_rel')

# +
# plotting percentage error in uw
clim = [-1e2, 1e2]

if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_scalar(mesh, (p_err.sym[0]/p_ana.sym[0])*100, 'p_err(%)', cmap=cmc.vik.resampled(41), clim=clim, save_png=True, 
                    dir_fname=output_dir+'p_p_err.png', clip_angle=0., cpos='xy')
    
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
        
        uw.pprint(0, 'Relative error in velocity in the L2 norm: ', v_err_l2)
            print('Relative error in pressure in the L2 norm: ', p_err_l2)

# +
# writing l2 norms to h5 file
if uw.mpi.size == 1 and os.path.isfile(output_dir+'error_norm.h5'):
    os.remove(output_dir+'error_norm.h5')
    print('Old file removed')

uw.pprint(0, 'Creating new h5 file')
    with h5py.File(output_dir+'error_norm.h5', 'w') as f:
        f.create_dataset("k", data=k)
        f.create_dataset("cellsize", data=cellsize)
        f.create_dataset("res", data=res)
        f.create_dataset("v_l2_norm", data=v_err_l2)
        f.create_dataset("p_l2_norm", data=p_err_l2)
# -
# saving h5 and xdmf file
mesh.petsc_save_checkpoint(index=0, meshVars=[v_uw, p_uw, v_ana, p_ana, v_err, p_err], outputPath=os.path.relpath(output_dir)+'/output')


# ### From here onwards we compute quantities to compare analytical and numerical solution.
# ### Feel free to comment out this section

# #### Plotting velocity and pressure on lower and outer boundaries

# +
# get indices
lower_indx = uw.discretisation.petsc_discretisation.petsc_dm_find_labeled_points_local(mesh.dm, 'Lower')
upper_indx = uw.discretisation.petsc_discretisation.petsc_dm_find_labeled_points_local(mesh.dm, 'Upper')

# get theta from from x, y
lower_theta = uw.function.evalf(th_uw, mesh.data[lower_indx])
lower_theta[lower_theta<0] += 2 * np.pi
upper_theta = uw.function.evalf(th_uw, mesh.data[upper_indx])
upper_theta[upper_theta<0] += 2 * np.pi
# -

# lower and upper bd velocities
if analytical:
    with mesh.access(v_uw, p_uw):
        def get_ana_soln_2(_var, _coords, _r_int, _soln_above, _soln_below):
            # get analytical solution into mesh variables
            r = uw.function.evalf(r_uw, _coords)
            for i, coord in enumerate(_coords):
                if r[i]>_r_int:
                    _var[i] = _soln_above(coord)
                else:
                    _var[i] = _soln_below(coord)
                    
        # pressure arrays
        p_ana_lower = np.zeros((len(lower_indx), 1))
        p_ana_upper = np.zeros((len(upper_indx), 1))
        p_uw_lower = np.zeros((len(lower_indx), 1))
        p_uw_upper = np.zeros((len(upper_indx), 1))

        # pressure analy and uw
        get_ana_soln_2(p_ana_upper, mesh.data[upper_indx], r_int, soln_above.pressure_cartesian, soln_below.pressure_cartesian)
        get_ana_soln_2(p_ana_lower, mesh.data[lower_indx], r_int, soln_above.pressure_cartesian, soln_below.pressure_cartesian)
        p_uw_lower[:,0] = uw.function.evalf(p_uw.sym, mesh.data[lower_indx])
        p_uw_upper[:,0] = uw.function.evalf(p_uw.sym, mesh.data[upper_indx]) 

        # velocity arrays
        v_ana_lower = np.zeros_like(mesh.data[lower_indx])
        v_ana_upper = np.zeros_like(mesh.data[upper_indx])
        v_uw_lower = np.zeros_like(mesh.data[lower_indx])
        v_uw_upper = np.zeros_like(mesh.data[upper_indx])
        
        # velocity analy and uw
        get_ana_soln_2(v_ana_upper, mesh.data[upper_indx], r_int, soln_above.velocity_cartesian, soln_below.velocity_cartesian)
        get_ana_soln_2(v_ana_lower, mesh.data[lower_indx], r_int, soln_above.velocity_cartesian, soln_below.velocity_cartesian)
        v_uw_lower = uw.function.evalf(v_uw.sym, mesh.data[lower_indx])
        v_uw_upper = uw.function.evalf(v_uw.sym, mesh.data[upper_indx])

# sort array for plotting
sort_lower = lower_theta.argsort()
sort_upper = upper_theta.argsort()


def plot_stats(_data_list='', _label_list='', _line_style='', _xlabel='', _ylabel='', _xlim='', _ylim='', _mod_xticks=False, 
               _save_pdf='', _output_path='', _fname=''):
    # plot some statiscs 
    fig, ax = plt.subplots()
    for i, data in enumerate(_data_list):
        ax.plot(data[:,0], data[:,1], label=_label_list[i], linestyle=_line_style[i])
    
    ax.set_xlabel(_xlabel)
    ax.set_ylabel(_ylabel)
    ax.grid(linestyle='--')
    ax.legend(loc=(1.01, 0.60), fontsize=14)

    if len(_xlim)!=0:
        ax.set_xlim(_xlim[0], _xlim[1])
        if _mod_xticks:
            ax.set_xticks(np.arange(_xlim[0], _xlim[1]+0.01, np.pi/2))
            labels = ['$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
            ax.set_xticklabels(labels)

    if len(_ylim)!=0:
        ax.set_ylim(_ylim[0], _ylim[1])

    if _save_pdf:
        plt.savefig(_output_path+_fname+'.pdf', format='pdf', bbox_inches='tight')


# +
# plotting pressure on lower and upper boundaries
if case in ('case1'):
    ylim = [-0.75, 0.75]
elif case in ('case2'):
    ylim = [-0.65, 0.65]
elif case in ('case3'):
    ylim = [-0.95, 0.95]
elif case in ('case4'):
    ylim = [-0.65, 0.65]
    
data_list = [np.hstack((np.c_[lower_theta[sort_lower]], p_ana_lower[sort_lower])),
             np.hstack((np.c_[upper_theta[sort_upper]], p_ana_upper[sort_upper])), 
             np.hstack((np.c_[lower_theta[sort_lower]], p_uw_lower[sort_lower])), 
             np.hstack((np.c_[upper_theta[sort_upper]], p_uw_upper[sort_upper]))]
label_list = ['k='+str(k)+' (analy.), '+r'$r=R_{1}$',
              'k='+str(k)+' (analy.), '+r'$r=R_{2}$',
              'k='+str(k)+' (UW), '+r'$r=R_{1}$', 
              'k='+str(k)+' (UW), '+r'$r=R_{2}$']
linestyle_list = ['-', '-', '--', '--']

plot_stats(_data_list=data_list, _label_list=label_list, _line_style=linestyle_list, _xlabel=r'$\theta$', _ylabel='Pressure', 
           _xlim=[0, 2*np.pi], _ylim=ylim, _mod_xticks=True, _save_pdf=True, _output_path=output_dir, _fname='p_r_i_o')


# -

def get_magnitude(_array):
    # compute velocity magnitude
    sqrd_sum = np.zeros((_array.shape[0], 1))
    for i in range(_array.shape[1]):
        sqrd_sum += _array[:, i:i+1]**2 
    return np.sqrt(sqrd_sum)


# compute velocity magnitude
v_ana_lower_mag = get_magnitude(v_ana_lower)
v_ana_upper_mag = get_magnitude(v_ana_upper)
v_uw_lower_mag = get_magnitude(v_uw_lower)
v_uw_upper_mag = get_magnitude(v_uw_upper)

# +
# plotting vel. mag on lower and upper boundaries
if case in ('case1'):
    ylim = [0, 5e-2]
elif case in ('case2'):
    ylim = [0, 4e-2]
elif case in ('case3'):
    ylim = [0, 3.7e-9]
elif case in ('case4'):
    ylim = [-1e-10, 6e-9]
    
data_list = [np.hstack((np.c_[lower_theta[sort_lower]], v_ana_lower_mag[sort_lower])),
             np.hstack((np.c_[upper_theta[sort_upper]], v_ana_upper_mag[sort_upper])), 
             np.hstack((np.c_[lower_theta[sort_lower]], v_uw_lower_mag[sort_lower])), 
             np.hstack((np.c_[upper_theta[sort_upper]], v_uw_upper_mag[sort_upper]))]
label_list = ['k='+str(k)+' (analy.), '+r'$r=R_{1}$',
              'k='+str(k)+' (analy.), '+r'$r=R_{2}$',
              'k='+str(k)+' (UW), '+r'$r=R_{1}$', 
              'k='+str(k)+' (UW), '+r'$r=R_{2}$']
linestyle_list = ['-', '-', '--', '--']

plot_stats(_data_list=data_list, _label_list=label_list, _line_style=linestyle_list, _xlabel=r'$\theta$', _ylabel='Velocity Magnitude', 
           _xlim=[0, 2*np.pi], _ylim=ylim, _mod_xticks=True, _save_pdf=True, _output_path=output_dir, _fname='vel_r_i_o')
# -

# uw velocity in (r, theta)
v_uw_r_th = mesh.CoordinateSystem.rRotN*v_uw.sym.T

# radial and theta components of velocity integrated over mesh. Theoretically output should be zero
if analytical:   
    v_r_rms_I = uw.maths.Integral(mesh, v_uw_r_th[0])
    print((1/(2*np.pi))*v_r_rms_I.evaluate())

    v_th_rms_I = uw.maths.Integral(mesh, v_uw_r_th[1])
    print((1/(2*np.pi))*v_th_rms_I.evaluate())    

# theta and r arrays
theta_0_2pi = np.linspace(0, 2*np.pi, 1000, endpoint=True)
# r_i_o = np.linspace(r_i, r_o-1e-4, 21, endpoint=True)
r_i_o = np.linspace(r_i, r_o-1e-3, 11, endpoint=True)


def get_vel_avg_r(_theta_arr, _r_arr, _vel_comp):
    'Return average velocity'
    vel_avg_arr = np.zeros_like(_r_arr)
    for i, r_val in enumerate(_r_arr):
        x_arr = r_val*np.cos(_theta_arr)
        y_arr = r_val*np.sin(_theta_arr)
        xy_arr = np.stack((x_arr, y_arr), axis=-1)
        vel_xy = uw.function.evaluate(_vel_comp, xy_arr)
        vel_avg_arr[i] = integrate.simpson(vel_xy, x=_theta_arr)/(2*np.pi)
    return vel_avg_arr


# velocity radial component average
vr_avg = get_vel_avg_r(theta_0_2pi, r_i_o, v_uw_r_th[0])

# +
# plotting velocity radial component average
ylim = [vr_avg.min(), vr_avg.max()] 
    
data_list = [np.hstack((np.c_[r_i_o], np.c_[np.zeros_like(r_i_o)])),
             np.hstack((np.c_[r_i_o], np.c_[vr_avg]))]
label_list = ['k='+str(k)+' (analy.)',
              'k='+str(k)+' (UW)',]
linestyle_list = ['-', '--']

plot_stats(_data_list=data_list, _label_list=label_list, _line_style=linestyle_list, _xlabel='r', _ylabel=r'$<v_{r}>$', 
           _xlim=[r_i, r_o], _ylim=ylim, _save_pdf=True, _output_path=output_dir, _fname='vel_r_avg')
# -

# velocity theta component average
vth_avg = get_vel_avg_r(theta_0_2pi, r_i_o, v_uw_r_th[1])

# +
# plotting velocity theta component average
ylim = [vth_avg.min(), vth_avg.max()] 

data_list = [np.hstack((np.c_[r_i_o], np.c_[np.zeros_like(r_i_o)])),
             np.hstack((np.c_[r_i_o], np.c_[vth_avg]))]
label_list = ['k='+str(k)+' (analy.)',
              'k='+str(k)+' (UW)',]
linestyle_list = ['-', '--']

plot_stats(_data_list=data_list, _label_list=label_list, _line_style=linestyle_list, _xlabel='r', _ylabel=r'$<v_{\theta}>$', 
           _xlim=[r_i, r_o], _ylim=ylim, _save_pdf=True, _output_path=output_dir, _fname='vel_th_avg')
# -


