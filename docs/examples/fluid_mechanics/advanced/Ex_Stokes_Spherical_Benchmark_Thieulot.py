# %% [markdown]
"""
# 🎓 Stokes Spherical Benchmark Thieulot

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
# ## Spherical Benchmark: Viscous Incompressible Stokes
#
# #### [Benchmark ASPECT results](https://aspect-documentation.readthedocs.io/en/latest/user/benchmarks/benchmarks/hollow_sphere/doc/hollow_sphere.html)
# #### [Benchmark paper](https://se.copernicus.org/articles/8/1181/2017/) 
#
# *Author: [Thyagarajulu Gollapalli](https://github.com/gthyagi)*

# ### Analytical solution

# This benchmark is based on [Thieulot](https://se.copernicus.org/articles/8/1181/2017/) in which an analytical solution to the isoviscous incompressible Stokes equations is derived in a spherical shell geometry. The velocity and pressure fields are as follows:
#
# $$ v_{\theta}(r, \theta) = f(r) \sin(\theta) $$
# $$ v_{\phi}(r, \theta) = f(r) \sin(\theta) $$
# $$ v_r(r, \theta) = g(r) \cos(\theta) $$
# $$ p(r, \theta) = h(r) \cos(\theta) $$
# $$ \mu(r) = \mu_{0}r^{m+1} $$
#
# where $m$ is an integer (positive or negative). Note that $m = −1$ yields a constant viscosity.
#
# $$ f(r) = {\alpha} r^{-(m+3)} + \beta r $$
#
# ##### Case $m = -1$
#
# $$ g(r) = -\frac{2}{r^2} \bigg(\alpha \ln r + \frac{\beta}{3}r^3 + \gamma \bigg) $$
# $$ h(r) = \frac{2}{r} \mu_{0} g(r) $$
# $$ \rho(r, \theta) = \bigg(\frac{\alpha}{r^4} (8\ln r - 6) + \frac{8\beta}{3r} + 8\frac{\gamma}{r^4} \bigg) \cos(\theta)$$
# $$ \alpha = -\gamma \frac{R_2^3 - R_1^3}{R_2^3 \ln R_1 - R_1^3 \ln R_2} $$
# $$ \beta = -3\gamma \frac{\ln R_2 - \ln R_1}{R_1^3 \ln R_2 - R_2^3 \ln R_1} $$
#
# ##### Case $m \neq -1$
#
# $$ g(r) = -\frac{2}{r^2} \bigg(-\frac{\alpha}{m+1} r^{-(m+1)} + \frac{\beta}{3}r^3 + \gamma \bigg) $$
# $$ h(r) = \frac{m+3}{r} \mu(r) g(r) $$
# $$ \rho(r, \theta) = \bigg[2\alpha r^{-(m+4)}\frac{m+3}{m+1}(m-1) - \frac{2\beta}{3}(m-1)(m+3) - m(m+5)\frac{2\gamma}{r^3} \bigg] \cos(\theta) $$
# $$ \alpha = \gamma (m+1) \frac{R_1^{-3} - R_2^{-3}}{R_1^{-(m+4)} - R_2^{-(m+4)}} $$
# $$ \beta = -3\gamma \frac{R_1^{m+1} - R_2^{m+1}}{R_1^{m+4} - R_2^{m+4}} $$
# Note that this imposes that $m \neq −4$.
#
# The radial component of the velocity is nul on the inside $r = R_1$ and outside $r = R_2$ of the domain, thereby insuring a
# tangential flow on the boundaries, i.e.
# $$ v_r(R_1, \theta) = v_r(R_2, \theta) = 0 $$
#
# The gravity vector is radial and of unit length. We set $R_1 = 0.5$ and $R_2 = 1$.
#
# In this work, the following spherical coordinates conventions are used: $r$ is the radial distance, $\theta \in [0,\pi]$ is the polar angle and $\phi \in [0, 2\pi]$ is the azimuthal angle.

from mpi4py import MPI
import underworld3 as uw
from underworld3.systems import Stokes

import numpy as np
import sympy as sp
import os
import assess
import h5py
import sys
from petsc4py import PETSc

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
r_o = 1.0
r_i = 0.5
r_int = 0.0

res = uw.options.getInt("res", default=4) # 4, 8, 16, 32, 64, 128
cellsize = 1/res
refine='None'

# m value -1 or 3
m = uw.options.getInt("m", default=-1)

# +
# fem stuff
vdegree  = uw.options.getInt("vdegree", default=2)
pdegree = uw.options.getInt("pdegree", default=1)
pcont = uw.options.getBool("pcont", default=True)
pcont_str = str(pcont).lower()

vel_penalty = uw.options.getReal('vel_penalty', default=1e8)
stokes_tol = uw.options.getReal('stokes_tol', default=1e-10)
vel_penalty_str = str("{:.1e}".format(vel_penalty))
stokes_tol_str = str("{:.1e}".format(stokes_tol))
# -

# compute analytical solution
analytical = True
visualize = False
timing = True

# ### Analytical solution in sympy

# The Cartesian unit vectors are related to the spherical unit vectors by
# $$ 
# \begin{pmatrix}
# \hat{\mathbf{e}}_x \\
# \hat{\mathbf{e}}_y \\
# \hat{\mathbf{e}}_z \\
# \end{pmatrix}
# =
# \begin{pmatrix}
# \sin(\theta) \cos(\phi) & \cos(\theta) \cos(\phi) & -\sin(\phi) \\
# \sin(\theta) \sin(\phi) & \cos(\theta) \sin(\phi) & \cos(\phi) \\
# \cos(\theta) & -\sin(\theta) & 0 \\
# \end{pmatrix}
# \begin{pmatrix}
# \hat{\mathbf{e}}_r \\
# \hat{\mathbf{e}}_{\theta} \\
# \hat{\mathbf{e}}_{\phi} \\
# \end{pmatrix}
# $$

# +
# analytical solution
r = sp.symbols('r')
theta = sp.Symbol('theta', real=True)
phi = sp.Symbol('phi', real=True)

gamma = 1.0
mu_0 = 1.0
mu = mu_0*(r**(m+1))

f = sp.Function('f')(r)
g = sp.Function('g')(r)
h = sp.Function('h')(r)
if m==-1:
    alpha = -gamma*((r_o**3 - r_i**3)/((r_o**3)*np.log(r_i) - (r_i**3)*np.log(r_o)))
    beta = -3*gamma*((np.log(r_o) - np.log(r_i))/((r_i**3)*np.log(r_o) - (r_o**3)*np.log(r_i)))
    f = alpha*(r**-(m+3)) + beta*r
    g = (-2/(r**2))*(alpha*sp.ln(r) + (beta/3)*(r**3) + gamma)
    h = (2/r)*mu_0*g

    # rho = -(alpha / r**4 * (8 * sp.log(r) - 6) + (8 * beta) / (3 * r) + 8 * gamma / r**4) * sp.cos(theta) + rho_0
    
    f_fd = sp.Derivative(f, r, evaluate=True)
    f_sd = sp.Derivative(f_fd, r, evaluate=True)
    f_td = sp.Derivative(f_sd, r, evaluate=True)
    g_fd = sp.Derivative(g, r, evaluate=True)
    g_sd = sp.Derivative(g_fd, r, evaluate=True)
    F_r = -(r*f_td) - (3*f_sd) + ((2*f_fd/r) - g_sd) + 2*((f+g)/r**2)
    
    rho_ = (F_r * sp.cos(theta)) 
    rho = rho_.simplify()
    
else:
    alpha = gamma*(m+1)*((r_i**-3 - r_o**-3)/((r_i**-(m+4)) - (r_o**-(m+4))))
    beta = -3*gamma*((r_i**(m+1)) - (r_o**(m+1)))/((r_i**(m+4)) - (r_o**(m+4)))
    f = alpha*(r**-(m+3)) + beta*r
    g = (-2/(r**2))*((-alpha/(m+1))*r**(-(m+1)) + (beta/3)*(r**3) + gamma)
    h = ((m+3)/r)*mu*g

    # rho = (2 * alpha * r**(-(m + 4)) * ((m + 3) / (m + 1)) * (m - 1)
    #            - (2 * beta / 3) * (m - 1) * (m + 3)
    #            - m * (m + 5) * 2 * gamma / r**3) * sp.cos(theta)
    
    f_fd = sp.Derivative(f, r, evaluate=True)
    f_sd = sp.Derivative(f_fd, r, evaluate=True)
    f_td = sp.Derivative(f_sd, r, evaluate=True)
    g_fd = sp.Derivative(g, r, evaluate=True)
    g_sd = sp.Derivative(g_fd, r, evaluate=True)
    F_r = (-r**2)*f_td - ((2*m)+5)*r*f_sd - (m*(m+3))*f_fd + (m*(m+3)+4)*((f+g)/r) - (m+1)*g_fd - r*g_sd
    rho_ = ((r**m)*F_r * sp.cos(theta))
    rho = rho_.simplify()
    
p = h*sp.cos(theta)

v_r = g*sp.cos(theta)
v_theta = f*sp.sin(theta)
v_phi = f*sp.sin(theta)

v_x = v_r*sp.sin(theta)*sp.cos(phi) + v_theta*sp.cos(theta)*sp.cos(phi) - v_phi*sp.sin(phi)
v_y = v_r*sp.sin(theta)*sp.sin(phi) + v_theta*sp.cos(theta)*sp.sin(phi) + v_phi*sp.cos(phi)
v_z = v_r*sp.cos(theta) - v_theta*sp.sin(theta)
# +
# output_dir = os.path.join(os.path.join("./output/Latex_Dir/"), f"{case}/")
output_dir = os.path.join(os.path.join("./output/"), 
                          f'case_m_{m}_res_{res}_vdeg_{vdegree}_pdeg_{pdegree}'\
                          f'_pcont_{pcont_str}_vel_penalty_{vel_penalty_str}_stokes_tol_{stokes_tol_str}/')

if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)
# -

if uw.mpi.size == 1 and visualize:
    # plot f, g, h, viscosity functions
    rad_np = np.linspace(1, 0.5, num=200, endpoint=True)
    f_np = np.zeros_like(rad_np)
    g_np = np.zeros_like(rad_np)
    h_np = np.zeros_like(rad_np)
    mu_np = np.zeros_like(rad_np)
    
    for i, r_val in enumerate(rad_np):
        f_np[i] = f.subs({r:r_val})
        g_np[i] = g.subs({r:r_val})
        h_np[i] = h.subs({r:r_val})
        mu_np[i] = mu.subs({r:r_val})

    fn_list = [f_np, g_np, h_np, mu_np]
    ylim_list = [[-10, 20], [-3, 4], [-10, 10], [1e-2, 1e2]]
    ylabel_list = [r'$f(r)$', r'$g(r)$', r'$h(r)$', 'Viscosity']
    
    # Set global font size
    plt.rcParams.update({'font.size': 14})
    
    # Create a 2x2 subplot layout
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    
    # Flatten the axs array to simplify iteration
    axs = axs.flatten()
    
    # Plot data on each subplot using a loop
    for i, ax in enumerate(axs):
        ax.plot(rad_np, fn_list[i], color='green', linewidth=1)
        ax.set_xlim(0.5, 1)
        ax.set_ylim(ylim_list[i])
        ax.grid(linewidth=0.7)
        ax.set_xlabel('r')
        ax.set_ylabel(ylabel_list[i])
    
        if i==3:
            # Set the y-axis to be logarithmic
            ax.set_yscale('log')
            
            # Set y axis label tickmark inward
            ax.tick_params(axis='y', direction='in')
    
        # Set the axis grid marks to point inward
        ax.tick_params(axis='both', direction='in', pad=8)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show the plots
    plt.savefig(output_dir+'analy_fns.pdf', format='pdf', bbox_inches='tight')

# ### Create Mesh

if timing:
    uw.timing.reset()
    uw.timing.start()

if r_int!=0.0:
    mesh = uw.meshing.SphericalShellInternalBoundary(radiusInner=r_i, radiusOuter=r_o, radiusInternal=r_int,
                                                     cellSize=cellsize, qdegree=max(pdegree, vdegree), 
                                                     filename=f'{output_dir}mesh.msh')
else:
    mesh = uw.meshing.SphericalShell(radiusInner=r_i, radiusOuter=r_o, cellSize=cellsize, 
                                     qdegree=max(pdegree, vdegree), filename=f'{output_dir}mesh.msh')

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

# +
# norm_v = uw.discretisation.MeshVariable("N", mesh, mesh.data.shape[1], degree=pdegree, varsymbol=r"{\hat{n}}")
# with mesh.access(norm_v):
#     norm_v.data[:,0] = uw.function.evaluate(mesh.CoordinateSystem.unit_e_0[0], norm_v.coords)
#     norm_v.data[:,1] = uw.function.evaluate(mesh.CoordinateSystem.unit_e_0[1], norm_v.coords)
#     norm_v.data[:,2] = uw.function.evaluate(mesh.CoordinateSystem.unit_e_0[2], norm_v.coords)
# -

# Some useful coordinate stuff
unit_rvec = mesh.CoordinateSystem.unit_e_0
r_uw, th_uw = mesh.CoordinateSystem.xR[0], mesh.CoordinateSystem.xR[1]
phi_uw =sp.Piecewise((2*sp.pi + mesh.CoordinateSystem.xR[2], mesh.CoordinateSystem.xR[2]<0), 
                        (mesh.CoordinateSystem.xR[2], True)
                       )

if analytical:
    with mesh.access(v_ana, p_ana, rho_ana):
        p_ana.data[:,0] = uw.function.evalf(p.subs({r:r_uw, theta:th_uw, phi:phi_uw}), p_ana.coords)
        rho_ana.data[:,0] = uw.function.evalf(rho.subs({r:r_uw, theta:th_uw, phi:phi_uw}), rho_ana.coords)
        v_ana.data[:,0] = uw.function.evalf(v_x.subs({r:r_uw, theta:th_uw, phi:phi_uw}), v_ana.coords)
        v_ana.data[:,1] = uw.function.evalf(v_y.subs({r:r_uw, theta:th_uw, phi:phi_uw}), v_ana.coords)
        v_ana.data[:,2] = uw.function.evalf(v_z.subs({r:r_uw, theta:th_uw, phi:phi_uw}), v_ana.coords)

# +
# plotting analytical velocities
if m==-1:
    clim, vmag, vfreq = [0., 5], 5e0, 75
elif m==3:
    clim, vmag, vfreq = [0., 20], 5e0, 75
    
if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_vector(mesh, v_ana, vector_name='v_ana', cmap=cmc.lapaz.resampled(21), clim=clim, vmag=vmag, vfreq=vfreq,
                    save_png=True, dir_fname=output_dir+'vel_ana.png', clip_angle=135, show_arrows=False, cpos='yz')
    
    vis.save_colorbar(colormap=cmc.lapaz.resampled(21), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18, 
                      cb_orient='horizontal', cb_axis_label='Velocity', cb_label_xpos=0.5, cb_label_ypos=-2.05, fformat='pdf', 
                      output_path=output_dir, fname='v_ana')

# +
# plotting analytical pressure
if m==-1:
    clim = [-2.5, 2.5]
elif m==3:
    clim = [-4, 4]
    
if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_scalar(mesh, p_ana.sym, 'p_ana', cmap=cmc.vik.resampled(41), clim=clim, save_png=True, clip_angle=135,
                    dir_fname=output_dir+'p_ana.png', cpos='yz')
    
    vis.save_colorbar(colormap=cmc.vik.resampled(41), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18, 
                      cb_orient='horizontal', cb_axis_label='Pressure', cb_label_xpos=0.5, cb_label_ypos=-2.0, fformat='pdf', 
                      output_path=output_dir, fname='p_ana')
# -

# plotting analytical density
if m==-1:
    clim = [-110, 110]
elif m==3:
    clim = [-35, 35]
if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_scalar(mesh, rho_ana.sym, 'Rho', cmap=cmc.roma.resampled(31), clim=clim, save_png=True, 
                    dir_fname=output_dir+'rho_ana.png', clip_angle=135, cpos='yz')
    
    vis.save_colorbar(colormap=cmc.roma.resampled(31), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18, 
                      cb_orient='horizontal', cb_axis_label='Rho', cb_label_xpos=0.5, cb_label_ypos=-2.0, fformat='pdf', 
                      output_path=output_dir, fname='rho_ana')

# Create Stokes object
stokes = Stokes(mesh, velocityField=v_uw, pressureField=p_uw)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = mu.subs({r:r_uw, theta:th_uw, phi:phi_uw})
stokes.saddle_preconditioner = 1.0/mu.subs({r:r_uw, theta:th_uw, phi:phi_uw})

# +
# gravity
gravity_fn = -1.0 * unit_rvec

# density
if m==-1:
    rho_uw = -rho.subs({r:r_uw, theta:th_uw, phi:phi_uw})
else:
    rho_uw = rho.subs({r:r_uw, theta:th_uw, phi:phi_uw})

# bodyforce term
stokes.bodyforce = rho_uw*gravity_fn # 0.0 * unit_rvec

# +
# boundary conditions

# method1
v_diff = (v_uw.sym - v_ana.sym)
stokes.add_natural_bc(vel_penalty*v_diff, "Upper")
stokes.add_natural_bc(vel_penalty*v_diff, "Lower")

# # method2
# ana_v_x = v_x.subs({r:r_uw, theta:th_uw, phi:phi_uw})
# ana_v_y = v_y.subs({r:r_uw, theta:th_uw, phi:phi_uw})
# ana_v_z = v_z.subs({r:r_uw, theta:th_uw, phi:phi_uw})
# stokes.add_essential_bc(sp.Matrix([ana_v_x, ana_v_y, ana_v_z]), "Upper")
# stokes.add_essential_bc(sp.Matrix([ana_v_x, ana_v_y, ana_v_z]), "Lower")

# imposing pressure boundary condition which seems is required
stokes.add_condition(p_uw.field_id, "dirichlet", 
                     sp.Matrix([0]), mesh.boundaries.Lower.name, 
                     components = (0))

stokes.add_condition(p_uw.field_id, "dirichlet", 
                     sp.Matrix([0]), mesh.boundaries.Upper.name, 
                     components = (0))
if r_int!=0.0:
    ana_p = p.subs({r:r_uw, theta:th_uw, phi:phi_uw})
    stokes.add_condition(p_uw.field_id, "dirichlet", 
                         sp.Matrix([ana_p]), mesh.boundaries.Internal.name, 
                         components = (0))

# +
# Stokes settings
stokes.tolerance = stokes_tol
stokes.petsc_options["ksp_monitor"] = None

# checking snes solve
stokes.petsc_options["ksp_monitor_true_residual"] = None
stokes.petsc_options["snes_monitor"] = None

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

# stokes.petsc_options["fieldsplit_pressure_pc_gasm_type"] = "svd"
# stokes.petsc_options["fieldsplit_pressure_mg_coarse_pc_type"] = "svd"
# stokes.penalty = 5 # values 1 to 10 recommended
# -

if timing:
    uw.timing.reset()
    uw.timing.start()

# +
stokes.solve(verbose=True, debug=False)

# check reason for convergence
print(stokes.snes.getConvergedReason())
print(stokes.snes.ksp.getConvergedReason())
# -

if timing:
    uw.timing.stop()
    uw.timing.print_table(group_by='line_routine', output_file=f"{output_dir}stokes_solve_time.txt", display_fraction=1.00)

# +
# # Null space evaluation

# I0 = uw.maths.Integral(mesh, v_theta_fn_xy.dot(v_uw.sym))
# norm = I0.evaluate()
# I0.fn = v_theta_fn_xy.dot(v_theta_fn_xy)
# vnorm = I0.evaluate()
# # print(norm/vnorm, vnorm)

# with mesh.access(v_uw):
#     dv = uw.function.evaluate(norm * v_theta_fn_xy, v_uw.coords) / vnorm
#     v_uw.data[...] -= dv 
# -

# compute error
if analytical:
    with mesh.access(v_uw, p_uw, v_err, p_err):
        v_err.data[:,0] = v_uw.data[:,0] - v_ana.data[:,0]
        v_err.data[:,1] = v_uw.data[:,1] - v_ana.data[:,1]
        v_err.data[:,2] = v_uw.data[:,2] - v_ana.data[:,2]
        p_err.data[:,0] = p_uw.data[:,0] - p_ana.data[:,0]

# +
# plotting velocities from uw
if m==-1:
    clim, vmag, vfreq = [0., 5.], 5e0, 75
elif m==3:
    clim, vmag, vfreq = [0., 20], 5e0, 75
    
if uw.mpi.size == 1 and visualize:
    vis.plot_vector(mesh, v_uw, vector_name='v_uw', cmap=cmc.lapaz.resampled(21), clim=clim, vmag=vmag, vfreq=vfreq,
                    save_png=True, dir_fname=output_dir+'vel_uw.png', clip_angle=135, cpos='yz', show_arrows=False)

    vis.save_colorbar(colormap=cmc.lapaz.resampled(21), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18, 
                      cb_orient='horizontal', cb_axis_label='Velocity', cb_label_xpos=0.5, cb_label_ypos=-2.05, fformat='pdf', 
                      output_path=output_dir, fname='v_uw')

# +
# plotting relative errror in velocities
if m==-1:
    clim, vmag, vfreq = [0., 0.05], 1e2, 75
elif m==3:
    clim, vmag, vfreq = [0., 4], 1e2, 75
         
if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_vector(mesh, v_err, vector_name='v_err(relative)', cmap=cmc.lapaz.resampled(11), clim=clim, vmag=vmag, vfreq=vfreq,
                    save_png=True, dir_fname=output_dir+'vel_r_err.png', clip_angle=135, cpos='yz', show_arrows=False)

    vis.save_colorbar(colormap=cmc.lapaz.resampled(11), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18, 
                      cb_orient='horizontal', cb_axis_label='Velocity', cb_label_xpos=0.5, cb_label_ypos=-2.05, fformat='pdf', 
                      output_path=output_dir, fname='v_err_rel')
# -

# plotting magnitude error in percentage
clim = [0, 5]
if uw.mpi.size == 1 and analytical and visualize: 
    vmag_expr = (sp.sqrt(v_err.sym.dot(v_err.sym))/sp.sqrt(v_ana.sym.dot(v_ana.sym)))*100
    vis.plot_scalar(mesh, vmag_expr, 'vmag_err(%)', cmap=cmc.oslo_r.resampled(21), clim=clim, save_png=True, 
                    dir_fname=output_dir+'vel_p_err.png', clip_angle=135, cpos='yz')
    
    vis.save_colorbar(colormap=cmc.oslo_r.resampled(21), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18, 
                      cb_orient='horizontal', cb_axis_label='Velocity Error (%)', cb_label_xpos=0.5, cb_label_ypos=-2.05, fformat='pdf', 
                      output_path=output_dir, fname='v_err_perc')

# +
# plotting pressure from uw
if m==-1:
    clim = [-2.5, 2.5]
elif m==3:
    clim = [-4, 4]
        
if uw.mpi.size == 1 and visualize:
    vis.plot_scalar(mesh, p_uw.sym, 'p_uw', cmap=cmc.vik.resampled(41), clim=clim, save_png=True, 
                    dir_fname=output_dir+'p_uw.png', clip_angle=135, cpos='yz')
    
    vis.save_colorbar(colormap=cmc.vik.resampled(41), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18, 
                      cb_orient='horizontal', cb_axis_label='Pressure', cb_label_xpos=0.5, cb_label_ypos=-2.0, fformat='pdf', 
                      output_path=output_dir, fname='p_uw')

# +
# plotting relative error in uw

clim = [-0.5, 0.5]  
if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_scalar(mesh, p_err.sym, 'p_err(relative)', cmap=cmc.vik.resampled(41), clim=clim, save_png=True, 
                    dir_fname=output_dir+'p_r_err.png', clip_angle=135, cpos='yz')

    vis.save_colorbar(colormap=cmc.vik.resampled(41), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18, 
                      cb_orient='horizontal', cb_axis_label='Pressure', cb_label_xpos=0.5, cb_label_ypos=-2.0, fformat='pdf', 
                      output_path=output_dir, fname='p_err_rel')

# +
# plotting percentage error in uw
clim=[-10, 10]

if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_scalar(mesh, (p_err.sym[0]/p_ana.sym[0])*100, 'p_err(%)', cmap=cmc.vik.resampled(41), clim=clim, save_png=True, 
                    dir_fname=output_dir+'p_p_err.png', clip_angle=135, cpos='yz')

    vis.save_colorbar(colormap=cmc.vik.resampled(41), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18, 
                      cb_orient='horizontal', cb_axis_label='Pressure', cb_label_xpos=0.5, cb_label_ypos=-2.0, fformat='pdf', 
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
# writing l2 norms to h5 file
if uw.mpi.size == 1 and os.path.isfile(output_dir+'error_norm.h5'):
    os.remove(output_dir+'error_norm.h5')
    print('Old file removed')

uw.pprint('Creating new h5 file')
    with h5py.File(output_dir+'error_norm.h5', 'w') as f:
        f.create_dataset("m", data=m)
        f.create_dataset("cellsize", data=cellsize)
        f.create_dataset("res", data=res)
        f.create_dataset("v_l2_norm", data=v_err_l2)
        f.create_dataset("p_l2_norm", data=p_err_l2)
# -

# saving h5 and xdmf file
mesh.petsc_save_checkpoint(index=0, meshVars=[v_uw, p_uw, v_ana, p_ana, v_err, p_err, rho_ana], outputPath=os.path.relpath(output_dir)+'/output')



