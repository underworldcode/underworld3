# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Stokes Spherical Shell Benchmark (Thieulot)

**PHYSICS:** fluid_mechanics
**DIFFICULTY:** advanced

## Description

Benchmark validation of the Stokes solver in a spherical shell geometry using
the manufactured solution from Thieulot (2017). Tests both constant (m=-1)
and variable (m=3) viscosity cases with analytical velocity and pressure fields.

## Key Concepts

- **Manufactured solution**: Analytical velocity/pressure fields that satisfy Stokes
- **Variable viscosity**: Power-law radial viscosity mu(r) = mu_0 * r^(m+1)
- **Spherical shell geometry**: 3D shell with inner and outer radii
- **Penalty boundary conditions**: Natural BCs matching analytical solution
- **L2 error analysis**: Convergence validation against analytical solution

## Mathematical Formulation

Velocity components in spherical coordinates:
$$v_r(r, \\theta) = g(r) \\cos(\\theta)$$
$$v_\\theta(r, \\theta) = f(r) \\sin(\\theta)$$
$$v_\\phi(r, \\theta) = f(r) \\sin(\\theta)$$

Viscosity profile:
$$\\mu(r) = \\mu_0 r^{m+1}$$

where m=-1 gives constant viscosity, m=3 gives strongly variable viscosity.

## Parameters

- `uw_res`: Mesh resolution (1/cellsize)
- `uw_m`: Viscosity exponent (-1 or 3)
- `uw_vdegree`: Velocity polynomial degree
- `uw_pdegree`: Pressure polynomial degree
- `uw_pcont`: Pressure continuity (True/False)
- `uw_vel_penalty`: Penalty for boundary conditions
- `uw_stokes_tol`: Solver tolerance

## References

- [Thieulot (2017)](https://se.copernicus.org/articles/8/1181/2017/)
- [ASPECT benchmark](https://aspect-documentation.readthedocs.io/en/latest/user/benchmarks/benchmarks/hollow_sphere/doc/hollow_sphere.html)

## Author

[Thyagarajulu Gollapalli](https://github.com/gthyagi)
"""

# %% [markdown]
"""
## Setup and Parameters
"""

# %%
import nest_asyncio
nest_asyncio.apply()

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
    import pyvista as pv
    import underworld3.visualisation as vis
    import matplotlib.pyplot as plt
    import cmcrameri.cm as cmc

os.environ["SYMPY_USE_CACHE"] = "no"
os.environ["UW_TIMING_ENABLE"] = "1"

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_Stokes_Spherical_Benchmark_Thieulot.py -uw_res 8
python Ex_Stokes_Spherical_Benchmark_Thieulot.py -uw_m 3
python Ex_Stokes_Spherical_Benchmark_Thieulot.py -uw_vdegree 3 -uw_pdegree 2
```
"""

# %%
params = uw.Params(
    uw_res = 4,                   # Mesh resolution (1/cellsize)
    uw_m = -1,                    # Viscosity exponent: -1 (constant) or 3 (variable)
    uw_vdegree = 2,               # Velocity polynomial degree
    uw_pdegree = 1,               # Pressure polynomial degree
    uw_pcont = 1,                 # Pressure continuity (1=True, 0=False)
    uw_vel_penalty = 1e8,         # Penalty for boundary conditions
    uw_stokes_tol = 1e-10,        # Solver tolerance
    uw_r_outer = 1.0,             # Outer radius
    uw_r_inner = 0.5,             # Inner radius
)

# Extract parameters
res = int(params.uw_res)
m = int(params.uw_m)
vdegree = int(params.uw_vdegree)
pdegree = int(params.uw_pdegree)
pcont = bool(params.uw_pcont)
vel_penalty = params.uw_vel_penalty
stokes_tol = params.uw_stokes_tol
r_o = params.uw_r_outer
r_i = params.uw_r_inner

# Derived parameters
cellsize = 1/res
r_int = 0.0  # No internal boundary for this benchmark
pcont_str = str(pcont).lower()
vel_penalty_str = str("{:.1e}".format(vel_penalty))
stokes_tol_str = str("{:.1e}".format(stokes_tol))
refine = 'None'

# Compute flags
analytical = True
visualize = False
timing = True

# %% [markdown]
"""
## Analytical Solution in Sympy

The Cartesian unit vectors are related to the spherical unit vectors by:
$$
\\begin{pmatrix}
\\hat{\\mathbf{e}}_x \\\\
\\hat{\\mathbf{e}}_y \\\\
\\hat{\\mathbf{e}}_z \\\\
\\end{pmatrix}
=
\\begin{pmatrix}
\\sin(\\theta) \\cos(\\phi) & \\cos(\\theta) \\cos(\\phi) & -\\sin(\\phi) \\\\
\\sin(\\theta) \\sin(\\phi) & \\cos(\\theta) \\sin(\\phi) & \\cos(\\phi) \\\\
\\cos(\\theta) & -\\sin(\\theta) & 0 \\\\
\\end{pmatrix}
\\begin{pmatrix}
\\hat{\\mathbf{e}}_r \\\\
\\hat{\\mathbf{e}}_{\\theta} \\\\
\\hat{\\mathbf{e}}_{\\phi} \\\\
\\end{pmatrix}
$$
"""

# %%
# Analytical solution symbols
r = sp.symbols('r')
theta = sp.Symbol('theta', real=True)
phi = sp.Symbol('phi', real=True)

gamma = 1.0
mu_0 = 1.0
mu = mu_0*(r**(m+1))

f = sp.Function('f')(r)
g = sp.Function('g')(r)
h = sp.Function('h')(r)

if m == -1:
    # Constant viscosity case
    alpha = -gamma*((r_o**3 - r_i**3)/((r_o**3)*np.log(r_i) - (r_i**3)*np.log(r_o)))
    beta = -3*gamma*((np.log(r_o) - np.log(r_i))/((r_i**3)*np.log(r_o) - (r_o**3)*np.log(r_i)))
    f = alpha*(r**-(m+3)) + beta*r
    g = (-2/(r**2))*(alpha*sp.ln(r) + (beta/3)*(r**3) + gamma)
    h = (2/r)*mu_0*g

    f_fd = sp.Derivative(f, r, evaluate=True)
    f_sd = sp.Derivative(f_fd, r, evaluate=True)
    f_td = sp.Derivative(f_sd, r, evaluate=True)
    g_fd = sp.Derivative(g, r, evaluate=True)
    g_sd = sp.Derivative(g_fd, r, evaluate=True)
    F_r = -(r*f_td) - (3*f_sd) + ((2*f_fd/r) - g_sd) + 2*((f+g)/r**2)

    rho_ = (F_r * sp.cos(theta))
    rho = rho_.simplify()
else:
    # Variable viscosity case (m != -1)
    alpha = gamma*(m+1)*((r_i**-3 - r_o**-3)/((r_i**-(m+4)) - (r_o**-(m+4))))
    beta = -3*gamma*((r_i**(m+1)) - (r_o**(m+1)))/((r_i**(m+4)) - (r_o**(m+4)))
    f = alpha*(r**-(m+3)) + beta*r
    g = (-2/(r**2))*((-alpha/(m+1))*r**(-(m+1)) + (beta/3)*(r**3) + gamma)
    h = ((m+3)/r)*mu*g

    f_fd = sp.Derivative(f, r, evaluate=True)
    f_sd = sp.Derivative(f_fd, r, evaluate=True)
    f_td = sp.Derivative(f_sd, r, evaluate=True)
    g_fd = sp.Derivative(g, r, evaluate=True)
    g_sd = sp.Derivative(g_fd, r, evaluate=True)
    F_r = (-r**2)*f_td - ((2*m)+5)*r*f_sd - (m*(m+3))*f_fd + (m*(m+3)+4)*((f+g)/r) - (m+1)*g_fd - r*g_sd
    rho_ = ((r**m)*F_r * sp.cos(theta))
    rho = rho_.simplify()

# Pressure and velocity in spherical coordinates
p = h*sp.cos(theta)

v_r = g*sp.cos(theta)
v_theta = f*sp.sin(theta)
v_phi = f*sp.sin(theta)

# Transform to Cartesian
v_x = v_r*sp.sin(theta)*sp.cos(phi) + v_theta*sp.cos(theta)*sp.cos(phi) - v_phi*sp.sin(phi)
v_y = v_r*sp.sin(theta)*sp.sin(phi) + v_theta*sp.cos(theta)*sp.sin(phi) + v_phi*sp.cos(phi)
v_z = v_r*sp.cos(theta) - v_theta*sp.sin(theta)

# %% [markdown]
"""
## Output Directory
"""

# %%
output_dir = os.path.join(
    "./output/",
    f'case_m_{m}_res_{res}_vdeg_{vdegree}_pdeg_{pdegree}'
    f'_pcont_{pcont_str}_vel_penalty_{vel_penalty_str}_stokes_tol_{stokes_tol_str}/'
)

if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)

# %% [markdown]
"""
## Visualization of Analytical Functions
"""

# %%
if uw.mpi.size == 1 and visualize:
    # Plot f, g, h, viscosity functions
    rad_np = np.linspace(1, 0.5, num=200, endpoint=True)
    f_np = np.zeros_like(rad_np)
    g_np = np.zeros_like(rad_np)
    h_np = np.zeros_like(rad_np)
    mu_np = np.zeros_like(rad_np)

    for i, r_val in enumerate(rad_np):
        f_np[i] = f.subs({r: r_val})
        g_np[i] = g.subs({r: r_val})
        h_np[i] = h.subs({r: r_val})
        mu_np[i] = mu.subs({r: r_val})

    fn_list = [f_np, g_np, h_np, mu_np]
    ylim_list = [[-10, 20], [-3, 4], [-10, 10], [1e-2, 1e2]]
    ylabel_list = [r'$f(r)$', r'$g(r)$', r'$h(r)$', 'Viscosity']

    plt.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        ax.plot(rad_np, fn_list[i], color='green', linewidth=1)
        ax.set_xlim(0.5, 1)
        ax.set_ylim(ylim_list[i])
        ax.grid(linewidth=0.7)
        ax.set_xlabel('r')
        ax.set_ylabel(ylabel_list[i])

        if i == 3:
            ax.set_yscale('log')
            ax.tick_params(axis='y', direction='in')

        ax.tick_params(axis='both', direction='in', pad=8)

    plt.tight_layout()
    plt.savefig(output_dir + 'analy_fns.pdf', format='pdf', bbox_inches='tight')

# %% [markdown]
"""
## Mesh Generation
"""

# %%
if timing:
    uw.timing.reset()
    uw.timing.start()

if r_int != 0.0:
    mesh = uw.meshing.SphericalShellInternalBoundary(
        radiusInner=r_i, radiusOuter=r_o, radiusInternal=r_int,
        cellSize=cellsize, qdegree=max(pdegree, vdegree),
        filename=f'{output_dir}mesh.msh'
    )
else:
    mesh = uw.meshing.SphericalShell(
        radiusInner=r_i, radiusOuter=r_o, cellSize=cellsize,
        qdegree=max(pdegree, vdegree), filename=f'{output_dir}mesh.msh'
    )

if timing:
    uw.timing.stop()
    uw.timing.print_table(
        group_by='line_routine',
        output_file=f"{output_dir}mesh_create_time.txt",
        display_fraction=1.00
    )

if uw.mpi.size == 1 and visualize:
    vis.plot_mesh(mesh, save_png=True, dir_fname=output_dir + 'mesh.png',
                  title='', clip_angle=135, cpos='yz')

# Print mesh size in each CPU
uw.pprint('-------------------------------------------------------------------------------')
mesh.dm.view()
uw.pprint('-------------------------------------------------------------------------------')

# %% [markdown]
"""
## Variables
"""

# %%
v_uw = uw.discretisation.MeshVariable('V_u', mesh, mesh.data.shape[1], degree=vdegree)
p_uw = uw.discretisation.MeshVariable('P_u', mesh, 1, degree=pdegree, continuous=pcont)

if analytical:
    v_ana = uw.discretisation.MeshVariable('V_a', mesh, mesh.data.shape[1], degree=vdegree)
    p_ana = uw.discretisation.MeshVariable('P_a', mesh, 1, degree=pdegree, continuous=pcont)
    rho_ana = uw.discretisation.MeshVariable('RHO_a', mesh, 1, degree=pdegree, continuous=True)

    v_err = uw.discretisation.MeshVariable('V_e', mesh, mesh.data.shape[1], degree=vdegree)
    p_err = uw.discretisation.MeshVariable('P_e', mesh, 1, degree=pdegree, continuous=pcont)

# %% [markdown]
"""
## Coordinate System
"""

# %%
unit_rvec = mesh.CoordinateSystem.unit_e_0
r_uw, th_uw = mesh.CoordinateSystem.xR[0], mesh.CoordinateSystem.xR[1]
phi_uw = sp.Piecewise(
    (2*sp.pi + mesh.CoordinateSystem.xR[2], mesh.CoordinateSystem.xR[2] < 0),
    (mesh.CoordinateSystem.xR[2], True)
)

# %% [markdown]
"""
## Initialize Analytical Solution
"""

# %%
if analytical:
    with mesh.access(v_ana, p_ana, rho_ana):
        p_ana.data[:, 0] = uw.function.evalf(p.subs({r: r_uw, theta: th_uw, phi: phi_uw}), p_ana.coords)
        rho_ana.data[:, 0] = uw.function.evalf(rho.subs({r: r_uw, theta: th_uw, phi: phi_uw}), rho_ana.coords)
        v_ana.data[:, 0] = uw.function.evalf(v_x.subs({r: r_uw, theta: th_uw, phi: phi_uw}), v_ana.coords)
        v_ana.data[:, 1] = uw.function.evalf(v_y.subs({r: r_uw, theta: th_uw, phi: phi_uw}), v_ana.coords)
        v_ana.data[:, 2] = uw.function.evalf(v_z.subs({r: r_uw, theta: th_uw, phi: phi_uw}), v_ana.coords)

# %% [markdown]
"""
## Visualization of Analytical Solution
"""

# %%
# Set visualization parameters based on m value
if m == -1:
    v_clim, vmag, vfreq = [0., 5], 5e0, 75
elif m == 3:
    v_clim, vmag, vfreq = [0., 20], 5e0, 75

if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_vector(
        mesh, v_ana, vector_name='v_ana', cmap=cmc.lapaz.resampled(21),
        clim=v_clim, vmag=vmag, vfreq=vfreq, save_png=True,
        dir_fname=output_dir + 'vel_ana.png', clip_angle=135,
        show_arrows=False, cpos='yz'
    )

    vis.save_colorbar(
        colormap=cmc.lapaz.resampled(21), cb_bounds=None,
        vmin=v_clim[0], vmax=v_clim[1], figsize_cb=(5, 5), primary_fs=18,
        cb_orient='horizontal', cb_axis_label='Velocity',
        cb_label_xpos=0.5, cb_label_ypos=-2.05, fformat='pdf',
        output_path=output_dir, fname='v_ana'
    )

# %%
# Pressure visualization parameters
if m == -1:
    p_clim = [-2.5, 2.5]
elif m == 3:
    p_clim = [-4, 4]

if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_scalar(
        mesh, p_ana.sym, 'p_ana', cmap=cmc.vik.resampled(41),
        clim=p_clim, save_png=True, clip_angle=135,
        dir_fname=output_dir + 'p_ana.png', cpos='yz'
    )

    vis.save_colorbar(
        colormap=cmc.vik.resampled(41), cb_bounds=None,
        vmin=p_clim[0], vmax=p_clim[1], figsize_cb=(5, 5), primary_fs=18,
        cb_orient='horizontal', cb_axis_label='Pressure',
        cb_label_xpos=0.5, cb_label_ypos=-2.0, fformat='pdf',
        output_path=output_dir, fname='p_ana'
    )

# %%
# Density visualization parameters
if m == -1:
    rho_clim = [-110, 110]
elif m == 3:
    rho_clim = [-35, 35]

if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_scalar(
        mesh, rho_ana.sym, 'Rho', cmap=cmc.roma.resampled(31),
        clim=rho_clim, save_png=True,
        dir_fname=output_dir + 'rho_ana.png', clip_angle=135, cpos='yz'
    )

    vis.save_colorbar(
        colormap=cmc.roma.resampled(31), cb_bounds=None,
        vmin=rho_clim[0], vmax=rho_clim[1], figsize_cb=(5, 5), primary_fs=18,
        cb_orient='horizontal', cb_axis_label='Rho',
        cb_label_xpos=0.5, cb_label_ypos=-2.0, fformat='pdf',
        output_path=output_dir, fname='rho_ana'
    )

# %% [markdown]
"""
## Stokes Solver
"""

# %%
stokes = Stokes(mesh, velocityField=v_uw, pressureField=p_uw)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = mu.subs({r: r_uw, theta: th_uw, phi: phi_uw})
stokes.saddle_preconditioner = 1.0 / mu.subs({r: r_uw, theta: th_uw, phi: phi_uw})

# %% [markdown]
"""
## Body Force
"""

# %%
gravity_fn = -1.0 * unit_rvec

# Density (sign differs between m=-1 and m!=1 cases)
if m == -1:
    rho_uw = -rho.subs({r: r_uw, theta: th_uw, phi: phi_uw})
else:
    rho_uw = rho.subs({r: r_uw, theta: th_uw, phi: phi_uw})

stokes.bodyforce = rho_uw * gravity_fn

# %% [markdown]
"""
## Boundary Conditions

Penalty-based boundary conditions matching analytical solution.
"""

# %%
v_diff = (v_uw.sym - v_ana.sym)
stokes.add_natural_bc(vel_penalty * v_diff, "Upper")
stokes.add_natural_bc(vel_penalty * v_diff, "Lower")

# Pressure boundary conditions
stokes.add_condition(
    p_uw.field_id, "dirichlet",
    sp.Matrix([0]), mesh.boundaries.Lower.name,
    components=(0)
)

stokes.add_condition(
    p_uw.field_id, "dirichlet",
    sp.Matrix([0]), mesh.boundaries.Upper.name,
    components=(0)
)

if r_int != 0.0:
    ana_p = p.subs({r: r_uw, theta: th_uw, phi: phi_uw})
    stokes.add_condition(
        p_uw.field_id, "dirichlet",
        sp.Matrix([ana_p]), mesh.boundaries.Internal.name,
        components=(0)
    )

# %% [markdown]
"""
## Solver Configuration
"""

# %%
stokes.tolerance = stokes_tol
stokes.petsc_options["ksp_monitor"] = None
stokes.petsc_options["ksp_monitor_true_residual"] = None
stokes.petsc_options["snes_monitor"] = None

stokes.petsc_options["snes_type"] = "newtonls"
stokes.petsc_options["ksp_type"] = "fgmres"

stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")

stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "fcg"
stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 5
stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# %% [markdown]
"""
## Solve
"""

# %%
if timing:
    uw.timing.reset()
    uw.timing.start()

stokes.solve(verbose=True, debug=False)

# Check convergence
print(stokes.snes.getConvergedReason())
print(stokes.snes.ksp.getConvergedReason())

if timing:
    uw.timing.stop()
    uw.timing.print_table(
        group_by='line_routine',
        output_file=f"{output_dir}stokes_solve_time.txt",
        display_fraction=1.00
    )

# %% [markdown]
"""
## Compute Error
"""

# %%
if analytical:
    with mesh.access(v_uw, p_uw, v_err, p_err):
        v_err.data[:, 0] = v_uw.data[:, 0] - v_ana.data[:, 0]
        v_err.data[:, 1] = v_uw.data[:, 1] - v_ana.data[:, 1]
        v_err.data[:, 2] = v_uw.data[:, 2] - v_ana.data[:, 2]
        p_err.data[:, 0] = p_uw.data[:, 0] - p_ana.data[:, 0]

# %% [markdown]
"""
## Visualization of Results
"""

# %%
# Velocity visualization
if m == -1:
    v_clim, vmag, vfreq = [0., 5.], 5e0, 75
elif m == 3:
    v_clim, vmag, vfreq = [0., 20], 5e0, 75

if uw.mpi.size == 1 and visualize:
    vis.plot_vector(
        mesh, v_uw, vector_name='v_uw', cmap=cmc.lapaz.resampled(21),
        clim=v_clim, vmag=vmag, vfreq=vfreq, save_png=True,
        dir_fname=output_dir + 'vel_uw.png', clip_angle=135,
        cpos='yz', show_arrows=False
    )

    vis.save_colorbar(
        colormap=cmc.lapaz.resampled(21), cb_bounds=None,
        vmin=v_clim[0], vmax=v_clim[1], figsize_cb=(5, 5), primary_fs=18,
        cb_orient='horizontal', cb_axis_label='Velocity',
        cb_label_xpos=0.5, cb_label_ypos=-2.05, fformat='pdf',
        output_path=output_dir, fname='v_uw'
    )

# %%
# Velocity error visualization
if m == -1:
    verr_clim, vmag, vfreq = [0., 0.05], 1e2, 75
elif m == 3:
    verr_clim, vmag, vfreq = [0., 4], 1e2, 75

if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_vector(
        mesh, v_err, vector_name='v_err(relative)', cmap=cmc.lapaz.resampled(11),
        clim=verr_clim, vmag=vmag, vfreq=vfreq, save_png=True,
        dir_fname=output_dir + 'vel_r_err.png', clip_angle=135,
        cpos='yz', show_arrows=False
    )

    vis.save_colorbar(
        colormap=cmc.lapaz.resampled(11), cb_bounds=None,
        vmin=verr_clim[0], vmax=verr_clim[1], figsize_cb=(5, 5), primary_fs=18,
        cb_orient='horizontal', cb_axis_label='Velocity',
        cb_label_xpos=0.5, cb_label_ypos=-2.05, fformat='pdf',
        output_path=output_dir, fname='v_err_rel'
    )

# %%
# Velocity magnitude error in percentage
clim = [0, 5]
if uw.mpi.size == 1 and analytical and visualize:
    vmag_expr = (sp.sqrt(v_err.sym.dot(v_err.sym)) / sp.sqrt(v_ana.sym.dot(v_ana.sym))) * 100
    vis.plot_scalar(
        mesh, vmag_expr, 'vmag_err(%)', cmap=cmc.oslo_r.resampled(21),
        clim=clim, save_png=True,
        dir_fname=output_dir + 'vel_p_err.png', clip_angle=135, cpos='yz'
    )

    vis.save_colorbar(
        colormap=cmc.oslo_r.resampled(21), cb_bounds=None,
        vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18,
        cb_orient='horizontal', cb_axis_label='Velocity Error (%)',
        cb_label_xpos=0.5, cb_label_ypos=-2.05, fformat='pdf',
        output_path=output_dir, fname='v_err_perc'
    )

# %%
# Pressure visualization
if m == -1:
    p_clim = [-2.5, 2.5]
elif m == 3:
    p_clim = [-4, 4]

if uw.mpi.size == 1 and visualize:
    vis.plot_scalar(
        mesh, p_uw.sym, 'p_uw', cmap=cmc.vik.resampled(41),
        clim=p_clim, save_png=True,
        dir_fname=output_dir + 'p_uw.png', clip_angle=135, cpos='yz'
    )

    vis.save_colorbar(
        colormap=cmc.vik.resampled(41), cb_bounds=None,
        vmin=p_clim[0], vmax=p_clim[1], figsize_cb=(5, 5), primary_fs=18,
        cb_orient='horizontal', cb_axis_label='Pressure',
        cb_label_xpos=0.5, cb_label_ypos=-2.0, fformat='pdf',
        output_path=output_dir, fname='p_uw'
    )

# %%
# Pressure error visualization
clim = [-0.5, 0.5]
if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_scalar(
        mesh, p_err.sym, 'p_err(relative)', cmap=cmc.vik.resampled(41),
        clim=clim, save_png=True,
        dir_fname=output_dir + 'p_r_err.png', clip_angle=135, cpos='yz'
    )

    vis.save_colorbar(
        colormap=cmc.vik.resampled(41), cb_bounds=None,
        vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18,
        cb_orient='horizontal', cb_axis_label='Pressure',
        cb_label_xpos=0.5, cb_label_ypos=-2.0, fformat='pdf',
        output_path=output_dir, fname='p_err_rel'
    )

# %%
# Pressure percentage error
clim = [-10, 10]
if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_scalar(
        mesh, (p_err.sym[0] / p_ana.sym[0]) * 100, 'p_err(%)',
        cmap=cmc.vik.resampled(41), clim=clim, save_png=True,
        dir_fname=output_dir + 'p_p_err.png', clip_angle=135, cpos='yz'
    )

    vis.save_colorbar(
        colormap=cmc.vik.resampled(41), cb_bounds=None,
        vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18,
        cb_orient='horizontal', cb_axis_label='Pressure',
        cb_label_xpos=0.5, cb_label_ypos=-2.0, fformat='pdf',
        output_path=output_dir, fname='p_err_perc'
    )

# %% [markdown]
"""
## L2 Error Analysis
"""

# %%
if analytical:
    with mesh.access(v_err, p_err, p_ana, v_ana):
        v_err_I = uw.maths.Integral(mesh, v_err.sym.dot(v_err.sym))
        v_ana_I = uw.maths.Integral(mesh, v_ana.sym.dot(v_ana.sym))
        v_err_l2 = np.sqrt(v_err_I.evaluate()) / np.sqrt(v_ana_I.evaluate())

        p_err_I = uw.maths.Integral(mesh, p_err.sym.dot(p_err.sym))
        p_ana_I = uw.maths.Integral(mesh, p_ana.sym.dot(p_ana.sym))
        p_err_l2 = np.sqrt(p_err_I.evaluate()) / np.sqrt(p_ana_I.evaluate())

        uw.pprint('Relative error in velocity in the L2 norm: ', v_err_l2)
        print('Relative error in pressure in the L2 norm: ', p_err_l2)

# %% [markdown]
"""
## Save Results
"""

# %%
# Write L2 norms to h5 file
if uw.mpi.size == 1 and os.path.isfile(output_dir + 'error_norm.h5'):
    os.remove(output_dir + 'error_norm.h5')
    print('Old file removed')

if uw.mpi.rank == 0:
    uw.pprint('Creating new h5 file')
    with h5py.File(output_dir + 'error_norm.h5', 'w') as f:
        f.create_dataset("m", data=m)
        f.create_dataset("cellsize", data=cellsize)
        f.create_dataset("res", data=res)
        f.create_dataset("v_l2_norm", data=v_err_l2)
        f.create_dataset("p_l2_norm", data=p_err_l2)

# %%
# Save checkpoint
mesh.petsc_save_checkpoint(
    index=0,
    meshVars=[v_uw, p_uw, v_ana, p_ana, v_err, p_err, rho_ana],
    outputPath=os.path.relpath(output_dir) + '/output'
)

# %%
print(f"Spherical Thieulot benchmark complete: m={m}, res={res}, v_err_l2={v_err_l2:.6e}")
