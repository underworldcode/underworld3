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
# Stokes Annulus Benchmark (Thieulot)

**PHYSICS:** fluid_mechanics
**DIFFICULTY:** advanced

## Description

Benchmark for isoviscous incompressible Stokes flow in an annulus geometry
using the manufactured solution from Thieulot et al. (2023). Compares numerical
results against analytical solutions for velocity and pressure fields.

## Key Concepts

- **Manufactured solution**: Analytical solution derived for benchmarking
- **Annulus geometry**: Flow between two concentric circles
- **Free-slip boundaries**: Velocity tangent to boundaries via penalty method
- **Convergence analysis**: L2 norm error calculation
- **Parameter study**: Variable convection cell count (k parameter)

## Mathematical Formulation

Velocity field:
$$v_\\theta(r, \\theta) = f(r) \\cos(k\\theta)$$
$$v_r(r, \\theta) = g(r)k \\sin(k\\theta)$$

Pressure:
$$p(r, \\theta) = kh(r) \\sin(k\\theta) + \\rho_0 g_r(R_2 - r)$$

## Parameters

- `uw_k`: Number of convection cells (wave number)
- `uw_res`: Mesh resolution (cells per unit length)
- `uw_vdegree`: Velocity polynomial degree
- `uw_pdegree`: Pressure polynomial degree
- `uw_vel_penalty`: Penalty parameter for boundary conditions
- `uw_stokes_tol`: Stokes solver tolerance

## References

- [ASPECT Benchmark](https://aspect-documentation.readthedocs.io/en/latest/user/benchmarks/benchmarks/annulus/doc/annulus.html)
- [Benchmark Paper](https://egusphere.copernicus.org/preprints/2023/egusphere-2023-2765/)

*Author: [Thyagarajulu Gollapalli](https://github.com/gthyagi)*
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
import sympy
import os
from enum import Enum
import h5py

if uw.mpi.size == 1:
    import matplotlib.pyplot as plt
    import cmcrameri.cm as cmc
    import pyvista as pv
    import underworld3.visualisation as vis
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    from scipy import integrate
    from sympy import lambdify

os.environ["SYMPY_USE_CACHE"] = "no"
os.environ["UW_TIMING_ENABLE"] = "1"

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_Stokes_Annulus_Benchmark_Thieulot.py -uw_k 4
python Ex_Stokes_Annulus_Benchmark_Thieulot.py -uw_res 32
python Ex_Stokes_Annulus_Benchmark_Thieulot.py -uw_stokes_tol 1e-12
```
"""

# %%
params = uw.Params(
    uw_k = 0,                    # Number of convection cells (wave number)
    uw_res = 16,                 # Mesh resolution
    uw_vdegree = 2,              # Velocity polynomial degree
    uw_pdegree = 1,              # Pressure polynomial degree
    uw_pcont = 1,                # Pressure continuity (1=True, 0=False)
    uw_vel_penalty = 2.5e8,      # Penalty for boundary conditions
    uw_stokes_tol = 1e-10,       # Solver tolerance
    uw_r_inner = 1.0,            # Inner radius
    uw_r_outer = 2.0,            # Outer radius
)

# Extract parameters
r_i = params.uw_r_inner
r_o = params.uw_r_outer

k = int(params.uw_k)
res = int(params.uw_res)
cellsize = 1 / res

vdegree = int(params.uw_vdegree)
pdegree = int(params.uw_pdegree)
pcont = bool(params.uw_pcont)
pcont_str = str(pcont).lower()

vel_penalty = params.uw_vel_penalty
stokes_tol = params.uw_stokes_tol

vel_penalty_str = str("{:.1e}".format(vel_penalty))
stokes_tol_str = str("{:.1e}".format(stokes_tol))

# %% [markdown]
"""
## Options
"""

# %%
# Compute analytical solutions
analytical = True
timing = True
visualize = True

# %% [markdown]
"""
## Output Directory
"""

# %%
output_dir = os.path.join(
    os.path.join("./output/Annulus_Benchmark_Thieulot/"),
    f'model_k_{k}_res_{res}_vdeg_{vdegree}_pdeg_{pdegree}'
    f'_pcont_{pcont_str}_vel_penalty_{vel_penalty_str}_stokes_tol_{stokes_tol_str}/'
)

if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)

# %% [markdown]
"""
## Analytical Solution in SymPy

The analytical solution uses:
- f(r) = Ar + B/r
- g(r) = (A/2)r + (B/r)ln(r) + C/r

With A and B chosen so that v_r = 0 at both boundaries.
"""

# %%
r = sympy.symbols('r')
theta = sympy.Symbol('theta', real=True)

C = -1
A = -C * (2 * (np.log(r_i) - np.log(r_o)) / ((r_o**2) * np.log(r_i) - (r_i**2) * np.log(r_o)))
B = -C * ((r_o**2 - r_i**2) / ((r_o**2) * np.log(r_i) - (r_i**2) * np.log(r_o)))
rho_0 = 0

f = sympy.Function('f')(r)
f = A * r + B / r

g = sympy.Function('g')(r)
g = ((A / 2) * r) + ((B / r) * sympy.ln(r)) + (C / r)

h = sympy.Function('h')(r)
h = (2 * g - f) / r

m = sympy.Function('m')(r)
m = g.diff(r, r) - (g.diff(r) / r) - (g / r**2) * (k**2 - 1) + (f / r**2) + (f.diff(r) / r)

v_r = g * k * sympy.sin(k * theta)
v_theta = f * sympy.cos(k * theta)
p = k * h * sympy.sin(k * theta) + rho_0 * (r_o - r)
rho = m * k * sympy.sin(k * theta) + rho_0
v_x = v_r * sympy.cos(theta) - v_theta * sympy.sin(theta)
v_y = v_r * sympy.sin(theta) + v_theta * sympy.cos(theta)

# %% [markdown]
"""
## Create Mesh
"""

# %%
if timing:
    uw.timing.reset()
    uw.timing.start()

mesh = uw.meshing.Annulus(
    radiusOuter=r_o,
    radiusInner=r_i,
    cellSize=cellsize,
    qdegree=max(pdegree, vdegree),
    degree=1,
    filename=f'{output_dir}mesh.msh'
)

if timing:
    uw.timing.stop()
    uw.timing.print_table(
        group_by='line_routine',
        output_file=f"{output_dir}mesh_create_time.txt",
        display_fraction=1.00
    )

if uw.mpi.size == 1 and visualize:
    vis.plot_mesh(mesh, save_png=True, dir_fname=output_dir + 'mesh.png', title='', clip_angle=0., cpos='xy')

# Print mesh size
uw.pprint('-------------------------------------------------------------------------------')
mesh.dm.view()
uw.pprint('-------------------------------------------------------------------------------')

# %% [markdown]
"""
## Variables
"""

# %%
v_uw = uw.discretisation.MeshVariable(r'{V_u}', mesh, mesh.dim, degree=vdegree)
p_uw = uw.discretisation.MeshVariable(r'{P_u}', mesh, 1, degree=pdegree, continuous=pcont)

if analytical:
    v_ana = uw.discretisation.MeshVariable(r'{V_a}', mesh, mesh.dim, degree=vdegree)
    p_ana = uw.discretisation.MeshVariable(r'{P_a}', mesh, 1, degree=pdegree, continuous=pcont)
    rho_ana = uw.discretisation.MeshVariable(r'{RHO_a}', mesh, 1, degree=pdegree, continuous=True)

    v_err = uw.discretisation.MeshVariable(r'{V_e}', mesh, mesh.dim, degree=vdegree)
    p_err = uw.discretisation.MeshVariable(r'{P_e}', mesh, 1, degree=pdegree, continuous=pcont)

# %% [markdown]
"""
## Coordinate System
"""

# %%
unit_rvec = mesh.CoordinateSystem.unit_e_0
r_uw, th_uw = mesh.CoordinateSystem.xR

# Null space in velocity (constant v_theta) expressed in x,y coordinates
v_theta_fn_xy = r_uw * mesh.CoordinateSystem.rRotN.T * sympy.Matrix((0, 1))

# %% [markdown]
"""
## Initialize Analytical Solution
"""

# %%
if analytical:
    with mesh.access(v_ana, p_ana, rho_ana):
        if k == 0:
            v_ana_expr = mesh.CoordinateSystem.rRotN.T * sympy.Matrix([0, v_theta.subs({r: r_uw, theta: th_uw})])
            p_ana.data[:, 0] = 0
            rho_ana.data[:, 0] = 0
        else:
            v_ana_expr = mesh.CoordinateSystem.rRotN.T * sympy.Matrix([
                v_r.subs({r: r_uw, theta: th_uw}),
                v_theta.subs({r: r_uw, theta: th_uw})
            ])
            p_ana.data[:, 0] = uw.function.evalf(p.subs({r: r_uw, theta: th_uw}), p_ana.coords)
            rho_ana.data[:, 0] = uw.function.evalf(rho.subs({r: r_uw, theta: th_uw}), rho_ana.coords)

        v_ana.data[:, 0] = uw.function.evalf(v_ana_expr[0], v_ana.coords)
        v_ana.data[:, 1] = uw.function.evalf(v_ana_expr[1], v_ana.coords)

# %% [markdown]
"""
## Visualize Analytical Solution
"""

# %%
clim_vel = [0., 2.5]
vmag = 1e-1
vfreq = 40

if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_vector(
        mesh, v_ana, vector_name='v_ana', cmap=cmc.lapaz.resampled(11),
        clim=clim_vel, vmag=vmag, vfreq=vfreq, save_png=True,
        dir_fname=output_dir + 'vel_ana.png', clip_angle=0., show_arrows=False, cpos='xy'
    )

    vis.save_colorbar(
        colormap=cmc.lapaz.resampled(11), cb_bounds=None, vmin=clim_vel[0], vmax=clim_vel[1],
        figsize_cb=(5, 5), primary_fs=18, cb_orient='horizontal', cb_axis_label='Velocity',
        cb_label_xpos=0.5, cb_label_ypos=-2.05, fformat='pdf', output_path=output_dir, fname='v_ana'
    )

# %%
clim_p = [-8.5, 8.5]

if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_scalar(
        mesh, p_ana.sym, 'p_ana', cmap=cmc.vik.resampled(41), clim=clim_p,
        save_png=True, clip_angle=0., dir_fname=output_dir + 'p_ana.png', cpos='xy'
    )

    vis.save_colorbar(
        colormap=cmc.vik.resampled(41), cb_bounds=None, vmin=clim_p[0], vmax=clim_p[1],
        figsize_cb=(5, 5), primary_fs=18, cb_orient='horizontal', cb_axis_label='Pressure',
        cb_label_xpos=0.5, cb_label_ypos=-2.0, fformat='pdf', output_path=output_dir, fname='p_ana'
    )

# %%
clim_rho = [-67.5, 67.5]

if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_scalar(
        mesh, -rho_ana.sym, 'Rho', cmap=cmc.roma.resampled(31), clim=clim_rho,
        save_png=True, dir_fname=output_dir + 'rho_ana.png', clip_angle=0., cpos='xy'
    )

    vis.save_colorbar(
        colormap=cmc.roma.resampled(31), cb_bounds=None, vmin=clim_rho[0], vmax=clim_rho[1],
        figsize_cb=(5, 5), primary_fs=18, cb_orient='horizontal', cb_axis_label='Rho',
        cb_label_xpos=0.5, cb_label_ypos=-2.0, fformat='pdf', output_path=output_dir, fname='rho_ana'
    )

# %% [markdown]
"""
## Stokes Solver
"""

# %%
stokes = Stokes(mesh, velocityField=v_uw, pressureField=p_uw, degree=max(pdegree, vdegree))
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0
stokes.saddle_preconditioner = 1.0

# Gravity (radial inward)
gravity_fn = -1.0 * unit_rvec

# Density
rho_uw = rho.subs({r: r_uw, theta: th_uw})

# Body force
stokes.bodyforce = rho_uw * gravity_fn

# %% [markdown]
"""
## Boundary Conditions

Using penalty method to enforce tangential velocity at boundaries.
"""

# %%
v_diff = v_uw.sym - v_ana.sym
stokes.add_natural_bc(vel_penalty * v_diff, mesh.boundaries.Upper.name)
stokes.add_natural_bc(vel_penalty * v_diff, mesh.boundaries.Lower.name)

# For k=0, pressure is indeterminate - add Dirichlet BC
if k == 0:
    stokes.add_condition(
        p_uw.field_id, "dirichlet",
        sympy.Matrix([0]), mesh.boundaries.Lower.name,
        components=(0)
    )
    stokes.add_condition(
        p_uw.field_id, "dirichlet",
        sympy.Matrix([0]), mesh.boundaries.Upper.name,
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

# Check convergence reason
print(f"SNES converged reason: {stokes.snes.getConvergedReason()}")
print(f"KSP converged reason: {stokes.snes.ksp.getConvergedReason()}")

if timing:
    uw.timing.stop()
    uw.timing.print_table(
        group_by='line_routine',
        output_file=f"{output_dir}/stokes_solve_time.txt",
        display_fraction=1.00
    )

# %% [markdown]
"""
## Compute Errors
"""

# %%
if analytical:
    with mesh.access(v_uw, p_uw, v_err, p_err):
        v_err.data[:, 0] = v_uw.data[:, 0] - v_ana.data[:, 0]
        v_err.data[:, 1] = v_uw.data[:, 1] - v_ana.data[:, 1]
        p_err.data[:, 0] = p_uw.data[:, 0] - p_ana.data[:, 0]

# %% [markdown]
"""
## Visualize Numerical Solution
"""

# %%
if uw.mpi.size == 1 and visualize:
    vis.plot_vector(
        mesh, v_uw, vector_name='v_uw', cmap=cmc.lapaz.resampled(11),
        clim=clim_vel, vmag=vmag, vfreq=vfreq, save_png=True,
        dir_fname=output_dir + 'vel_uw.png', clip_angle=0., cpos='xy', show_arrows=False
    )

    vis.save_colorbar(
        colormap=cmc.lapaz.resampled(11), cb_bounds=None, vmin=clim_vel[0], vmax=clim_vel[1],
        figsize_cb=(5, 5), primary_fs=18, cb_orient='horizontal', cb_axis_label='Velocity',
        cb_label_xpos=0.5, cb_label_ypos=-2.05, fformat='pdf', output_path=output_dir, fname='v_uw'
    )

# %%
if uw.mpi.size == 1 and visualize:
    vis.plot_scalar(
        mesh, p_uw.sym, 'p_uw', cmap=cmc.vik.resampled(41), clim=clim_p,
        save_png=True, dir_fname=output_dir + 'p_uw.png', clip_angle=0., cpos='xy'
    )

    vis.save_colorbar(
        colormap=cmc.vik.resampled(41), cb_bounds=None, vmin=clim_p[0], vmax=clim_p[1],
        figsize_cb=(5, 5), primary_fs=18, cb_orient='horizontal', cb_axis_label='Pressure',
        cb_label_xpos=0.5, cb_label_ypos=-2.0, fformat='pdf', output_path=output_dir, fname='p_uw'
    )

# %% [markdown]
"""
## Visualize Errors
"""

# %%
clim_v_err = [0., 2.5e-4]

if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_vector(
        mesh, v_err, vector_name='v_err(relative)', cmap=cmc.lapaz.resampled(11),
        clim=clim_v_err, vmag=10, vfreq=20, save_png=True,
        dir_fname=output_dir + 'vel_r_err.png', clip_angle=0., cpos='xy', show_arrows=False
    )

    vis.save_colorbar(
        colormap=cmc.lapaz.resampled(11), cb_bounds=None, vmin=clim_v_err[0], vmax=clim_v_err[1],
        figsize_cb=(5, 5), primary_fs=18, cb_orient='horizontal', cb_axis_label='Velocity Error (relative)',
        cb_label_xpos=0.5, cb_label_ypos=-2.05, fformat='pdf', output_path=output_dir, fname='v_err_rel'
    )

# %%
clim_v_err_pct = [0, 1]

if uw.mpi.size == 1 and visualize and analytical:
    vmag_expr = (sympy.sqrt(v_err.sym.dot(v_err.sym)) / sympy.sqrt(v_ana.sym.dot(v_ana.sym))) * 100
    vis.plot_scalar(
        mesh, vmag_expr, 'vmag_err(%)', cmap=cmc.oslo_r.resampled(21), clim=clim_v_err_pct,
        save_png=True, dir_fname=output_dir + 'vel_p_err.png', clip_angle=0., cpos='xy'
    )

    vis.save_colorbar(
        colormap=cmc.oslo_r.resampled(21), cb_bounds=None, vmin=clim_v_err_pct[0], vmax=clim_v_err_pct[1],
        figsize_cb=(5, 5), primary_fs=18, cb_orient='horizontal', cb_axis_label='Velocity Error (%)',
        cb_label_xpos=0.5, cb_label_ypos=-2.05, fformat='pdf', output_path=output_dir, fname='v_err_perc'
    )

# %%
clim_p_err = [-0.006, 0.006]

if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_scalar(
        mesh, p_err.sym, 'p_err(relative)', cmap=cmc.vik.resampled(41), clim=clim_p_err,
        save_png=True, dir_fname=output_dir + 'p_r_err.png', clip_angle=0., cpos='xy'
    )

    vis.save_colorbar(
        colormap=cmc.vik.resampled(41), cb_bounds=None, vmin=clim_p_err[0], vmax=clim_p_err[1],
        figsize_cb=(5, 5), primary_fs=18, cb_orient='horizontal', cb_axis_label='Pressure Error (relative)',
        cb_label_xpos=0.5, cb_label_ypos=-2.0, fformat='pdf', output_path=output_dir, fname='p_err_rel'
    )

# %%
if k == 0:
    pass
elif uw.mpi.size == 1 and analytical and visualize:
    clim_p_err_pct = [-100, 100]
    vis.plot_scalar(
        mesh, (p_err.sym[0] / p_ana.sym[0]) * 100, 'p_err(%)', cmap=cmc.vik.resampled(41),
        clim=clim_p_err_pct, save_png=True, dir_fname=output_dir + 'p_p_err.png', clip_angle=0., cpos='xy'
    )

    vis.save_colorbar(
        colormap=cmc.vik.resampled(41), cb_bounds=None, vmin=clim_p_err_pct[0], vmax=clim_p_err_pct[1],
        figsize_cb=(5, 5), primary_fs=18, cb_orient='horizontal', cb_axis_label='Pressure Error (%)',
        cb_label_xpos=0.5, cb_label_ypos=-2.0, fformat='pdf', output_path=output_dir, fname='p_err_perc'
    )

# %% [markdown]
"""
## L2 Norm Error Analysis
"""

# %%
if analytical:
    with mesh.access(v_err, p_err, p_ana, v_ana):
        v_err_I = uw.maths.Integral(mesh, v_err.sym.dot(v_err.sym))
        v_ana_I = uw.maths.Integral(mesh, v_ana.sym.dot(v_ana.sym))
        v_err_l2 = np.sqrt(v_err_I.evaluate()) / np.sqrt(v_ana_I.evaluate())
        uw.pprint('Relative error in velocity in the L2 norm: ', v_err_l2)

        if k == 0:
            uw.pprint('For k=0, analytical pressure integral is zero - L2 norm not meaningful.')
            p_err_l2 = np.inf
        else:
            p_err_I = uw.maths.Integral(mesh, p_err.sym.dot(p_err.sym))
            p_ana_I = uw.maths.Integral(mesh, p_ana.sym.dot(p_ana.sym))
            p_err_l2 = np.sqrt(p_err_I.evaluate()) / np.sqrt(p_ana_I.evaluate())
            uw.pprint('Relative error in pressure in the L2 norm: ', p_err_l2)

# %% [markdown]
"""
## Save Results
"""

# %%
# Write L2 norms to HDF5
if uw.mpi.size == 1 and os.path.isfile(output_dir + 'error_norm.h5'):
    os.remove(output_dir + 'error_norm.h5')
    print('Old file removed')

if uw.mpi.rank == 0:
    uw.pprint('Creating new h5 file')
    with h5py.File(output_dir + 'error_norm.h5', 'w') as f_h5:
        f_h5.create_dataset("k", data=k)
        f_h5.create_dataset("cellsize", data=cellsize)
        f_h5.create_dataset("res", data=res)
        f_h5.create_dataset("v_l2_norm", data=v_err_l2)
        if k == 0:
            f_h5.create_dataset("p_l2_norm", data=np.inf)
        else:
            f_h5.create_dataset("p_l2_norm", data=p_err_l2)

# %%
# Save checkpoint
mesh.petsc_save_checkpoint(
    index=0,
    meshVars=[v_uw, p_uw, v_ana, p_ana, rho_ana, v_err, p_err],
    outputPath=os.path.relpath(output_dir) + '/output'
)

# %% [markdown]
"""
## Boundary Comparison (Optional)

The following sections compare analytical and numerical solutions on boundaries.
"""

# %%
# Get boundary indices
lower_indx = uw.discretisation.petsc_discretisation.petsc_dm_find_labeled_points_local(mesh.dm, 'Lower')
upper_indx = uw.discretisation.petsc_discretisation.petsc_dm_find_labeled_points_local(mesh.dm, 'Upper')

# Get theta from x, y
lower_theta = uw.function.evalf(th_uw, mesh.X.coords[lower_indx])
lower_theta[lower_theta < 0] += 2 * np.pi
upper_theta = uw.function.evalf(th_uw, mesh.X.coords[upper_indx])
upper_theta[upper_theta < 0] += 2 * np.pi

# %%
# Lower and upper boundary velocities
if analytical:
    with mesh.access(v_uw, p_uw):
        # Pressure arrays
        p_ana_lower = np.zeros((len(lower_indx), 1))
        p_ana_upper = np.zeros((len(upper_indx), 1))
        p_uw_lower = np.zeros((len(lower_indx), 1))
        p_uw_upper = np.zeros((len(upper_indx), 1))

        if k == 0:
            p_ana_lower[:, 0] = 0
            p_ana_upper[:, 0] = 0
        else:
            p_ana_lower[:, 0] = uw.function.evalf(p.subs({r: r_uw, theta: th_uw}), mesh.X.coords[lower_indx])
            p_ana_upper[:, 0] = uw.function.evalf(p.subs({r: r_uw, theta: th_uw}), mesh.X.coords[upper_indx])
        p_uw_lower[:, 0] = uw.function.evalf(p_uw.sym, mesh.X.coords[lower_indx])
        p_uw_upper[:, 0] = uw.function.evalf(p_uw.sym, mesh.X.coords[upper_indx])

        # Velocity arrays
        v_ana_lower = np.zeros_like(mesh.X.coords[lower_indx])
        v_ana_upper = np.zeros_like(mesh.X.coords[upper_indx])
        v_uw_lower = np.zeros_like(mesh.X.coords[lower_indx])
        v_uw_upper = np.zeros_like(mesh.X.coords[upper_indx])

        v_ana_lower[:, 0] = uw.function.evalf(v_ana_expr[0], mesh.X.coords[lower_indx])
        v_ana_lower[:, 1] = uw.function.evalf(v_ana_expr[1], mesh.X.coords[lower_indx])
        v_ana_upper[:, 0] = uw.function.evalf(v_ana_expr[0], mesh.X.coords[upper_indx])
        v_ana_upper[:, 1] = uw.function.evalf(v_ana_expr[1], mesh.X.coords[upper_indx])
        v_uw_lower = uw.function.evalf(v_uw.sym, mesh.X.coords[lower_indx])
        v_uw_upper = uw.function.evalf(v_uw.sym, mesh.X.coords[upper_indx])

# Sort arrays for plotting
sort_lower = lower_theta.argsort()
sort_upper = upper_theta.argsort()


# %%
def plot_stats(_data_list='', _label_list='', _line_style='', _xlabel='', _ylabel='',
               _xlim='', _ylim='', _mod_xticks=False, _save_pdf='', _output_path='', _fname=''):
    """Plot statistics comparison."""
    fig, ax = plt.subplots()
    for i, data in enumerate(_data_list):
        ax.plot(data[:, 0], data[:, 1], label=_label_list[i], linestyle=_line_style[i])

    ax.set_xlabel(_xlabel)
    ax.set_ylabel(_ylabel)
    ax.grid(linestyle='--')
    ax.legend(loc=(1.01, 0.60), fontsize=14)

    if len(_xlim) != 0:
        ax.set_xlim(_xlim[0], _xlim[1])
        if _mod_xticks:
            ax.set_xticks(np.arange(_xlim[0], _xlim[1] + 0.01, np.pi / 2))
            labels = ['$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
            ax.set_xticklabels(labels)

    if len(_ylim) != 0:
        ax.set_ylim(_ylim[0], _ylim[1])

    if _save_pdf:
        plt.savefig(_output_path + _fname + '.pdf', format='pdf', bbox_inches='tight')


# %%
# Plot pressure on boundaries
if uw.mpi.size == 1 and analytical:
    data_list = [
        np.hstack((np.c_[lower_theta[sort_lower]], p_ana_lower[sort_lower])),
        np.hstack((np.c_[upper_theta[sort_upper]], p_ana_upper[sort_upper])),
        np.hstack((np.c_[lower_theta[sort_lower]], p_uw_lower[sort_lower])),
        np.hstack((np.c_[upper_theta[sort_upper]], p_uw_upper[sort_upper]))
    ]
    label_list = [
        'k=' + str(k) + ' (analy.), ' + r'$r=R_{1}$',
        'k=' + str(k) + ' (analy.), ' + r'$r=R_{2}$',
        'k=' + str(k) + ' (UW), ' + r'$r=R_{1}$',
        'k=' + str(k) + ' (UW), ' + r'$r=R_{2}$'
    ]
    linestyle_list = ['-', '-', '--', '--']

    plot_stats(
        _data_list=data_list, _label_list=label_list, _line_style=linestyle_list,
        _xlabel=r'$\theta$', _ylabel='Pressure', _xlim=[0, 2 * np.pi], _ylim=[-2.5, 2.5], _mod_xticks=True
    )


# %%
def get_magnitude(_array):
    """Compute velocity magnitude."""
    sqrd_sum = np.zeros((_array.shape[0], 1))
    for i in range(_array.shape[1]):
        sqrd_sum += _array[:, i:i + 1]**2
    return np.sqrt(sqrd_sum)


# %%
# Compute velocity magnitude
if analytical:
    v_ana_lower_mag = get_magnitude(v_ana_lower)
    v_ana_upper_mag = get_magnitude(v_ana_upper)
    v_uw_lower_mag = get_magnitude(v_uw_lower)
    v_uw_upper_mag = get_magnitude(v_uw_upper)

# %%
# Plot velocity magnitude on boundaries
if uw.mpi.size == 1 and analytical:
    data_list = [
        np.hstack((np.c_[lower_theta[sort_lower]], v_ana_lower_mag[sort_lower])),
        np.hstack((np.c_[upper_theta[sort_upper]], v_ana_upper_mag[sort_upper])),
        np.hstack((np.c_[lower_theta[sort_lower]], v_uw_lower_mag[sort_lower])),
        np.hstack((np.c_[upper_theta[sort_upper]], v_uw_upper_mag[sort_upper]))
    ]
    label_list = [
        'k=' + str(k) + ' (analy.), ' + r'$r=R_{1}$',
        'k=' + str(k) + ' (analy.), ' + r'$r=R_{2}$',
        'k=' + str(k) + ' (UW), ' + r'$r=R_{1}$',
        'k=' + str(k) + ' (UW), ' + r'$r=R_{2}$'
    ]
    linestyle_list = ['-', '-', '--', '--']

    plot_stats(
        _data_list=data_list, _label_list=label_list, _line_style=linestyle_list,
        _xlabel=r'$\theta$', _ylabel='Velocity Magnitude', _xlim=[0, 2 * np.pi], _ylim=[0, 2.5], _mod_xticks=True
    )

# %%
# Velocity in (r, theta) coordinates
v_uw_r_th = mesh.CoordinateSystem.rRotN * v_uw.sym.T

# Radial and theta components integrated over mesh (should be zero)
if analytical:
    v_r_rms_I = uw.maths.Integral(mesh, v_uw_r_th[0])
    print(f"v_r integral: {(1 / (2 * np.pi)) * v_r_rms_I.evaluate()}")

    v_th_rms_I = uw.maths.Integral(mesh, v_uw_r_th[1])
    print(f"v_theta integral: {(1 / (2 * np.pi)) * v_th_rms_I.evaluate()}")

# %%
# Arrays for radial profiles
theta_0_2pi = np.linspace(0, 2 * np.pi, 1000, endpoint=True)
r_i_o = np.hstack((
    np.linspace(r_i, np.pi / 2, 7, endpoint=True),
    np.linspace(np.pi / 1.92, r_o - 1e-3, 4, endpoint=True)
))


def get_vel_avg_r(_theta_arr, _r_arr, _vel_comp):
    """Return average velocity over theta at each radius."""
    vel_avg_arr = np.zeros_like(_r_arr)
    for i, r_val in enumerate(_r_arr):
        x_arr = r_val * np.cos(_theta_arr)
        y_arr = r_val * np.sin(_theta_arr)
        xy_arr = np.stack((x_arr, y_arr), axis=-1)
        vel_xy = uw.function.evaluate(_vel_comp, xy_arr)
        vel_avg_arr[i] = integrate.simpson(vel_xy, x=_theta_arr) / (2 * np.pi)
    return vel_avg_arr


# %%
# Velocity radial component average
if uw.mpi.size == 1 and analytical:
    vr_avg = get_vel_avg_r(theta_0_2pi, r_i_o, v_uw_r_th[0])

    data_list = [
        np.hstack((np.c_[r_i_o], np.c_[np.zeros_like(r_i_o)])),
        np.hstack((np.c_[r_i_o], np.c_[vr_avg]))
    ]
    label_list = ['k=' + str(k) + ' (analy.)', 'k=' + str(k) + ' (UW)']
    linestyle_list = ['-', '--']

    plot_stats(
        _data_list=data_list, _label_list=label_list, _line_style=linestyle_list,
        _xlabel='r', _ylabel=r'$<v_{r}>$', _xlim=[r_i, r_o], _ylim=[vr_avg.min(), vr_avg.max()]
    )

# %%
# Velocity theta component average
if uw.mpi.size == 1 and analytical:
    vth_avg = get_vel_avg_r(theta_0_2pi, r_i_o, v_uw_r_th[1])

    data_list = [
        np.hstack((np.c_[r_i_o], np.c_[np.zeros_like(r_i_o)])),
        np.hstack((np.c_[r_i_o], np.c_[vth_avg]))
    ]
    label_list = ['k=' + str(k) + ' (analy.)', 'k=' + str(k) + ' (UW)']
    linestyle_list = ['-', '--']

    plot_stats(
        _data_list=data_list, _label_list=label_list, _line_style=linestyle_list,
        _xlabel='r', _ylabel=r'$<v_{\theta}>$', _xlim=[r_i, r_o], _ylim=[vth_avg.min(), vth_avg.max()]
    )


# %%
def get_vel_rms_r(_theta_arr, _r_arr, _vel_comp):
    """Return RMS velocity over theta at each radius."""
    vel_rms_arr = np.zeros_like(_r_arr)
    for i, r_val in enumerate(_r_arr):
        x_arr = r_val * np.cos(_theta_arr)
        y_arr = r_val * np.sin(_theta_arr)
        xy_arr = np.stack((x_arr, y_arr), axis=-1)
        vel_xy = uw.function.evaluate(_vel_comp, xy_arr)
        vel_rms_arr[i] = np.sqrt(integrate.simpson(vel_xy, x=_theta_arr) / (2 * np.pi))
    return vel_rms_arr


# %%
# Lambdified functions for analytical comparison
g_lbd = lambdify([r], g)
f_lbd = lambdify([r], f)

# %%
# Velocity radial component RMS
if uw.mpi.size == 1 and analytical:
    vr_rms = get_vel_rms_r(theta_0_2pi, r_i_o, v_uw_r_th[0]**2)

    data_list = [
        np.hstack((np.c_[r_i_o], np.c_[k * np.abs(g_lbd(r_i_o)) / np.sqrt(2)])),
        np.hstack((np.c_[r_i_o], np.c_[vr_rms]))
    ]
    label_list = ['k=' + str(k) + ' (analy.)', 'k=' + str(k) + ' (UW)']
    linestyle_list = ['-', '--']

    plot_stats(
        _data_list=data_list, _label_list=label_list, _line_style=linestyle_list,
        _xlabel='r', _ylabel=r'$<v_{r}>_{rms}$', _xlim=[r_i, r_o], _ylim=[0, 2.1]
    )

# %%
# Velocity theta component RMS
if uw.mpi.size == 1 and analytical:
    vth_rms = get_vel_rms_r(theta_0_2pi, r_i_o, v_uw_r_th[1]**2)

    data_list = [
        np.hstack((np.c_[r_i_o], np.c_[np.abs(f_lbd(r_i_o)) / np.sqrt(2)])),
        np.hstack((np.c_[r_i_o], np.c_[vth_rms]))
    ]
    label_list = ['k=' + str(k) + ' (analy.)', 'k=' + str(k) + ' (UW)']
    linestyle_list = ['-', '--']

    plot_stats(
        _data_list=data_list, _label_list=label_list, _line_style=linestyle_list,
        _xlabel='r', _ylabel=r'$<v_{\theta}>_{rms}$', _xlim=[r_i, r_o], _ylim=[0, 2]
    )

# %%
print(f"Annulus Benchmark (Thieulot) complete: k={k}, res={res}")
