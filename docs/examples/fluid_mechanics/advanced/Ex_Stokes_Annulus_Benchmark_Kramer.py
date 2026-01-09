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
# Stokes Annulus Benchmark (Kramer)

**PHYSICS:** fluid_mechanics
**DIFFICULTY:** advanced

## Description

Benchmark for isoviscous incompressible Stokes flow in an annulus geometry
using the analytical solutions from Kramer et al. (2021). Supports multiple
test cases with different boundary conditions and density perturbations.

## Key Concepts

- **Analytical benchmark**: Comparison with ASSESS library solutions
- **Multiple test cases**: Free-slip/no-slip with delta/smooth density
- **Annulus with internal boundary**: Mesh refinement at density interface
- **Null space handling**: Removing rigid body rotation mode
- **Convergence analysis**: L2 norm error calculation

## Test Cases

- **Case 1**: Free-slip boundaries, delta function density
- **Case 2**: Free-slip boundaries, smooth density distribution
- **Case 3**: No-slip boundaries, delta function density
- **Case 4**: No-slip boundaries, smooth density distribution

## Parameters

- `uw_case`: Test case (case1, case2, case3, case4)
- `uw_n`: Wave number for density perturbation
- `uw_k`: Power exponent for smooth density
- `uw_res`: Mesh resolution
- `uw_vel_penalty`: Penalty for boundary conditions
- `uw_stokes_tol`: Solver tolerance

## References

- [Benchmark Paper](https://gmd.copernicus.org/articles/14/1899/2021/)

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
import assess
import h5py
from enum import Enum

if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis
    import matplotlib.pyplot as plt
    import cmcrameri.cm as cmc
    from scipy import integrate

os.environ["SYMPY_USE_CACHE"] = "no"
os.environ["UW_TIMING_ENABLE"] = "1"

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_Stokes_Annulus_Benchmark_Kramer.py -uw_case case2
python Ex_Stokes_Annulus_Benchmark_Kramer.py -uw_n 4 -uw_k 5
python Ex_Stokes_Annulus_Benchmark_Kramer.py -uw_res 32
```
"""

# %%
params = uw.Params(
    uw_case = "case2",           # Test case: case1, case2, case3, case4
    uw_n = 2,                    # Wave number for density perturbation
    uw_k = 3,                    # Power exponent for smooth density
    uw_res = 16,                 # Mesh resolution
    uw_vdegree = 2,              # Velocity polynomial degree
    uw_pdegree = 1,              # Pressure polynomial degree
    uw_pcont = 1,                # Pressure continuity (1=True, 0=False)
    uw_vel_penalty = 2.5e8,      # Penalty for boundary conditions
    uw_stokes_tol = 1e-10,       # Solver tolerance
    uw_r_outer = 2.22,           # Outer radius
    uw_r_internal = 2.0,         # Internal interface radius
    uw_r_inner = 1.22,           # Inner radius
)

# Extract parameters
r_o = params.uw_r_outer
r_int = params.uw_r_internal
r_i = params.uw_r_inner

res = int(params.uw_res)
cellsize = 1 / res
csize_int_fac = 1 / 2  # Internal layer cellsize factor

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
## Case Configuration
"""

# %%
# Which normals to use
ana_normal = False  # Unit radial vector
petsc_normal = True  # Gamma function

# Compute analytical solutions
analytical = True
timing = True
visualize = True

# %%
# Specify the case
case = params.uw_case
n = int(params.uw_n)  # Wave number
k = int(params.uw_k)  # Power exponent

# %%
# Boundary condition and density perturbation
freeslip, noslip, delta_fn, smooth = False, False, False, False

if case in ('case1'):
    freeslip, delta_fn = True, True
elif case in ('case2'):
    freeslip, smooth = True, True
elif case in ('case3'):
    noslip, delta_fn = True, True
elif case in ('case4'):
    noslip, smooth = True, True

uw.pprint(f"Case: {case}, freeslip={freeslip}, noslip={noslip}, delta_fn={delta_fn}, smooth={smooth}")

# %% [markdown]
"""
## Output Directory
"""

# %%
output_dir = os.path.join(
    os.path.join("./output/Annulus_Benchmark_Kramer/"),
    f'{case}_n_{n}_k_{k}_res_{res}_vdeg_{vdegree}_pdeg_{pdegree}'
    f'_pcont_{pcont_str}_vel_penalty_{vel_penalty_str}_stokes_tol_{stokes_tol_str}/'
)

if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)

# %% [markdown]
"""
## Analytical Solution (ASSESS Library)

The ASSESS library provides analytical solutions for cylindrical Stokes problems.
"""

# %%
if analytical:
    if freeslip:
        if delta_fn:
            soln_above = assess.CylindricalStokesSolutionDeltaFreeSlip(n, +1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
            soln_below = assess.CylindricalStokesSolutionDeltaFreeSlip(n, -1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
        elif smooth:
            # For smooth density, single solution exists but we create two for code consistency
            soln_above = assess.CylindricalStokesSolutionSmoothFreeSlip(n, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)
            soln_below = assess.CylindricalStokesSolutionSmoothFreeSlip(n, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)
    elif noslip:
        if delta_fn:
            soln_above = assess.CylindricalStokesSolutionDeltaZeroSlip(n, +1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
            soln_below = assess.CylindricalStokesSolutionDeltaZeroSlip(n, -1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
        elif smooth:
            soln_above = assess.CylindricalStokesSolutionSmoothZeroSlip(n, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)
            soln_below = assess.CylindricalStokesSolutionSmoothZeroSlip(n, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0)

# %% [markdown]
"""
## Create Mesh
"""

# %%
if timing:
    uw.timing.reset()
    uw.timing.start()

if delta_fn:
    mesh = uw.meshing.AnnulusInternalBoundary(
        radiusOuter=r_o,
        radiusInternal=r_int,
        radiusInner=r_i,
        cellSize_Inner=cellsize,
        cellSize_Internal=cellsize * csize_int_fac,
        cellSize_Outer=cellsize,
        filename=f'{output_dir}/mesh.msh'
    )
elif smooth:
    mesh = uw.meshing.Annulus(
        radiusOuter=r_o,
        radiusInner=r_i,
        cellSize=cellsize,
        qdegree=max(pdegree, vdegree),
        degree=1,
        filename=f'{output_dir}/mesh.msh',
        refinement=None
    )

if timing:
    uw.timing.stop()
    uw.timing.print_table(
        group_by='line_routine',
        output_file=f"{output_dir}/mesh_create_time.txt",
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
v_uw = uw.discretisation.MeshVariable('V_u', mesh, mesh.dim, degree=vdegree)
p_uw = uw.discretisation.MeshVariable('P_u', mesh, 1, degree=pdegree, continuous=pcont)

if analytical:
    v_ana = uw.discretisation.MeshVariable('V_a', mesh, mesh.dim, degree=vdegree)
    p_ana = uw.discretisation.MeshVariable('P_a', mesh, 1, degree=pdegree, continuous=pcont)
    rho_ana = uw.discretisation.MeshVariable('RHO_a', mesh, 1, degree=pdegree, continuous=True)

    v_err = uw.discretisation.MeshVariable('V_e', mesh, mesh.dim, degree=vdegree)
    p_err = uw.discretisation.MeshVariable('P_e', mesh, 1, degree=pdegree, continuous=pcont)

# %%
norm_v = uw.discretisation.MeshVariable("N", mesh, 2, degree=1, varsymbol=r"{\hat{n}}")
with mesh.access(norm_v):
    norm_v.data[:, 0] = uw.function.evaluate(mesh.CoordinateSystem.unit_e_0[0], norm_v.coords)
    norm_v.data[:, 1] = uw.function.evaluate(mesh.CoordinateSystem.unit_e_0[1], norm_v.coords)

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
    with mesh.access(v_ana, p_ana):

        def get_ana_soln(_var, _r_int, _soln_above, _soln_below):
            """Get analytical solution into mesh variables."""
            r = uw.function.evalf(r_uw, _var.coords)
            for i, coord in enumerate(_var.coords):
                if r[i] > _r_int:
                    _var.data[i] = _soln_above(coord)
                else:
                    _var.data[i] = _soln_below(coord)

        # Velocities
        get_ana_soln(v_ana, r_int, soln_above.velocity_cartesian, soln_below.velocity_cartesian)

        # Pressure
        get_ana_soln(p_ana, r_int, soln_above.pressure_cartesian, soln_below.pressure_cartesian)

# %% [markdown]
"""
## Visualize Analytical Solution
"""

# %%
# Case-dependent visualization parameters
if case in ('case1'):
    clim_vel, vmag, vfreq = [0., 0.05], 5e0, 75
elif case in ('case2'):
    clim_vel, vmag, vfreq = [0., 0.04], 6e0, 75
elif case in ('case3'):
    clim_vel, vmag, vfreq = [0., 0.01], 2.5e1, 75
elif case in ('case4'):
    clim_vel, vmag, vfreq = [0., 0.00925], 3e1, 75

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
if case in ('case1'):
    clim_p = [-0.65, 0.65]
elif case in ('case2'):
    clim_p = [-0.5, 0.5]
elif case in ('case3'):
    clim_p = [-0.65, 0.65]
elif case in ('case4'):
    clim_p = [-0.5, 0.5]

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

# %% [markdown]
"""
## Stokes Solver
"""

# %%
stokes = Stokes(mesh, velocityField=v_uw, pressureField=p_uw)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0
stokes.saddle_preconditioner = 1.0

# Define density function
if delta_fn:
    rho = sympy.cos(n * th_uw) * sympy.exp(-1e5 * ((r_uw - r_int) ** 2))
    stokes.add_natural_bc(-rho * unit_rvec, "Internal")
    stokes.bodyforce = sympy.Matrix([0., 0.])
elif smooth:
    rho = ((r_uw / r_o)**k) * sympy.cos(n * th_uw)
    gravity_fn = -1.0 * unit_rvec
    stokes.bodyforce = rho * gravity_fn

# %%
# Store density in mesh variable
if analytical:
    with mesh.access(rho_ana):
        rho_ana.data[:] = np.c_[uw.function.evaluate(rho, rho_ana.coords)]

# %%
clim_rho = [-1, 1]

if uw.mpi.size == 1 and visualize:
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
## Boundary Conditions
"""

# %%
if freeslip:
    if ana_normal:
        Gamma = mesh.CoordinateSystem.unit_e_0
    elif petsc_normal:
        Gamma = mesh.Gamma

    v_diff = v_uw.sym - v_ana.sym
    stokes.add_natural_bc(vel_penalty * v_diff, mesh.boundaries.Upper.name)
    stokes.add_natural_bc(vel_penalty * v_diff, mesh.boundaries.Lower.name)

elif noslip:
    v_diff = v_uw.sym - v_ana.sym
    stokes.add_natural_bc(vel_penalty * v_diff, mesh.boundaries.Upper.name)
    stokes.add_natural_bc(vel_penalty * v_diff, mesh.boundaries.Lower.name)

# %% [markdown]
"""
## Solver Configuration
"""

# %%
stokes.tolerance = stokes_tol
stokes.petsc_options["ksp_monitor"] = None

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

if timing:
    uw.timing.stop()
    uw.timing.print_table(
        group_by='line_routine',
        output_file=f"{output_dir}/stokes_solve_time.txt",
        display_fraction=1.00
    )

# %% [markdown]
"""
## Null Space Removal

For free-slip boundaries, remove the rigid body rotation mode.
"""

# %%
I0 = uw.maths.Integral(mesh, v_theta_fn_xy.dot(v_uw.sym))
norm = I0.evaluate()
I0.fn = v_theta_fn_xy.dot(v_theta_fn_xy)
vnorm = I0.evaluate()

with mesh.access(v_uw):
    dv = uw.function.evaluate(norm * v_theta_fn_xy, v_uw.coords) / vnorm
    v_uw.data[...] -= dv

# %% [markdown]
"""
## Compute Errors
"""

# %%
if analytical:
    with mesh.access(v_uw, p_uw, v_err, p_err):

        def get_error(_var_err, _var_uw, _r_int, _soln_above, _soln_below):
            """Get error in numerical solution."""
            r = uw.function.evalf(r_uw, _var_err.coords)
            for i, coord in enumerate(_var_err.coords):
                if r[i] > _r_int:
                    _var_err.data[i] = _var_uw.data[i] - _soln_above(coord)
                else:
                    _var_err.data[i] = _var_uw.data[i] - _soln_below(coord)

        # Error in velocities
        get_error(v_err, v_uw, r_int, soln_above.velocity_cartesian, soln_below.velocity_cartesian)

        # Error in pressure
        get_error(p_err, p_uw, r_int, soln_above.pressure_cartesian, soln_below.pressure_cartesian)

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
# Case-dependent error visualization parameters
if case in ('case1'):
    clim_v_err, vmag_err, vfreq_err = [0., 0.005], 1e2, 75
elif case in ('case2'):
    clim_v_err, vmag_err, vfreq_err = [0., 7e-4], 1e2, 75
elif case in ('case3'):
    clim_v_err, vmag_err, vfreq_err = [0., 1e-4], 2e2, 75
elif case in ('case4'):
    clim_v_err, vmag_err, vfreq_err = [0., 1e-5], 5e4, 75

if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_vector(
        mesh, v_err, vector_name='v_err(relative)', cmap=cmc.lapaz.resampled(11),
        clim=clim_v_err, vmag=vmag_err, vfreq=vfreq_err, save_png=True,
        dir_fname=output_dir + 'vel_r_err.png', clip_angle=0., cpos='xy', show_arrows=False
    )

    vis.save_colorbar(
        colormap=cmc.lapaz.resampled(11), cb_bounds=None, vmin=clim_v_err[0], vmax=clim_v_err[1],
        figsize_cb=(5, 5), primary_fs=18, cb_orient='horizontal', cb_axis_label='Velocity Error (relative)',
        cb_label_xpos=0.5, cb_label_ypos=-2.05, fformat='pdf', output_path=output_dir, fname='v_err_rel'
    )

# %%
if case in ('case1'):
    clim_v_err_pct = [0, 20]
elif case in ('case2'):
    clim_v_err_pct = [0, 20]
elif case in ('case3'):
    clim_v_err_pct = [0, 5]
elif case in ('case4'):
    clim_v_err_pct = [0, 1]

if uw.mpi.size == 1 and analytical and visualize:
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
if case in ('case1'):
    clim_p_err = [-0.065, 0.065]
elif case in ('case2'):
    clim_p_err = [-0.003, 0.003]
elif case in ('case3'):
    clim_p_err = [-0.0065, 0.0065]
elif case in ('case4'):
    clim_p_err = [-0.0045, 0.0045]

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
clim_p_err_pct = [-1e2, 1e2]

if uw.mpi.size == 1 and analytical and visualize:
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

        p_err_I = uw.maths.Integral(mesh, p_err.sym.dot(p_err.sym))
        p_ana_I = uw.maths.Integral(mesh, p_ana.sym.dot(p_ana.sym))
        p_err_l2 = np.sqrt(p_err_I.evaluate()) / np.sqrt(p_ana_I.evaluate())

        uw.pprint('Relative error in velocity in the L2 norm: ', v_err_l2)
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
    with h5py.File(output_dir + 'error_norm.h5', 'w') as f:
        f.create_dataset("k", data=k)
        f.create_dataset("cellsize", data=cellsize)
        f.create_dataset("res", data=res)
        f.create_dataset("v_l2_norm", data=v_err_l2)
        f.create_dataset("p_l2_norm", data=p_err_l2)

# %%
# Save checkpoint
mesh.petsc_save_checkpoint(
    index=0,
    meshVars=[v_uw, p_uw, v_ana, p_ana, v_err, p_err],
    outputPath=os.path.relpath(output_dir) + '/output'
)

# %% [markdown]
"""
## Boundary Comparison (Optional)
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
# Lower and upper boundary values
if analytical:
    with mesh.access(v_uw, p_uw):

        def get_ana_soln_2(_var, _coords, _r_int, _soln_above, _soln_below):
            """Get analytical solution at given coordinates."""
            r = uw.function.evalf(r_uw, _coords)
            for i, coord in enumerate(_coords):
                if r[i] > _r_int:
                    _var[i] = _soln_above(coord)
                else:
                    _var[i] = _soln_below(coord)

        # Pressure arrays
        p_ana_lower = np.zeros((len(lower_indx), 1))
        p_ana_upper = np.zeros((len(upper_indx), 1))
        p_uw_lower = np.zeros((len(lower_indx), 1))
        p_uw_upper = np.zeros((len(upper_indx), 1))

        # Pressure analytical and numerical
        get_ana_soln_2(p_ana_upper, mesh.X.coords[upper_indx], r_int, soln_above.pressure_cartesian, soln_below.pressure_cartesian)
        get_ana_soln_2(p_ana_lower, mesh.X.coords[lower_indx], r_int, soln_above.pressure_cartesian, soln_below.pressure_cartesian)
        p_uw_lower[:, 0] = uw.function.evalf(p_uw.sym, mesh.X.coords[lower_indx])
        p_uw_upper[:, 0] = uw.function.evalf(p_uw.sym, mesh.X.coords[upper_indx])

        # Velocity arrays
        v_ana_lower = np.zeros_like(mesh.X.coords[lower_indx])
        v_ana_upper = np.zeros_like(mesh.X.coords[upper_indx])
        v_uw_lower = np.zeros_like(mesh.X.coords[lower_indx])
        v_uw_upper = np.zeros_like(mesh.X.coords[upper_indx])

        # Velocity analytical and numerical
        get_ana_soln_2(v_ana_upper, mesh.X.coords[upper_indx], r_int, soln_above.velocity_cartesian, soln_below.velocity_cartesian)
        get_ana_soln_2(v_ana_lower, mesh.X.coords[lower_indx], r_int, soln_above.velocity_cartesian, soln_below.velocity_cartesian)
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
    if case in ('case1'):
        ylim = [-0.75, 0.75]
    elif case in ('case2'):
        ylim = [-0.65, 0.65]
    elif case in ('case3'):
        ylim = [-0.95, 0.95]
    elif case in ('case4'):
        ylim = [-0.65, 0.65]

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
        _xlabel=r'$\theta$', _ylabel='Pressure', _xlim=[0, 2 * np.pi], _ylim=ylim,
        _mod_xticks=True, _save_pdf=True, _output_path=output_dir, _fname='p_r_i_o'
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
    if case in ('case1'):
        ylim = [0, 5e-2]
    elif case in ('case2'):
        ylim = [0, 4e-2]
    elif case in ('case3'):
        ylim = [0, 3.7e-9]
    elif case in ('case4'):
        ylim = [-1e-10, 6e-9]

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
        _xlabel=r'$\theta$', _ylabel='Velocity Magnitude', _xlim=[0, 2 * np.pi], _ylim=ylim,
        _mod_xticks=True, _save_pdf=True, _output_path=output_dir, _fname='vel_r_i_o'
    )

# %%
# Velocity in (r, theta) coordinates
v_uw_r_th = mesh.CoordinateSystem.rRotN * v_uw.sym.T

# Radial and theta components integrated over mesh
if analytical:
    v_r_rms_I = uw.maths.Integral(mesh, v_uw_r_th[0])
    print(f"v_r integral: {(1 / (2 * np.pi)) * v_r_rms_I.evaluate()}")

    v_th_rms_I = uw.maths.Integral(mesh, v_uw_r_th[1])
    print(f"v_theta integral: {(1 / (2 * np.pi)) * v_th_rms_I.evaluate()}")

# %%
# Arrays for radial profiles
theta_0_2pi = np.linspace(0, 2 * np.pi, 1000, endpoint=True)
r_i_o = np.linspace(r_i, r_o - 1e-3, 11, endpoint=True)


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
        _xlabel='r', _ylabel=r'$<v_{r}>$', _xlim=[r_i, r_o], _ylim=[vr_avg.min(), vr_avg.max()],
        _save_pdf=True, _output_path=output_dir, _fname='vel_r_avg'
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
        _xlabel='r', _ylabel=r'$<v_{\theta}>$', _xlim=[r_i, r_o], _ylim=[vth_avg.min(), vth_avg.max()],
        _save_pdf=True, _output_path=output_dir, _fname='vel_th_avg'
    )

# %%
print(f"Annulus Benchmark (Kramer) complete: {case}, n={n}, k={k}, res={res}")
