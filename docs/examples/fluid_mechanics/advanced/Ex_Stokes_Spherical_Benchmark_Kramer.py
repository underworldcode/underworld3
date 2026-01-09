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
# Stokes Spherical Shell Benchmark (Kramer)

**PHYSICS:** fluid_mechanics
**DIFFICULTY:** advanced

## Description

Benchmark validation of the Stokes solver in a spherical shell geometry using
analytical solutions from Kramer et al. (2021). Tests four cases combining
free-slip/no-slip boundaries with delta function/smooth density perturbations.

## Key Concepts

- **ASSESS library**: Analytical Stokes solutions for spherical geometries
- **Four test cases**: Free-slip/no-slip × delta/smooth density
- **Spherical harmonics**: Y_lm density perturbations
- **Internal boundary**: Delta function density at r_int interface
- **Null space handling**: Removal of rigid body rotation modes
- **L2 error analysis**: Convergence validation against analytical solution

## Test Cases

| Case  | Boundaries | Density Type | Description |
|-------|------------|--------------|-------------|
| case1 | Free-slip  | Delta function | Thin dense layer at interface |
| case2 | Free-slip  | Smooth | Power-law radial density |
| case3 | No-slip    | Delta function | Thin dense layer, fixed walls |
| case4 | No-slip    | Smooth | Power-law radial density, fixed walls |

## Parameters

- `uw_case`: Test case selection (case1, case2, case3, case4)
- `uw_l`: Spherical harmonic degree
- `uw_m`: Spherical harmonic order
- `uw_res`: Mesh resolution (1/cellsize)
- `uw_vdegree`: Velocity polynomial degree
- `uw_pdegree`: Pressure polynomial degree
- `uw_vel_penalty`: Penalty for boundary conditions
- `uw_stokes_tol`: Solver tolerance

## References

- [Kramer et al. (2021)](https://gmd.copernicus.org/articles/14/1899/2021/)

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
import sympy
import os
import assess
import h5py
import sys

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
python Ex_Stokes_Spherical_Benchmark_Kramer.py -uw_case case1
python Ex_Stokes_Spherical_Benchmark_Kramer.py -uw_res 32
python Ex_Stokes_Spherical_Benchmark_Kramer.py -uw_l 3 -uw_m 2
```
"""

# %%
params = uw.Params(
    uw_case = "case1",            # Test case: case1, case2, case3, case4
    uw_l = 2,                     # Spherical harmonic degree
    uw_m = 1,                     # Spherical harmonic order
    uw_res = 16,                  # Mesh resolution (1/cellsize)
    uw_vdegree = 2,               # Velocity polynomial degree
    uw_pdegree = 1,               # Pressure polynomial degree
    uw_pcont = 1,                 # Pressure continuity (1=True, 0=False)
    uw_vel_penalty = 1e8,         # Penalty for boundary conditions
    uw_stokes_tol = 1e-5,         # Solver tolerance
    uw_r_outer = 2.22,            # Outer radius
    uw_r_internal = 2.0,          # Internal interface radius
    uw_r_inner = 1.22,            # Inner radius
)

# Extract parameters
case = params.uw_case
l = int(params.uw_l)
m = int(params.uw_m)
k = l + 1  # Power exponent for smooth density

res = int(params.uw_res)
vdegree = int(params.uw_vdegree)
pdegree = int(params.uw_pdegree)
pcont = bool(params.uw_pcont)
vel_penalty = params.uw_vel_penalty
stokes_tol = params.uw_stokes_tol

r_o = params.uw_r_outer
r_int = params.uw_r_internal
r_i = params.uw_r_inner

# Derived parameters
cellsize = 1 / res
refine = None
pcont_str = str(pcont).lower()
vel_penalty_str = str("{:.1e}".format(vel_penalty))
stokes_tol_str = str("{:.1e}".format(stokes_tol))

# Compute flags
analytical = True
visualize = False
timing = True

# %% [markdown]
"""
## Case Selection

Determine boundary conditions and density perturbation type from case string.
"""

# %%
freeslip, noslip, delta_fn, smooth = False, False, False, False

if case in ('case1'):
    freeslip, delta_fn = True, True
elif case in ('case2'):
    freeslip, smooth = True, True
elif case in ('case3'):
    noslip, delta_fn = True, True
elif case in ('case4'):
    noslip, smooth = True, True

# %% [markdown]
"""
## Output Directory
"""

# %%
output_dir = os.path.join(
    "./output/Latex_Dir/",
    f'{case}_l_{l}_m_{m}_k_{k}_res_{res}_refine_{refine}_vdeg_{vdegree}_pdeg_{pdegree}'
    f'_pcont_{pcont_str}_vel_penalty_{vel_penalty_str}_stokes_tol_{stokes_tol_str}/'
)

if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)

# %% [markdown]
"""
## Analytical Solution from ASSESS Library
"""

# %%
if analytical:
    if freeslip:
        if delta_fn:
            soln_above = assess.SphericalStokesSolutionDeltaFreeSlip(
                l, m, +1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0
            )
            soln_below = assess.SphericalStokesSolutionDeltaFreeSlip(
                l, m, -1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0
            )
        elif smooth:
            # For smooth density, single solution exists but we create two for code optimization
            soln_above = assess.SphericalStokesSolutionSmoothFreeSlip(
                l, m, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0
            )
            soln_below = assess.SphericalStokesSolutionSmoothFreeSlip(
                l, m, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0
            )
    elif noslip:
        if delta_fn:
            soln_above = assess.SphericalStokesSolutionDeltaZeroSlip(
                l, m, +1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0
            )
            soln_below = assess.SphericalStokesSolutionDeltaZeroSlip(
                l, m, -1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0
            )
        elif smooth:
            soln_above = assess.SphericalStokesSolutionSmoothZeroSlip(
                l, m, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0
            )
            soln_below = assess.SphericalStokesSolutionSmoothZeroSlip(
                l, m, k, Rp=r_o, Rm=r_i, nu=1.0, g=1.0
            )

# %% [markdown]
"""
## Mesh Generation
"""

# %%
if timing:
    uw.timing.reset()
    uw.timing.start()

if case in ('case1', 'case3'):
    # Delta function cases need internal boundary
    mesh = uw.meshing.SphericalShellInternalBoundary(
        radiusInner=r_i, radiusOuter=r_o, radiusInternal=r_int,
        cellSize=cellsize, qdegree=max(pdegree, vdegree),
        filename=f'{output_dir}mesh.msh', refinement=refine
    )
else:
    # Smooth density cases use standard shell
    mesh = uw.meshing.SphericalShell(
        radiusInner=r_i, radiusOuter=r_o, cellSize=cellsize,
        qdegree=max(pdegree, vdegree),
        filename=f'{output_dir}mesh.msh', refinement=refine
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
v_uw = uw.discretisation.MeshVariable('V_u', mesh, mesh.dim, degree=vdegree)
p_uw = uw.discretisation.MeshVariable('P_u', mesh, 1, degree=pdegree, continuous=pcont)

if analytical:
    v_ana = uw.discretisation.MeshVariable('V_a', mesh, mesh.dim, degree=vdegree)
    p_ana = uw.discretisation.MeshVariable('P_a', mesh, 1, degree=pdegree, continuous=pcont)
    rho_ana = uw.discretisation.MeshVariable('RHO_a', mesh, 1, degree=pdegree, continuous=True)

    v_err = uw.discretisation.MeshVariable('V_e', mesh, mesh.dim, degree=vdegree)
    p_err = uw.discretisation.MeshVariable('P_e', mesh, 1, degree=pdegree, continuous=pcont)

# %% [markdown]
"""
## Normal Vector for Null Space
"""

# %%
norm_v = uw.discretisation.MeshVariable("N", mesh, mesh.dim, degree=pdegree, varsymbol=r"{\hat{n}}")
with mesh.access(norm_v):
    norm_v.data[:, 0] = uw.function.evaluate(mesh.CoordinateSystem.unit_e_0[0], norm_v.coords)
    norm_v.data[:, 1] = uw.function.evaluate(mesh.CoordinateSystem.unit_e_0[1], norm_v.coords)
    norm_v.data[:, 2] = uw.function.evaluate(mesh.CoordinateSystem.unit_e_0[2], norm_v.coords)

# %% [markdown]
"""
## Coordinate System
"""

# %%
unit_rvec = mesh.CoordinateSystem.unit_e_0
r_uw, th_uw = mesh.CoordinateSystem.xR[0], mesh.CoordinateSystem.xR[1]
phi_uw = sympy.Piecewise(
    (2 * sympy.pi + mesh.CoordinateSystem.xR[2], mesh.CoordinateSystem.xR[2] < 0),
    (mesh.CoordinateSystem.xR[2], True)
)

# Null space in velocity expressed in x,y,z coordinates
v_theta_phi_fn_xyz = sympy.Matrix(((0, 1, 1), (-1, 0, 1), (-1, -1, 0))) * mesh.CoordinateSystem.N.T

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
## Visualization of Analytical Solution
"""

# %%
# Set visualization parameters based on case
if case in ('case1'):
    v_clim, vmag, vfreq = [0., 0.015], 5e0, 75
elif case in ('case2'):
    v_clim, vmag, vfreq = [0., 0.007], 1e1, 75
elif case in ('case3'):
    v_clim, vmag, vfreq = [0., 0.003], 2.5e1, 75
elif case in ('case4'):
    v_clim, vmag, vfreq = [0., 0.001], 5e2, 75

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
# Pressure visualization
if case in ('case1'):
    p_clim = [-0.25, 0.25]
elif case in ('case2'):
    p_clim = [-0.1, 0.1]
elif case in ('case3'):
    p_clim = [-0.3, 0.3]
elif case in ('case4'):
    p_clim = [-0.1, 0.1]

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

# %% [markdown]
"""
## Stokes Solver
"""

# %%
stokes = Stokes(mesh, velocityField=v_uw, pressureField=p_uw)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0
stokes.saddle_preconditioner = 1.0

# %% [markdown]
"""
## Density and Body Force

Spherical harmonic density perturbation: Y_lm(theta, phi)
"""

# %%
# Real part of spherical harmonic Y_lm
y_lm_real = (
    sympy.sqrt((2 * l + 1) / (4 * sympy.pi)
               * sympy.factorial(l - m) / sympy.factorial(l + m))
    * sympy.cos(m * phi_uw)
    * sympy.assoc_legendre(l, m, sympy.cos(th_uw))
)

gravity_fn = -1.0 * unit_rvec

if delta_fn:
    # Delta function density at interface (approximated by Gaussian)
    rho = sympy.exp(-1e3 * ((r_uw - r_int) ** 2)) * y_lm_real
    stokes.add_natural_bc(-rho * unit_rvec, mesh.boundaries.Internal.name)
    stokes.bodyforce = sympy.Matrix([0., 0., 0.])

if smooth:
    # Smooth power-law density
    rho = ((r_uw / r_o) ** k) * y_lm_real
    stokes.bodyforce = rho * gravity_fn

# %%
if analytical:
    with mesh.access(rho_ana):
        rho_ana.data[:] = np.c_[uw.function.evaluate(rho, rho_ana.coords)]

# %%
# Density visualization
clim = [-0.4, 0.4]
if uw.mpi.size == 1 and visualize:
    vis.plot_scalar(
        mesh, -rho_ana.sym, 'Rho', cmap=cmc.roma.resampled(31),
        clim=clim, save_png=True,
        dir_fname=output_dir + 'rho_ana.png', clip_angle=135, cpos='yz'
    )

    vis.save_colorbar(
        colormap=cmc.roma.resampled(31), cb_bounds=None,
        vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18,
        cb_orient='horizontal', cb_axis_label='Rho',
        cb_label_xpos=0.5, cb_label_ypos=-2.0, fformat='pdf',
        output_path=output_dir, fname='rho_ana'
    )

# %% [markdown]
"""
## Boundary Conditions
"""

# %%
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
        output_file=f"{output_dir}stokes_solve_time.txt",
        display_fraction=1.00
    )

# %% [markdown]
"""
## Null Space Removal

Remove rigid body rotation modes from the solution.
"""

# %%
I0 = uw.maths.Integral(mesh, v_theta_phi_fn_xyz.dot(v_uw.sym))
norm = I0.evaluate()

I0.fn = v_theta_phi_fn_xyz.dot(v_theta_phi_fn_xyz)
vnorm = I0.evaluate()

with mesh.access(v_uw):
    dv = uw.function.evaluate(norm * v_theta_phi_fn_xyz, v_uw.coords) / vnorm
    v_uw.data[...] -= dv

# %% [markdown]
"""
## Compute Error
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
## Visualization of Results
"""

# %%
# Velocity from solver
if case in ('case1'):
    v_clim, vmag, vfreq = [0., 0.015], 5e0, 75
elif case in ('case2'):
    v_clim, vmag, vfreq = [0., 0.007], 1e1, 75
elif case in ('case3'):
    v_clim, vmag, vfreq = [0., 0.003], 2.5e1, 75
elif case in ('case4'):
    v_clim, vmag, vfreq = [0., 0.001], 5e2, 75

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
# Velocity error
if case in ('case1'):
    verr_clim, vmag, vfreq = [0., 0.005], 1e2, 75
elif case in ('case2'):
    verr_clim, vmag, vfreq = [0., 1e-4], 1e2, 75
elif case in ('case3'):
    verr_clim, vmag, vfreq = [0., 6e-3], 2e2, 75
elif case in ('case4'):
    verr_clim, vmag, vfreq = [0., 1e-4], 1e5, 75

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
        cb_orient='horizontal', cb_axis_label='Velocity Error (relative)',
        cb_label_xpos=0.5, cb_label_ypos=-2.05, fformat='pdf',
        output_path=output_dir, fname='v_err_rel'
    )

# %%
# Velocity magnitude error in percentage
clim = [0, 100]
if uw.mpi.size == 1 and analytical and visualize:
    vmag_expr = (sympy.sqrt(v_err.sym.dot(v_err.sym)) / sympy.sqrt(v_ana.sym.dot(v_ana.sym))) * 100
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
# Pressure from solver
if case in ('case1'):
    p_clim = [-0.25, 0.25]
elif case in ('case2'):
    p_clim = [-0.1, 0.1]
elif case in ('case3'):
    p_clim = [-0.3, 0.3]
elif case in ('case4'):
    p_clim = [-0.1, 0.1]

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
# Pressure error
if case in ('case1', 'case3'):
    perr_clim = [-0.065, 0.065]
elif case in ('case2', 'case4'):
    perr_clim = [-0.01, 0.01]

if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_scalar(
        mesh, p_err.sym, 'p_err(relative)', cmap=cmc.vik.resampled(41),
        clim=perr_clim, save_png=True,
        dir_fname=output_dir + 'p_r_err.png', clip_angle=135, cpos='yz'
    )

    vis.save_colorbar(
        colormap=cmc.vik.resampled(41), cb_bounds=None,
        vmin=perr_clim[0], vmax=perr_clim[1], figsize_cb=(5, 5), primary_fs=18,
        cb_orient='horizontal', cb_axis_label='Pressure Error (relative)',
        cb_label_xpos=0.5, cb_label_ypos=-2.0, fformat='pdf',
        output_path=output_dir, fname='p_err_rel'
    )

# %%
# Pressure percentage error
clim = [-1e2, 1e2]
if uw.mpi.size == 1 and analytical and visualize:
    vis.plot_scalar(
        mesh, (p_err.sym[0] / p_ana.sym[0]) * 100, 'p_err(%)',
        cmap=cmc.vik.resampled(41), clim=clim, save_png=True,
        dir_fname=output_dir + 'p_p_err.png', clip_angle=135, cpos='yz'
    )

    vis.save_colorbar(
        colormap=cmc.vik.resampled(41), cb_bounds=None,
        vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18,
        cb_orient='horizontal', cb_axis_label='Pressure Error (%)',
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
        f.create_dataset("k", data=k)
        f.create_dataset("res", data=res)
        f.create_dataset("cellsize", data=cellsize)
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
print(f"Spherical Kramer benchmark complete: {case}, l={l}, m={m}, res={res}, v_err_l2={v_err_l2:.6e}")
