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
# Stokes Flow in a Spherical Domain

**PHYSICS:** fluid_mechanics
**DIFFICULTY:** intermediate

## Description

Stokes flow in a spherical domain (solid sphere or spherical shell) with
free-slip boundary conditions. This example demonstrates:
- Different spherical mesh generators (CubedSphere, SphericalShell, etc.)
- Free-slip boundary conditions using penalty method on curved surfaces
- Null-space handling for rigid body rotations

## Mathematical Formulation

The Navier-Stokes equation with the Boussinesq approximation:

$$
\\frac{1}{Pr} \\frac{\\partial u}{\\partial t} + \\nabla^2 u - \\nabla p = Ra T' \\hat{g}
$$

Where Ra (Rayleigh number) and Pr (Prandtl number) are dimensionless.
For large Pr (typical in geodynamics), inertial terms are negligible.

## Parameters

- `uw_problem_size`: Controls mesh resolution (1-6)
- `uw_grid_refinement`: Additional mesh refinement levels
- `uw_grid_type`: Mesh type (simplex, cubed_sphere, ball, etc.)
- `uw_rayleigh`: Rayleigh number
"""

# %% [markdown]
"""
## Setup and Parameters
"""

# %%
import nest_asyncio
nest_asyncio.apply()

import underworld3 as uw
from underworld3 import timing
import numpy as np
import sympy
import os

os.environ["UW_TIMING_ENABLE"] = "1"

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_Stokes_Spherical_FreeSlipBCs.py -uw_problem_size 4
python Ex_Stokes_Spherical_FreeSlipBCs.py -uw_grid_type cubed_sphere
```
"""

# %%
params = uw.Params(
    uw_problem_size = 3,           # 1-6: resolution level
    uw_grid_refinement = 0,        # Additional refinement levels
    uw_grid_type = "cubed_sphere", # simplex, cubed_sphere, ball, segmented
    uw_rayleigh = 1000000,         # Rayleigh number
    uw_radius_outer = 1.0,         # Outer radius
    uw_radius_inner = 0.547,       # Inner radius (for shells)
)

# Create output directory
if uw.mpi.size == 1:
    os.makedirs("output", exist_ok=True)
else:
    os.makedirs(f"output_np{uw.mpi.size}", exist_ok=True)

# %% [markdown]
"""
## Mesh Generation

Different mesh types available:
- `simplex`: Unstructured tetrahedral mesh
- `cubed_sphere`: Structured mesh from cube projection
- `ball`: Solid sphere (no inner boundary)
- `segmented`: Segmented spherical shell
"""

# %%
r_o = params.uw_radius_outer
r_i = params.uw_radius_inner
grid_refinement = params.uw_grid_refinement
grid_type = params.uw_grid_type

Rayleigh = uw.function.expression(R"\mathrm{Ra}", params.uw_rayleigh, "Rayleigh number")

# Map problem_size to element count
els_map = {1: 3, 2: 6, 3: 12, 4: 24, 5: 48, 6: 96}
els = els_map.get(params.uw_problem_size, 12)
cell_size = 1 / els

expt_name = f"Stokes_Sphere_free_slip_{els}"

# %%
timing.reset()
timing.start()

if "ball" in grid_type:
    meshball = uw.meshing.SegmentedSphericalBall(
        radius=r_o,
        cellSize=cell_size,
        numSegments=5,
        qdegree=2,
        refinement=grid_refinement,
    )
elif "cubed" in grid_type:
    meshball = uw.meshing.CubedSphere(
        radiusInner=r_i,
        radiusOuter=r_o,
        numElements=els,
        simplex=True,
        refinement=grid_refinement,
        qdegree=2,
    )
elif "simplex" in grid_type:
    meshball = uw.meshing.SphericalShell(
        radiusInner=r_i,
        radiusOuter=r_o,
        cellSize=cell_size,
        refinement=grid_refinement,
        qdegree=2,
    )
else:
    meshball = uw.meshing.SegmentedSphericalShell(
        radiusInner=r_i,
        radiusOuter=r_o,
        cellSize=cell_size,
        numSegments=5,
        qdegree=2,
        refinement=grid_refinement,
    )

meshball.dm.view()

# %% [markdown]
"""
## Mesh Visualization
"""

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)

    clipped = pvmesh.clip(
        origin=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), invert=True, crinkle=True
    )

    pl = pv.Plotter(window_size=[750, 750])
    pl.add_axes()

    pl.add_mesh(
        clipped,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        use_transparency=False,
        show_scalar_bar=False,
        opacity=1.0,
    )

    pl.show(cpos="xy")

# %% [markdown]
"""
## Stokes Solver Setup
"""

# %%
stokes = uw.systems.Stokes(meshball, verbose=False)

v_soln = stokes.Unknowns.u
p_soln = stokes.Unknowns.p

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1
stokes.penalty = 0

# %% [markdown]
"""
## Coordinate System and Buoyancy
"""

# %%
x, y, z = meshball.CoordinateSystem.N
ra, l1, l2 = meshball.CoordinateSystem.R

bc_penalty = uw.function.expression(
    r"\Pi", sympy.sympify(100000), "BC enforcement penalty factor"
)

# Radial functions
radius_fn = sympy.sqrt(meshball.rvec.dot(meshball.rvec))
unit_rvec = meshball.X / radius_fn
gravity_fn = radius_fn

# %% [markdown]
"""
## Temperature Forcing

Gaussian blobs of buoyancy at three locations on the sphere.
"""

# %%
t_soln = uw.discretisation.MeshVariable(r"\Delta T", meshball, 1, degree=2)

t_forcing_fn = 1.0 * (
    sympy.exp(-10.0 * (x**2 + (y - 0.8) ** 2 + z**2))
    + sympy.exp(-10.0 * ((x - 0.8) ** 2 + y**2 + z**2))
    + sympy.exp(-10.0 * (x**2 + y**2 + (z - 0.8) ** 2))
)

with meshball.access(t_soln):
    t_soln.data[...] = uw.function.evaluate(
        t_forcing_fn, t_soln.coords, meshball.N
    ).reshape(-1, 1)

# %% [markdown]
"""
## Rigid Body Rotations (Null Space)

For free-slip boundaries on a sphere, there are three null-space modes
corresponding to rigid rotations about the x, y, and z axes. We need
to monitor and potentially remove these from the solution.
"""

# %%
orientation_wrt_z = sympy.atan2(y + 1.0e-10, x + 1.0e-10)
v_rbm_z_x = -ra * sympy.sin(orientation_wrt_z)
v_rbm_z_y = ra * sympy.cos(orientation_wrt_z)
v_rbm_z = sympy.Matrix([v_rbm_z_x, v_rbm_z_y, 0]).T

orientation_wrt_x = sympy.atan2(z + 1.0e-10, y + 1.0e-10)
v_rbm_x_y = -ra * sympy.sin(orientation_wrt_x)
v_rbm_x_z = ra * sympy.cos(orientation_wrt_x)
v_rbm_x = sympy.Matrix([0, v_rbm_x_y, v_rbm_x_z]).T

orientation_wrt_y = sympy.atan2(z + 1.0e-10, x + 1.0e-10)
v_rbm_y_x = -ra * sympy.sin(orientation_wrt_y)
v_rbm_y_z = ra * sympy.cos(orientation_wrt_y)
v_rbm_y = sympy.Matrix([v_rbm_y_x, 0, v_rbm_y_z]).T

# %% [markdown]
"""
## Solver Configuration
"""

# %%
stokes.tolerance = 1.0e-3
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

stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "additive")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# %% [markdown]
"""
## Boundary Conditions and Body Force

Free-slip implemented via penalty method on the normal velocity component.
"""

# %%
Gamma = meshball.CoordinateSystem.unit_e_0
stokes.add_natural_bc(bc_penalty * Gamma.dot(v_soln.sym) * Gamma, "Upper")

if "ball" not in grid_type:
    stokes.add_natural_bc(bc_penalty * Gamma.dot(v_soln.sym) * Gamma, "Lower")

stokes.bodyforce = unit_rvec * Rayleigh * gravity_fn * t_forcing_fn

# %% [markdown]
"""
## Solve
"""

# %%
timing.reset()
timing.start()

stokes.solve(zero_init_guess=True)

# %%
# Check null-space contamination
I0 = uw.maths.Integral(meshball, v_rbm_y.dot(v_rbm_y))
norm = I0.evaluate()
I0.fn = v_soln.sym.dot(v_soln.sym)
vnorm = np.sqrt(I0.evaluate())

I0.fn = v_soln.sym.dot(v_rbm_x)
x_ns = I0.evaluate() / norm
I0.fn = v_soln.sym.dot(v_rbm_y)
y_ns = I0.evaluate() / norm
I0.fn = v_soln.sym.dot(v_rbm_z)
z_ns = I0.evaluate() / norm

null_space_err = np.sqrt(x_ns**2 + y_ns**2 + z_ns**2) / vnorm

print(
    "Rigid body: {:.4}, {:.4}, {:.4} / {:.4}  (x,y,z axis / total)".format(
        x_ns, y_ns, z_ns, null_space_err
    )
)

timing.print_table()

# %% [markdown]
"""
## Output
"""

# %%
outdir = "output"

meshball.write_timestep(
    expt_name,
    meshUpdates=True,
    meshVars=[p_soln, v_soln],
    outputPath=outdir,
    index=0,
)

# %% [markdown]
"""
## Visualization
"""

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)

    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    clipped = pvmesh.clip(origin=(0.0, 0.0, 0.0), normal=(0.0, 1, 0), invert=True)
    clipped.point_data["V"] = vis.vector_fn_to_pv_points(clipped, v_soln.sym)

    # Streamlines
    skip = 20
    points = np.zeros((meshball._centroids[::skip].shape[0], 3))
    points[:, 0] = meshball._centroids[::skip, 0]
    points[:, 1] = meshball._centroids[::skip, 1]
    points[:, 2] = meshball._centroids[::skip, 2]
    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud,
        vectors="V",
        integration_direction="both",
        integrator_type=45,
        surface_streamlines=False,
        initial_step_length=0.01,
        max_time=2.0,
        max_steps=200,
    )

    pl = pv.Plotter(window_size=[1000, 750])
    pl.add_axes()

    pl.add_mesh(
        clipped,
        cmap="Reds",
        edge_color="Black",
        show_edges=True,
        scalars="T",
        use_transparency=False,
        show_scalar_bar=False,
        opacity=0.9,
    )

    pl.add_mesh(pvstream, show_scalar_bar=False, render_lines_as_tubes=False)

    pl.camera_position = "yz"
    pl.camera.azimuth = 45
    pl.camera.elevation = 25

    pl.show()
