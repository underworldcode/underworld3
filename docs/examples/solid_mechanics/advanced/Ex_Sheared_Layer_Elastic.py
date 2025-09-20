# %% [markdown]
"""
# 🎓 Sheared Layer Elastic

**PHYSICS:** solid_mechanics  
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
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python (Pixi)
#     language: python
#     name: pixi-kernel-python3
# ---

# # Elastic Shearing
#
# Simple shear with material defined by particle swarm (based on inclusion model), position, pressure, strain rate etc.  Check the implementation of the Jacobians using various non-linear terms.
#
# Check elastic stress terms
#

# +
# to fix trame issue
import nest_asyncio

nest_asyncio.apply()

# +
import os

os.environ["UW_TIMING_ENABLE"] = "1"

import numpy as np
import petsc4py
import pyvista as pv
import sympy
import underworld3 as uw
import vtk
from underworld3 import timing

resolution = uw.options.getReal("model_resolution", default=0.05)
mu = uw.options.getInt("mu", default=0.5)
maxsteps = uw.options.getInt("max_steps", default=500)


## Define units here and physical timestep numbers etc.

observation_timescale = 0.0033


# +
from enum import Enum

import meshio
import pygmsh

# Mesh a 2D pipe with a circular hole

csize = 0.075
csize_circle = 0.025
res = csize_circle

width = 3.0
height = 1.0
radius = 0.1


class boundaries(Enum):
    bottom = 1
    right = 2
    top = 3
    left = 4
    inclusion = 5
    All_Boundaries = 1001


if uw.mpi.rank == 0:
    # Generate local mesh on boss process

    with pygmsh.geo.Geometry() as geom:
        geom.characteristic_length_max = csize

        inclusion = geom.add_circle(
            (0.0, 0.0, 0.0), radius, make_surface=False, mesh_size=csize_circle
        )
        domain = geom.add_rectangle(
            xmin=-width / 2,
            ymin=-height / 2,
            xmax=width / 2,
            ymax=height / 2,
            z=0,
            holes=[inclusion],
            mesh_size=csize)

        geom.add_physical(
            domain.surface.curve_loop.curves[0], label=boundaries.bottom.name
        )
        geom.add_physical(
            domain.surface.curve_loop.curves[1], label=boundaries.right.name
        )
        geom.add_physical(
            domain.surface.curve_loop.curves[2], label=boundaries.top.name
        )
        geom.add_physical(
            domain.surface.curve_loop.curves[3], label=boundaries.left.name
        )
        geom.add_physical(inclusion.curve_loop.curves, label=boundaries.inclusion.name)
        geom.add_physical(domain.surface, label="Elements")

        geom.generate_mesh(dim=2, verbose=False)
        geom.save_geometry("tmp_shear_inclusion.msh")


## Restore inflow samples to inflow points
def mesh_return_coords_to_bounds(coords):
    lefty_troublemakers = coords[:, 0] < -width / 2
    righty_troublemakers = coords[:, 0] > width / 2
    coords[lefty_troublemakers, 0] = -width / 2 + 0.0001
    coords[righty_troublemakers, 0] = width / 2 - 0.0001

    return coords


mesh1 = uw.discretisation.Mesh(
    "tmp_shear_inclusion.msh",
    markVertices=True,
    useRegions=True,
    refinement=0,
    # refinement_callback=_mesh_refinement_callback,
    return_coords_to_bounds=mesh_return_coords_to_bounds,
    boundaries=boundaries,
    qdegree=3)


# +
v_soln = uw.discretisation.MeshVariable("U", mesh1, mesh1.dim, degree=2)
p_soln = uw.discretisation.MeshVariable(
    "P", mesh1, 1, vtype=uw.VarType.SCALAR, degree=1, continuous=True
)
Stress = uw.discretisation.MeshVariable(
    r"Stress",
    mesh1,
    (2, 2),
    vtype=uw.VarType.SYM_TENSOR,
    degree=2,
    continuous=True,
    varsymbol=r"{\sigma}")

work = uw.discretisation.MeshVariable(
    "W", mesh1, 1, vtype=uw.VarType.SCALAR, degree=2, continuous=True
)
strain_rate_inv2 = uw.discretisation.MeshVariable(
    "eps_dot", mesh1, 1, degree=2, varsymbol=r"{\dot\varepsilon}"
)
strain_rate_inv2_pl = uw.discretisation.MeshVariable(
    "eps_dot_pl", mesh1, 1, degree=2, varsymbol=r"{\dot\varepsilon_{pl}}"
)
dev_stress_inv2 = uw.discretisation.MeshVariable("tau", mesh1, 1, degree=2)

# +
# mesh1.view()
# -

# Some useful coordinate stuff
x, y = mesh1.X


# +
# Create Solver object

stokes = uw.systems.VE_Stokes(
    mesh1, velocityField=v_soln, pressureField=p_soln, verbose=False, order=1
)

# viscosity_L = sympy.Piecewise(
#     (1, material.sym[0] > 0.5),
#     (1000, True),
# )

stokes.petsc_options["snes_monitor"] = None
stokes.petsc_options["ksp_monitor"] = None


# -

stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel

stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
stokes.constitutive_model.Parameters.shear_modulus = 10
stokes.constitutive_model.Parameters.dt_elastic = sympy.sympify(observation_timescale)

uw.systems.Stokes.view()




sigma_projector = uw.systems.Tensor_Projection(
    mesh1, tensor_Field=Stress, scalar_Field=work
)
sigma_projector.uw_function = stokes.stress



# +
nodal_strain_rate_inv2 = uw.systems.Projection(
    mesh1,
    strain_rate_inv2)

nodal_strain_rate_inv2.uw_function = stokes.Unknowns.Einv2
nodal_strain_rate_inv2.smoothing = 1.0e-3

nodal_tau_inv2 = uw.systems.Projection(
    mesh1,
    dev_stress_inv2)
nodal_tau_inv2.uw_function = (
    2 * stokes.constitutive_model.viscosity * stokes.Unknowns.Einv2
)
nodal_tau_inv2.smoothing = 1.0e-3
# +
# Set solve options here (or remove default values
# stokes.petsc_options.getAll()

# Constant visc

stokes.penalty = 1.0
stokes.tolerance = 1.0e-4

# Velocity boundary conditions

stokes.add_dirichlet_bc((0.0, 0.0), "inclusion")
stokes.add_dirichlet_bc((1.0, 0.0), "top")
stokes.add_dirichlet_bc((-1.0, 0.0), "bottom")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "left")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "right")
# -
stokes.view()

stokes.delta_t

stokes.constitutive_model.viscosity.sym.diff(
    stokes.constitutive_model.Parameters.shear_viscosity_0)

uw.function.derivative(
    stokes.constitutive_model.viscosity.sym,
    stokes.constitutive_model.Parameters.shear_viscosity_0)

stokes.constitutive_model.viscosity.sym

a = uw.function.derivative(
    stokes.constitutive_model.viscosity,
    stokes.constitutive_model.Parameters.shear_viscosity_0, 
    evaluate=False)

a.sym

a.doit()

a.expr

a.diff_variable

b = uw.function.derivative(
    v_soln.sym,
    mesh1.CoordinateSystem.X,
    evaluate=False)

b = uw.function.expressions.UWDerivativeExpression("tester", v_soln.sym, mesh1.CoordinateSystem.X)



# +
# a.diff_variable = stokes.constitutive_model.Parameters.shear_viscosity_0
# -

a.doit()

0/0



a.expr.diff(a.diff_variable)

stokes.constitutive_model.Parameters.ve_effective_viscosity.sym.diff(stokes.constitutive_model.Parameters.shear_viscosity_0)

stokes.constitutive_model.viscosity.diff(
    stokes.constitutive_model.Parameters.shear_viscosity_0
)

uw.function.derivative(
    stokes.constitutive_model.viscosity,
    stokes.constitutive_model.Parameters.shear_viscosity_0)


stokes.constitutive_model.viscosity.diff(
    stokes.constitutive_model.Parameters.shear_viscosity_0
)

(stokes.constitutive_model.flux + stokes.constitutive_model.flux) / 2

0 / 0

stokes.Unknowns.DFDt._psi_star_projection_solver.Unknowns.L

nodal_strain_rate_inv2.solve()
sigma_projector.uw_function = stokes.stress_deviator
sigma_projector.solve()

# +
# with swarm.access(stress_star_p), mesh1.access():
#     stress_star_p.data[
#         ...
#     ] = 0.0  # Stress.rbf_interpolate(swarm._particle_coordinates.data)


mesh1.view()
# -


timing.reset()
timing.start()

stokes.solve(zero_init_guess=False, evalf=False)
timing.print_table(display_fraction=1)
print(stokes.Unknowns.u.max(), stokes.Unknowns.p.max())


# +
stokes.solve(zero_init_guess=False, verbose=False, evalf=False)
timing.print_table(display_fraction=1)
print(stokes.Unknowns.u.max(), stokes.Unknowns.p.max())

stokes.DFDt.psi_star[0]

# +
nodal_strain_rate_inv2.uw_function = stokes.Unknowns.Einv2
nodal_strain_rate_inv2.solve()

S = stokes.stress_deviator
nodal_tau_inv2.uw_function = (
    stokes.constitutive_model.viscosity * 2 * stokes.Unknowns.Einv2
)
nodal_tau_inv2.solve()
# +
# check it - NOTE - for the periodic mesh, points which have crossed the coordinate sheet are plotted somewhere
# unexpected. This is a limitation we are stuck with for the moment.

if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh1)

    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym[0])
    pvmesh.point_data["Edot"] = vis.scalar_fn_to_pv_points(
        pvmesh, strain_rate_inv2.sym[0]
    )
    pvmesh.point_data["Strs"] = vis.scalar_fn_to_pv_points(
        pvmesh, dev_stress_inv2.sym[0]
    )

    # Velocity arrows
    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(
        velocity_points, v_soln.sym
    )

    pl = pv.Plotter(window_size=(1000, 500))

    arrows0 = pl.add_arrows(
        velocity_points.points,
        velocity_points.point_data["V"],
        mag=0.1,
        opacity=1,
        show_scalar_bar=False)

    pl.add_mesh(
        pvmesh,
        cmap="Blues",
        edge_color="Grey",
        show_edges=True,
        # clim=[0.0,1.0],
        scalars="Strs",
        use_transparency=False,
        opacity=0.5)

    # pl.add_points(point_cloud, colormap="coolwarm", scalars="strain", point_size=10.0, opacity=0.5)

    pl.camera.SetPosition(0.0, 0.0, 3.0)
    pl.camera.SetFocalPoint(0.0, 0.0, 0.0)
    pl.camera.SetClippingRange(1.0, 8.0)

    pl.show(jupyter_backend="client")
# -
ts = 0
time = 0.0
delta_t = stokes.delta_t
maxsteps = 100

# +
expt_name = f"shear_band_sw_nonp_{mu}"

for step in range(0, maxsteps):
    stokes.solve(zero_init_guess=False, evalf=False)

    nodal_strain_rate_inv2.uw_function = stokes.Unknowns.Einv2
    nodal_strain_rate_inv2.solve()

    S = stokes.stress_deviator
    nodal_tau_inv2.uw_function = sympy.simplify(sympy.sqrt(((S**2).trace()) / 2))
    nodal_tau_inv2.solve()

    if uw.mpi.rank == 0:
        print(f"Stress Inv II -  {dev_stress_inv2.mean()}")

    mesh1.write_timestep(
        expt_name,
        meshUpdates=False,
        meshVars=[p_soln, v_soln],
        outputPath="output",
        index=ts)

    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(step, delta_t))

    ts += 1
    time += delta_t
# +
# check it - NOTE - for the periodic mesh, points which have crossed the coordinate sheet are plotted somewhere
# unexpected. This is a limitation we are stuck with for the moment.

if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh1)

    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym[0])
    pvmesh.point_data["Edot"] = vis.scalar_fn_to_pv_points(
        pvmesh, strain_rate_inv2.sym[0]
    )
    pvmesh.point_data["Strs"] = vis.scalar_fn_to_pv_points(
        pvmesh, dev_stress_inv2.sym[0]
    )

    # Velocity arrows
    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(
        velocity_points, v_soln.sym
    )

    pl = pv.Plotter(window_size=(1000, 500))

    arrows0 = pl.add_arrows(
        velocity_points.points,
        velocity_points.point_data["V"],
        mag=0.1,
        opacity=1,
        show_scalar_bar=False)

    pl.add_mesh(
        pvmesh,
        cmap="Blues",
        edge_color="Grey",
        show_edges=True,
        # clim=[0.0,1.0],
        scalars="Strs",
        use_transparency=False,
        opacity=0.5)

    # pl.add_points(point_cloud, colormap="coolwarm", scalars="strain", point_size=10.0, opacity=0.5)

    pl.camera.SetPosition(0.0, 0.0, 3.0)
    pl.camera.SetFocalPoint(0.0, 0.0, 0.0)
    pl.camera.SetClippingRange(1.0, 8.0)

    pl.show(jupyter_backend="client")
# -
stokes
