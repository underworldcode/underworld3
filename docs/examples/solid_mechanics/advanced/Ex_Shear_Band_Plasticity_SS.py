# %% [markdown]
"""
# 🎓 Shear Band Plasticity SS

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

# # Shear bands around a circular inclusion in a simple shear flow
#
# No slip conditions
#
#

# +
# to fix trame issue
import nest_asyncio

nest_asyncio.apply()
# -

expt_name = "ShearBand"

import numpy as np
import petsc4py
import underworld3 as uw


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
# -


mesh1 = uw.discretisation.Mesh(
    "tmp_shear_inclusion.msh",
    markVertices=True,
    useMultipleTags=True,
    useRegions=True,
    refinement=0,
    # refinement_callback=_mesh_refinement_callback,
    # return_coords_to_bounds= _return_coords_to_bounds,
    boundaries=boundaries,
    qdegree=3)



# +
# Define some functions on the mesh

import sympy

# Some useful coordinate stuff

x, y = mesh1.X

# relative to the centre of the inclusion
r = sympy.sqrt(x**2 + y**2)
th = sympy.atan2(y, x)

# need a unit_r_vec equivalent

inclusion_rvec = mesh1.X
inclusion_unit_rvec = inclusion_rvec / inclusion_rvec.dot(inclusion_rvec)
inclusion_unit_rvec = mesh1.vector.to_matrix(inclusion_unit_rvec)


# +
v_soln = uw.discretisation.MeshVariable("U", mesh1, mesh1.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", mesh1, 1, degree=1, continuous=True)
p_cont = uw.discretisation.MeshVariable("Pc", mesh1, 1, degree=1, continuous=True)
p_null = uw.discretisation.MeshVariable(r"P2", mesh1, 1, degree=1, continuous=True)

vorticity = uw.discretisation.MeshVariable("omega", mesh1, 1, degree=1)
strain_rate_inv2 = uw.discretisation.MeshVariable("eps", mesh1, 1, degree=1)
dev_stress_inv2 = uw.discretisation.MeshVariable("tau", mesh1, 1, degree=1)
node_viscosity = uw.discretisation.MeshVariable("eta", mesh1, 1, degree=1)
r_inc = uw.discretisation.MeshVariable("R", mesh1, 1, degree=1)


# +
# Create NS/Stokes object

stokes = uw.systems.Stokes(
    mesh1,
    velocityField=v_soln,
    pressureField=p_soln,
    verbose=False)

mu = uw.function.expression(R"\mu", 0.1, "Friction")
C = uw.function.expression(R"C", 1000, "Cohesion")

stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
stokes.constitutive_model.Parameters.yield_stress = mu * p_soln.sym[0] + C
stokes.constitutive_model.Parameters.yield_min = 1000
stokes.penalty = 1

stokes.petsc_options["ksp_monitor"] = None
stokes.petsc_options["snes_monitor"] = None
stokes.petsc_options["snes_atol"] = 0.001

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


# +
nodal_strain_rate_inv2 = uw.systems.Projection(
    mesh1,
    strain_rate_inv2)
nodal_strain_rate_inv2.uw_function = stokes.Unknowns.Einv2
nodal_strain_rate_inv2.smoothing = 1.0e-3
nodal_strain_rate_inv2.petsc_options.delValue("ksp_monitor")

nodal_tau_inv2 = uw.systems.Projection(mesh1, dev_stress_inv2)
S = stokes.stress_deviator
nodal_tau_inv2.uw_function = (
    sympy.simplify(sympy.sqrt(((S**2).trace()) / 2)) - p_soln.sym[0]
)
nodal_tau_inv2.smoothing = 1.0e-3
nodal_tau_inv2.petsc_options.delValue("ksp_monitor")

nodal_visc_calc = uw.systems.Projection(mesh1, node_viscosity)
nodal_visc_calc.uw_function = stokes.constitutive_model.viscosity
nodal_visc_calc.smoothing = 1.0e-3
nodal_visc_calc.petsc_options.delValue("ksp_monitor")

# nodal_pres_calc = uw.systems.Projection(mesh1, p_cont)
# nodal_pres_calc.uw_function = p_soln.sym[0]
# nodal_pres_calc.smoothing = 1.0e-3
# nodal_pres_calc.petsc_options.delValue("ksp_monitor")


# +
# Set solve options here (or remove default values
# stokes.petsc_options.getAll()

# Constant visc

stokes.bodyforce = mesh1.CoordinateSystem.unit_e_1 * 1.0e-5

# Velocity boundary conditions

Gamma = mesh1.Gamma
stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) * Gamma, "inclusion")

# Velocity boundary conditions

# stokes.add_dirichlet_bc((0.0, 0.0), "inclusion")
stokes.add_dirichlet_bc((1.0, 0.0), "top")
stokes.add_dirichlet_bc((-1.0, 0.0), "bottom")
stokes.add_dirichlet_bc((None, 0.0), "left")
stokes.add_dirichlet_bc((None, 0.0), "right")
# -


# linear solve first
stokes.constitutive_model.Parameters.yield_stress_min = sympy.oo
stokes.constitutive_model.Parameters.yield_stress = sympy.oo
stokes.solve()



# +
# Approach the required non-linear value by gradually adjusting the parameters

steps = 4
for i in range(steps+1):
    mu = 0.5
    C = 2 + (steps - i) * 0.5
    print(f"Mu - {mu}, C = {C}")
    # viscosity = sympy.Min(tau_y / (2 * stokes.Unknowns.Einv2 + 0.01), 1.0)
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
    stokes.constitutive_model.Parameters.yield_stress = C + mu * stokes.p.sym[0]
    stokes.constitutive_model.Parameters.yield_stress_min = 1

    stokes.saddle_preconditioner = 1 /  stokes.constitutive_model.viscosity
    stokes.solve(zero_init_guess=False)


# +
nodal_tau_inv2.uw_function = (
    stokes.constitutive_model.viscosity * stokes.Unknowns.Einv2
)
nodal_tau_inv2.solve()


nodal_visc_calc.uw_function = stokes.constitutive_model.viscosity
nodal_visc_calc.solve()

nodal_strain_rate_inv2.solve()


# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh1)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_cont.sym)
    pvmesh.point_data["Edot"] = vis.scalar_fn_to_pv_points(pvmesh, strain_rate_inv2.sym)
    pvmesh.point_data["Visc"] = vis.scalar_fn_to_pv_points(pvmesh, node_viscosity.sym)
    pvmesh.point_data["Str"] = vis.scalar_fn_to_pv_points(pvmesh, dev_stress_inv2.sym)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(
        pvmesh, v_soln.sym.dot(v_soln.sym)
    )

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(
        velocity_points, v_soln.sym
    )

    # point sources at cell centres

    subsample = 10
    points = np.zeros((mesh1._centroids[::subsample].shape[0], 3))
    points[:, 0] = mesh1._centroids[::subsample, 0]
    points[:, 1] = mesh1._centroids[::subsample, 1]
    point_cloud = pv.PolyData(points)

    # pvstream = pvmesh.streamlines_from_source(
    #     point_cloud, vectors="V", integration_direction="both", max_steps=100
    # )

    pl = pv.Plotter(window_size=(1000, 500))

    pl.add_arrows(
        velocity_points.points, velocity_points.point_data["V"], mag=0.1, opacity=0.75
    )
    pl.camera_position = "xy"

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Grey",
        show_edges=True,
        clim=[0.5, 2.0],
        scalars="Edot",
        use_transparency=False,
        opacity=1.0)

    # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.75)
    # pl.add_mesh(pvstream)

    # pl.remove_scalar_bar("mag")

    pl.show()
# -

