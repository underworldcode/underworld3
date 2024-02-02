# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Stokes flow in a Spherical Domain
#
#
# ## Mathematical formulation
#
# The Navier-Stokes equation describes the time-dependent flow of a viscous fluid in response to buoyancy forces and pressure gradients:
#
# $$
# \rho \frac{\partial \mathbf{u}}{\partial t} + \eta\nabla^2 \mathbf{u} -\nabla p = \rho \mathbf{g}
# $$
#
# Where $\rho$ is the density, $\eta$ is dynamic viscosity and $\mathbf{g}$ is the gravitational acceleration vector. We here assume that density changes are due to temperature and are small enough to be consistent with an assumption of incompressibility (the Boussinesq approximation). We can rescale this equation of motion using units for length, time, temperature and mass that are specific to the problem and, in this way, obtain a scale-independent form:
#
# $$
# \frac{1}{\mathrm{Pr}} \frac{\partial \mathbf{u}}{\partial t} + \nabla^2 \mathbf{u} -\nabla p = \mathrm{Ra} T' \hat{\mathbf{g}}
# $$
#

# where we have assumed that buoyancy forces on the right hand side are due to temperature variations, and the two dimensionless numbers, $\mathrm{Ra}$ and $\mathrm{Pr}$ are measures of the importance of buoyancy forcing and intertial terms, respectively.
#
# $$
# \mathrm{Ra} = \frac{g\rho_0 \alpha \Delta T d^3}{\kappa \eta}
# \quad \textrm{and} \quad
# \mathrm{Pr} = \frac{\eta}{\rho \kappa}
# $$
#
# Here $\alpha$ is the thermal expansivity, $\Delta T$ is the range of the temperature variation, $d$ is the typical length scale over which temperature varies, $\kappa$ is the thermal diffusivity ( $\kappa = k / \rho_0 C_p$; $k$ is thermal conductivity, and $C_p$ is heat capacity).

# If we assume that the Prandtl number is large, then the inertial terms will not contribute significantly to the balance of forces in the equation of motion because we have rescaled the equations so that the velocity and pressure gradient terms are of order 1. This assumption eliminates the time dependent terms in the equations and tells us that the flow velocity and pressure field are always in equilibrium with the pattern of density variations and this also tells us that we can evaluate the flow without needing to know the history or origin of the buoyancy forces. When the viscosity is independent of velocity and dynamic pressure, the velocity and pressure scale proportionally with $\mathrm{Ra}$ but the flow pattern itself is unchanged.
#
# The scaling that we use for the non-dimensionalisation is as follows:
#
# $$
#     x = d x', \quad t = \frac{d^2}{\kappa} t', \quad T=\Delta T T', \quad
#     p = p_0 + \frac{\eta \kappa}{d^2} p'
# $$
#
# where the stress (pressure) scaling using viscosity ($\eta$) determines how the mass scales. In the above, $d$ is the radius of the inner core, a typical length scale for the problem, $\Delta T$ is the order-of-magnitude range of the temperature variation from our observations, and $\kappa$ is thermal diffusivity. The scaled velocity is obtained as $v = \kappa / d v'$.

# ## Formulation & model
#
#
# The model consists of a spherical ball divided into an unstructured tetrahedral mesh of quadratic velocity, linear pressure elements with a free slip upper boundary and with a buoyancy force pre-defined :
#
# $$
# T(r,\theta,\phi) =  T_\textrm{TM}(\theta, \phi) \cdot r  \sin(\pi r)
# $$

# ## Computational script in python

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# +
import petsc4py
from petsc4py import PETSc
import mpi4py
import os

os.environ["UW_TIMING_ENABLE"] = "1"

import underworld3 as uw
import numpy as np
import sympy

if uw.mpi.size == 1:
    os.makedirs("output", exist_ok=True)
else:
    os.makedirs(f"output_np{uw.mpi.size}", exist_ok=True)


# +
# Define the problem size
#      1 - ultra low res for automatic checking
#      2 - low res problem to play with this notebook
#      3 - medium resolution (be prepared to wait)
#      4 - highest resolution (benchmark case from Spiegelman et al)

problem_size = uw.options.getInt("problem_size", default=4)
grid_refinement = uw.options.getInt("grid_refinement", default=0)
grid_simplex = uw.options.getBool("simplex", default=True)


# +
visuals = 1
output_dir = "output"

# Some gmsh issues, so we'll use a pre-built one
r_o = 1.0
r_i = 0.547

Rayleigh = 1.0e6  # Doesn't actually matter to the solution pattern,

# +
if problem_size <= 1:
    els = 3
elif problem_size == 2:
    els = 6
elif problem_size == 3:
    els = 12
elif problem_size == 4:
    els = 24
elif problem_size == 5:  # Pretty extreme to mesh this on proc0
    els = 48
elif problem_size >= 6:  # should consider refinement (or prebuild)
    els = 96

cell_size = 1/els
res = cell_size

expt_name = f"Stokes_Spherical_Cap_free_slip_{els}"

from underworld3 import timing

timing.reset()
timing.start()

# +

meshball = uw.meshing.RegionalSphericalBox(
        radiusInner=r_i,
        radiusOuter=r_o,
        numElements=els,
        refinement=grid_refinement,
        qdegree=2,
        simplex=grid_simplex,
    )

meshball.dm.view()
# -
stokes = uw.systems.Stokes(
    meshball,
    verbose=False,
    solver_name="stokes",
)


v_soln = stokes.Unknowns.u
p_soln = stokes.Unknowns.p

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1
stokes.penalty = 1.0

# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre
# of the sphere to (say) 1 at the surface

# Some useful coordinate stuff

x, y, z = meshball.CoordinateSystem.N
ra, l1, l2 = meshball.CoordinateSystem.R

## Mesh Variables for T ## 

radius_fn = sympy.sqrt(
    meshball.rvec.dot(meshball.rvec)
)  # normalise by outer radius if not 1.0

unit_rvec = meshball.X / (radius_fn)
gravity_fn = radius_fn

## Buoyancy (T) field

t_soln = uw.discretisation.MeshVariable(r"\Delta T", meshball, 1, degree=2)

t_forcing_fn = 1.0 * (
    sympy.exp(-10.0 * (x**2 + (y - 0.8) ** 2 + z**2))
    + sympy.exp(-10.0 * ((x - 0.8) ** 2 + y**2 + z**2))
    + sympy.exp(-10.0 * (x**2 + y**2 + (z + 0.8) ** 2))
)

with meshball.access(t_soln):
    t_soln.data[...] = uw.function.evaluate(
        t_forcing_fn, t_soln.coords, meshball.N
    ).reshape(-1, 1)


# +
# Stokes settings

stokes.tolerance = 1.0e-3
stokes.petsc_options["ksp_monitor"] = None

# stokes.petsc_options["ksp_type"] = "fgmres"

# # stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
# stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
# stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")

# stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"

# stokes.petsc_options[f"fieldsplit_velocity_ksp_type"] = "fcg"
# stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
# stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_max_it"] = 5
# stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

# # gasm is super-fast ... but mg seems to be bulletproof
# # gamg is toughest wrt viscosity

# stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "additive")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# # # mg, multiplicative - very robust ... similar to gamg, additive

# stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")
# thermal buoyancy force

Gamma = meshball.Gamma
stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) *  Gamma, "Upper")
stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) *  Gamma, "Lower")
stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) *  Gamma, "North")
stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) *  Gamma, "East")
stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) *  Gamma, "South")
stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) *  Gamma, "West")

stokes.bodyforce = unit_rvec * Rayleigh * gravity_fn * t_forcing_fn 

# +
timing.reset()
timing.start()

stokes.solve(zero_init_guess=True)
# -
timing.print_table()

# +
outdir="output"

meshball.write_timestep(
    expt_name,
    meshUpdates=True,
    meshVars=[p_soln, v_soln],
    outputPath=outdir,
    index=0,
)



# +
# OR
# check the mesh if in a notebook / serial

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

    points = np.zeros((meshball._centroids.shape[0], 3))
    points[:, 0] = meshball._centroids[:, 0]
    points[:, 1] = meshball._centroids[:, 1]
    points[:, 2] = meshball._centroids[:, 2]

    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="V", 
        integration_direction="forward", 
        integrator_type=45,
        surface_streamlines=False,
        initial_step_length=0.01,
        max_time=0.25,
        max_steps=1000
    )

    pl = pv.Plotter(window_size=[1000, 750])
    pl.add_axes()

    pl.add_mesh(
        clipped,
        # cmap="RdGy_r",
        cmap="coolwarm",
        edge_color="Black",
        show_edges=False,
        scalars="T",
        use_transparency=False,
        show_scalar_bar = False,
        opacity=1,
    )

    # pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T",
    #               use_transparency=False, opacity=1.0)

    
    pl.add_mesh(pvstream)

    arrows = pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], 
                           show_scalar_bar = False, 
                           mag=10/Rayleigh, )

    # pl.screenshot(filename="sphere.png", window_size=(1000, 1000), return_img=False)
    # OR
    pl.show(cpos="xy")

# -

