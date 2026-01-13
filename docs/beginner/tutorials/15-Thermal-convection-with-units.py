# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python (Pixi)
#     language: python
#     name: pixi-kernel-python3
# ---

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Notebook 15: Rayleigh-Bénard Convection (with physical units)
#
# <div style="float: right; width: 40%">
#     
# ![](media/AnnulusConvectionModel.png)
#
# </div>
#
#
# We'll look at a convection problem which couples Stokes Flow with time-dependent advection/diffusion to give simple Rayleigh-Bénard convection model. 
#
# $$
# -\nabla \cdot
#     \left[
#             \frac{\eta}{2}\left( \nabla \mathbf{u} + \nabla \mathbf{u}^T \right) -  p \mathbf{I} \right] =
#      -\rho_0 \alpha T \mathbf{g} 
# $$
# $$
# \nabla \cdot \mathbf{u} = 0
# $$
#
# $\eta$ is viscosity, $p$ is pressure, $\rho_0$ is a reference density, $\alpha$ is thermal expansivity, and $T$ is the temperature. Here we explicitly express density variations in terms of temperature variations.
#
# Thermal evolution is given by
# $$
# \frac{\partial T}{\partial t} - \mathbf{u}\cdot\nabla T = \kappa \nabla^2 T 
# $$
# where the velocity, $\mathbf{u}$ is the result of the Stokes flow calculation. $\kappa$ is the thermal diffusivity (compare this with Notebook 4).  
#
# The starting point is our previous notebook where we solved for Stokes
# flow in a cylindrical annulus geometry. We then add an advection-diffusion 
# solver to evolve temperature. The Stokes buoyancy force is proportional to the
# temperature anomaly, and the velocity solution is fed back into the 
# temperature advection term. The timestepping loop is written by
# hand because usually you will want to do some analysis or output some checkpoints.
#
# To read more about the applications of simple mantle convection models like this one, see (for example) Schubert et al, 2001.
#
#

# %% editable=true slideshow={"slide_type": ""}
#|  echo: false  # Hide in html version

# This is required to fix pyvista
# (visualisation) crashes in interactive notebooks (including on binder)

import nest_asyncio
nest_asyncio.apply()

# %%
#| output: false # Suppress warnings in html version

import numpy as np
import sympy
import underworld3 as uw

# %%
uw.__file__

# %%
# Step 1: Create Model FIRST
model = uw.Model()

# Step 2: Define the unit system

model.set_reference_quantities(
    length=uw.quantity(1000, "km"),
    diffusivity=uw.quantity(1e-6, "m**2/s"),
    viscosity=uw.quantity(1e21, "Pa.s"),
    temperature=uw.quantity(1000, "K"), 
    verbose=False,
    nondimensional_scaling=True, # The default !
)

uw.use_nondimensional_scaling(True)

outer_radius = uw.expression(r"r_o", uw.quantity(6370, "km"), "outer radius")
inner_radius = uw.expression(r"r_i", outer_radius * 0.55, "inner radius")
mantle_thickness = uw.expression(r"d_m", outer_radius.sym - inner_radius.sym, "mantle thickness")

velocity_phys = uw.quantity(1, "cm/year")    # Horizontal velocity
T_outer = uw.quantity(273, "K")               # Left boundary temperature (cold)
T_inner = uw.quantity(283, "K")             # Right boundary temperature (hot)

alpha = uw.expression(r"\alpha", uw.quantity(1e-5, "1/K"), "thermal expansivity")
kappa = uw.expression(r"\kappa", uw.quantity(1e-6, "m**2/s"), "thermal diffusivity")    
rho = uw.expression(r"\rho", uw.quantity(3000, "kg/m**3"), "density")    
eta_0 = uw.expression(r"\eta_0", uw.quantity(7e21, "Pa.s"), "mantle viscosity")    
gravity = uw.expression(r"g", uw.quantity(10, "m/s**2"), "gravitational acceleration")
deltaT = uw.expression(r"\Delta T", T_inner - T_outer, "temperature drop")


# %%
rayleigh_number = gravity * rho * alpha * deltaT * mantle_thickness**3 / (kappa * eta_0)

# %%
rayleigh_number

# %%
mantle_thickness = outer_radius - inner_radius

res = 5

meshball = uw.meshing.Annulus(
    radiusOuter=outer_radius,
    radiusInner=inner_radius,
    cellSize= mantle_thickness / res,
    qdegree=3,
)

# Coordinate directions etc
x, y = meshball.CoordinateSystem.X
r, th = meshball.CoordinateSystem.R
unit_rvec = meshball.CoordinateSystem.unit_e_0

# Orientation of surface normals
Gamma_N = meshball.Gamma / sympy.sqrt(meshball.Gamma.dot(meshball.Gamma))
Gamma_N = unit_rvec

# %%
Gamma_N

# %%
uw.function.evaluate(rayleigh_number, meshball.X.coords[0]).squeeze()

# %%
# Mesh variables for the unknowns

v_soln = uw.discretisation.MeshVariable("V0", meshball, 2, degree=2, varsymbol=r"{v_0}", units="cm/yr")
p_soln = uw.discretisation.MeshVariable("p", meshball, 1, degree=1, continuous=False, units="MPa")
t_soln = uw.discretisation.MeshVariable("T", meshball, 1, degree=3, continuous=True, units="K")
t_0    = uw.discretisation.MeshVariable("T0", meshball, 1, degree=3, continuous=True, units="K")

# %% [markdown]
# ### Create linked solvers
#
# We create the Stokes solver as we did in the previous notebook. 
# The buoyancy force is proportional to the temperature anomaly
# (`t_soln`). Solvers can either be provided with unknowns as pre-defined
# meshVariables, or they will define their own. When solvers are coupled,
# explicitly defining unknowns makes everything clearer.
#
# The advection-diffusion solver evolved `t_soln` using the Stokes
# velocity `v_soln` in the fluid-transport term. 
#
# ### Curved, free-slip boundaries
#
# In the annulus, a free slip boundary corresponds to zero radial 
# velocity. However, in this mesh, $v_r$ is not one of the unknowns
# ($\mathbf{v} = (v_x, v_y)$). We apply a non linear boundary condition that
# penalises $v_r$ on the boundary as discussed previously in Example 5. 

# %%
stokes = uw.systems.Stokes(
    meshball,
    velocityField=v_soln,
    pressureField=p_soln,
)

stokes.bodyforce = gravity * rho * alpha * t_soln * unit_rvec

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_0
stokes.tolerance = 1.0e-3

# stokes.petsc_options.setValue("ksp_monitor", None)
# stokes.petsc_options.setValue("snes_monitor", None)
stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"

penalty = 1000000 * uw.non_dimensionalise(eta_0)

stokes.add_natural_bc(penalty * Gamma_N.dot(v_soln) * Gamma_N, "Upper")
stokes.add_natural_bc(penalty * Gamma_N.dot(v_soln) * Gamma_N, "Lower")
# stokes.add_dirichlet_bc((uw.quantity(0,"mm/Myr"),uw.quantity(0,"mm/Myr")), "Lower")


# %%
# Create solver for the energy equation (Advection-Diffusion of temperature)

adv_diff = uw.systems.AdvDiffusion(
    meshball,
    u_Field=t_soln,
    V_fn=v_soln,
    order=2,
    verbose=False,
)

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = kappa

## Boundary conditions for this solver

adv_diff.add_dirichlet_bc(T_inner, "Lower")
adv_diff.add_dirichlet_bc(T_outer, "Upper")

adv_diff.petsc_options.setValue("snes_rtol", 0.001)
adv_diff.petsc_options.setValue("ksp_rtol", 0.0001)
# adv_diff.petsc_options.setValue("snes_monitor", None)
# adv_diff.petsc_options.setValue("ksp_monitor", None)

# %% [markdown]
# ### Initial condition
#
# We need to set an initial condition for the temperature field as the 
# coupled system is an initial value problem. Choose whatever works but
# remember that the boundary conditions will over-rule values you set on 
# the lower and upper boundaries.

# %%
# Initial temperature

r_prime = uw.expression(r"r'", (r - inner_radius) / (mantle_thickness), "R prime")

init_t = T_outer + deltaT * (1 - r_prime) + 0.1 * deltaT * (sympy.sin(3 * th) * sympy.cos(np.pi * r_prime)) 

t_soln.array[...] = uw.function.evaluate(init_t, t_soln.coords)
t_0.data[...] = t_soln.data[...]


# %% [markdown]
# #### Initial velocity solve
#
# The first solve allows us to determine the magnitude of the velocity field 
# and is useful to keep separated to check convergence rates etc. 
#
# For non-linear problems, we usually need an initial guess using a 
# reasonably close linear problem. 
#
# `zero_init_guess` is used to reset any information in the vector of 
# unknowns (i.e. do not use any initial information if `zero_init_guess==True`).

# %%
stokes.solve(verbose=False, debug=False, zero_init_guess=True)

# %% jupyter={"source_hidden": true}
if 0 and uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym)

    pvmesh_t = vis.meshVariable_to_pv_mesh_object(t_soln)
    pvmesh_t.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh_t, t_soln.sym)

    pvmesh_v = vis.meshVariable_to_pv_mesh_object(v_soln)
    pvmesh_v.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh_v, v_soln.sym)

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(
        pvmesh_t,
        cmap="RdBu_r",
        edge_color="Grey",
        edge_opacity=0.33,
        scalars="T",
        show_edges=True,
        use_transparency=False,
        opacity=1.0,
        show_scalar_bar=True,
    )

    pl.add_arrows(pvmesh_v.points, pvmesh_v.point_data["V"], mag=3e16 )


    # pl.show()

# %%
# Keep the initialisation separate
# so we can run the loop again in a notebook

max_steps = 25
timestep = 0
elapsed_time = uw.quantity(0, "Myr")

# %%
adv_diff.view(class_documentation=True)

# %%
delta_t = adv_diff.estimate_dt()
delta_t.to("Myr")

# %%
adv_diff.solve(timestep=delta_t , zero_init_guess=True)

# %%
# Null space ?

for step in range(0, max_steps):
    print(f"Timestep: {timestep}, dt: {delta_t.to('Myr')}, time: {elapsed_time}")
    
    stokes.solve(zero_init_guess=True)
    delta_t = 2 * adv_diff.estimate_dt()
    adv_diff.solve(timestep=delta_t, zero_init_guess=False, verbose=False)

    timestep += 1
    elapsed_time += delta_t



# %%
#0/0 

# %%
# visualise it


if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln)

    pvmesh_t = vis.meshVariable_to_pv_mesh_object(t_soln)
    pvmesh_t.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh_t, t_soln.sym)

    pvmesh_v = vis.meshVariable_to_pv_mesh_object(v_soln)
    pvmesh_v.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh_v, v_soln.sym)

    

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(
        pvmesh_t,
        cmap="RdBu_r",
        edge_color="Grey",
        edge_opacity=0.33,
        scalars="T",
        show_edges=True,
        use_transparency=False,
        opacity=1.0,
        show_scalar_bar=True,
    )

    pl.add_arrows(pvmesh_v.points, pvmesh_v.point_data["V"], mag=7e15 )


    pl.export_html("html5/annulus_convection_scaled.html")
    pl.show(cpos="xy", jupyter_backend="trame")

# %%
#| fig-cap: "Interactive Image: Convection model output"
from IPython.display import IFrame

IFrame(src="html5/annulus_convection_scaled.html", width=500, height=400)

# %% [markdown]
# ## Exercise - Null space
#
# Based on our previous notebook, can you see how to calculate and (if necessary) remove rigid-body the rotation 
# null-space from the solution ? 
#
# The use of a coarse-level singular-value decomposition for the velocity solver should help, in this case, but sometimes
# you can see that there is a rigid body rotation (look at the streamlines). It's wise to check and quantify the presence of 
# the null space.
#
# ```python
#     stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
# ```
#
# ## Exercise - Heat flux
#
# Could you calculate the radial heat flux field ? Its surface average value plotted against
# time tells you if you have reached a steady state.
#
# Hint:
#
# $$
#     Q_\textrm{surf} = \nabla T \cdot \hat{r} + T (\mathbf{v} \cdot \hat{r} )
# $$ 
#
# ```python
#     Q_surf = -meshball.vector.gradient(t_soln.sym).dot(unit_rvec) +\
#                     t_soln.sym[0] * v_soln.sym.dot(unit_rvec)
# ```
#
#
#

# %% [markdown]
# ### References
#
# Schubert, G., Turcotte, D. L., & Olson, P. (2001). Mantle Convection in the Earth and Planets (1st ed.). Cambridge University Press. https://doi.org/10.1017/CBO9780511612879
#

# %%
