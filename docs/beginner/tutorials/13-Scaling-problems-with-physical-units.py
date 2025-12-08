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

# %% [markdown]
# # Notebook 13: Non-Dimensional Scaling
#
# When working with physical problems, the range of scales can cause numerical difficulties. For example, mantle convection involves:
# - Velocities of ~10⁻⁹ m/s (tiny)
# - Viscosities of ~10²¹ Pa·s (huge)
# - Pressures of ~10⁹ Pa (large)
#
# Non-dimensional (ND) scaling transforms the problem so all quantities are order-one, improving numerical conditioning and stability.
#
# In this notebook you'll learn:
# - Setting reference quantities for automatic scaling
# - Solving Poisson and Stokes equations with ND scaling
# - Understanding what happens under the hood
# - Validating that ND and dimensional solutions match

# %%
import nest_asyncio
nest_asyncio.apply()

import os
os.environ["SYMPY_USE_CACHE"] = "no"

import underworld3 as uw
import numpy as np
import sympy

# %% [markdown]
# ## Example 1: Heat Equation with ND Scaling
#
# We'll solve a steady-state heat conduction problem with internal heat production (like radioactive heating) twice:
# 1. Without ND scaling (manual approach)
# 2. With ND scaling (automatic)
#
# Both should give the same non-dimensional values in `.data`.

# %% [markdown]
# ### Without ND Scaling (Manual approach)
#
# First, we'll solve in non-dimensional form with no units (and no reference quantities set). The units are implicit and the user has to keep track of them correctly.

# %%
# Ensure clean slate - NO reference quantities, NO ND scaling
uw.reset_default_model()
uw.use_nondimensional_scaling(False)

# Create mesh (ND coordinates: 0 to 1)
mesh_manual = uw.meshing.UnstructuredSimplexBox(cellSize=0.125)
x, y = mesh_manual.X

# Create temperature variable without units (will store ND values)
T_manual = uw.discretisation.MeshVariable("T_manual", mesh_manual, 1, degree=2, varsymbol=r"T_\textrm{man}")

# Set up Poisson solver for heat equation: ∇²T = Q
poisson_manual = uw.systems.Poisson(mesh_manual, u_Field=T_manual)
poisson_manual.constitutive_model = uw.constitutive_models.DiffusionModel
poisson_manual.constitutive_model.Parameters.diffusivity = 1.0  # ND thermal diffusivity
poisson_manual.f = 2.0  # ND heat source

# Boundary conditions (ND values - cold boundaries)
poisson_manual.add_dirichlet_bc(0.0, "Bottom")
poisson_manual.add_dirichlet_bc(0.0, "Top")

# Solve
poisson_manual.solve()

# Store ND solution
T_manual_data = np.copy(T_manual.data)

print(f"Manual ND: T_max = {T_manual.max()}")

# %% [markdown]
# ### With ND Scaling (Automatic based on reference scales)
#
# Now solve the same problem with ND scaling. We'll use simple reference scales (L₀=1 m, V₀=1 m/s) so the numbers are the same as far as the numerical solvers are concerned, but underworld will track the units for you and will help you convert from one to another.

# %%
# Reset and set up ND scaling
uw.reset_default_model()
model = uw.get_default_model()

# Set reference quantities (triggers unit requirement everywhere)
model.set_reference_quantities(
    domain_depth=uw.quantity(1, "m"),  # L₀ = 1 m
    plate_velocity=uw.quantity(1, "m/s"),  # V₀ = 1 m/s (needed for system)
    mantle_viscosity=uw.quantity(1, "Pa*s"),  # η₀ = 1 Pa·s (needed for system)
    temperature_difference=uw.quantity(1, "K")  # T₀ = 1 K
)

# Create mesh
mesh_nd = uw.meshing.UnstructuredSimplexBox(cellSize=0.125)
x, y = mesh_nd.X

# Create temperature variable WITH units
T_nd = uw.discretisation.MeshVariable("T_nd", mesh_nd, 1, degree=2, units="K", varsymbol=r"T_\textrm{ND}")

# Enable ND scaling
uw.use_nondimensional_scaling(True)

# Set up Poisson solver for heat equation
poisson_nd = uw.systems.Poisson(mesh_nd, u_Field=T_nd)
poisson_nd.constitutive_model = uw.constitutive_models.DiffusionModel
poisson_nd.constitutive_model.Parameters.diffusivity = 1.0
# Heat source WITH units: for ∇²T = Q, where T is in [K], Q must be in [K/m²]
# This represents internal heating (like radioactive decay)
# With L₀ = 1 m and T₀ = 1 K, dimensional Q = 2.0 K/m² gives ND value of 2.0
poisson_nd.f = uw.quantity(2.0, "K/m**2")

# Boundary conditions WITH units (cold boundaries at T = 0 K relative to reference)
poisson_nd.add_dirichlet_bc(uw.quantity(0.0, "K"), "Bottom")
poisson_nd.add_dirichlet_bc(uw.quantity(0.0, "K"), "Top")

# Solve
poisson_nd.solve()

# Store solution
T_nd_solution = np.copy(T_nd.data)

# T_nd.min(), T_nd.max()

# %% [markdown]
# ### Comparison
#
# The solutions should match perfectly:

# %%
# Compute difference
difference = np.max(np.abs(T_manual_data - T_nd_solution))
relative_error = difference / np.max(np.abs(T_manual_data))

difference, relative_error

# %% [markdown]
# The difference is at machine precision - as expected since we have only added unit-tracking to the problem, we haven't actually changed any values. You can see that the units are carried and the system will happily convert quantities for you. It is capable of more though.
#
# Typically, we'd like to specify real-world values for all quantities. For a planetary modelling problem, lengths will be in thousands of km, viscosities will be in zetta Pascal seconds, and stresses will be in MPa or GPa. We can use that information to set reference quantities (e.g. reference viscosity is $10^{21}$ Pa.s) and specify all viscosities relative to that value. This avoids potential computational inaccuracy associated with having large forces applied to near-rigid materials and calculating tiny responses.
#

# %% [markdown]
# ## Example 2: Stokes Flow with Realistic Scales
#
# Now let's solve a lid-driven cavity flow problem with realistic mantle convection parameters. This demonstrates why ND scaling is important.

# %% [markdown]
# ### Manual Non-Dimensionalization
#
# First, solve with manual non-dimensionalisation (the traditional approach). We work directly with ND values and, when we have finished solving, we interpret the raw results by rescaling them to represent physical values. 

# %%
# Reset - NO ND scaling, manual non-dimensionalization
uw.reset_default_model()
uw.use_nondimensional_scaling(False)

# Create mesh (ND coordinates: 0 to 1)
resolution = 8
mesh_manual = uw.meshing.UnstructuredSimplexBox(cellSize=1/resolution)

# Create variables without units (will store ND values)
v_manual = uw.discretisation.MeshVariable("v_manual", mesh_manual, 2, degree=2, varsymbol=r"v_\textrm{man}")
p_manual = uw.discretisation.MeshVariable("p_manual", mesh_manual, 1, degree=1, varsymbol=r"p_\textrm{man}")

# Set up Stokes solver with ND values
stokes_manual = uw.systems.Stokes(mesh_manual, velocityField=v_manual, pressureField=p_manual)
stokes_manual.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes_manual.constitutive_model.Parameters.viscosity = 1.0  # ND viscosity

stokes_manual.petsc_options.setValue("ksp_monitor", None)
stokes_manual.petsc_options.setValue("snes_monitor", None)

# Lid-driven cavity: top moves with ND velocity = 1.0
stokes_manual.add_dirichlet_bc((1.0, 0.0), "Top")
stokes_manual.add_dirichlet_bc((0.0, 0.0), "Bottom")
stokes_manual.add_dirichlet_bc((0.0, 0.0), "Left")  # v_x free, v_y = 0
stokes_manual.add_dirichlet_bc((0.0, 0.0), "Right")  # v_x free, v_y = 0

# %%
# Solve
stokes_manual.solve()

# Store ND solution
v_manual_data = np.copy(v_manual.data)
p_manual_data = np.copy(p_manual.data)

print(f"Manual ND: v_max = {v_manual.max()}, p_max = {p_manual.max()}")

# %%
uw.unwrap(stokes_manual.F1.sym, keep_constants=True, apply_scaling=True)

# %% [markdown]
# ### Automatic Non-Dimensionalization
#
# Now solve the same problem using automatic ND scaling. We provide dimensional values, and the system converts to ND form automatically:

# %%
# Reset and set reference quantities
uw.reset_default_model()
model = uw.get_default_model()

# Define reference scales - same ND numbers as manual case
# We want V₀ = 1.0 (in ND form), so we just use reference=1.0 in compatible units
model.set_reference_quantities(
    domain_depth=uw.quantity(1000, "km"),            # Reference length = 1 m
    plate_velocity=uw.quantity(1, "cm/yr"),        # Reference velocity = 1 m/s
    mantle_viscosity=uw.quantity(1e19, "Pa*s")      # Reference viscosity = 1 Pa·s
)

# Create mesh (unit square like manual case, but now with units)
mesh_auto = uw.meshing.UnstructuredSimplexBox(
    cellSize=uw.quantity(125, "km"),
    minCoords=(0,0),
    maxCoords=(uw.quantity(1000, "km"), uw.quantity(1000, "km"))
)

# Create variables WITH units
v_auto = uw.discretisation.MeshVariable("v_auto", mesh_auto, 2, degree=2, units="m/s", varsymbol=r"v_\textrm{auto}")
p_auto = uw.discretisation.MeshVariable("p_auto", mesh_auto, 1, degree=1, units="Pa", varsymbol=r"p_\textrm{auto}")

# Check what the scaling coefficients are
V0_val = v_auto.scaling_coefficient
P0_val = p_auto.scaling_coefficient

print(f"Scaling coefficients (with reference scales = 1.0):")
print(f"  V₀ = {V0_val:.6e} m/s")
print(f"  P₀ = {P0_val:.6e} Pa")

# %%
model.get_fundamental_scales()

# %%
# Enable ND scaling
uw.use_nondimensional_scaling(True)

# Set up Stokes solver
stokes_auto = uw.systems.Stokes(mesh_auto, velocityField=v_auto, pressureField=p_auto)
stokes_auto.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes_auto.constitutive_model.Parameters.shear_viscosity_0 = uw.quantity(1e19, uw.units("Pa.s"))  # ND viscosity (reference = 1e19 Pa·s)

# Boundary conditions with reference units: To get ND value of 1.0,
# we provide dimensional value = 10.0 m/s (which equals 1.0 × V₀ when V₀ = 10 m/s)
stokes_auto.add_dirichlet_bc((uw.quantity(1.0, "cm/yr"), uw.quantity(0.0, "cm/yr")), "Top")
stokes_auto.add_dirichlet_bc((uw.quantity(0.0, "cm/yr"), uw.quantity(0.0, "cm/yr")), "Bottom")
stokes_auto.add_dirichlet_bc((uw.quantity(0.0, "cm/yr"), uw.quantity(0.0, "cm/yr")), "Left")
stokes_auto.add_dirichlet_bc((uw.quantity(0.0, "cm/yr"), uw.quantity(0.0, "cm/yr")), "Right")

# Solve
stokes_auto.petsc_options.setValue("ksp_monitor", None)
stokes_auto.petsc_options.setValue("snes_monitor", None)
stokes_auto.solve(verbose=False)

# Store solution (will be in ND form in .data)
v_auto_data = np.copy(v_auto.data)
p_auto_data = np.copy(p_auto.data)

print(f"Automatic scaling: v_max = {np.nanmax(np.abs(v_auto_data)):.6e}, p_max = {np.nanmax(np.abs(p_auto_data)):.6e}")

# %% [markdown]
# ### Validation: Same ND Values
#
# Both approaches solve the same ND problem. Let's verify that `.data` contains identical ND values:

# %%
# Compare ND values in .data arrays
v_diff = np.max(np.abs(v_manual_data - v_auto_data))
v_rel_error = v_diff / np.max(np.abs(v_manual_data))

p_diff = np.max(np.abs(p_manual_data - p_auto_data))
p_rel_error = p_diff / (np.max(np.abs(p_manual_data)) + 1e-15)

print(f"Velocity difference: {v_diff:.3e} (relative: {v_rel_error:.3e})")
print(f"Pressure difference: {p_diff:.3e} (relative: {p_rel_error:.3e})")

v_diff, v_rel_error, p_diff, p_rel_error

# %% [markdown]
# The values of all raw values in the uw data containers are identical whether we scale by hand or automatically, but the view we have of the data in the scaled case is more flexible - it can be converted to any units of the correct dimensions. The `array` property of the variable reflects units and scaling, whereas the `data` property is a view into the raw numbers that are used by PETSc. 

# %%
v_manual.array[100:110].squeeze()

# %%
v_auto.array[100:110].squeeze()

# %%
v_manual.data[100:110].squeeze()

# %%
v_auto.data[100:110].squeeze()

# %%
uw.function.evaluate(v_auto, v_auto.coords[100:110]).squeeze()

# %%
p_manual.data[20:30]

# %%
p_auto.data[20:30]

# %%
stokes_auto.PF0.sym

# %%

# %% [markdown]
# ### Behind the scenes
#
# What just happened:
#
# **Manual ND (Traditional Approach):**
#  - User manually non-dimensionalizes the problem
#  - Sets up equations with ND values (viscosity=1.0, BC=1.0, etc.)
#  - `.data` contains ND values directly and `.array` has the same values
#  - User must track what the values mean physically
#
# **Automatic ND (Underworld3 System):**
#
#  - User provides reference quantities and units
#  - System derives scaling coefficients (V₀, P₀, etc.)
#  - User provides in physical units, the system automatically non-dimensionalises them when they are used
#  - `.data` contains the **same ND values** as manual approach, but the `.array` is in convenient units
#  - System tracks physical meaning (dimensionality of quantities) automatically
#
# Both cases give PETSc identical ND matrices to solve. The difference is whether the user or the system manages the unit tracking and scaling.
#
# This is why `.data` arrays are identical - they both contain the non-dimensional values that PETSc actually works with !
#
#
# **Note:** It would be a fair criticism to say that this is not a particularly taxing test for the reason that this is a lid driven flow with velocity entirely controlled by the surface value. We will see more complicated examples in the next Notebook.

# %% [markdown]
# ## Things to try 
#
# Exercises to explore:
#
# ```python
# # 1. Try different reference quantities
# model.set_reference_quantities(
#     domain_depth=uw.quantity(100, "km"),  # Lithosphere scale
#     plate_velocity=uw.quantity(1, "cm/year"),
#     mantle_viscosity=uw.quantity(1e23, "Pa*s")  # Lithosphere viscosity
# )
#
# # 2. Check scaling coefficients
# v.scaling_coefficient  # What is V₀?
# p.scaling_coefficient  # What is P₀?
#
# # 3. Validate with different resolutions
# # Do the solutions still match with elementRes=(16, 16)?
# ```

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh_auto)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_auto)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_auto)
    
    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Grey",
        edge_opacity=0.33,
        scalars="P",
        show_edges=True,
        use_transparency=False,
        opacity=1.0,
        show_scalar_bar=True
    )

    pl.add_arrows(pvmesh.points, pvmesh.point_data["V"]*1e15, )



    pl.show()
