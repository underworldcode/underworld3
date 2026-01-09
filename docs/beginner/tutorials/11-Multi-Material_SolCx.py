# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notebook 9: Multi-Material Constitutive Models
#
# **PHYSICS:** fluid_mechanics  
# **DIFFICULTY:** intermediate  
# **PURPOSE:** demonstration
#
# ## Description
#
# This notebook demonstrates the new multi-material constitutive model by recreating the classic SolCx benchmark using two different materials instead of a piecewise viscosity function.
#
# **Key Features:**
# - Multi-material constitutive model with level-set averaging
# - Index swarm variable for material tracking
# - Comparison with analytical SolCx solution
# - Simple setup suitable for quickstart examples
#
# **Physical Setup:**
# - Two materials: low viscosity ($\eta=1$) and high viscosity ($\eta=10^6$)
# - Material boundary at x = 0.5
# - SolCx harmonic forcing: $f_y = -\cos(\pi x) \sin(2\pi y)$
# - Free slip boundary conditions
#
# ## Mathematical Foundation
#
# The multi-material model uses level-set weighted flux averaging:
# $$\mathbf{f}_{\text{composite}}(\mathbf{x}) = \sum_{i=0}^{N-1} \phi_i(\mathbf{x}) \cdot \mathbf{f}_i(\mathbf{x})$$
#
# where $\phi_i(\mathbf{x})$ are level-set functions from IndexSwarmVariable.

# %% [markdown]
# ## Setup and Imports

# %%
import petsc4py
from petsc4py import PETSc
import nest_asyncio
nest_asyncio.apply()

import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
import sympy

# %%
uw.doctor()

# %% [markdown]
# ## Mesh and Variables Setup
#
# We create a simple structured mesh and define velocity and pressure fields.

# %%
# Create mesh - same as original SolCx
n_els = 6
refinement = 1

mesh = uw.meshing.UnstructuredSimplexBox(
    regular=True,
    minCoords=(0.0, 0.0), 
    maxCoords=(1.0, 1.0), 
    cellSize=1 / n_els, 
    qdegree=3, 
    refinement=refinement
)

x, y = mesh.X

print(f"Mesh created with {mesh.X.coords.shape[0]} nodes")

# %%
# Mesh variables for velocity and pressure
v = uw.discretisation.MeshVariable("V", mesh, vtype=uw.VarType.VECTOR, degree=2, varsymbol=r"{v}")
p = uw.discretisation.MeshVariable("P", mesh, vtype=uw.VarType.SCALAR, degree=1, continuous=False, varsymbol=r"{p}")


# %% [markdown]
# ## Material Setup with Index Swarm
#
# Instead of using a piecewise viscosity function, we create a particle swarm with two materials tracked by an IndexSwarmVariable.

# %%
# Create swarm for material tracking
swarm = uw.swarm.Swarm(mesh=mesh)
material_var = uw.swarm.IndexSwarmVariable(r"\cal{M}", swarm, indices=2, proxy_degree=1)

# Populate swarm with particles
swarm.populate(fill_param=3)

print(f"Swarm created with {swarm.data.shape[0]} particles")

# %%
# Assign materials based on x-coordinate 
x_c = 0.5

material_indices = np.where(swarm.data[:, 0] > x_c, 1, 0)
material_var.data[:, 0] = material_indices.reshape(-1)

print(f"Material 0 (low viscosity): {np.sum(material_indices == 0)} particles")
print(f"Material 1 (high viscosity): {np.sum(material_indices == 1)} particles")

# %% [markdown]
# ## Multi-Material Constitutive Model
#
# Now we create individual constitutive models for each material and combine them using the MultiMaterialConstitutiveModel.

# %%
# Create Stokes solver first to get unknowns
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p, verbose=False)

# Create individual constitutive models for each material
# Material 0: Low viscosity (η = 1.0)
viscous_material_0 = uw.constitutive_models.ViscousFlowModel(stokes.Unknowns)
viscous_material_0.Parameters.shear_viscosity_0.rename(r"\eta_{\mathrm{w}}")
viscous_material_0.Parameters.shear_viscosity_0 = 1.0

# Material 1: High viscosity (η = 1.0e6)  
viscous_material_1 = uw.constitutive_models.ViscousFlowModel(stokes.Unknowns)
viscous_material_1.Parameters.shear_viscosity_0 = 1.0e6
viscous_material_1.Parameters.shear_viscosity_0.rename(r"\eta_{\mathrm{s}}")

print(f"Material 0 viscosity: {viscous_material_0.Parameters.shear_viscosity_0.sym}")
print(f"Material 1 viscosity: {viscous_material_1.Parameters.shear_viscosity_0.sym}")

# %%
viscous_material_0.Parameters.shear_viscosity_0

# %%
# Create multi-material constitutive model
multi_material_model = uw.MultiMaterialConstitutiveModel(
    unknowns=stokes.Unknowns,
    material_swarmVariable=material_var,
    constitutive_models=[viscous_material_0, viscous_material_1]
)

# Assign to Stokes solver
stokes.constitutive_model = multi_material_model

print("Multi-material constitutive model created successfully")
multi_material_model.flux

# %% [markdown]
# ## Boundary Conditions and Forcing
#
# We apply the same boundary conditions and body force as the original SolCx benchmark.

# %%
# Set up SolCx body force: f_y = -cos(πx)sin(2πy)
stokes.bodyforce = sympy.Matrix([
    0, 
    -sympy.cos(sympy.pi * x) * sympy.sin(2 * sympy.pi * y)
])

print(f"Body force: {stokes.bodyforce.sym}")

# %%
# Free slip boundary conditions
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Top")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
stokes.add_dirichlet_bc((0.0, sympy.oo), "Left")
stokes.add_dirichlet_bc((0.0, sympy.oo), "Right")

print("Free slip boundary conditions applied")

# %%
stokes.view()

# %% editable=true slideshow={"slide_type": ""}
uw.pause("Stokes setup complete", explanation="Run next cell to continue with plasticity")

# %%
# Solver settings
stokes.tolerance = 1e-3
stokes.petsc_options["snes_monitor"] = None
stokes.petsc_options["ksp_monitor"] = None

# Enhanced solver options for high viscosity contrast
stokes.petsc_options["snes_type"] = "newtonls"
stokes.petsc_options["ksp_type"] = "fgmres"
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")
stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")

print("Boundary conditions and solver options configured")

# %% [markdown]
# ## Solve the Multi-Material Stokes Problem

# %%
print("Solving multi-material Stokes problem...")
stokes.solve(verbose=False)
print("✅ Multi-material solve completed successfully!")

# Compute some diagnostics
v_rms = np.sqrt(np.mean(v.data**2))
p_range = np.max(p.data) - np.min(p.data)

print(f"Velocity RMS: {v_rms:.6e}")
print(f"Pressure range: {p_range:.6e}")

# %% [markdown]
# ## Comparison with Original SolCx
#
# Let's compare our multi-material result with the original piecewise viscosity approach.

# %%
# Create reference solution with original piecewise viscosity
print("\nCreating reference solution with piecewise viscosity...")

# Create a second Stokes solver for comparison
v_ref = uw.discretisation.MeshVariable("V_ref", mesh, vtype=uw.VarType.VECTOR, degree=2, varsymbol=r"v_r")
p_ref = uw.discretisation.MeshVariable("P_ref", mesh, vtype=uw.VarType.SCALAR, degree=1, continuous=False, varsymbol=r"p_r")

stokes_ref = uw.systems.Stokes(mesh, velocityField=v_ref, pressureField=p_ref, verbose=False)
stokes_ref.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes_ref.constitutive_model.Parameters.shear_viscosity_0 = sympy.Piecewise(
    (1.0e6, x > x_c),
    (1.0, True)
)

print(f"Reference viscosity: {stokes_ref.constitutive_model.Parameters.shear_viscosity_0.sym}")

# %%
# Same boundary conditions and forcing
stokes_ref.bodyforce = stokes.bodyforce
stokes_ref.add_dirichlet_bc((sympy.oo, 0.0), "Top")
stokes_ref.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
stokes_ref.add_dirichlet_bc((0.0, sympy.oo), "Left")
stokes_ref.add_dirichlet_bc((0.0, sympy.oo), "Right")
stokes_ref.tolerance = 1e-3

stokes_ref.saddle_preconditioner = 1/stokes_ref.constitutive_model.Parameters.shear_viscosity_0

# Copy solver options
for key, value in stokes.petsc_options.getAll().items():
    stokes_ref.petsc_options[key] = value

# Solve reference
stokes_ref.solve()
print("✅ Reference solve completed!")

# Compute diagnostics
v_rms_ref = np.sqrt(np.mean(v_ref.data**2))
p_range_ref = np.max(p_ref.data) - np.min(p_ref.data)

print(f"Reference velocity RMS: {v_rms_ref:.6e}")
print(f"Reference pressure range: {p_range_ref:.6e}")

# %% [markdown]
# ## Results Comparison and Validation

# %%
# Compare solutions
velocity_diff = np.linalg.norm(v.data - v_ref.data)
pressure_diff = np.linalg.norm(p.data - p_ref.data) 

# Relative errors
v_ref_norm = np.linalg.norm(v_ref.data)
p_ref_norm = np.linalg.norm(p_ref.data)

rel_vel_error = velocity_diff / v_ref_norm if v_ref_norm > 0 else 0
rel_pres_error = pressure_diff / p_ref_norm if p_ref_norm > 0 else 0

print(f"\n📊 COMPARISON RESULTS:")
print(f"Velocity L2 difference: {velocity_diff:.6e}")
print(f"Pressure L2 difference: {pressure_diff:.6e}")
print(f"Relative velocity error: {rel_vel_error:.6e}")
print(f"Relative pressure error: {rel_pres_error:.6e}")

# Validation thresholds
if rel_vel_error < 1e-10 and rel_pres_error < 1e-10:
    print("✅ VALIDATION PASSED: Multi-material model matches piecewise viscosity!")
else:
    print("⚠️  Large differences detected - take a look at the solution to see if it is OK !")

# %% [markdown]
# ## Visualization

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    # Create visualization mesh
    pvmesh = vis.mesh_to_pv_mesh(mesh)
    
    # Add multi-material solution data
    pvmesh.point_data["V_multi"] = vis.vector_fn_to_pv_points(pvmesh, v.sym)
    pvmesh.point_data["P_multi"] = vis.scalar_fn_to_pv_points(pvmesh, p.sym)
    pvmesh.point_data["Vmag_multi"] = vis.scalar_fn_to_pv_points(pvmesh, v.sym.dot(v.sym))
    
    # Add reference solution data
    pvmesh.point_data["V_ref"] = vis.vector_fn_to_pv_points(pvmesh, v_ref.sym)
    pvmesh.point_data["P_ref"] = vis.scalar_fn_to_pv_points(pvmesh, p_ref.sym)
    pvmesh.point_data["Vmag_ref"] = vis.scalar_fn_to_pv_points(pvmesh, v_ref.sym.dot(v_ref.sym))
    
    # Add material distribution (level-sets)
    pvmesh.point_data["Material_0"] = vis.scalar_fn_to_pv_points(pvmesh, material_var.sym[0])
    pvmesh.point_data["Material_1"] = vis.scalar_fn_to_pv_points(pvmesh, material_var.sym[1])
    
    # Add viscosity field for comparison
    pvmesh.point_data["Viscosity"] = vis.scalar_fn_to_pv_points(pvmesh, stokes_ref.constitutive_model.Parameters.shear_viscosity_0)
    
    # Difference fields
    v_diff_field = (v.sym - v_ref.sym).dot(v.sym - v_ref.sym)
    pvmesh.point_data["V_diff"] = vis.scalar_fn_to_pv_points(pvmesh, v_diff_field)

    # Create visualization
    pl = pv.Plotter(window_size=(1500, 500), shape=(1, 3))
    
    # Plot 1: Multi-material velocity
    pl.subplot(0, 0)
    pl.add_text("Multi-Material Solution", position="upper_edge", font_size=12)
    pl.add_mesh(pvmesh, scalars="Vmag_multi", cmap="plasma", show_edges=True, 
                edge_color="white", line_width=0.5, opacity=0.8)
    pl.add_arrows(pvmesh.points, pvmesh.point_data["V_multi"], mag=200, color="Red" )
    pl.add_arrows(pvmesh.points, pvmesh.point_data["V_ref"], mag=100, color="Blue" )
    
    # Plot 2: Material distribution  
    pl.subplot(0, 1)
    pl.add_text("Material Distribution", position="upper_edge", font_size=12)
    pl.add_mesh(pvmesh, scalars="Material_0", cmap="RdBu", show_edges=True,
                edge_color="black", line_width=1.0)
    
    # Plot 3: Velocity difference
    pl.subplot(0, 2)
    pl.add_text("Velocity Difference (Multi vs Ref)", position="upper_edge", font_size=12)
    pl.add_mesh(pvmesh, scalars="V_diff", cmap="viridis", show_edges=True,
                edge_color="white", line_width=0.5, log_scale=True)
    
    pl.show()
else:
    print("Visualization skipped in parallel mode")

# %% [markdown]
# ## Summary and Conclusions
#
# This demonstration successfully shows:
#
# 1. **Multi-material constitutive model works correctly**
#    - Proper flux averaging using level-set functions
#    - Seamless integration with Stokes solver
#    - Maintains solver-authoritative stress history architecture
#
# 2. **Accurate reproduction of SolCx benchmark**  
#    - Results are similar to the piecewise viscosity but differences exist due to different representation
#    - Demonstrates the effectiveness and the limitations of this approach
#
# 3. **Simple, extensible approach**
#    - Easy to set up additional materials
#    - Clear separation between material properties and solver
#
#

# %% [markdown]
# ## Exercise 9.1
#
# Try modifying the viscosity contrast by changing the high viscosity material from 1e6 to different values (1e3, 1e9). How does this affect:
# - The convergence of the solver?
# - The velocity patterns in the solution?
# - The accuracy compared to the piecewise viscosity solution?
#
# ## Exercise 9.2
#
# Add a third material by:
# 1. Changing `indices=2` to `indices=3` in the IndexSwarmVariable
# 2. Creating a third ViscousFlowModel with intermediate viscosity (e.g., η=1e3)
# 3. Assigning material indices based on x-coordinate: material 0 for x<0.33, material 1 for 0.33<x<0.67, material 2 for x>0.67
#
# How does the three-material system compare to the two-material case?
#
# ## Exercise 9.3
#
# Instead of a sharp material boundary at x=0.5, create a transitional zone:
# 1. Use a smooth function like `tanh((x-0.5)/0.1)` to create gradual material mixing
# 2. Assign material indices based on this smooth transition
#
# Compare the results with the sharp boundary case. What are the advantages and disadvantages of each approach?
