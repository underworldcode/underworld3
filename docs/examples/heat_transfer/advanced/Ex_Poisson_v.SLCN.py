# %% [markdown]
"""
# 🎓 Poisson v.SLCN

**PHYSICS:** heat_transfer  
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
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Poisson Equation (simple)
#
# First we show how this works using the generic class and then the minor differences for
# the `Poisson` class
#
# ## Generic scalar solver class


# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# +
from petsc4py import PETSc

import os

os.environ["UW_TIMING_ENABLE"] = "1"

import underworld3 as uw
from underworld3 import timing

import numpy as np
import sympy

from IPython.display import display


# +
mesh1 = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(3.0, 1.0), 
    cellSize=1.0 / 8, 
    refinement=0)

mesh2 = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(3.0, 1.0),
    cellSize=1.0 / 8,
    regular=True,
    refinement=0)

mesh3 = uw.meshing.StructuredQuadBox(
    elementRes=(12,4),
    minCoords=(0.0, 0.0),
    maxCoords=(3.0,1.0),
    gmsh_verbosity=0,
    refinement=0)


# +
# pick a mesh
mesh = mesh2

x,y  = mesh.CoordinateSystem.X
x_vector = mesh.CoordinateSystem.unit_e_0
y_vector = mesh.CoordinateSystem.unit_e_1

# -

phi = uw.discretisation.MeshVariable("Phi", mesh, 1, degree=2, varsymbol=r"\phi")
V = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2, varsymbol=r"\mathbf{V}")
scalar = uw.discretisation.MeshVariable(
    "Theta", mesh, 1, degree=1, continuous=False, varsymbol=r"\Theta"
)

# +
init_value = 0.25 * sympy.cos(8.0 * sympy.pi * x) * sympy.sin(sympy.pi * y) + (1-y)

with mesh.access(V, phi):
    V.data[...] = 0.0
    phi.data[...] = uw.function.evaluate(init_value, phi.coords).reshape(-1, 1)

# -

# Create Poisson object

# +
poisson1 = uw.systems.Poisson(mesh, 
                             u_Field=phi)

poisson2 = uw.systems.AdvDiffusion(mesh, 
                                   u_Field=phi, 
                                   V_fn = V.sym,
                                   order = 1)
                                   
                                       
# -

poisson = poisson2

# Constitutive law (diffusivity)

poisson.constitutive_model = uw.constitutive_models.DiffusionModel
poisson.constitutive_model.Parameters.diffusivity = 1


# +
# Set some things
poisson.f = 0.0
poisson.add_dirichlet_bc(1.0, "Bottom", components=0)
poisson.add_dirichlet_bc(0.0, "Top", components=0)

poisson.tolerance = 1.0e-6
# poisson.petsc_options["snes_type"] = "newtonls"
# poisson.petsc_options["ksp_type"] = "fgmres"

# poisson.petsc_options["snes_monitor"] = None
# poisson.petsc_options["ksp_monitor"] = None
poisson.petsc_options.setValue("pc_type", "mg")
poisson.petsc_options.setValue("pc_mg_type", "multiplicative")
poisson.petsc_options.setValue("pc_mg_type", "kaskade")
# poisson.petsc_options["mg_levels"] = mesh.dm.getRefineLevel()-2
# poisson.petsc_options["mg_levels_ksp_type"] = "fgmres"
# poisson.petsc_options["mg_levels_ksp_max_it"] = 100
poisson.petsc_options["mg_levels_ksp_converged_maxits"] = None
# poisson.petsc_options["mg_coarse_pc_type"] = "svd"

# -

poisson.estimate_dt()

poisson.view()

# +
# timing.reset()
# timing.start()

# +
# %%
# Solve time
poisson.solve(timestep=poisson.estimate_dt())

time = 0.0
steps = 0

# +
for step in range(0, 50):

    dt = poisson.estimate_dt()
    
    poisson.solve( zero_init_guess=False,
                   timestep=poisson.estimate_dt())
    time += dt
    steps += 1

    print(f"{steps}: Time {time}")

    if time > 1:
        break
    

# -



mesh_numerical_soln = uw.function.evalf(poisson.u.fn, mesh.X.coords)
mesh_analytic_soln = uw.function.evalf(1.0 - mesh.N.y, mesh.X.coords)
if not np.allclose(mesh_analytic_soln, mesh_numerical_soln, rtol=0.0001):
    print("Unexpected values encountered.")


# Validate

if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)

    pvmesh.point_data["T"] = mesh_analytic_soln
    pvmesh.point_data["T2"] = mesh_numerical_soln
    pvmesh.point_data["DT"] = pvmesh.point_data["T"] - pvmesh.point_data["T2"]


    pvmesh_t = vis.meshVariable_to_pv_mesh_object(phi, alpha=None)
    pvmesh_t.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh_t, phi.sym[0])


    pl = pv.Plotter(window_size=(750, 750))

    # pl.add_mesh(
    #     pvmesh,
    #     cmap="coolwarm",
    #     edge_color="Black",
    #     show_edges=True,
    #     scalars="T2",
    #     use_transparency=False,
    #     opacity=0.5,
    #     # scalar_bar_args=sargs,
    # )


    pl.add_mesh(
        pvmesh_t,
        cmap="RdBu_r",
        edge_color="Black",
        edge_opacity=0.1,
        show_edges=True,
        scalars="T",
        use_transparency=False,
        opacity=1.0,
        # clim=[0.0,1.0],
        # show_scalar_bar=False
    )

    pl.camera_position = "xy"

    pl.show(cpos="xy")
    # pl.screenshot(filename="test.png")

# Create some arbitrary function using one of the base scalars x,y[,z] = mesh.X


