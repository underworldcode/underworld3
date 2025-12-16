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
# # Notebook 2: Variables
#
# We can add discrete "variables" (unknowns associated with the mesh points) to a mesh, assign values to them and build expressions that `sympy` can understand, manipulate and simplify.
#
# This notebook introduces the concept of `MeshVariables` in `Underworld3`. These are both data containers and `sympy` symbolic objects. We show you how to inspect a `meshVariable`, set the data values in the `MeshVariable` and visualise them, and build expressions that `sympy` can understand, manipulate and simplify.
#
#

# %% editable=true slideshow={"slide_type": ""}
#|  echo: false 
# This is required to fix pyvista 
# (visualisation) crashes in interactive notebooks (including on binder)

import nest_asyncio
nest_asyncio.apply()

# %% editable=true slideshow={"slide_type": ""}
#| output: false # Suppress warnings in html version

import underworld3 as uw
import numpy as np
import sympy 

# %% editable=true slideshow={"slide_type": ""}
mesh = uw.meshing.CubedSphere(
    radiusOuter=1.0,
    radiusInner=0.547,
    numElements=8,
    simplex=True,
    verbose=False,
)

x,y,z = mesh.CoordinateSystem.X
r,th,phi = mesh.CoordinateSystem.R
r_vec = mesh.CoordinateSystem.unit_e_0
th_vec =  mesh.CoordinateSystem.unit_e_1
phi_vec =  mesh.CoordinateSystem.unit_e_2


# %% [markdown]
# This example shows how we can add a scalar field with a single value associated with each mesh node, and a vector field which has quadratic interpolation (points at the nodes plus interpolating points along mesh edges). 

# %%
# mesh variable example / test

scalar_var = uw.discretisation.MeshVariable(
    varname="Radius",
    mesh=mesh, 
    vtype = uw.VarType.SCALAR,
    varsymbol=r"r"
)

vector_var = uw.discretisation.MeshVariable(
    varname="Vertical_Vec",
    mesh=mesh, 
    degree=2, #quadratic interpolation
    vtype = uw.VarType.VECTOR,
    varsymbol=r"\mathbf{v}",
)



# %% [markdown]
# To set values of the variable, we can use the `array` property and `evaluate` a function at the coordinates appropriate to fill up each variable. 
#
#

# %%
scalar_var.array[...] = uw.function.evaluate(r, scalar_var.coords)
vector_var.array[...] = uw.function.evaluate(r_vec, vector_var.coords)

# %% [markdown]
# When updating multiple variables at once, use the `synchronised_array_update()` context manager to batch the updates for better performance and ensure they are all synchronized across processors at the same time (important for parallel execution).

# %%
with uw.synchronised_array_update():
    scalar_var.array[...] = uw.function.evaluate(r, scalar_var.coords)
    vector_var.array[...] = uw.function.evaluate(r_vec, vector_var.coords)

# %% [markdown]
# Variables are like most `underworld` and `PETSc` objects - they can be examined using  their `view()` method. The information that you will see is split into the underworld representation (listed under **MeshVariable**) and the PETSc representation (listed under **FE Data** which also includes the numerical values).

# %%
scalar_var.view()

# %%

# %% editable=true slideshow={"slide_type": ""}
# Visualise it / them

import pyvista as pv
import underworld3.visualisation as vis

pvmesh = vis.mesh_to_pv_mesh(mesh)
pvmesh.point_data["z"] = vis.scalar_fn_to_pv_points(pvmesh, mesh.CoordinateSystem.X[2])
pvmesh.point_data["r"] = vis.scalar_fn_to_pv_points(pvmesh, scalar_var.sym[0])
pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, (1-scalar_var.sym[0]) * vector_var.sym)

if mesh.dim==3:
    pvmesh_clipped = pvmesh.clip( normal='z', crinkle=False,origin=(0.0,0.0,0.01))

# pvmesh.plot(show_edges=True, show_scalar_bar=False)

pl = pv.Plotter(window_size=(750, 750))

pl.add_mesh(pvmesh_clipped, 
            show_edges=True,
            scalars="z", 
            opacity=0.6,
            show_scalar_bar=False)

pl.add_arrows(pvmesh.points, 
              pvmesh.point_data["V"],
              mag=0.25,
              opacity=0.6,
              color="Black",
              show_scalar_bar=False)

pl.export_html("html5/echidna_plot.html")

# %% editable=true slideshow={"slide_type": ""}
#| fig-cap: "Interactive Image: Spherical shell mesh cut away to show radial arrows with length decreasing away from the centre."

from IPython.display import IFrame
IFrame(src="html5/echidna_plot.html", width=600, height=400)

# %%

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## More information
#
# The meshVariable code is described [**API docs** here.](https://underworldcode.github.io/underworld3/development_api/underworld3/discretisation.html#underworld3.discretisation.MeshVariable)
