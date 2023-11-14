# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()
# + [markdown] magic_args="[markdown]"
# # Periodic Mesh Example (WIP)
#
# This is a periodic, Cartesian mesh with the periodic bcs specified using gmsh itself.
# Compare this to the Cylindrical Stokes example that has periodic coordinates in a mesh
# that is continuously connected.

# + [markdown] magic_args="[markdown]"
# ## Generate Periodic mesh using GMSH
# -

# %%
import gmsh

gmsh.initialize()

# %%
gmsh.model.add("Periodic x")

# %%
minCoords = (0.0, 0.0)
maxCoords = (1.0, 1.0)
cellSize = 0.1

# %%
boundaries = {
    "Bottom": 1,
    "Top": 2,
    "Right": 3,
    "Left": 4,
}

# %%
xmin, ymin = minCoords
xmax, ymax = maxCoords

# %%
p1 = gmsh.model.geo.add_point(xmin, ymin, 0.0, meshSize=cellSize)
p2 = gmsh.model.geo.add_point(xmax, ymin, 0.0, meshSize=cellSize)
p3 = gmsh.model.geo.add_point(xmin, ymax, 0.0, meshSize=cellSize)
p4 = gmsh.model.geo.add_point(xmax, ymax, 0.0, meshSize=cellSize)

l1 = gmsh.model.geo.add_line(p1, p2, tag=boundaries["Bottom"])
l2 = gmsh.model.geo.add_line(p2, p4, tag=boundaries["Right"])
l3 = gmsh.model.geo.add_line(p4, p3, tag=boundaries["Top"])
l4 = gmsh.model.geo.add_line(p3, p1, tag=boundaries["Left"])

cl = gmsh.model.geo.add_curve_loop((l1, l2, l3, l4))
surface = gmsh.model.geo.add_plane_surface([cl])

# %%
gmsh.model.geo.synchronize()

# %%
translation = [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

# %%
gmsh.model.mesh.setPeriodic(1, [boundaries["Right"]], [boundaries["Left"]], translation)

# %%
# Add Physical groups
for name, tag in boundaries.items():
    gmsh.model.add_physical_group(1, [tag], tag)
    gmsh.model.set_physical_name(1, tag, name)

gmsh.model.addPhysicalGroup(2, [surface], surface)
gmsh.model.setPhysicalName(2, surface, "Elements")

# %%
gmsh.model.mesh.generate(2)
gmsh.write("tmp_periodicx.msh")
gmsh.finalize()

# + [markdown] magic_args="[markdown]"
# ## Import Mesh into PETSc
# -

# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np

options = PETSc.Options()


# %%
plex = PETSc.DMPlex().createFromFile("tmp_periodicx.msh")

# %%
for name, tag in boundaries.items():
    plex.createLabel(name)
    label = plex.getLabel(name)
    indexSet = plex.getStratumIS("Face Sets", tag)
    if indexSet:
        label.insertIS(indexSet, 1)
    else:
        plex.removeLabel(name)

plex.removeLabel("Face Sets")

# %%
plex.view()

# %%
from underworld3.discretisation import Mesh

# %%
mesh = Mesh(plex, degree=1)

# %%
mesh.dm.view()

# %%
swarm = uw.swarm.Swarm(mesh=mesh)
material = uw.swarm.IndexSwarmVariable("M", swarm, indices=2, proxy_degree=1)
swarm.populate(fill_param=3)

# Create Stokes object

v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1

# No slip boundary conditions
stokes.add_dirichlet_bc((0.5, 0.0), "Top", (0, 1))
stokes.add_dirichlet_bc((-0.5, 0.0), "Bottom", (0, 1))

# %%
# Write density into a variable for saving
densvar = uw.discretisation.MeshVariable("density", mesh, 1)
with mesh.access(densvar):
    densvar.data[:, 0] = 1.0

# %%
swarm.dm.getCoordinates().array

# %%
# body force
import sympy

x, y = mesh.X

unit_rvec = mesh.rvec / sympy.sqrt(mesh.rvec.dot(mesh.rvec))
stokes.bodyforce = 0 * mesh.X  # -mesh.X / sympy.sqrt(x**2 + y**2)

# %%
# Solve time
stokes.solve()

mesh.petsc_save_checkpoint(index=0, meshVars=[stokes.u, stokes.p, densvar], outputPath='./output/')
swarm.petsc_save_checkpoint(swarmName='swarm', index=0, outputPath='./output/')

# check if that works

if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh("tmp_periodicx.msh")
    # pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v.sym)

    velocity_points = vis.meshVariable_to_pv_cloud(v)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v.sym)
    
    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_mesh(pvmesh, "Black", "wireframe")
    # pl.add_arrows(pvmesh.points, pvmesh.point_data["V"], mag=5.0e-1, opacity=0.5)
    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=5.0e-1, opacity=0.5)
    
    pl.show(cpos="xy")

if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh("tmp_periodicx.msh")
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v.sym)

    # velocity_points = vis.meshVariable_to_pv_cloud(v)
    # velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v.sym)
    
    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_mesh(pvmesh, "Black", "wireframe")
    pl.add_arrows(pvmesh.points, pvmesh.point_data["V"], mag=5.0e-1, opacity=0.5)
    # pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=5.0e-1, opacity=0.5)
    
    pl.show(cpos="xy")


