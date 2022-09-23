#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # Reading a gmsh file (and checking the labels)
#
# This example is for the notch-localization test of Spiegelman et al. For which they supply a geometry file which gmsh can use to construct meshes at various resolutions. NOTE: we are just demonstrating the mesh here, not the solver configuration / benchmarking.
#
# The `.geo` file is provided and we show how to make this into a `.msh` file and 
# how to read that into a `uw.discretisation.Mesh` object. The `.geo` file has header parameters to control the mesh refinement, and we provide a coarse version and the original version. 
#
# After that, there is some cell data which we can assign to a data structure on the elements (such as a swarm).

# %%
import gmsh
import meshio

import petsc4py
from petsc4py import PETSc
import underworld3 as uw
import numpy as np
import sympy

from underworld3.cython import petsc_discretisation


# %%
mesh_res = "coarse"  # For tests
build_mesh = False

# %%
if build_mesh:
    if mesh_res == "coarse":
        gmsh.initialize()
        gmsh.model.add("Notch")
        gmsh.open("meshes/compression_mesh_rounded_coarse.geo")
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.write("meshes/notch_coarse.msh")
        gmsh.finalize()
        
    else:
        gmsh.initialize()
        gmsh.model.add("Notch")
        gmsh.open("meshes/compression_mesh_rounded_refine.geo")
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.write("meshes/notch_refine.msh")
        gmsh.finalize()


# %%
mesh1 = uw.discretisation.Mesh("meshes/notch_coarse.msh", simplex=True)
mesh1.dm.view()


# %%
# This always seems to fail in parallel

cellType = uw.discretisation.MeshVariable(r"C_c", mesh1, 1, degree=0, continuous=False)

# %%
v_soln = uw.discretisation.MeshVariable("U", mesh1, mesh1.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", mesh1, 1, degree=1)

# %% [markdown]
# This is how we extract cell data from the mesh. We can map it to the swarm data structure

# %%
plex = mesh1.dm

## Cell labels

plex.createLabel("weak")
label = plex.getLabel("weak")
indexSetW = plex.getStratumIS("Cell Sets", 0)
if indexSetW:
    label.insertIS(indexSetW, 1)
else:
    plex.removeLabel("weak")
    
plex.createLabel("strong")
label = plex.getLabel("strong")
indexSetS = plex.getStratumIS("Cell Sets", 1)
if indexSetS:
    label.insertIS(indexSetS, 1)
else:
    plex.removeLabel("strong")
    
## Boundary labels 

plex.createLabel("Left")
label = plex.getLabel("Left")
indexSetBD= plex.getStratumIS("Face Sets", 1)
if indexSetBD:
    label.insertIS(indexSetBD, 1)
else:
    plex.removeLabel("Left")

plex.createLabel("Right")
label = plex.getLabel("Right")
indexSetBD = plex.getStratumIS("Face Sets", 2)
if indexSetBD:
    label.insertIS(indexSetBD, 1)
else:
    plex.removeLabel("Right")
    
    
plex.createLabel("Bottom")
label = plex.getLabel("Bottom")
indexSetBD = plex.getStratumIS("Face Sets", 3)
if indexSetBD:
    label.insertIS(indexSetBD, 1)
else:
    plex.removeLabel("Bottom")
    
plex.createLabel("Top")
label = plex.getLabel("Top")
indexSetBD = plex.getStratumIS("Face Sets", 4)
if indexSetBD:
    label.insertIS(indexSetBD, 1)
else:
    plex.removeLabel("Top")
    


# %%
# This is very hacky ... will it work in parallel ?

with mesh1.access(cellType):
    lvec = cellType._lvec
    lvec.isset(indexSetW, 0.0)
    lvec.isset(indexSetS, 1.0)
    print(cellType.data)

# %%
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [1050, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, 1.0]

    mesh1.vtk("tmp_notch.vtk")
    pvmesh = pv.read("tmp_notch.vtk")

    pl = pv.Plotter()

    points = np.zeros((mesh1._centroids.shape[0], 3))
    points[:, 0] = mesh1._centroids[:, 0]
    points[:, 1] = mesh1._centroids[:, 1]

    point_cloud = pv.PolyData(points)
    
    with mesh1.access():
        point_cloud.point_data["M"] = cellType.data.copy()

    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, use_transparency=False, opacity=0.5,)
    pl.add_points(point_cloud, cmap="coolwarm", render_points_as_spheres=False, point_size=2, opacity=0.66)

    pl.show(cpos="xy")


# %%
# Check that this mesh can be solved for a simple, linear problem

# Create Stokes object

stokes = uw.systems.Stokes(
    mesh1,
    velocityField=v_soln,
    pressureField=p_soln,
    solver_name="stokes",
)

# Set solve options here (or remove default values
stokes.petsc_options["ksp_monitor"] = None


# Level set approach to rheology: 
viscosity_L = sympy.Piecewise((1.0, cellType.sym[0] < 0.5), (1000.0, True),)

stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh1.dim)
stokes.constitutive_model.material_properties = stokes.constitutive_model.Parameters(viscosity=viscosity_L)
stokes.saddle_preconditioner = 1 / viscosity_L
stokes.penalty = 0.0 

# Velocity boundary conditions
stokes.add_dirichlet_bc((+1.0,0), "Left", (0,1))
stokes.add_dirichlet_bc((-1.0,0), "Right", (0,1))
# stokes.add_dirichlet_bc((0.0,), "Top", (1,)) # leave top open
stokes.add_dirichlet_bc((0.0,), "Bottom", (1,))

stokes.bodyforce = sympy.Matrix([0,-1])

# %%
# And a non-linear version ?



# %%
# Check that this level-set approach is ok

stokes._setup_terms()
stokes._u_f1

# %%

# %%
stokes.solve(zero_init_guess=True)

# %%
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [1050, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, 1.0]

    mesh1.vtk("tmp_notch.vtk")
    pvmesh = pv.read("tmp_notch.vtk")

    pl = pv.Plotter()

    points = np.zeros((mesh1._centroids.shape[0], 3))
    points[:, 0] = mesh1._centroids[:, 0]
    points[:, 1] = mesh1._centroids[:, 1]
    
    with mesh1.access():
        usol = v_soln.data.copy()

    arrow_loc = np.zeros((v_soln.coords.shape[0], 3))
    arrow_loc[:, 0:2] = v_soln.coords[...]

    arrow_length = np.zeros((v_soln.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]

    pl.add_arrows(arrow_loc, arrow_length, mag=0.1, opacity=0.75)

    point_cloud = pv.PolyData(points)
    
    with mesh1.access():
        point_cloud.point_data["M"] = cellType.data.copy()

    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, use_transparency=False, opacity=0.1,)
    pl.add_points(point_cloud, cmap="coolwarm", render_points_as_spheres=False, point_size=2, opacity=0.66)

    pl.show(cpos="xy")

# %%

# %%
