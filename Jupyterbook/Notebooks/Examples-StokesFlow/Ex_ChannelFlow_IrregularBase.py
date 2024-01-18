# # 3D channel flow
#
# Potentially applicable to ice-sheet flow models

# +
import underworld3 as uw
import numpy as np
import sympy

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

import os
import sys

if uw.mpi.size == 1:
    import matplotlib.pyplot as plt 


# +
## This is adapted from terrain mesh example provided
## in the gmsh examples.

from enum import Enum

class boundaries(Enum):
    Upper = 1
    Lower = 2
    Left  = 3
    Right = 4
    Front = 5
    Back  = 6

import gmsh
import math
import sys

gmsh.initialize()
gmsh.model.add("terrain")

# create the terrain surface from N x N input data points (here simulated using
# a simple function):
N = 100
coords = []  # x, y, z coordinates of all the points
nodes = []  # tags of corresponding nodes
tris = []  # connectivities (node tags) of triangle elements
lin = [[], [], [], []]  # connectivities of boundary line elements


def tag(i, j):
    return (N + 1) * i + j + 1

for i in range(N + 1):
    for j in range(N + 1):
        nodes.append(tag(i, j))
        coords.extend([
            2 * float(i) / N,
            float(j) / N, 
            0.1 * math.sin(20 * float(i + j) / N)
        ])
        if i > 0 and j > 0:
            tris.extend([tag(i - 1, j - 1), tag(i, j - 1), tag(i - 1, j)])
            tris.extend([tag(i, j - 1), tag(i, j), tag(i - 1, j)])
        if (i == 0 or i == N) and j > 0:
            lin[3 if i == 0 else 1].extend([tag(i, j - 1), tag(i, j)])
        if (j == 0 or j == N) and i > 0:
            lin[0 if j == 0 else 2].extend([tag(i - 1, j), tag(i, j)])
pnt = [tag(0, 0), tag(N, 0), tag(N, N), tag(0, N)]  # corner points element

# create 4 corner points
gmsh.model.geo.addPoint(0, 0, coords[3 * tag(0, 0) - 1], 1)
gmsh.model.geo.addPoint(2, 0, coords[3 * tag(N, 0) - 1], 2)
gmsh.model.geo.addPoint(2, 1, coords[3 * tag(N, N) - 1], 3)
gmsh.model.geo.addPoint(0, 1, coords[3 * tag(0, N) - 1], 4)
gmsh.model.geo.synchronize()

# create 4 discrete bounding curves, with their boundary points
for i in range(4):
    gmsh.model.addDiscreteEntity(1, i + 1, [i + 1, i + 2 if i < 3 else 1])

# create one discrete surface, with its bounding curves
topo = gmsh.model.addDiscreteEntity(2, 1, [1, 2, -3, -4])

# add all the nodes on the surface (for simplicity... see below)
gmsh.model.mesh.addNodes(2, 1, nodes, coords)

# add elements on the 4 points, the 4 curves and the surface
for i in range(4):
    # type 15 for point elements:
    gmsh.model.mesh.addElementsByType(i + 1, 15, [], [pnt[i]])
    # type 1 for 2-node line elements:
    gmsh.model.mesh.addElementsByType(i + 1, 1, [], lin[i])
# type 2 for 3-node triangle elements:
gmsh.model.mesh.addElementsByType(1, 2, [], tris)

# reclassify the nodes on the curves and the points (since we put them all on
# the surface before for simplicity)
gmsh.model.mesh.reclassifyNodes()

# note that for more complicated meshes, e.g. for on input unstructured STL, we
# could use gmsh.model.mesh.classifySurfaces() to automatically create the
# discrete entities and the topology; but we would have to extract the
# boundaries afterwards

# create a geometry for the discrete curves and surfaces, so that we can remesh
# them

gmsh.model.mesh.createGeometry()

# create other CAD entities to form one volume below the terrain surface, and
# one volume on top; beware that only built-in CAD entities can be hybrid,
# i.e. have discrete entities on their boundary: OpenCASCADE does not support
# this feature
# p1 = gmsh.model.geo.addPoint(0, 0, -0.5)
# p2 = gmsh.model.geo.addPoint(1, 0, -0.5)
# p3 = gmsh.model.geo.addPoint(1, 1, -0.5)
# p4 = gmsh.model.geo.addPoint(0, 1, -0.5)

p5 = gmsh.model.geo.addPoint(0, 0, 0.5)
p6 = gmsh.model.geo.addPoint(2, 0, 0.5)
p7 = gmsh.model.geo.addPoint(2, 1, 0.5)
p8 = gmsh.model.geo.addPoint(0, 1, 0.5)

# c1 = gmsh.model.geo.addLine(p1, p2)
# c2 = gmsh.model.geo.addLine(p2, p3)
# c3 = gmsh.model.geo.addLine(p3, p4)
# c4 = gmsh.model.geo.addLine(p4, p1)

c5 = gmsh.model.geo.addLine(p5, p6)
c6 = gmsh.model.geo.addLine(p6, p7)
c7 = gmsh.model.geo.addLine(p7, p8)
c8 = gmsh.model.geo.addLine(p8, p5)

# c10 = gmsh.model.geo.addLine(p1, 1)
# c11 = gmsh.model.geo.addLine(p2, 2)
# c12 = gmsh.model.geo.addLine(p3, 3)
# c13 = gmsh.model.geo.addLine(p4, 4)

c14 = gmsh.model.geo.addLine(1, p5)
c15 = gmsh.model.geo.addLine(2, p6)
c16 = gmsh.model.geo.addLine(3, p7)
c17 = gmsh.model.geo.addLine(4, p8)

# bottom and top
# ll1 = gmsh.model.geo.addCurveLoop([c1, c2, c3, c4])
# s1 = gmsh.model.geo.addPlaneSurface([ll1])

ll2 = gmsh.model.geo.addCurveLoop([c5, c6, c7, c8])
s2 = gmsh.model.geo.addPlaneSurface([ll2])

# # lower
# ll3 = gmsh.model.geo.addCurveLoop([c1, c11, -1, -c10])
# s3 = gmsh.model.geo.addPlaneSurface([ll3])
# ll4 = gmsh.model.geo.addCurveLoop([c2, c12, -2, -c11])
# s4 = gmsh.model.geo.addPlaneSurface([ll4])
# ll5 = gmsh.model.geo.addCurveLoop([c3, c13, 3, -c12])
# s5 = gmsh.model.geo.addPlaneSurface([ll5])
# ll6 = gmsh.model.geo.addCurveLoop([c4, c10, 4, -c13])
# s6 = gmsh.model.geo.addPlaneSurface([ll6])
# sl1 = gmsh.model.geo.addSurfaceLoop([s1, s3, s4, s5, s6, 1])
# v1 = gmsh.model.geo.addVolume([sl1])

# upper
ll7 = gmsh.model.geo.addCurveLoop([c5, -c15, -1, c14])
s7 = gmsh.model.geo.addPlaneSurface([ll7])
ll8 = gmsh.model.geo.addCurveLoop([c6, -c16, -2, c15])
s8 = gmsh.model.geo.addPlaneSurface([ll8])
ll9 = gmsh.model.geo.addCurveLoop([c7, -c17, 3, c16])
s9 = gmsh.model.geo.addPlaneSurface([ll9])
ll10 = gmsh.model.geo.addCurveLoop([c8, -c14, 4, c17])
s10 = gmsh.model.geo.addPlaneSurface([ll10])

sl2 = gmsh.model.geo.addSurfaceLoop([s2, s7, s8, s9, s10, 1])
v2 = gmsh.model.geo.addVolume([sl2])

gmsh.model.geo.synchronize()

gmsh.model.addPhysicalGroup(2, [s2], boundaries.Upper.value, name=boundaries.Upper.name,)
gmsh.model.addPhysicalGroup(2, [s7], boundaries.Front.value, name=boundaries.Front.name,)
gmsh.model.addPhysicalGroup(2, [s8], boundaries.Right.value, name=boundaries.Right.name,)
gmsh.model.addPhysicalGroup(2, [s9], boundaries.Back.value, name=boundaries.Back.name,)
gmsh.model.addPhysicalGroup(2, [s10], boundaries.Left.value, name=boundaries.Left.name,)

gmsh.model.addPhysicalGroup(2, [topo], boundaries.Lower.value, name=boundaries.Lower.name,)


gmsh.model.addPhysicalGroup(3, [v2], 666666, "Elements")


# set this to True to build a fully hex mesh
#transfinite = True
transfinite = False

if transfinite:
    NN = 30
    for c in gmsh.model.getEntities(1):
        gmsh.model.mesh.setTransfiniteCurve(c[1], NN)
    for s in gmsh.model.getEntities(2):
        gmsh.model.mesh.setTransfiniteSurface(s[1])
        gmsh.model.mesh.setRecombine(s[0], s[1])
        gmsh.model.mesh.setSmoothing(s[0], s[1], 100)
    gmsh.model.mesh.setTransfiniteVolume(v2)
else:
    gmsh.option.setNumber('Mesh.MeshSizeMin', 0.075)
    gmsh.option.setNumber('Mesh.MeshSizeMax', 0.075)

gmsh.model.mesh.generate(3)
gmsh.write('.meshes/tmp_terrain.msh')

# gmsh.fltk.run()

gmsh.finalize()


# +
terrain_mesh = uw.discretisation.Mesh(
        ".meshes/tmp_terrain.msh",
        degree=1,
        qdegree=3,
        useMultipleTags=True,
        useRegions=True,
        markVertices=True,
        boundaries=None,
        coordinate_system_type=None,
        refinement=0,
        refinement_callback=None,
        return_coords_to_bounds=None,
    )

x,y,z = terrain_mesh.X

# -

terrain_mesh.dm.view()

# +
outputPath = f'./output/terrain_mesh_stokes'

if uw.mpi.rank==0:      
    os.makedirs(outputPath, exist_ok=True)

# +
stokes = uw.systems.Stokes(terrain_mesh, solver_name="stokes_terrain")
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
stokes.penalty = 1.0

v = stokes.Unknowns.u
p = stokes.Unknowns.p

stokes.add_essential_bc( [sympy.oo, 0.0, 0.0 ], "Left") 
stokes.add_essential_bc( [sympy.oo, 0.0, 0.0 ], "Right") 

stokes.add_essential_bc( [0.0, 0.0, 0.0 ], "Front") 
stokes.add_essential_bc( [0.0, 0.0, 0.0 ], "Back") 

stokes.add_essential_bc( [sympy.oo, sympy.oo, 0.0 ], "Upper") 
# stokes.add_essential_bc( [0.0, 0.0, 0.0 ], "Upper") 

## Free slip base
Gamma = terrain_mesh.Gamma
stokes.add_natural_bc(10000 * Gamma.dot(v.sym) *  Gamma, "Lower")

## Buoyancy

theta = 2 * sympy.pi / 180
stokes.bodyforce = -sympy.Matrix([[sympy.sin(theta), 0.0, 0.0*sympy.cos(theta)]])
# -


Gamma.dot(v.sym)

# +
# Stokes settings

stokes.tolerance = 1.0e-3
stokes.petsc_options["ksp_monitor"] = None
stokes.petsc_options["snes_monitor"] = None

stokes.petsc_options["snes_type"] = "newtonls"
stokes.petsc_options["ksp_type"] = "fgmres"

# stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
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

# # # mg, multiplicative - very robust ... similar to gamg, additive

# stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

stokes.solve()

# +
## Visualise the mesh

# OR
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(terrain_mesh)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v.sym)

    velocity_points = vis.meshVariable_to_pv_cloud(v)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v.sym)

    clipped = pvmesh.clip(origin=(0.0, 0.0, 0.1), normal=(0.0, 0, 1), invert=False)

    points = np.zeros((terrain_mesh._centroids.shape[0], 3))
    points[:, 0] = terrain_mesh._centroids[:, 0]
    points[:, 1] = terrain_mesh._centroids[:, 1]
    points[:, 2] = terrain_mesh._centroids[:, 2]

    point_cloud = pv.PolyData(points[np.logical_and(points[:, 2] < 0.125, points[:, 2] > 0.075)]  )

    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="V", 
        integration_direction="forward", 
        integrator_type=45,
        surface_streamlines=False,
        initial_step_length=0.01,
        max_time=1.0,
        max_steps=1000
    )

    point_cloud2 = pv.PolyData(points[np.logical_and(points[:, 2] < 0.5, points[:, 2] > 0.45)]  )

    pvstream2 = pvmesh.streamlines_from_source(
        point_cloud2, vectors="V", 
        integration_direction="forward", 
        integrator_type=45,
        surface_streamlines=False,
        initial_step_length=0.01,
        max_time=1.0,
        max_steps=1000
    )

    pl = pv.Plotter(window_size=[1000, 1000])
    pl.add_axes()

    pl.add_mesh(pvmesh,'Grey', 'wireframe', opacity=0.25)
    # pl.add_mesh(clipped,'white', show_edges=False, opacity=0.75)
    # pl.add_mesh(pvmesh, 'white', show_edges=True, opacity=0.5)

    # pl.add_mesh(pvstream)
    pl.add_mesh(pvstream2)


    arrows = pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], 
                           show_scalar_bar = False, opacity=1
                           ,
                           mag=100, )


    # pl.screenshot(filename="sphere.png", window_size=(1000, 1000), return_img=False)
    # OR
    
    pl.show(cpos="xy")



# -


