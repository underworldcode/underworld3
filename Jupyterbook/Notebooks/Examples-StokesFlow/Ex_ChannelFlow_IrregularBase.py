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



elements = 7
resolution = 1/elements

outputPath = f"./output/ChannelFlow3D"
expt_name = f"WigglyBottom_{elements}"

os.makedirs(outputPath, exist_ok=True)



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
    All_Boundaries = 1001 # Petsc Boundary Label

import gmsh
import math
import sys

gmsh.initialize()
gmsh.option.setNumber('General.Verbosity', 1)

gmsh.model.add("terrain")

# create the terrain surface from N x N input data points (here simulated using
# a simple function):
N = 200
coords = []  # x, y, z coordinates of all the points
nodes = []  # tags of corresponding nodes
tris = []  # connectivities (node tags) of triangle elements
lin = [[], [], [], []]  # connectivities of boundary line elements


def tag(i, j):
    return (N + 1) * i + j + 1

for i in range(N + 1):
    X = float(i) / N
    for j in range(N + 1):
        Y = float(j) / N
        
        nodes.append(tag(i, j))
        coords.extend([
            2 * X,
            Y, 
            0.05 * math.sin(20 * (X + 0.1 * Y)) * math.cos(sympy.pi * Y) - 0.1 * math.sin(sympy.pi * Y)
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

p5 = gmsh.model.geo.addPoint(0, 0, 0.5)
p6 = gmsh.model.geo.addPoint(2, 0, 0.5)
p7 = gmsh.model.geo.addPoint(2, 1, 0.5)
p8 = gmsh.model.geo.addPoint(0, 1, 0.5)

c5 = gmsh.model.geo.addLine(p5, p6)
c6 = gmsh.model.geo.addLine(p6, p7)
c7 = gmsh.model.geo.addLine(p7, p8)
c8 = gmsh.model.geo.addLine(p8, p5)

c14 = gmsh.model.geo.addLine(1, p5)
c15 = gmsh.model.geo.addLine(2, p6)
c16 = gmsh.model.geo.addLine(3, p7)
c17 = gmsh.model.geo.addLine(4, p8)

# bottom and top
# ll1 = gmsh.model.geo.addCurveLoop([c1, c2, c3, c4])
# s1 = gmsh.model.geo.addPlaneSurface([ll1])

ll2 = gmsh.model.geo.addCurveLoop([c5, c6, c7, c8])
s2 = gmsh.model.geo.addPlaneSurface([ll2])

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

gmsh.option.setNumber('Mesh.MeshSizeMin', resolution)
gmsh.option.setNumber('Mesh.MeshSizeMax', resolution)
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
        boundaries=boundaries,
        coordinate_system_type=None,
        refinement=1,
        refinement_callback=None,
        return_coords_to_bounds=None,
    )

x,y,z = terrain_mesh.X

# +
l = terrain_mesh.dm.getLabel("Lower")
i = l.getStratumSize(2)
ii = uw.utilities.gather_data(np.array([float(i)]))

if uw.mpi.rank == 0:
    print(f"Nodes in LOWER by rank: {ii.astype(int)}", flush=True)

uw.mpi.barrier()
# -






0/0

terrain_mesh.dm.view()

# +
n_vect = uw.discretisation.MeshVariable("Gamma", terrain_mesh, vtype=uw.VarType.VECTOR,
                                        degree=2, varsymbol="{\Gamma_N}")

projection = uw.systems.Vector_Projection(terrain_mesh, n_vect)
projection.uw_function = sympy.Matrix([[0,0,0]])

GammaNorm = 1 / sympy.sqrt(terrain_mesh.Gamma.dot(terrain_mesh.Gamma))

projection.add_natural_bc(terrain_mesh.Gamma * GammaNorm, "Lower")
projection.smoothing = 1.0e-6
projection.solve(verbose=False)

# Ensure n_vect are unit vectors 
with terrain_mesh.access(n_vect):
    n_vect.data[:,:] /= np.sqrt(n_vect.data[:,0]**2 + n_vect.data[:,1]**2 + n_vect.data[:,2]**2).reshape(-1,1)

# -

with terrain_mesh.access(n_vect):
    print(n_vect.data.max(), flush=True)



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

## Free slip base (conditional)
Gamma = n_vect.sym # terrain_mesh.Gamma
bc_mask0 = sympy.Piecewise((1.0, z < -0.09), (0.0, True))
bc_mask1 = sympy.Piecewise((1.0, -0.09 < z ), (0.0, True))
bc_mask2 = sympy.Piecewise((1.0, z < 0.0 ), (0.0, True))

nbc = 10000 *  sympy.simplify( 
                bc_mask0 * Gamma.dot(v.sym) *  Gamma +
                (bc_mask1 * bc_mask2)  * v.sym 
                )

stokes.add_natural_bc(nbc, "Lower")

## Buoyancy

theta = 2 * sympy.pi / 180
stokes.bodyforce = -sympy.Matrix([[sympy.sin(theta), 0.0, 0.0*sympy.cos(theta)]])

stokes.petsc_options.setValue("ksp_monitor", None)
stokes.petsc_options.setValue("snes_monitor", None)

stokes.solve()
# -


terrain_mesh.write_timestep(
    expt_name,
    meshUpdates=True,
    meshVars=[p, v],
    outputPath=outputPath,
    index=0,
)


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

    clipped = pvmesh.clip(origin=(0.0, 0.0, -0.09), normal=(0.0, 0, 1), invert=True)
    clipped.point_data["V"] = vis.vector_fn_to_pv_points(clipped, v.sym)

    clipped2 = pvmesh.clip(origin=(0.0, 0.0, -0.05), normal=(0.0, 0, 1), invert=True)
    clipped2.point_data["V"] = vis.vector_fn_to_pv_points(clipped2, v.sym)
    
    clipped3 = pvmesh.clip(origin=(0.0, 0.0, 0.4), normal=(0.0, 0, 1), invert=False)
    clipped3.point_data["V"] = vis.vector_fn_to_pv_points(clipped3, v.sym)


    skip = 10
    points = np.zeros((terrain_mesh._centroids[::skip].shape[0], 3))
    points[:, 0] = terrain_mesh._centroids[::skip, 0]
    points[:, 1] = terrain_mesh._centroids[::skip, 1]
    points[:, 2] = terrain_mesh._centroids[::skip, 2]

    point_cloud = pv.PolyData(points[np.logical_and(points[:, 0] < 2.0, points[:, 0] > 0.0)]  )

    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="V", 
        integration_direction="forward", 
        integrator_type=45,
        surface_streamlines=False,
        initial_step_length=0.1,
        max_time=0.5,
        max_steps=1000
    )

    point_cloud2 = pv.PolyData(points[np.logical_and(points[:, 2] < 0.5, points[:, 2] > 0.45)]  )

    pvstream2 = pvmesh.streamlines_from_source(
        point_cloud2, vectors="V", 
        integration_direction="forward", 
        integrator_type=45,
        surface_streamlines=False,
        initial_step_length=0.01,
        max_time=0.5,
        max_steps=1000
    )

    pl = pv.Plotter(window_size=[1000, 1000])
    pl.add_axes()

    pl.add_mesh(pvmesh,'Grey', 'wireframe', opacity=0.1)
    pl.add_mesh(clipped,'Blue', show_edges=False, opacity=0.25)
    # pl.add_mesh(pvmesh, 'white', show_edges=True, opacity=0.5)

    #pl.add_mesh(pvstream)
    pl.add_mesh(pvstream2)


    arrows = pl.add_arrows(clipped2.points, clipped2.point_data["V"], 
                           show_scalar_bar = False, opacity=1,
                           mag=100, )
    
    # arrows = pl.add_arrows(clipped3.points, clipped3.point_data["V"], 
    #                        show_scalar_bar = False, opacity=1,
    #                        mag=33, )


    # pl.screenshot(filename="sphere.png", window_size=(1000, 1000), return_img=False)
    # OR
    
    pl.show(cpos="xy")



# -



