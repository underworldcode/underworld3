# %% [markdown]
"""
# 🎓 Darcy 3D Loop Mesh Fault

**PHYSICS:** utilities  
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
# ### Darcy flow in 3D with Loop mesh and fault
#
# To fully understand this example, it is recommended to run and study the following examples:
#
# 1. `Ex_Darcy_1D_benchmark.py`
# 2. `Ex_Darcy_3D_flow_z_axis.py`
# 3. `Ex_Darcy_3D_flow_x_axis.py`
# 4. `Ex_Darcy_3D_Loop_Mesh.py`

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

import underworld3 as uw
import numpy as np
from enum import Enum
from petsc4py import PETSc
import sympy
from sympy import Piecewise, ceiling, Abs
import os
options = PETSc.Options()

# vis tools
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis
    import matplotlib.pyplot as plt

# importing loop meshing tools
from underworld3.utilities import create_dmplex_from_medit

# +
# create output dir
if uw.mpi.size==1:
    output_dir = './output/darcy_loop_mesh/serial/'
else:
    output_dir = './output/darcy_loop_mesh/parallel/'

if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)
# -

# loading mesh data from dmplex
medit_plex = create_dmplex_from_medit('./meshout.mesh')


# +
class boundaries_3D(Enum):
    Bottom = 395
    Top = 396
    Right = 392
    Left = 391
    Front = 393
    Back = 394
    
class boundary_normals_3D(Enum):
    Bottom = sympy.Matrix([0, 0, 1])
    Top = sympy.Matrix([0, 0, 1])
    Right = sympy.Matrix([1, 0, 0])
    Left = sympy.Matrix([1, 0, 0])
    Front = sympy.Matrix([0, 1, 0])
    Back = sympy.Matrix([0, 1, 0])


# -

# mesh length
minX, maxX = 0.0, 10.0
minY, maxY = 0.0, 10.0
minZ, maxZ = 0.0, 2.0

# creating mesh from plex
mesh = uw.meshing.Mesh(medit_plex, boundaries=boundaries_3D, boundary_normals=boundary_normals_3D)
mesh.dm.view()

# +
# mesh.view()
# -

# ### Following commented code is related to visualization in pyvista

# +
# def get_labels(plex, depth, label_name):
#     # Get all label values
#     tet_pStart, tet_pEnd = plex.getDepthStratum(depth)
#     label_values = []
#     for point in range(tet_pStart, tet_pEnd):
#         label_value = plex.getLabelValue(label_name, point)
#         if label_value is not None:
#             label_values.append(label_value)

#     return np.array(label_values)

# +
# pvmesh = vis.mesh_to_pv_mesh(mesh)
# pvmesh.cell_data['tetra_data'] = get_labels(mesh.dm, 3, 'TetraLabels') - 400

# +
# # print values in tetra data
# print(np.unique(get_labels(mesh.dm, 3, 'TetraLabels')))

# +
# # plotting mesh and cell data 
# pl = pv.Plotter() # pv.Plotter(window_size=(550, 550))
# pl.add_mesh(pvmesh, edge_color="Grey", show_edges=True, use_transparency=False, opacity=1.0, 
#             scalars='tetra_data', cmap='tab10', clim=(0, 7))
# pl.show(cpos="xy")

# +
# # print values in line data
# print(np.unique(np.unique(get_labels(mesh.dm, 1, 'LineLabels'))))

# +
# # Create a PolyData object for the edges
# points = pvmesh.points  # Get the vertices of the mesh
# edge_polydata = pv.PolyData()
# edge_polydata.points = points

# # Create lines from the edge connections
# edges = mesh_data.get('line')
# lines = np.hstack([np.array([2, edge[0], edge[1]]) for edge in edges])
# edge_polydata.lines = lines

# # Assign the edge data to the PolyData object
# edge_polydata["line_data"] = mesh_data.get('line_data')

# +
# # print values in triangle data
# print(np.unique(mesh_data.get('triangle_data')))

# +
# # Create a PolyData object for the faces
# cond = np.where(np.logical_or(mesh_data.get('triangle_data')==2, mesh_data.get('triangle_data')==4))
# # cond = np.where(mesh_data.get('triangle_data')==10)
# # cond = ...

# faces = mesh_data['triangle'][cond]
# face_polydata = pv.PolyData()
# face_polydata.points = points
# face_conn = np.hstack([np.hstack(([len(face)], face)) for face in faces])

# # Set the faces in the PolyData object
# face_polydata.faces = face_conn

# # Assign the face data to the PolyData object
# face_polydata["triangle_data"] = mesh_data.get('triangle_data')[cond]

# +
# # Mask the mesh based on cell data
# threshold_value = 6
# masked_mesh = pvmesh.threshold(value=threshold_value, scalars='tetra_data', invert=True)

# # Visualize the masked mesh
# pl = pv.Plotter()
# pl.add_mesh(masked_mesh, scalars='tetra_data', show_edges=True, cmap='tab10', clim=(0,7))
# # pl.add_mesh(edge_polydata, scalars='line_data', line_width=5, cmap='viridis', show_scalar_bar=False)
# pl.add_mesh(face_polydata, scalars='triangle_data', opacity=0.85, cmap=['darkblue', 'red'], show_scalar_bar=False)
# pl.show()

# # # Create animation
# # viewup = [-0.05, -0.25, 1]
# # pl.show(auto_close=False)
# # path = pl.generate_orbital_path(factor=2.0, n_points=36, viewup=viewup, shift=0.2)
# # pl.open_gif("mesh.gif")
# # pl.orbit_on_path(path, write_frames=True, viewup=viewup, step=0.1)
# # pl.close()

# +
# # todo: write function to separate line data into single line

# line_edge_list = []
# for line_edge in mesh_data.get('line'):
#     pt1 = mesh_data.get('points')[line_edge[0]]
#     pt2 = mesh_data.get('points')[line_edge[1]]
#     if np.isclose(pt1[1], 0) and np.isclose(pt2[1], 0) and np.isclose(pt1[2], 2) and np.isclose(pt1[2], 2):
#         # print(line_edge, pt1, pt2)
#         line_edge_list.append(line_edge)

# line_edge_arr = np.array(line_edge_list)
# -
# x and y coordinates
x, y, z = mesh.CoordinateSystem.N

# Create Darcy Solver
darcy = uw.systems.SteadyStateDarcy(mesh)
p_soln = darcy.Unknowns.u
v_soln = darcy.v

# Needs to be smaller than the contrast in properties
darcy.petsc_options["snes_rtol"] = 1.0e-6  
darcy.constitutive_model = uw.constitutive_models.DarcyFlowModel

p_soln_0 = p_soln.clone("P_no_g", r"{p_\textrm{(no g)}}")
v_soln_0 = v_soln.clone("V_no_g", r"{v_\textrm{(no g)}}")


def plot_P_V(_mesh, _p_soln, _v_soln):
    '''
    Plot pressure and velcity streamlines
    '''
    pvmesh = vis.mesh_to_pv_mesh(_mesh)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, _p_soln.sym)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, _v_soln.sym)

    velocity_points = vis.meshVariable_to_pv_cloud(_v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, _v_soln.sym)

    # point sources at cell centres
    points = np.zeros((_mesh._centroids.shape[0], 3))
    points[:, 0] = _mesh._centroids[:, 0]
    points[:, 1] = _mesh._centroids[:, 1]
    points[:, 2] = _mesh._centroids[:, 2]
    point_cloud = pv.PolyData(points[::3])

    pvstream = pvmesh.streamlines_from_source(point_cloud, vectors="V", integrator_type=45, 
                                              integration_direction="both", max_steps=1000,
                                              max_time=0.1, initial_step_length=0.001, 
                                              max_step_length=0.01)

    pl = pv.Plotter(window_size=(750, 750))
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="P",
                use_transparency=False, opacity=0.5)
    pl.add_mesh(pvstream, line_width=1.0)
    # pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.005, opacity=0.75)

    pl.show(cpos="xy")


# +
# set up two materials
interface = 2.5
k1 = 1e-1
k2 = 1e-5

# Groundwater pressure boundary condition on the left wall
max_pressure = 0.5

# The piecewise version
kFunc = Piecewise((k1, x < interface), (k2, x >= interface), (1.0, True))

# A smooth version
# darcy.constitutive_model.Parameters.permeability = kFunc
darcy.constitutive_model.Parameters.s = sympy.Matrix([0, 0, 0]).T
darcy.f = 0.0

# set up boundary conditions
darcy.add_dirichlet_bc(0.0, "Left")
darcy.add_dirichlet_bc(1.0 * maxX * max_pressure, "Right")
# -

# create cell centred mesh variable (disc and 0th order)
permeability = uw.discretisation.MeshVariable('K', mesh, 1, degree=0, continuous=False)

# setting a array to store k values
perm_arr = np.zeros_like(permeability.data)

# +
# # dealing with vertices

# pStart, pEnd = mesh.dm.getDepthStratum(0)
# pNum = pEnd-pStart

# # Get coordinates array
# coords = mesh.dm.getCoordinates().array.reshape(pNum, mesh.cdim)

# # dealing with tets
# tet_Start, tet_End = mesh.dm.getDepthStratum(3)

# # fault tetra
# fault_tetra = mesh.dm.getStratumIS("TetraLabels", 404).array

# for t in range(tet_Start, tet_End):
#     if t in fault_tetra:
#         coneclose, orient = mesh.dm.getTransitiveClosure(t)
#         if np.any(coords[coneclose[-4:]-tet_pEnd][:,1]>5):
#             perm_arr[t] = k1
#         else:
#             perm_arr[t] = 0.0       
#     else:
#         perm_arr[t] = k2

# +
comm = mesh.dm.getComm()  # PETSc communicator

# Get vertex depth range
pStart, pEnd = mesh.dm.getDepthStratum(0)
pNum = pEnd - pStart  # Number of local vertices

# Get coordinates
coord_sec = mesh.dm.getCoordinateSection()
coord_vec = mesh.dm.getCoordinatesLocal().array

# Ensure proper reshaping by getting the actual number of vertices
actual_pNum = coord_vec.shape[0] // mesh.cdim
coords = coord_vec.reshape(actual_pNum, mesh.cdim)  # Use computed pNum

# Get tetrahedral depth range
tet_Start, tet_End = mesh.dm.getDepthStratum(3)

# Get fault tetra indices
fault_tetra_set = set(mesh.dm.getStratumIS("TetraLabels", 404).array)

# Parallel-safe permeability assignment
for t in range(tet_Start, tet_End):
    if t in fault_tetra_set:
        coneclose, orient = mesh.dm.getTransitiveClosure(t, useCone=True)
        
        # Get the last 4 entries (vertices) and their y-coordinates
        vertex_indices = [v for v in coneclose if pStart <= v < pEnd]
        if len(vertex_indices) != 4:
            continue  # Ensure we have exactly 4 vertices for the tetrahedron

        y_coords = [coords[v - pStart][1] for v in vertex_indices]

        # Assign permeability based on y-coordinate
        if np.any(np.array(y_coords) > 5):
            perm_arr[t] = k1
        else:
            perm_arr[t] = 0.0
    else:
        perm_arr[t] = k2

# Ensure consistency across MPI ranks
comm.Barrier()
# -

# assigning k values to mesh variable
permeability.data[...] = perm_arr

darcy.constitutive_model.Parameters.permeability = permeability.sym[0]

# darcy solve without gravity
darcy.solve(verbose=True)

# +
# # saving output

mesh.petsc_save_checkpoint(index=0, meshVars=[p_soln, v_soln], 
                           outputPath=f'{output_dir}darcy_3d_loop_mesh_fault_no_g')

# mesh.write_timestep(f'darcy_3d_loop_mesh_fault_no_g', meshUpdates=True, 
#                     meshVars=[v_soln, p_soln], 
#                     outputPath='./output/', index=0)
# -

# plotting soln without gravity
if uw.mpi.size==1:
    plot_P_V(mesh, p_soln, v_soln)

# # copy soln
# TODO: Consider uw.synchronised_array_update() for multi-variable assignment
p_soln_0.data[...] = p_soln.data[...]
v_soln_0.data[...] = v_soln.data[...]

# now switch on gravity
darcy.constitutive_model.Parameters.s = sympy.Matrix([0, 0, -1]).T
darcy.solve(verbose=True)

# +
# saving output

mesh.petsc_save_checkpoint(index=0, meshVars=[p_soln, v_soln, permeability], 
                           outputPath=f'{output_dir}darcy_3d_loop_mesh_fault_g')
# -

# plotting soln without gravity
if uw.mpi.size==1:
    plot_P_V(mesh, p_soln, v_soln)

# set up interpolation coordinates
if uw.mpi.size==1:
    xcoords = np.linspace(minX + 0.001 * (maxX - minX), maxX - 0.001 * (maxX - minX), 100)
    ycoords = np.full_like(xcoords, 5)
    zcoords = np.full_like(xcoords, 2)
    xyz_coords = np.column_stack([xcoords, ycoords, zcoords])
    
    pressure_interp = uw.function.evaluate(p_soln.sym[0], xyz_coords)
    pressure_interp_0 = uw.function.evaluate(p_soln_0.sym[0], xyz_coords)

# plotting numerical and analytical solution
if uw.mpi.size==1:
    fig = plt.figure(figsize=(15,7))
    ax1 = fig.add_subplot(111, xlabel="X-Distance", ylabel="Pressure")
    ax1.plot(xcoords, pressure_interp, linewidth=3, label="Numerical solution")
    ax1.plot(xcoords, pressure_interp_0, linewidth=3, label="Numerical solution (no G)")
    # ax1.plot(pressure_analytic, xcoords, linewidth=3, linestyle="--", label="Analytic solution")
    # ax1.plot(pressure_analytic_noG, xcoords, linewidth=3, linestyle="--", label="Analytic (no gravity)")
    ax1.grid("on")
    ax1.legend()


