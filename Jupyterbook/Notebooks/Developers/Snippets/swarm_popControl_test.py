# %% [markdown]
# # Population control test

# %%
from petsc4py import PETSc
import underworld3 as uw
import numpy as np
import sympy
from mpi4py import MPI

import os

from underworld3.utilities import generateXdmf, swarm_h5, swarm_xdmf
import petsc4py

# %%
petsc4py.__version__

# %%
petsc4py.get_config()

# %% [markdown]
# ### Create mesh

# %%
# mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0,0.0), 
#                                               maxCoords=(1.0,1.0), 
#                                               cellSize=1.0/res, 
#                                               regular=True)

# mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(xmin, ymin), maxCoords=(xmax, ymax), cellSize=1.0 / res, regular=False)


mesh = uw.meshing.StructuredQuadBox(elementRes =(int(5),int(5)),
                                    minCoords=(0,0), 
                                    maxCoords=(1,1))


v = uw.discretisation.MeshVariable('U',    mesh,  mesh.dim, degree=2 )


if uw.mpi.rank == 0:
    print('finished mesh')


# %% [markdown]
# ### Create Stokes object

# %% [markdown]
# #### Setup swarm

# %%
swarm     = uw.swarm.Swarm(mesh=mesh)


# %%
# test2      = swarm.add_variable(name="test2", num_components=1, dtype=)
# test3      = swarm.add_variable(name="test3", num_components=2, dtype=np.float64)

material = swarm.add_variable('M')

swarm.populate_petsc(2)
pop_control = uw.swarm.PopulationControl(swarm)

if uw.mpi.rank == 0:
    print('populate swarm')

# %%
with swarm.access(material):
    material.data[:,0] = 0 
    material.data[(swarm.data[:,1] < 0.2) | (swarm.data[:,1] > 0.8) ] = 1


# %%
def plot_mat():

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'panel'
    pv.global_theme.smooth_shading = True


    mesh.vtk("tempMsh.vtk")
    pvmesh = pv.read("tempMsh.vtk") 

    with swarm.access():
        points = np.zeros((swarm.data.shape[0],3))
        points[:,0] = swarm.data[:,0]
        points[:,1] = swarm.data[:,1]
        points[:,2] = 0.0

    point_cloud = pv.PolyData(points)


    with swarm.access():
        point_cloud.point_data["M"] = material.data.copy()



    pl = pv.Plotter(notebook=True)

    pl.add_mesh(pvmesh,'Black', 'wireframe')

    # pl.add_points(point_cloud, color="Black",
    #                   render_points_as_spheres=False,
    #                   point_size=2.5, opacity=0.75)       



    pl.add_mesh(point_cloud, cmap="coolwarm", edge_color="Black", show_edges=False, scalars='M',
                        use_transparency=False, opacity=0.95, point_size= 20)



    pl.show(cpos="xy")
    
plot_mat()

# %%
with swarm.access():
    s_cID, s_cID_counts = np.unique(swarm.particle_cellid.data, return_counts=True)
print(f'cell particles min: {s_cID_counts.min()}\ncell particles max: {s_cID_counts.max()}')

# %%
with mesh.access(v):
    v.data[:,0] = 1
    
### Advect particles out to create some empty/under resolved cells
swarm.advection(V_fn=v.sym, delta_t = 0.2)

plot_mat()

with swarm.access():
    s_cID, s_cID_counts = np.unique(swarm.particle_cellid.data, return_counts=True)
print(f'cell particles min: {s_cID_counts.min()}\ncell particles max: {s_cID_counts.max()}')

# %%
# pop_control.repopulate_loop(material)
pop_control.repopulate_fast(material)
with swarm.access():
    s_cID, s_cID_counts = np.unique(swarm.particle_cellid.data, return_counts=True)
print(f'cell particles min: {s_cID_counts.min()}\ncell particles max: {s_cID_counts.max()}')

# %%
plot_mat()

# %%

with mesh.access(v):
    v.data[:,0] = 1
    
### Advect particles out to create some empty/under resolved cells
swarm.advection(V_fn=v.sym, delta_t = 0.5)

plot_mat()

with swarm.access():
    s_cID, s_cID_counts = np.unique(swarm.particle_cellid.data, return_counts=True)
print(f'cell particles min: {s_cID_counts.min()}\ncell particles max: {s_cID_counts.max()}')


# %%
# pop_control.repopulate_loop(material)
pop_control.repopulate_fast(material)
plot_mat()

with swarm.access():
    s_cID, s_cID_counts = np.unique(swarm.particle_cellid.data, return_counts=True)
print(f'cell particles min: {s_cID_counts.min()}\ncell particles max: {s_cID_counts.max()}')

# %%
with mesh.access(v):
    v.data[:,0] = 1
    
### Advect particles out to create some empty/under resolved cells
swarm.advection(V_fn=v.sym, delta_t = 0.5)

plot_mat()

with swarm.access():
    s_cID, s_cID_counts = np.unique(swarm.particle_cellid.data, return_counts=True)
print(f'cell particles min: {s_cID_counts.min()}\ncell particles max: {s_cID_counts.max()}')

# %%
# pop_control.repopulate_loop(material)
pop_control.repopulate_fast(material)
plot_mat()

with swarm.access():
    s_cID, s_cID_counts = np.unique(swarm.particle_cellid.data, return_counts=True)
print(f'cell particles min: {s_cID_counts.min()}\ncell particles max: {s_cID_counts.max()}')

# %%
with mesh.access(v):
    v.data[:,0] = 1
    
### Advect particles out to create some empty/under resolved cells
swarm.advection(V_fn=v.sym, delta_t = 0.22)

plot_mat()

with swarm.access():
    s_cID, s_cID_counts = np.unique(swarm.particle_cellid.data, return_counts=True)
print(f'cell particles min: {s_cID_counts.min()}\ncell particles max: {s_cID_counts.max()}')

# %%
# pop_control.repopulate_loop(material)
pop_control.repopulate_fast(material)
plot_mat()

with swarm.access():
    s_cID, s_cID_counts = np.unique(swarm.particle_cellid.data, return_counts=True)
print(f'cell particles min: {s_cID_counts.min()}\ncell particles max: {s_cID_counts.max()}')

# %%

# %%

# %%

# %%

# %%
