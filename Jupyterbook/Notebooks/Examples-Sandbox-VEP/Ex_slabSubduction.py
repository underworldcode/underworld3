#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # Slab subduction
#
#
# #### [From Dan Sandiford](https://github.com/dansand/uw3_models/blob/main/slabsubduction.ipynb)
#
#
#
# UW2 example ported to UW3 

# %%
import numpy as np
import os
import math
import underworld3


from underworld3.utilities import uw_petsc_gen_xdmf


# %%
expt_name = 'output/slabSubduction/'

# Make output directory if necessary.
from mpi4py import MPI
if MPI.COMM_WORLD.rank==0:
    ### delete previous model run
    if os.path.exists(expt_name):
        for i in os.listdir(expt_name):
            os.remove(expt_name + i)
            
    ### create folder if not run before
    if not os.path.exists(expt_name):
        os.makedirs(expt_name)


# %%
### For visualisation
render = True


# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes

options = PETSc.Options()

options["snes_converged_reason"] = None
options["snes_monitor_short"] = None

if uw.mpi.size == 1:
    options["pc_type"]  = "lu"
    options["ksp_type"] = "preonly"

sys = PETSc.Sys()
sys.pushErrorHandler("traceback")

# %%
n_els     =  10
dim       =   2
boxLength = 4.0
boxHeight = 1.0
ppcell    =   5

# %% [markdown]
# ### Create mesh and mesh vars

# %%
mesh = uw.meshing.StructuredQuadBox(elementRes=(    4*n_els,n_els), 
                    minCoords =(       0.,)*dim, 
                    maxCoords =(boxLength,boxHeight),)


v = uw.discretisation.MeshVariable("Velocity", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("Pressure", mesh, 1, degree=1)

strain_rate_inv2 = uw.discretisation.MeshVariable("SR", mesh, 1, degree=1)
node_viscosity   = uw.discretisation.MeshVariable("Viscosity", mesh, 1, degree=1)
materialField    = uw.discretisation.MeshVariable("Material", mesh, 1, degree=1)

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p, verbose=True)
stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh.dim)


# %% [markdown]
# ### Create swarm and swarm vars
# - 'swarm.add_variable' is a traditional swarm, can't be used to map material properties. Can be used for sympy operations, similar to mesh vars.
# - 'uw.swarm.IndexSwarmVariable', creates a mask for each material and can be used to map material properties. Can't be used for sympy operations.
#

# %%
swarm  = uw.swarm.Swarm(mesh)

# %%
## # Add variable for material
materialVariable      = swarm.add_variable(name="materialVariable", size=1, dtype=PETSc.IntType)
material              = uw.swarm.IndexSwarmVariable("M", swarm, indices=5) 

swarm.populate()

# Add some randomness to the particle distribution
import numpy as np
np.random.seed(0)

with swarm.access(swarm.particle_coordinates):
    factor = 0.5*boxLength/n_els/ppcell
    swarm.particle_coordinates.data[:] += factor*np.random.rand(*swarm.particle_coordinates.data.shape)
      


# %% [markdown]
# #### Project fields to mesh vars
# Useful for visualising stuff on the mesh (Viscosity, material, strain rate etc) and saving to a grouped xdmf file


# %%
nodal_strain_rate_inv2 = uw.systems.Projection(mesh, strain_rate_inv2)
nodal_strain_rate_inv2.uw_function = stokes._Einv2
# nodal_strain_rate_inv2.smoothing = 1.0e-3
nodal_strain_rate_inv2.petsc_options.delValue("ksp_monitor")

nodal_visc_calc = uw.systems.Projection(mesh, node_viscosity)
nodal_visc_calc.uw_function = stokes.constitutive_model.Parameters.viscosity
# nodal_visc_calc.smoothing = 1.0e-3
nodal_visc_calc.petsc_options.delValue("ksp_monitor")

meshMat = uw.systems.Projection(mesh, materialField)
meshMat.uw_function = materialVariable.sym[0]
# meshMat.smoothing = 1.0e-3
meshMat.petsc_options.delValue("ksp_monitor")

def updateFields():
    ### update strain rate
    nodal_strain_rate_inv2.uw_function = stokes._Einv2
    nodal_strain_rate_inv2.solve()

    ### update viscosity
    nodal_visc_calc.uw_function = stokes.constitutive_model.Parameters.shear_viscosity_0
    nodal_visc_calc.solve(_force_setup=True)
    
    ### update material field from swarm
    meshMat.uw_function = materialVariable.sym[0] 
    meshMat.solve(_force_setup=True)



# %% [markdown]
# ## Setup the material distribution


# %%
import matplotlib.path as mpltPath

### initialise the 'materialVariable' data to represent two different materials. 
upperMantleIndex = 0
lowerMantleIndex = 1
upperSlabIndex   = 2
lowerSlabIndex   = 3
coreSlabIndex    = 4

### Initial material layout has a flat lying slab with at 15\degree perturbation
lowerMantleY   = 0.4
slabLowerShape = np.array([ (1.2,0.925 ), (3.25,0.925 ), (3.20,0.900), (1.2,0.900), (1.02,0.825), (1.02,0.850) ])
slabCoreShape  = np.array([ (1.2,0.975 ), (3.35,0.975 ), (3.25,0.925), (1.2,0.925), (1.02,0.850), (1.02,0.900) ])
slabUpperShape = np.array([ (1.2,1.000 ), (3.40,1.000 ), (3.35,0.975), (1.2,0.975), (1.02,0.900), (1.02,0.925) ])


# %%
slabLower  = mpltPath.Path(slabLowerShape)
slabCore   = mpltPath.Path(slabCoreShape)
slabUpper  = mpltPath.Path(slabUpperShape)


# %% [markdown]
# ### Update the material variable of the swarm

# %%
with swarm.access(swarm.particle_coordinates, materialVariable, material):
    
    #### for the piecewise functions for material properties
    materialVariable.data[:] = upperMantleIndex
    materialVariable.data[swarm.particle_coordinates.data[:,1] < lowerMantleY]           = lowerMantleIndex
    materialVariable.data[slabLower.contains_points(swarm.particle_coordinates.data[:])] = lowerSlabIndex
    materialVariable.data[slabCore.contains_points(swarm.particle_coordinates.data[:])]  = coreSlabIndex
    materialVariable.data[slabUpper.contains_points(swarm.particle_coordinates.data[:])] = upperSlabIndex
    
    ### for the symbolic mapping of material properties
    material.data[:] = upperMantleIndex
    material.data[swarm.particle_coordinates.data[:,1] < lowerMantleY]           = lowerMantleIndex
    material.data[slabLower.contains_points(swarm.particle_coordinates.data[:])] = lowerSlabIndex
    material.data[slabCore.contains_points(swarm.particle_coordinates.data[:])]  = coreSlabIndex
    material.data[slabUpper.contains_points(swarm.particle_coordinates.data[:])] = upperSlabIndex
    
    
    


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
        point_cloud.point_data["M"] = materialVariable.data.copy()



    pl = pv.Plotter(notebook=True)

    pl.add_mesh(pvmesh,'Black', 'wireframe')

    # pl.add_points(point_cloud, color="Black",
    #                   render_points_as_spheres=False,
    #                   point_size=2.5, opacity=0.75)       



    pl.add_mesh(point_cloud, cmap="coolwarm", edge_color="Black", show_edges=False, scalars="M",
                        use_transparency=False, opacity=0.95)



    pl.show(cpos="xy")
 
if render == True:
    plot_mat()


# %% [markdown]
# ### Function to save output of model
# Saves both the mesh vars and swarm vars

# %%
def saveData(step, outputPath):
    
    ### save mesh vars
    fname = f"./{expt_name}{'step_'}{step:02d}.h5"
    xfname = f"./{expt_name}{'step_'}{step:02d}.xdmf"
    viewer = PETSc.ViewerHDF5().createHDF5(fname, mode=PETSc.Viewer.Mode.WRITE,  comm=PETSc.COMM_WORLD)

    viewer(mesh.dm)

    ### add mesh vars to viewer to save as one h5/xdmf file. Has to be a PETSc object (?)
    viewer(stokes.u._gvec)         # add velocity
    viewer(stokes.p._gvec)         # add pressure
    viewer(materialField._gvec)    # add material projection
    viewer(strain_rate_inv2._gvec) # add strain rate
    viewer(node_viscosity._gvec)   # add viscosity
    viewer.destroy() 
    uw_petsc_gen_xdmf.generateXdmf(fname, xfname)
    
    if uw.mpi.size == 1:
        import pyvista as pv
        with swarm.access():
            points = np.zeros((swarm.data.shape[0],3))
            points[:,0] = swarm.data[:,0]
            points[:,1] = swarm.data[:,1]
            points[:,2] = 0.0

        point_cloud = pv.PolyData(points)


        with swarm.access():
            point_cloud.point_data["M"] = materialVariable.data.copy()

        point_cloud.save(f"./{outputPath}{'swarm_step_'}{step:02d}.vtk")

# %% [markdown]
# ### Rheology

# %%
from sympy import Piecewise, ceiling, Abs, Min, sqrt, eye, Matrix, Max


# %%
# upperMantleViscosity =    1.0
# lowerMantleViscosity =   50.0

# slabViscosity        =  100.0
# coreViscosity        =  250.0

### viscosity from UW2 example
upperMantleViscosity =    1.0
lowerMantleViscosity =  100.0
slabViscosity        =  500.0
coreViscosity        =  500.0


strainRate_2ndInvariant = stokes._Einv2 #sqrt(inv2)


cohesion = 0.06
vonMises = 0.5 * cohesion / (strainRate_2ndInvariant+1.0e-18)


# The upper slab viscosity is the minimum of the 'slabViscosity' or the 'vonMises' 
slabYieldvisc =  Max(0.1, Min(vonMises, slabViscosity))

# %% [markdown]
# #### Density

# %%
mantleDensity = 0.0
slabDensity   = 1.0 

density = material.createMask( [ mantleDensity, 
                                 mantleDensity, 
                                 slabDensity, 
                                 slabDensity, 
                                 slabDensity ])

stokes.bodyforce =  Matrix([0, -1 * density]) # -density*mesh.N.j


# %%
material.viewMask(density)

# %%
density

# %% [markdown]
# ### Boundary conditions
#
# Free slip by only constraining one component of velocity 

# %%
#free slip
stokes.add_dirichlet_bc( (0.,0.), ['Top',  'Bottom'], 1)  # top/bottom: function, boundaries, components 
stokes.add_dirichlet_bc( (0.,0.), ['Left', 'Right' ], 0)  # left/right: function, boundaries, components

# ## Initial Solve

# %% [markdown]
# ###### initial first guess of constant viscosity

# %%
### initial linear solve
stokes.constitutive_model.Parameters.shear_viscosity_0  = 1.
stokes.petsc_options["pc_type"] = "lu"
stokes.solve(zero_init_guess=True)


# %% [markdown]
# #### add in NL viscosity for solve loop

# %%
viscosity = material.createMask([ upperMantleViscosity,
                                  lowerMantleViscosity,
                                  slabYieldvisc,
                                  slabYieldvisc,
                                  coreViscosity ] )

stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity

# %% [markdown]
# ### Main loop
# Stokes solve loop

# %%
step      = 0
max_steps = 50
time      = 0


#timing setup
#viewer.getTimestep()
#viewer.setTimestep(1)


while step<max_steps:
    
    print(f'\nstep: {step}, time: {time}')
          
    #viz for parallel case - write the hdf5s/xdmfs 
    if step%2==0:
        if uw.mpi.rank == 0:
            print(f'\nVisualisation: ')
            
        ### updates projection of fields to the mesh
        updateFields()
        
        ### saves the mesh and swarm
        saveData(step, expt_name)
        

            
    
    if uw.mpi.rank == 0:
        print(f'\nStokes solve: ')  
        
    stokes.solve(zero_init_guess=False)
    
    ### get the timestep
    dt = stokes.estimate_dt()
 
    ### advect the particles according to the timestep
    swarm.advection(V_fn=stokes.u.sym, delta_t=dt, corrector=False)
        
    step += 1
    
    time += dt



    #viewer.setTimestep(step)



# %%
