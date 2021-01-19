# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.stokes import Stokes
import numpy as np

options = PETSc.Options()
# options["help"] = None

# options["pc_type"]  = "svd"

options["ksp_rtol"] =  1.0e-6
options["ksp_atol"] =  1.0e-11
# options["ksp_monitor_short"] = None

# options["snes_type"]  = "fas"
options["snes_converged_reason"] = None
options["snes_monitor_short"] = None
# options["snes_view"]=None
# options["snes_test_jacobian"] = None
# options["snes_rtol"] = 1.0e-2  # set this low to force single SNES it. 
options["snes_max_it"] = 1

sys = PETSc.Sys()
sys.pushErrorHandler("traceback")
# %%
dim = 2
n_els = 64
boxLength      = 0.9142
viscosityRatio = 1.0
dx = boxLength/n_els
ppcell = 2

mesh = uw.Mesh(elementRes=(    n_els,)*dim, 
               minCoords =(       0.,)*dim, 
               maxCoords =(boxLength,1.) )
# %%
u_degree = 1
stokes = Stokes(mesh, u_degree=u_degree )
# %%
# Create a variable to store material variable
matMeshVar = uw.MeshVariable(mesh, 1, "matmeshvar", uw.mesh.VarType.SCALAR, degree=u_degree)

#%%
# Create swarm
swarm  = uw.Swarm(mesh)
# Add variable for material
matSwarmVar      = swarm.add_variable(name="matSwarmVar",      num_components=1, dtype=PETSc.IntType)
matSwarmVarFloat = swarm.add_variable(name="matSwarmVarFloat", num_components=1, dtype=PETSc.ScalarType)
velSwarmVar = swarm.add_variable(name="velSwarmVar", num_components=mesh.dim, dtype=PETSc.ScalarType)
# Note that `ppcell` specifies particles per cell per dim.
swarm.populate(ppcell=ppcell)

#%%
# with mesh.access(stokes.u):
#     stokes.u.data[:] = 1.234
#     print(stokes.u.data)

# velSwarmVar.project_from(stokes.u)

#%%
# Add some randomness to the particle distribution
import numpy as np
with swarm.access(swarm.particle_coordinates):
    factor = 0.25*boxLength/n_els/ppcell
    swarm.particle_coordinates.data[:] += factor*np.random.rand(*swarm.particle_coordinates.data.shape)

#%%
# define these for convenience. 
lightIndex = 0
denseIndex = 1

# material perturbation from van Keken et al. 1997
wavelength = 2.0*boxLength
amplitude  = 0.06
offset     = 0.2
k = 2. * np.pi / wavelength

# init material variable
with swarm.access(matSwarmVar):
    perturbation = offset + amplitude*np.cos( k*swarm.particle_coordinates.data[:,0] )
    matSwarmVar.data[:,0] = np.where( perturbation>swarm.particle_coordinates.data[:,1], lightIndex, denseIndex )


# %%
# do kdtree
from scipy import spatial
with swarm.access():
    tree = spatial.KDTree(swarm.particle_coordinates.data)
nn_map = tree.query(mesh.data)[1]


# %%
# set NN vals on mesh var
with swarm.access(),mesh.access(matMeshVar):
    matMeshVar.data[:,0] = matSwarmVar.data[nn_map,0]


# %%
import plot

fig = plot.Plot(rulers=True)
# fig.edges(mesh)
with swarm.access(),mesh.access():
    fig.swarm_points(swarm, matSwarmVar.data, pointsize=4, colourmap="blue green", colourbar=True)
    # fig.nodes(mesh,matMeshVar.data,colourmap="blue green", pointsize=6, pointtype=4)
fig.display((1000,800))

# %%
matSwarmVarFloat.project_from(matMeshVar)
#%%
fig = plot.Plot(rulers=True)
# fig.edges(mesh)
with swarm.access(),mesh.access():
    fig.swarm_points(swarm, matSwarmVarFloat.data, pointsize=4, colourmap="blue green", colourbar=True)
    # fig.nodes(mesh,matMeshVar.data,colourmap="blue green", pointsize=6, pointtype=4)
fig.display((1000,800))
# %%
# velSwarmVar.project_from(stokes.u)
# %%
from sympy import Piecewise, ceiling, Abs

density = Piecewise( ( 0., Abs(matMeshVar.fn - lightIndex)<0.5 ),
                     ( 1., Abs(matMeshVar.fn - denseIndex)<0.5 ),
                     ( 0.,                                True ) )

stokes.bodyforce = -density*mesh.N.j

stokes.viscosity = Piecewise( ( viscosityRatio, Abs(matMeshVar.fn - lightIndex)<0.5 ),
                              (             1., Abs(matMeshVar.fn - denseIndex)<0.5 ),
                              (             1.,                                True ) )

# %%
# note with petsc we always need to provide a vector of correct cardinality. 
bnds = mesh.boundary
stokes.add_dirichlet_bc( (0.,0.), [bnds.TOP,  bnds.BOTTOM], (0,1) )  # top/bottom: function, boundaries, components 
stokes.add_dirichlet_bc( (0.,0.), [bnds.LEFT, bnds.RIGHT ], 0  )  # left/right: function, boundaries, components

# %%
import time
# %%
# Solve time
start_time = time.time()
stokes.solve(zero_init_guess=False, _force_setup=False)
print(f'solve time {time.time()-start_time}\n')

# %%
fig = plot.Plot(rulers=True)
with mesh.access():
    fig.vector_arrows(mesh, stokes.u.data)
fig.display((800,800))


# %%
dt = stokes.dt()
# %%
s_time = time.time()
with swarm.access():
    vel_on_particles = matMeshVar.evaluate(swarm.particle_coordinates.data)
print(f"eval time {time.time()-s_time}")
# %%

# s_time = time.time()
# with swarm.access(swarm.particle_coordinates):
#     swarm.particle_coordinates.data[:]+=dt*vel_on_particles
# print(f"advect time {time.time()-s_time}")

# # %%
# print("done")

