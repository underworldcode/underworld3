# %% [markdown]
# # Test swarm IO

# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
import sympy
from mpi4py import MPI

import os

from underworld3.utilities import generateXdmf, swarm_h5, swarm_xdmf

# %%
outputPath = './output/swarmTest/'


if uw.mpi.rank == 0:
    # checking if the directory demo_folder 
    # exist or not.
    if not os.path.exists(outputPath):

        # if the demo_folder directory is not present 
        # then create it.
        os.makedirs(outputPath)

# %% [markdown]
# ### Create mesh

# %%
# mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0,0.0), 
#                                               maxCoords=(1.0,1.0), 
#                                               cellSize=1.0/res, 
#                                               regular=True)

# mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(xmin, ymin), maxCoords=(xmax, ymax), cellSize=1.0 / res, regular=False)


mesh = uw.meshing.StructuredQuadBox(elementRes =(int(32),int(32)),
                                    minCoords=(0,0), 
                                    maxCoords=(1,1))


# %% [markdown]
# ### Create Stokes object

# %% [markdown]
# #### Setup swarm

# %%
swarm     = uw.swarm.Swarm(mesh=mesh)

# %%
# material  = uw.swarm.IndexSwarmVariable("M", swarm, indices=2, proxy_continuous=False, proxy_degree=0)
material  = uw.swarm.IndexSwarmVariable("material", swarm, indices=2)

materialVariable      = swarm.add_variable(name="materialVariable", num_components=1, dtype=PETSc.IntType)

test0      = swarm.add_variable(name="test0", num_components=1, dtype=PETSc.RealType)
test1      = swarm.add_variable(name="test1", num_components=2, dtype=PETSc.RealType)

rank      = swarm.add_variable(name="rank", num_components=2, dtype=PETSc.RealType)

# test2      = swarm.add_variable(name="test2", num_components=1, dtype=)
# test3      = swarm.add_variable(name="test3", num_components=2, dtype=np.float64)

swarm.populate(2)

# %%
with swarm.access(rank):
    rank.data[:] = uw.mpi.rank

# %% [markdown]
# #### create a block at the base of the model

# %%
for i in [material, materialVariable]:
        with swarm.access(i):
            i.data[:] = 0
            i.data[(swarm.data[:,1] <= 0.1) & 
                  (swarm.data[:,0] >= (((1 - 0) / 2.) - (0.1 / 2.)) ) & 
                  (swarm.data[:,0] <= (((1 - 0) / 2.) + (0.1 / 2.)) )] = 1


# %% [markdown]
# Save the swarm fields

# %%
fields = [material,materialVariable, test0, test1, rank]


swarm_h5(swarm=swarm, fileName='swarm', fields=fields, timestep=0, outputPath=outputPath)

swarm_xdmf(fields=fields, fileName='swarm', timestep=0, outputPath=outputPath)

# %% [markdown]
# #### create a block at that is smaller at base of the model

# %%
for i in [material, materialVariable]:
        with swarm.access(i):
            i.data[:] = 0
            i.data[(swarm.data[:,1] <= 0.25) & 
                  (swarm.data[:,0] >= (((1 - 0) / 2.) - (0.25 / 2.)) ) & 
                  (swarm.data[:,0] <= (((1 - 0) / 2.) + (0.25 / 2.)) )] = 1

# %% [markdown]
# Save the swarm fields with the updated material field (but not materialVarible field)

# %%
fields = [material,materialVariable, test0, test1, rank]


swarm_h5(swarm=swarm, fileName='swarm', fields=fields, timestep=1, outputPath=outputPath)

swarm_xdmf(fields=fields, fileName='swarm', timestep=1, outputPath=outputPath)

# %% [markdown]
# #### Load the original material distribution back to the material field

# %%
material.load(outputPath+'material-0000.h5', outputPath+'swarm-0000.h5')




# %% [markdown]
# Save the swarm fields again

# %%
fields = [material,materialVariable, test0, test1, rank]


swarm_h5(swarm=swarm, fileName='swarm', fields=fields, timestep=2, outputPath=outputPath)

swarm_xdmf(fields=fields, fileName='swarm', timestep=2, outputPath=outputPath)

# %% [markdown]
# Visualise in paraview to check:
#
# - Should be big block --> small block --> big block for the 'material' field (as this is the field we re-load)
# - should be big block --> small block --> small block for the 'materialVariable' field

# %%
