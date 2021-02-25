#!/usr/bin/env python
# coding: utf-8

# # Steady State Solve
# 
# Example is based on Julian's Laplace example.

# In[1]:


import petsc4py.PETSc as PETSc
import underworld3 as uw


# In[2]:


options = PETSc.Options()


# In[3]:


options.setValue("elRes", "10,10")
options.setValue("simplex", 1)
options.setValue("dm_plex_separate_marker", None)
options.setValue("temperature_petscspace_degree", 1)
options.setValue("dm_view", "hdf5:sol.h5")
options.setValue("sol_vec_view", "hdf5:sol.h5::append")
options.setValue("options_view",None)


# In[4]:


user = {"y0": -1.6,
        "y1": 1.3,
        "k": 0.5,
        "h": 10,
        "T0": 4.0,
        "T1": 8.0,
        "simplex": True}


# In[5]:


from underworld3.systems import *


# ## Create Mesh

# In[6]:


plex = PETSc.DMPlex().createBoxMesh(faces=(10,10), 
                                    lower=(-1.0, user["y0"]),
                                    upper=(1.0, user["y1"]),
                                    simplex=user["simplex"])
part = plex.getPartitioner()
part.setFromOptions()
plex.distribute()
plex.localizeCoordinates()
plex.setFromOptions()
plex.viewFromOptions('-dm_view')


# In[7]:


pySetupDiscretization(plex,user)


# In[8]:


plex.createClosureIndex(None)
plex.setSNESLocalFEM()


# In[10]:


# Build snes
snes = PETSc.SNES().create(comm=plex.getComm())
snes.setDM(plex)
snes.setFromOptions()

# Build vector
vec = plex.createGlobalVector()
vec.array.shape

snes.solve(None, vec)

vec.viewFromOptions('-sol_vec_view')


# In[13]:


from subprocess import call


# In[15]:


# For some reason this doesn't work. I think because petsc4py is holding some reference 
# to .h5 until the python script finishes


# rank = PETSc.COMM_WORLD.getRank()
# if rank == 0:
#     print("Converting h5 -> xmf ", end="")
    
#     cmd = "$PETSC_DIR/lib/petsc/bin/petsc_gen_xdmf.py sol.h5"
#     try:
#         retcode = call(cmd, shell=True)
#         if retcode < 0:
#             print("... was terminated by signal {}".format(-retcode), file=sys.stderr)
#     except OSError as e:
#         print("... failed: {}".format(e), file=sys.stderr)
    
#     print("... done")


# In[ ]:




