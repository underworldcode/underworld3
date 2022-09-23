#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc
import swarm2D

OptDB = PETSc.Options()

#lambda_ = OptDB.getReal('lambda', 6.0)
#do_plot = OptDB.getBool('plot', False)
dim = OptDB.getInt('dim', 2)

plex = PETSc.DMPlex().createBoxMesh([4]*dim, simplex=False)
plex.view()

dim = plex.getDimension()
vStart, vEnd = plex.getDepthStratum(0)
numVertices = vEnd-vStart

print("The number of vertices are {}".format(numVertices))


# In[2]:


swarm2D.pyBuildField(plex, False)


# In[3]:


fields = ('velocity', 'pressure')


# In[4]:


swarm = PETSc.DM().create()


# In[5]:


swarm


# In[6]:


swarm2D.pyBuildSwarm(plex, 2, fields, swarm)


# In[6]:


vec = plex.getGlobalVec()


# In[10]:


plex.getNumFields()


# In[10]:


swarm.view()

