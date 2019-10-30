#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

OptDB = PETSc.Options()

lambda_ = OptDB.getReal('lambda', 6.0)
do_plot = OptDB.getBool('plot', False)
dim = OptDB.getInt('dim', 2)

plex = PETSc.DMPlex().createBoxMesh([6]*dim, simplex=False)
plex.setFromOptions()
plex.setUp()
plex.distribute()
plex.view()


dim = plex.getDimension()
vStart, vEnd = plex.getDepthStratum(0)
numVertices = vEnd-vStart

print("The number of vertices are {}".format(numVertices))

import swarm2D

swarm2D.pyBuildField(plex, False)
vec = plex.getGlobalVec()
print("Build {} fields, global vec size is {}\n".format(plex.getNumFields(), vec.size))

fields = ('velocity', 'pressure')
swarm = PETSc.DM().create()
swarm2D.pyBuildSwarm(plex, 2, fields, swarm)
swarm.view()
