# %% [markdown]
# # Write checkpoint
#
# Write out some checkpoint data:
#   - A scalar variable with X coordinates
#   - A vector variable with X,Y coordinates
#   - A scalar indexing variable (the original vertex numbering)
#   - The original grid file / vertex ordering
#   - The grid file as dumped by dmplex.view()
#   
# Record this for a number of different decompositions (e.g. 1, 2, 4 ... )
#
# We will keep this data and try to read it back in various ways.

# %%
# Enable timing (before uw imports)

import os

os.environ["UW_TIMING_ENABLE"] = "0"

from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import timing

import numpy as np
import sympy
from mpi4py import MPI

import h5py

if uw.mpi.size == 1:
    os.makedirs("output", exist_ok=True)
else:
    os.makedirs(f"output_np{uw.mpi.size}", exist_ok=True)

# %%
# Define the problem size
#      1  - ultra low res for automatic checking
#      2  - low res problem to play with this notebook
#      3  - medium resolution (be prepared to wait)
#      4  - highest resolution
#      5+ - v. high resolution (parallel only)

problem_size = uw.options.getInt("model_size", default=2)

if problem_size <= 1:
    res = 8
elif problem_size == 2:
    res = 16
elif problem_size == 3:
    res = 32
elif problem_size == 4:
    res = 48
elif problem_size == 5:
    res = 64
elif problem_size >= 6:
    res = 128

# %%
expt_name = f"test_checkpointing_np{uw.mpi.size}"

# %%
## Mesh - checkpointed and we are going to read it regardless of the number of 
## processes when we restore the data

mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(-1.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=1.0 / res,
    regular=False,
    qdegree=3,
    filename=f"{expt_name}.orig.msh",
)

# meshfilename = "test_checkpointing_np1.mesh.0.h5"
# mesh = uw.discretisation.Mesh(meshfilename)

# Variables (vector, scalar) and copies
v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True)
index = uw.discretisation.MeshVariable("Index", mesh, 1, degree=1, continuous=True)
index1p = uw.discretisation.MeshVariable("Index1proc", mesh, 1, degree=1, continuous=True)

if mesh.sf is None:
    mesh.sf = mesh.dm.getDefaultSF()


# %%
iset = mesh.dm.getVertexNumbering()
indices = iset.getIndices()

with mesh.access(p,v, index, index1p):
    p.data[:,0] = p.coords[:,0]
    v.data[:,:] = v.coords[:,:]
    index.data[:,0] = indices[:]   
    index1p.data[:,0] = indices[:]   

if uw.mpi.size != 1:
    filename = "test_checkpointing_np1.Index.0.h5"
    index1p.load_from_checkpoint(filename, data_name='Index')  


# %%
# Do this for the serial case as a reference point

mesh.write_visualisation_xdmf(expt_name, 
                          meshUpdates=True, 
                          meshVars=[p,v,index, index1p], 
                          index=0)

   

# %%
mesh.sf.getGraph()[1]

# %%
mesh.dm.getVertexNumbering().view()

# %%
sf = mesh.sf
sf_gr = sf.getGraph()

sfi = mesh.sf.createInverse()
sfi_gr = sfi.getGraph()

if uw.mpi.rank == uw.mpi.size-1:
    print(sf_gr[1], flush=True)    
    print(sf_gr[2], flush=True)
    
if uw.mpi.rank == uw.mpi.size-1:
    print(sfi_gr[1], flush=True)    
    print(sfi_gr[2], flush=True)


# %%
0/0

# %%
# mesh.dm.getOrdering(PETSc.Mat.OrderingType.NATURAL).getIndices()

# %%
# mesh.dm.getVertexNumbering().getIndices()


# %%
with mesh.access(p,v, vc, pc):
    if uw.mpi.rank == 0:
        print(f"P   - {p.data[0:10].T}", flush=True)
        print(f"Pc  - {pc.data[0:10].T}", flush=True)
        print(f"Ixc - {index1pc.data[0:10].astype(int).T}", flush=True)
        print(f"Ix  - {index1p.data[0:10].astype(int).T}", flush=True)


# %%
mesh.write_visualisation_xdmf(expt_name+"_c", 
                              meshUpdates=True, 
                              meshVars=[p,v,index1p,index1pc], 
                              index=0)

# %%
if mesh.sf == None:
    mesh.sf = mesh.dm.getDefaultSF()

isf = mesh.sf.createInverse()

sec = mesh.dm.getDefaultSection()
lvec = mesh.dm.getLocalVec()
gvec = mesh.dm.getGlobalVec()

# push this vec into that one 

isset, subdm = mesh.dm.createSubDM(index1p.field_id)
pcvec = gvec.getSubVector(isset)
subdm.localToGlobal(index1p._lvec, pcvec)
gvec.restoreSubVector(isset, pcvec)
mesh.dm.globalToLocal(gvec, lvec)

# Do the permutation

newsec, newvec = mesh.dm.distributeField(mesh.sf, sec, lvec)
mesh.dm.localToGlobal(newvec, gvec)
pcvec = gvec.getSubVector(isset)
index1p._gvec.array[...] = pcvec.array[...]
subdm.globalToLocal(index1p._gvec, index1p._lvec)

if uw.mpi.rank == 0:
    print(f"Index Permutation - {index1p._lvec.array[0:10].T}", flush=True)


# %%
0/0

# %%
# newsec.view()

# %%
if uw.mpi.rank == 0:
    print(f"Pcp - {pc._lvec.array[0:10].T}", flush=True)

# %%
if not uw.is_notebook:
    exit()
else:
    0/0

# %%
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(-1.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=1.0 / res,
    regular=False,
    qdegree=3,
    filename=f"{expt_name}.orig.msh",
)

# Variables (vector, scalar) and copies
v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
vc = uw.discretisation.MeshVariable("Uc", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True)
pc = uw.discretisation.MeshVariable("Pc", mesh, 1, degree=1, continuous=True)


# %%
with mesh.access(p,v, vc, pc):
    p.data[:,0] = p.coords[:,0]
    v.data[:,:] = v.coords[:,:]
    vc.data[:,:] = 0.0
    pc.data[:,0] = 0.0


# %%
mesh.write_visualisation_xdmf(expt_name, 
                              meshUpdates=True, 
                              meshVars=[p,v], 
                              index=0)

# %%
if mesh.sf == None:
    mesh.sf = mesh.dm.getDefaultSF()
