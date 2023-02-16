# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Checkpoint restore - vertex values via kdtree
#
# The `chkpt` directory has data that was saved for various parallel decompositions (just a few procs). The data we have saved is actually the x coordinate in a scalar variable and the x,y coordinates in a vector. This makes it very simple to check that the mesh variables are reconstructed correctly.
#
# The data ordering differs from that in the original mesh file in each case ... here we just use a kd-tree to rebuild the vertex data and interpolate. This is quite a flexible workflow but its not a genuine checkpoint so we will see how well it works. 
#
#

import underworld3 as uw
from petsc4py import PETSc
import numpy as np
import h5py

from underworld3 import kdtree

# +
mesh1 = uw.discretisation.Mesh("./chkpt/test_checkpointing_np4.mesh.0.h5")

pm = uw.discretisation.MeshVariable("P", mesh1, 1, degree=1, continuous=True)
pm2 = uw.discretisation.MeshVariable("P2", mesh1, 1, degree=2, continuous=True)
um = uw.discretisation.MeshVariable("U", mesh1, mesh1.dim, degree=1, continuous=True)
um2 = uw.discretisation.MeshVariable("U2", mesh1, mesh1.dim, degree=2, continuous=True)

# -

uw.utilities.h5_scan("./chkpt/test_checkpointing_np8.P.0.h5")

pm.read_from_vertex_checkpoint("./chkpt/test_checkpointing_np4.P.0.h5", "P")
pm2.read_from_vertex_checkpoint("./chkpt/test_checkpointing_np4.P.0.h5", "P")
um.read_from_vertex_checkpoint("./chkpt/test_checkpointing_np4.U.0.h5", "U")
um2.read_from_vertex_checkpoint("./chkpt/test_checkpointing_np4.U.0.h5", "U")


# +
# This is the validation - every entry in this
# array should be "small"

with mesh1.access():
    local_mean = (pm.data[:,0]-pm.coords[:,0]).mean()
    print("MEAN P1->1:", local_mean)
    if local_mean > 1.0e-3:
        print((pm.data[::13,0]-pm.coords[::13,0]).T)
    
    local_mean = (pm2.data[:,0]-pm2.coords[:,0]).mean()
    print("MEAN P1->2:", local_mean)
    if local_mean > 1.0e-3:
        print((pm2.data[::13,0]-pm2.coords[::13,0]).T)     

    local_mean = (um.data[:,0]-um.coords[:,0]).mean()
    print("MEAN: U2->1", local_mean)
    if local_mean > 1.0e-3:
        print((um.data[::13,0]-um.coords[::13,0]).T)

    local_mean = (um2.data[:,0]-um2.coords[:,0]).mean()
    print("MEAN: U2->2", local_mean)
    if local_mean > 1.0e-3:
        print((um2.data[::13,0]-um2.coords[::13,0]).T)

# +
## This is the way to read back the vertex field checkpoint which is 
## usually saved for a visualisation with paraview (need original mesh information for this)


pm.read_from_vertex_checkpoint("./chkpt/test_checkpointing_np4.P.0.h5", "P", 
                               vertex_mesh_file="./chkpt/test_checkpointing_np4.mesh.0.h5", 
                               vertex_field=True, vertex_field_degree=1)

pm2.read_from_vertex_checkpoint("./chkpt/test_checkpointing_np4.P.0.h5", "P", 
                               vertex_mesh_file="./chkpt/test_checkpointing_np4.mesh.0.h5", 
                               vertex_field=True, vertex_field_degree=1)

um.read_from_vertex_checkpoint("./chkpt/test_checkpointing_np4.U.0.h5", "U", 
                               vertex_mesh_file="./chkpt/test_checkpointing_np4.mesh.0.h5", 
                               vertex_field=True, vertex_field_degree=2)

um2.read_from_vertex_checkpoint("./chkpt/test_checkpointing_np4.U.0.h5", "U", 
                               vertex_mesh_file="./chkpt/test_checkpointing_np4.mesh.0.h5", 
                               vertex_field=True, vertex_field_degree=2)


# +
# This is the validation - every entry in this
# array should be "small"

with mesh1.access():
    local_mean = (pm.data[:,0]-pm.coords[:,0]).mean()
    print("MEAN P1->1:", local_mean)
    if local_mean > 1.0e-3:
        print((pm.data[::13,0]-pm.coords[::13,0]).T)
    
    local_mean = (pm2.data[:,0]-pm2.coords[:,0]).mean()
    print("MEAN P1->2:", local_mean)
    if local_mean > 1.0e-3:
        print((pm2.data[::13,0]-pm2.coords[::13,0]).T)     

    local_mean = (um.data[:,0]-um.coords[:,0]).mean()
    print("MEAN: U2->1", local_mean)
    if local_mean > 1.0e-3:
        print((um.data[::13,0]-um.coords[::13,0]).T)

    local_mean = (um2.data[:,0]-um2.coords[:,0]).mean()
    print("MEAN: U2->2", local_mean)
    if local_mean > 1.0e-3:
        print((um2.data[::13,0]-um2.coords[::13,0]).T)
# -


