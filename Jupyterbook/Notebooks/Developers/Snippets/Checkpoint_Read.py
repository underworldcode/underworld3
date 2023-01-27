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

# +
import underworld3 as uw
from petsc4py import PETSc
import numpy as np
import h5py



# -

from underworld3 import kdtree

# +
mesh1 = uw.discretisation.Mesh("./chkpt/test_checkpointing_np4.mesh.0.h5")

pm = uw.discretisation.MeshVariable("P", mesh1, 1, degree=1, continuous=True)
pm2 = uw.discretisation.MeshVariable("P2", mesh1, 1, degree=2, continuous=True)
um = uw.discretisation.MeshVariable("U", mesh1, mesh1.dim, degree=1, continuous=True)
um2 = uw.discretisation.MeshVariable("U2", mesh1, mesh1.dim, degree=2, continuous=True)


# + jupyter={"source_hidden": true} tags=[]
def load_from_vertex_checkpoint(self, mesh_file, data_file, data_name, data_degree):
    '''
    Read a mesh variable from an arbitrary vertex-based checkpoint file
    and reconstruct/interpolate the data field accordingly. The data sizes / meshes can be
    different and will be matched using a kd-tree / inverse-distance weighting
    to the new mesh. Mesh projection is used to map from linear variables to higher-degree
    ones. Currently, scalar and vector variables can be recovered.
    
    Note: data_name, data_degree refer to the checkpointed data - this can be inferred from
    the file if the checkpoints are one file per unknown, but that's not necessarily what
    we will be using.
    
    '''
    
    ## Sub functions that are used to read / interpolate the mesh.
    

    def field_from_checkpoint_vertices(mesh_file, 
                                       data_file=None, 
                                       data_name=None, 
                                       data_degree=1,):

        h5f = h5py.File(data_file)

        data_field_name = data_name + f"_P{data_degree}"  # What if it's not continuous ?

        D_vertex = h5f['vertex_fields'][data_field_name][()]
        h5f.close()

        file = "test_checkpointing_np4.mesh.0.h5"
        h5f = h5py.File(file)
        X_vertex = h5f['geometry']['vertices'][()]
        h5f.close()

        if len(D_vertex.shape) == 1:
            D_vertex = D_vertex.reshape(-1,1)

        return X_vertex, D_vertex


    def map_to_vertex_values(X_vertex, D_vertex):

        # An inverse-distance mapping is quite robust here ... as long
        # as we take care of the case where nodes coincide (quite likely)

        mesh1kdt = uw.kdtree.KDTree(mesh1.data)
        mesh1kdt.build_index()

        closest, distance, found = mesh1kdt.find_closest_point(X_vertex)

        num_local_vertices = mesh1.data.shape[0]
        data_size = D_vertex.shape[1]

        with mesh1.access():   
            Values = np.zeros((num_local_vertices, data_size))
            Weights = np.zeros((num_local_vertices, 1))

        epsilon = 1.0e-8
        for i in range(D_vertex.shape[0]):
            nearest = closest[i]
            Values[nearest,:] += D_vertex[i,:] / (epsilon+distance[i])
            Weights[nearest] += 1.0 / (epsilon+distance[i])

        Values[...] /= Weights[:]

        return Values


    def values_to_mesh_var(mesh_variable, Values):
        
        degree = mesh_variable.degree
        mesh = mesh_variable.mesh

        if degree == 1:
            with mesh.access(mesh_variable):
                mesh_variable.data[...] = Values[...]

        else:

            if mesh_variable.num_components == 1:

                vertex_var = mesh._work_MeshVar
                with mesh.access(vertex_var):
                    vertex_var.data[...] = Values[...]

                projector = uw.systems.Projection(mesh, mesh_variable)
                projector.uw_function = vertex_var.sym[0]
                projector.solve()

            elif mesh_variable.num_components == mesh.dim:

                vertex_var = mesh._work_MeshVec
                with mesh.access(vertex_var):
                    vertex_var.data[...] = Values[...]

                projector = uw.systems.Vector_Projection(mesh, mesh_variable)
                projector.uw_function = vertex_var.sym
                projector.solve()

            else:
                raise NotImplementedError()

            return
        
        
    X_vertex, D_vertex = field_from_checkpoint_vertices(
        mesh_file, data_file, data_name, data_degree)

    D = map_to_vertex_values(X_vertex, D_vertex)

    values_to_mesh_var(self, D)

    return


# +
pm.load_from_vertex_checkpoint("./chkpt/test_checkpointing_np4.mesh.0.h5", 
                                "./chkpt/test_checkpointing_np4.P.0.h5", "P", 1)

pm2.load_from_vertex_checkpoint("./chkpt/test_checkpointing_np4.mesh.0.h5", 
                                "./chkpt/test_checkpointing_np4.P.0.h5", "P", 1)

um.load_from_vertex_checkpoint("./chkpt/test_checkpointing_np4.mesh.0.h5", 
                                "./chkpt/test_checkpointing_np4.U.0.h5", "U", 2)

um2.load_from_vertex_checkpoint("./chkpt/test_checkpointing_np4.mesh.0.h5", 
                                "./chkpt/test_checkpointing_np4.U.0.h5", "U", 2)


# +
with mesh1.access():
    print((pm.data[::13,0]-pm.coords[::13,0]).T)
    
with mesh1.access():
    print((pm2.data[::43,0]-pm2.coords[::43,0]).T)
     
with mesh1.access():
    print((um.data[::13,1]-um.coords[::13,1]).T)

with mesh1.access():
    print((um2.data[::43,1]-um2.coords[::43,1]).T)

