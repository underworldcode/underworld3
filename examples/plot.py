# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
#Underworld3 plotting prototype
import lavavu
import numpy as np

def mesh_coords(mesh):
    cdim = mesh.dm.getCoordinateDim()
    #lcoords = mesh.dm.getCoordinatesLocal().array.reshape(-1,cdim)
    coords = mesh.dm.getCoordinates().array.reshape(-1,cdim)
    return coords

def mesh_edges(mesh):
    #1) EDGES
    #print('-- EDGES ----------')
    coords = mesh_coords(mesh)
    S = 1
    starti,endi = mesh.dm.getDepthStratum(S)
    #print(starti,endi,S)
    #for j in range(starti, endi):
    #    print(i, j, mesh.dm.getConeSize(j), mesh.dm.getCone(j), mesh.dm.getSupportSize(j), mesh.dm.getSupport(j))
    DIM = mesh.dm.getConeSize(starti)
    edges = np.zeros((endi-starti,DIM), dtype=PETSc.IntType)
    #print("DIM:",DIM, "SHAPE:",edges.shape)
    for c in range(starti, endi):
        #point_closure = mesh.dm.getTransitiveClosure(c)[0]
        #edges[c-starti,:] = point_closure[-DIM:] #-endi
        edges[c-starti,:] = mesh.dm.getCone(c)

    edges -= edges.min() #Why the offset?
    #print(edges)
    #print(edges.min(), edges.max(), coords.shape)
    return edges

def mesh_faces(mesh):
    #Faces / 2d cells
    coords = mesh_coords(mesh)
    cdim = mesh.dm.getCoordinateDim()

    #2) FACES (quads or tris)
    #print('-- FACES ----------')
    S = 2
    starti,endi = mesh.dm.getDepthStratum(S)
    #print(starti,endi,S)
    DIM = mesh.dm.getConeSize(starti)
    faces = np.zeros((endi-starti,DIM), dtype=PETSc.IntType)
    #print("DIM:",DIM, "SHAPE:",faces.shape)
    for c in range(starti, endi):
        point_closure = mesh.dm.getTransitiveClosure(c)[0]
        faces[c-starti,:] = point_closure[-DIM:] #-endi
        #This works for edges, but screws up ordering of faces
        #faces[c-starti,:] = mesh.dm.getCone(c)

    #print("RANGE",faces.min(), faces.max())
    faces -= faces.min() #Why the offset?

    if faces.shape[-1] == 3:
        #print('Plotting tris')
        tverts = de_index(faces, coords)
    elif faces.shape[-1] == 4:
        #print('Plotting quads')
        tfaces = quad2tri_indices(faces)
        tverts = de_index(tfaces, coords)
    return tverts, faces.shape[-1]

def quad2tri_indices(indices):
    #Convert quad indices to tri indices
    length = indices.shape[0]
    tris = np.zeros(dtype=np.uint32, shape=(length*2,3))
    for i in range(length):
        tris[i*2] = [indices[i][0], indices[i][1], indices[i][2]]
        tris[i*2+1] = [indices[i][2], indices[i][3], indices[i][0]]
    return tris

def de_index(indices, vertices):
    # Convert vertices+indices to vertices only
    # (can be points/lines/tris/quads/whatever)
    cdim = vertices.shape[-1]
    indices = indices.ravel()
    length = indices.size
    out = np.zeros(dtype=np.float32, shape=(length,cdim))
    for i in range(length):
        out[i] = vertices[indices[i]]
    return out

class Plot(lavavu.Viewer):
    def __init__(self, *args, **kwargs):
        super(Plot, self).__init__(*args, **kwargs)

    def nodes(self, mesh, **kwargs):
        #Plot nodes only as points
        return self.points('nodes', vertices=mesh_coords(mesh), **kwargs)

    def edges(self, mesh, **kwargs):
        #Mesh lines
        print(**kwargs)
        return self.lines('edges', vertices=mesh_coords(mesh), indices=mesh_edges(mesh), **kwargs)

    def faces(self, mesh, colourbar=True, **kwargs):
        faces, n = mesh_faces(mesh)
        if n == 3:
            return self.triangles('faces', vertices=faces, **kwargs)
        elif n == 4:
            return self.quads('faces', vertices=faces, **kwargs)

    def vector_arrows(self, mesh, vectors, **kwargs):
        #Create viewer
        coords = mesh_coords(mesh)
        return self.vectors('vectors', vertices=coords, vectors=vectors, **kwargs)

    def cells(self, mesh, **kwargs):
        #3d cells - volume
        return #TODO self.volume()

