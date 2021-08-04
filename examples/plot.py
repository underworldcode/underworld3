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
    # coords = mesh_coords(mesh)
    # import pdb; pdb.set_trace()
    starti,endi = mesh.dm.getDepthStratum(1)
    #Offset of the node indices (level 0)
    coffset = mesh.dm.getDepthStratum(0)[0]
    edgesize = mesh.dm.getConeSize(starti)
    edges = np.zeros((endi-starti,edgesize), dtype=np.uint32)
    for c in range(starti, endi):
        edges[c-starti,:] = mesh.dm.getCone(c) - coffset

    #edges -= edges.min() #Why the offset?
    #print(edges)
    #print(edges.min(), edges.max(), coords.shape)
    return edges

def mesh_faces(mesh):
    #Faces / 2d cells
    coords = mesh_coords(mesh)
    #cdim = mesh.dm.getCoordinateDim()

    #Index range in mesh.dm of level 2
    starti,endi = mesh.dm.getDepthStratum(2)
    #Offset of the node indices (level 0)
    coffset = mesh.dm.getDepthStratum(0)[0]
    FACES=(endi-starti)
    facesize = mesh_facesize(mesh) # Face elements 3(tri) or 4(quad)
    faces = np.zeros((FACES,facesize), dtype=np.uint32)
    for c in range(starti, endi):
        point_closure = mesh.dm.getTransitiveClosure(c)[0]
        faces[c-starti,:] = point_closure[-facesize:] - coffset
    return faces

def face_count(mesh):
    starti,endi = mesh.dm.getDepthStratum(2)
    return endi-starti

def cell_count(mesh):
    depth = mesh.dm.getDepth()
    S = 3
    if depth < 3:
        S = 2
    starti,endi = mesh.dm.getDepthStratum(S)
    return endi-starti

def mesh_facesize(mesh):
    return mesh.dm.getConeSize(mesh.dm.getDepthStratum(2)[0]) #Face elements 3(tri) or 4(quad)

def mesh_cellsize(mesh):
    depth = mesh.dm.getDepth()
    if depth < 3:
        return mesh_facesize(mesh) #Cells are faces
    return mesh.dm.getConeSize(mesh.dm.getDepthStratum(3)[0])  #Cell elements 4(tet) or 6(cuboid)

def mesh_info(mesh):
    depth = mesh.dm.getDepth()
    sz = mesh.dm.getChart()
    print('getChart (index range)', sz, 'getDepth', depth)
    for i in range(depth+1):
        starti,endi = mesh.dm.getDepthStratum(i)
        conesize = mesh.dm.getConeSize(starti)
        print(i, "range: [", starti, endi, "] coneSize", conesize)

def mesh_cells(mesh):
    depth = mesh.dm.getDepth()
    if depth < 3:
        return mesh_faces(mesh)

    #Index range in mesh.dm of level 3
    starti,endi = mesh.dm.getDepthStratum(3)
    #Offset of the node indices (level 0)
    coffset = mesh.dm.getDepthStratum(0)[0]
    CELLS=(endi-starti)
    facesize = mesh_facesize(mesh) # Face elements 3(tri) or 4(quad)
    cellsize = mesh_cellsize(mesh) # Cell elements 4(tet) or 6(cuboid)
    FACES = CELLS * cellsize
    VERTS = FACES * 3
    INDICES = VERTS * 3

    #List of faces (vertex indices)
    faces = np.zeros((FACES,facesize), dtype=np.uint32)
    #print("CELLSIZE:", cellsize, "FACESIZE:",facesize, "SHAPE:",faces.shape)
    for c in range(CELLS):
        #The "cone" is the list of face indices for this cell
        cone = mesh.dm.getCone(c+starti)
        #print("CONE",cone)
        #Iterate through each face element of the cone
        for co in range(cellsize):
            #This contains the face vertex indices in correct order at the end
            point_closure = mesh.dm.getTransitiveClosure(cone[co])[0]
            #print("  CO", cellsize*c+co, co, cone[co], face, point_closure[-TRI_QUAD:] - coffset)
            faces[cellsize*c + co,:] = point_closure[-facesize:] - coffset

    return faces

def quad2tri_indices(indices):
    #Convert quad indices to tri indices
    length = indices.shape[0]
    tris = np.zeros(dtype=np.uint32, shape=(length*2,3))
    for i in range(length):
        tris[i*2] = [indices[i][0], indices[i][1], indices[i][2]]
        tris[i*2+1] = [indices[i][2], indices[i][3], indices[i][0]]
    return tris

def de_index(indices, vertices, values=None):
    # Convert vertices+indices to vertices only
    # (can be points/lines/tris/quads/whatever)
    cdim = vertices.shape[-1]
    indices = indices.ravel()
    length = indices.size
    out = np.zeros((length,cdim), dtype=np.float32)
    outvalues = np.zeros((length), dtype=np.float32)
    for i in range(length):
        out[i] = vertices[indices[i]]
        if values is not None:
            outvalues[i] = values[indices[i]]
    return out, outvalues

def cell_centres(mesh):
    #Return mesh cell centre vertices instead of nodes
    size = np.product(mesh.elementRes)
    cells = np.zeros((size, mesh.dim), dtype=np.float32)
    faces, n = mesh_faces(mesh)
    faces = faces.reshape((size, -1, mesh.dim))
    elsPerCell = faces.shape[1]
    for i in range(size):
        f = faces[i]
        cells[i] = np.mean(faces[i], axis=0)
    return cells

class Plot(lavavu.Viewer):
    def __init__(self, *args, **kwargs):
        super(Plot, self).__init__(*args, **kwargs)

    def nodes(self, mesh, values, **kwargs):
        #Plot nodes only as points
        return self.points('nodes', vertices=mesh_coords(mesh), values=values, **kwargs)

    def swarm_points(self, swarm, values=None, **kwargs):
        #Plot particles as points
        ptsobj = self.points('swarm_points', **kwargs)
        with swarm.access():
            ptsobj.vertices(swarm.particle_coordinates.data)
            ptsobj.values(values)
        return ptsobj

    def edges(self, mesh, **kwargs):
        #Mesh lines
        return self.lines('edges', vertices=mesh_coords(mesh), indices=mesh_edges(mesh), **kwargs)

    def _plot_elements(self, mesh, faces, values=None, **kwargs):
        #Plots faces or cells as triangles
        coords = mesh_coords(mesh)
        facesize = mesh_facesize(mesh) # Face elements 3(tri) or 4(quad)

        #Convert quads to triangles
        if facesize == 4:
            faces = quad2tri_indices(faces)

        #If value per vertex provided, need to duplicate values alongside vertices
        if values is not None and values.size == coords.shape[0]:
            tverts,values = de_index(faces, coords, values)
        else:
            tverts,_ = de_index(faces, coords)

        return self.mesh('faces', vertices=tverts, values=values, **kwargs)

    def faces(self, mesh, values=None, **kwargs):
        coords = mesh_coords(mesh)
        faces = mesh_faces(mesh)
        facesize = mesh_facesize(mesh) # Face elements 3(tri) or 4(quad)

        #Label elements by colour (can be face or cell, pass size)
        if values is None and "colourmap" in kwargs:
            values = np.arange(face_count(mesh))

        return self._plot_elements(mesh, faces, values=values, **kwargs)

    def cells(self, mesh, values=None, **kwargs):
        coords = mesh_coords(mesh)
        faces = mesh_cells(mesh)
        facesize = mesh_facesize(mesh) # Face elements 3(tri) or 4(quad)
        cellsize = mesh_cellsize(mesh) # Cell elements 4(tet) or 6(cuboid)

        #Label elements by colour (can be face or cell, pass size)
        if values is None and "colourmap" in kwargs:
            values = np.arange(cell_count(mesh))

        return self._plot_elements(mesh, faces, values=values, **kwargs)

    def vector_arrows(self, mesh, vectors, **kwargs):
        #Create viewer
        coords = mesh_coords(mesh)
        return self.vectors('vectors', vertices=coords, vectors=vectors, **kwargs)

    def cellvolume(self, mesh, values=None, **kwargs):
        #3d cells - volume
        return #TODO self.volume()


class _xvfb_runner(object):
    """
    This class will initialise the X virtual framebuffer (Xvfb).
    Xvfb is useful on headless systems. Note that xvfb will need to be 
    installed, as will pyvirtualdisplay.

    This class also manages the lifetime of the virtual display driver. When
    the object is garbage collected, the driver is stopped.
    """
    def __init__(self):
        from pyvirtualdisplay import Display
        self._xvfb = Display(visible=0, size=(1600, 1200))
        self._xvfb.start()

    def __del__(self):
        try:
            if not self._xvfb is None :
                self._xvfb.stop()
        except:
            pass

import os as _os
if "UW_XVFB_ENABLE" in _os.environ:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        _display = _xvfb_runner()
