#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# ## Creating DMPlex from .mesh file
#
# #### The purpose of this notebook is to provide a tutorial to load .mesh file data into DMPlex in PETSc.

# %%
from __future__ import print_function
import sys,petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
import time


# %%
from underworld3.utilities.read_medit_ascii import read_medit_ascii
from underworld3.utilities.read_medit_ascii import print_medit_mesh_info


# %%
# Mesh_from_Medit
# Mesh_from_Gmsh

# %%
def create_dmplex_from_medit(medit_file, print_medit_mesh_info=False):
    """
    Reads medit file (.mesh)
    Returns DMPlex of medit file with labels 
    """
    # print mesh file info
    if print_medit_mesh_info:
        print_medit_mesh_info(medit_file)
    
    # reading mesh vertices and indices
    vertices, vert_indx = read_medit_ascii(medit_file, 'Vertices')

    # reading mesh cells and indices
    cells, cells_indx = read_medit_ascii(medit_file, 'Tetrahedra')

    # reading mesh triangles and indices
    triangles, tria_indx = read_medit_ascii(medit_file, 'Triangles')

    # reading mesh edges and indices
    edges, edges_indx = read_medit_ascii(medit_file, 'Edges')

    # Note: default petsc installation requires int32 
    dim = vertices.shape[1]
    plex = PETSc.DMPlex().createFromCellList(dim, cells, vertices)

    # Label tet's
    tet_pStart, tet_pEnd = plex.getDepthStratum(3)
    plex.createLabel("TetraLabels")
    for i in range(tet_pStart, tet_pEnd):
        coneclose, orient = plex.getTransitiveClosure(i)
        conepoints = coneclose[-4:] - tet_pEnd
        
        if tuple(conepoints) == tuple(cells[i]):
            plex.setLabelValue("TetraLabels", i, 400 + cells_indx[i])
        else:
            print('orientation different')

    # following dmplex boundingbox formatting
    vert_labels_values = ((('LeftPts', 101), ('RightPts', 102)), 
                          (('FrontPts', 103), ('BackPts', 104)), 
                          (('BottomPts', 105), ('TopPts', 106)))

    def label_vertices(dm, vert_labels_values='', dim=3, atol=1e-6, ):
        """
        Label vertices
        """
        # Create labels
        for j in range(dim): # looping on dim
            for k in range(2): # looping on min, max
                dm.createLabel(vert_labels_values[j][k][0])
    
        # Get start and end indices of points in dmplex
        pStart, pEnd = plex.getDepthStratum(0)
        pNum = pEnd-pStart
        
        # Get coordinates array
        coords = dm.getCoordinates().array.reshape(pNum, dim)
        
        # Loop over all vertices
        for i, point in enumerate(range(pStart, pEnd)):
            coord = coords[i]
            
            # labeling vertices
            for j in range(dim): # looping on dim
                for k in range(2): # looping on min, max
                    bb_value = dm.getBoundingBox()[j][k]
                    if np.isclose(coord[j], bb_value, atol=atol):
                        label, value = vert_labels_values[j][k]
                        dm.setLabelValue(label, point, value)
        return

    # label vertices on 'top', 'bottom', 'left', 'right', 'front', and 'back' faces
    label_vertices(plex, vert_labels_values=vert_labels_values)

    def get_label_indices(dm, label_name, label_value):
        """
        Get label indices
        """
        # Get the label
        label = dm.getLabel(label_name)
        if label is None:
            print(f"Label '{label_name}' not found")
            return
    
        # Get the indices of all points with the label
        indices = dm.getStratumIS(label_name, label_value)
        if indices is None:
            print(f"No indices found for label '{label_name}'")
            return
    
        return indices.getIndices()

    def struct_arr(arr, int_typ):
        """
        Given an array and data type
        Returns structured array
        """
        # create data type
        dtype = []
        for i in range(arr.shape[1]):
            dtype.append((f'f{i}', int_typ))
    
        # Create structured arrays
        structured_arr = np.array([tuple(row) for row in np.sort(arr)], dtype=dtype)
        
        return structured_arr

    # Label triangles
    # face labels: following dmplex boundingbox formatting
    face_labels_values = ((('Left', 391), ('Right', 392)), 
                          (('Front', 393), ('Back', 394)), 
                          (('Bottom', 395), ('Top', 396)))
    
    # Create labels
    plex.createLabel("TriangleLabels")
    for j in range(dim): # looping on dim
        for k in range(2): # looping on min, max
            plex.createLabel(face_labels_values[j][k][0])
    
    # Create structured arrays
    struct_tria = struct_arr(triangles, np.int32)
    
    tri_pStart, tri_pEnd = plex.getDepthStratum(2)
    
    for i in range(tri_pStart, tri_pEnd):
        coneclose,orient = plex.getTransitiveClosure(i)
        conepoints = coneclose[-3:] - tet_pEnd
    
        # Create structured array of conepoints
        struct_cpts_arr = struct_arr(np.array([conepoints]), np.int32)
        
        # Check if rows in tri_mesh_data exist in conepoints array
        result = np.isin(struct_tria, struct_cpts_arr)
    
        if len(tria_indx[result])!=0:
            tri_label = tria_indx[result]
            plex.setLabelValue("TriangleLabels", i, 300 + tri_label[0])
        
        # labeling faces
        for j in range(dim): # looping on dim
            for k in range(2): # looping on min, max
                vert_label, vert_value = vert_labels_values[j][k]
                face_label, face_value = face_labels_values[j][k]
                tri_pt1 = conepoints[0] in get_label_indices(plex, vert_label, vert_value) - tet_pEnd
                tri_pt2 = conepoints[1] in get_label_indices(plex, vert_label, vert_value) - tet_pEnd
                tri_pt3 = conepoints[2] in get_label_indices(plex, vert_label, vert_value) - tet_pEnd
                if tri_pt1 & tri_pt2 & tri_pt3:
                    plex.setLabelValue(face_label, i, face_value)

    # Label edges
    # Create structured arrays
    struct_edges = struct_arr(edges, np.int32)
    
    edg_pStart, edg_pEnd = plex.getDepthStratum(1)
    plex.createLabel("LineLabels")
    for i in range(edg_pStart, edg_pEnd):
        coneclose,orient = plex.getTransitiveClosure(i)
        conepoints = coneclose[-2:] - tet_pEnd
    
        # Create structured array of conepoints
        struct_cpts_arr = struct_arr(np.array([conepoints]), np.int32)
        
        # Check if rows in edges exist in conepoints array
        result = np.isin(struct_edges, struct_cpts_arr)
    
        if len(edges_indx[result])!=0:
            line_label = edges_indx[result]
            plex.setLabelValue("LineLabels", i, 200 + line_label[0])
    
    return plex
