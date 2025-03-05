#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
import underworld3 as uw
from mpi4py import MPI

from underworld3.utilities.read_medit_ascii import read_medit_ascii
from underworld3.utilities.read_medit_ascii import print_medit_mesh_info


def create_dmplex_from_medit(medit_file, print_info=False):
    """
    Reads a medit (.mesh) file and returns a distributed DMPlex with labels.
    """
    if print_info:
        print_medit_mesh_info(medit_file)

    # Read mesh data
    vertices, _ = read_medit_ascii(medit_file, 'Vertices')
    cells, cells_indx = read_medit_ascii(medit_file, 'Tetrahedra')
    triangles, triangles_indx = read_medit_ascii(medit_file, 'Triangles')
    edges, edges_indx = read_medit_ascii(medit_file, 'Edges')

    dim = vertices.shape[1]
    plex = PETSc.DMPlex().createFromCellList(dim, cells, vertices)

    # Label Tetrahedra
    plex.createLabel("TetraLabels")
    tet_Start, tet_End = plex.getDepthStratum(3)

    for i in range(tet_Start, tet_End):
        plex.setLabelValue("TetraLabels", i, 400 + cells_indx[i])

    # Define boundary labels
    vertices_labels_values = (
        (('LeftPts', 101), ('RightPts', 102)),
        (('FrontPts', 103), ('BackPts', 104)),
        (('BottomPts', 105), ('TopPts', 106))
    )

    def label_vertices(dm, labels_values, dim=3, atol=1e-6):
        """
        Efficiently label vertices in a distributed way.
        """
        pStart, pEnd = dm.getDepthStratum(0)
        coords = dm.getCoordinates().array.reshape(pEnd - pStart, dim)

        # PETSc requires all ranks to call getBoundingBox()
        uw.mpi.comm.barrier()
        bb_values = dm.getBoundingBox()

        for j in range(dim):
            for k in range(2):
                dm.createLabel(labels_values[j][k][0])

        # Label points
        for i, point in enumerate(range(pStart, pEnd)):
            coord = coords[i]
            for j in range(dim):
                for k in range(2):
                    if np.isclose(coord[j], bb_values[j][k], atol=atol):
                        label, value = labels_values[j][k]
                        dm.setLabelValue(label, point, value)

    # Label vertices
    label_vertices(plex, vertices_labels_values)

    # Fetch Label Indices Once (Cache them)
    def get_label_indices(dm, label_name, label_value):
        label = dm.getLabel(label_name)
        if label is None:
            return None
        indices = dm.getStratumIS(label_name, label_value)
        return indices.getIndices() if indices else None

    # Cache face and vertex labels to avoid repeated function calls
    label_cache = {}
    for dim_labels in vertices_labels_values:
        for name, value in dim_labels:
            indices = get_label_indices(plex, name, value) - tet_End
            label_cache[(name, value)] = indices

    # Label Triangles 
    face_labels_values = (
        (('Left', 391), ('Right', 392)),
        (('Front', 393), ('Back', 394)),
        (('Bottom', 395), ('Top', 396))
    )

    plex.createLabel("TriangleLabels")
    for j in range(dim):
        for k in range(2):
            plex.createLabel(face_labels_values[j][k][0])

    sorted_triangles = np.sort(triangles, axis=1)
    tri_Start, tri_End = plex.getDepthStratum(2)

    for i in range(tri_Start, tri_End):
        coneclose, _ = plex.getTransitiveClosure(i)
        conepoints = np.sort(coneclose[-3:] - tet_End)

        if np.any(np.all(sorted_triangles == conepoints, axis=1)):
            index_tri = np.where(np.all(sorted_triangles == conepoints, axis=1))[0][0] # fetch the where conepoints matches in sorted_triangles list
            plex.setLabelValue("TriangleLabels", i, 300 + triangles_indx[index_tri])

        # boundary labeling using cached indices
        for j in range(dim):
            for k in range(2):
                vertices_label, vertices_value = vertices_labels_values[j][k]
                face_label, face_value = face_labels_values[j][k]
                vertices_indices = label_cache.get((vertices_label, vertices_value), [])

                if conepoints[0] in vertices_indices and conepoints[1] in vertices_indices and conepoints[2] in vertices_indices:
                    plex.setLabelValue(face_label, i, face_value)

    # Label Edges
    sorted_edges = np.sort(edges, axis=1)
    edge_Start, edge_End = plex.getDepthStratum(1)
    plex.createLabel("LineLabels")

    for i in range(edge_Start, edge_End):
        coneclose, _ = plex.getTransitiveClosure(i)
        conepoints = np.sort(coneclose[-2:] - tet_End)

        if np.any(np.all(sorted_edges == conepoints, axis=1)):
            index_edge = np.where(np.all(sorted_edges == conepoints, axis=1))[0][0] # fetch the where conepoints matches in sorted_edges list
            plex.setLabelValue("LineLabels", i, 200 + edges_indx[index_edge])

    return plex