# cython: profile=False
from typing import Optional, Tuple
import os
import tempfile

import  numpy
import  numpy as np
cimport numpy as np

from mpi4py import MPI
from petsc4py import PETSc

include "./petsc_extras.pxi"

import underworld3
import underworld3 as uw 
from   underworld3 import _api_tools
from   underworld3.mesh import Mesh
import underworld3.timing as timing


def Unstructured_Simplex_Box(
                minCoords    :Tuple = (0., 0.),
                maxCoords    :Tuple = (1.0, 1.0),
                cell_size    :float = 0.1,
                regular      :bool  = True):
  
    """
    Generates a 2 or 3-dimensional box mesh.

    Parameters
    ----------
    minCoord:
        Tuple specifying minimum mesh location.
    maxCoord:
        Tuple specifying maximum mesh location.
    """

    boundaries = {
     "Bottom": 1,
     "Top": 2,
     "Right": 3,
     "Left": 4,
     "Front": 5,
     "Back": 6}

    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("Box")
    
    # Create Box Geometry
    dim = len(minCoords)

    if dim == 2:
        
        xmin, ymin = minCoords
        xmax, ymax = maxCoords
        
        p1 = gmsh.model.geo.add_point(xmin,ymin,0., meshSize=cell_size)
        p2 = gmsh.model.geo.add_point(xmax,ymin,0., meshSize=cell_size)
        p3 = gmsh.model.geo.add_point(xmin,ymax,0., meshSize=cell_size)
        p4 = gmsh.model.geo.add_point(xmax,ymax,0., meshSize=cell_size)

        l1 = gmsh.model.geo.add_line(p1, p2, tag=boundaries["Bottom"])
        l2 = gmsh.model.geo.add_line(p2, p4, tag=boundaries["Right"])
        l3 = gmsh.model.geo.add_line(p4, p3, tag=boundaries["Top"])
        l4 = gmsh.model.geo.add_line(p3, p1, tag=boundaries["Left"])

        cl = gmsh.model.geo.add_curve_loop((l1, l2, l3, l4))
        surface = gmsh.model.geo.add_plane_surface([cl])
        
        gmsh.model.geo.synchronize()

        # Add Physical groups
        for name, tag in boundaries.items():
            gmsh.model.add_physical_group(1, [tag] , tag)
            gmsh.model.set_physical_name(1, tag, name)

        gmsh.model.addPhysicalGroup(2, [surface], surface)
        gmsh.model.setPhysicalName(2, surface, "Elements")

        if regular:
            gmsh.model.mesh.set_transfinite_surface(surface, arrangement="Alternate", cornerTags=[p1,p2,p3,p4])

    else:
        
        xmin, ymin, zmin = minCoords
        xmax, ymax, zmax = maxCoords
        
        p1 = gmsh.model.geo.add_point(xmin,ymin,zmin, meshSize=cell_size)
        p2 = gmsh.model.geo.add_point(xmax,ymin,zmin, meshSize=cell_size)
        p3 = gmsh.model.geo.add_point(xmin,ymax,zmin, meshSize=cell_size)
        p4 = gmsh.model.geo.add_point(xmax,ymax,zmin, meshSize=cell_size)
        p5 = gmsh.model.geo.add_point(xmin,ymin,zmax, meshSize=cell_size)
        p6 = gmsh.model.geo.add_point(xmax,ymin,zmax, meshSize=cell_size)
        p7 = gmsh.model.geo.add_point(xmin,ymax,zmax, meshSize=cell_size)
        p8 = gmsh.model.geo.add_point(xmax,ymax,zmax, meshSize=cell_size)    

        l1 = gmsh.model.geo.add_line(p1, p2)
        l2 = gmsh.model.geo.add_line(p2, p4)
        l3 = gmsh.model.geo.add_line(p4, p3)
        l4 = gmsh.model.geo.add_line(p3, p1)
        l5 = gmsh.model.geo.add_line(p5, p6)
        l6 = gmsh.model.geo.add_line(p6, p8)
        l7 = gmsh.model.geo.add_line(p8, p7)
        l8 = gmsh.model.geo.add_line(p7, p5)    
        l9 = gmsh.model.geo.add_line(p5, p1)
        l10 = gmsh.model.geo.add_line(p2, p6)
        l11 = gmsh.model.geo.add_line(p7, p3)
        l12 = gmsh.model.geo.add_line(p4, p8) 
    
        
        cl = gmsh.model.geo.add_curve_loop((l1, l2, l3, l4))
        bottom = gmsh.model.geo.add_plane_surface([cl], tag=boundaries["Bottom"])

        cl = gmsh.model.geo.add_curve_loop((l5, l6, l7, l8))
        top = gmsh.model.geo.add_plane_surface([cl], tag=boundaries["Top"]) 
        
        cl = gmsh.model.geo.add_curve_loop((l10, l6, -l12, -l2))
        right = gmsh.model.geo.add_plane_surface([cl], tag=boundaries["Right"]) 
        
        cl = gmsh.model.geo.add_curve_loop((l9, -l4, -l11, l8))
        left = gmsh.model.geo.add_plane_surface([cl], tag=boundaries["Left"]) 
        
        cl = gmsh.model.geo.add_curve_loop((l1, l10, -l5, l9))
        front = gmsh.model.geo.add_plane_surface([cl], tag=boundaries["Front"])  
        
        cl = gmsh.model.geo.add_curve_loop((-l3, l12, l7, l11))
        back = gmsh.model.geo.add_plane_surface([cl], tag=boundaries["Back"]) 
        
            
        sloop = gmsh.model.geo.add_surface_loop([front, right, back, top, left, bottom])
        volume = gmsh.model.geo.add_volume([sloop])
        
        gmsh.model.geo.synchronize()
        
        # Add Physical groups
        for name, tag in boundaries.items():
            gmsh.model.add_physical_group(2, [tag] , tag)
            gmsh.model.set_physical_name(2, tag, name)

        gmsh.model.addPhysicalGroup(3, [volume], volume)
        gmsh.model.setPhysicalName(3, volume, "Elements")
        
    # Generate Mesh
    with tempfile.NamedTemporaryFile(mode="w", suffix=".msh") as fp:
        gmsh.model.mesh.generate(dim) 
        gmsh.write(fp.name)
        gmsh.finalize()
        plex = PETSc.DMPlex().createFromFile(fp.name)
    
    for name, tag in boundaries.items():
        plex.createLabel(name)
        label = plex.getLabel(name)
        indexSet = plex.getStratumIS("Face Sets", tag)
        if indexSet:
            label.insertIS(indexSet, 1)
        else:
            plex.removeLabel(name)

    plex.removeLabel("Face Sets")     

    return Mesh(plex, simplex=True)
 

def Structured_Quad_Box(elementRes, minCoords, maxCoords):
    
    import gmsh

    boundaries = {
     "Bottom": 1,
     "Top": 2,
     "Right": 3,
     "Left": 4,
     "Front": 5,
     "Back": 6}
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("Box")
    
    # Create Box Geometry
    dim = len(minCoords)

    if dim == 2:
        
        xmin, ymin = minCoords
        xmax, ymax = maxCoords
        
        p1 = gmsh.model.geo.add_point(xmin,ymin,0., tag=1)
        p2 = gmsh.model.geo.add_point(xmax,ymin,0., tag=2)
        p3 = gmsh.model.geo.add_point(xmin,ymax,0., tag=3)
        p4 = gmsh.model.geo.add_point(xmax,ymax,0., tag=4)

        l1 = gmsh.model.geo.add_line(p1, p2, tag=boundaries["Bottom"])
        l2 = gmsh.model.geo.add_line(p2, p4, tag=boundaries["Right"])
        l3 = gmsh.model.geo.add_line(p4, p3, tag=boundaries["Top"])
        l4 = gmsh.model.geo.add_line(p3, p1, tag=boundaries["Left"])

        cl = gmsh.model.geo.add_curve_loop((l1, l2, l3, l4))
        surface = gmsh.model.geo.add_plane_surface([cl])

        gmsh.model.geo.synchronize()

        # Add Physical groups
        gmsh.model.add_physical_group(1, [l1] , l1)
        gmsh.model.set_physical_name(1, l1, "Bottom")
        gmsh.model.add_physical_group(1, [l2] , l2)
        gmsh.model.set_physical_name(1, l2, "Right")
        gmsh.model.add_physical_group(1, [l3] , l3)
        gmsh.model.set_physical_name(1, l3, "Top")
        gmsh.model.add_physical_group(1, [l4] , l4)
        gmsh.model.set_physical_name(1, l4, "Left")

        gmsh.model.add_physical_group(2, [surface] , surface)
        gmsh.model.set_physical_name(2, surface, "Elements")

        nx, ny = elementRes

        gmsh.model.mesh.set_transfinite_curve(tag=l1, numNodes=nx+1, meshType="Progression", coef=1.0)
        gmsh.model.mesh.set_transfinite_curve(tag=l2, numNodes=ny+1, meshType="Progression", coef=1.0)
        gmsh.model.mesh.set_transfinite_curve(tag=l3, numNodes=nx+1, meshType="Progression", coef=1.0)
        gmsh.model.mesh.set_transfinite_curve(tag=l4, numNodes=ny+1, meshType="Progression", coef=1.0)
        gmsh.model.mesh.set_transfinite_surface(tag=surface, arrangement="Left", cornerTags=[p1,p2,p3,p4])
        gmsh.model.mesh.set_recombine(2, surface) 
        

    else:
        
        xmin, ymin, zmin = minCoords
        xmax, ymax, zmax = maxCoords
        
        p1 = gmsh.model.geo.add_point(xmin,ymin,zmin)
        p2 = gmsh.model.geo.add_point(xmax,ymin,zmin)
        p3 = gmsh.model.geo.add_point(xmin,ymax,zmin)
        p4 = gmsh.model.geo.add_point(xmax,ymax,zmin)
        p5 = gmsh.model.geo.add_point(xmin,ymin,zmax)
        p6 = gmsh.model.geo.add_point(xmax,ymin,zmax)
        p7 = gmsh.model.geo.add_point(xmin,ymax,zmax)
        p8 = gmsh.model.geo.add_point(xmax,ymax,zmax)    

        l1 = gmsh.model.geo.add_line(p1, p2)
        l2 = gmsh.model.geo.add_line(p2, p4)
        l3 = gmsh.model.geo.add_line(p4, p3)
        l4 = gmsh.model.geo.add_line(p3, p1)
        l5 = gmsh.model.geo.add_line(p5, p6)
        l6 = gmsh.model.geo.add_line(p6, p8)
        l7 = gmsh.model.geo.add_line(p8, p7)
        l8 = gmsh.model.geo.add_line(p7, p5)    
        l9 = gmsh.model.geo.add_line(p5, p1)
        l10 = gmsh.model.geo.add_line(p2, p6)
        l11 = gmsh.model.geo.add_line(p7, p3)
        l12 = gmsh.model.geo.add_line(p4, p8) 
    
        
        cl = gmsh.model.geo.add_curve_loop((l1, l2, l3, l4))
        bottom = gmsh.model.geo.add_plane_surface([cl], tag=boundaries["Bottom"])

        cl = gmsh.model.geo.add_curve_loop((l5, l6, l7, l8))
        top = gmsh.model.geo.add_plane_surface([cl], tag=boundaries["Top"]) 
        
        cl = gmsh.model.geo.add_curve_loop((l10, l6, -l12, -l2))
        right = gmsh.model.geo.add_plane_surface([cl], tag=boundaries["Right"]) 
        
        cl = gmsh.model.geo.add_curve_loop((l9, -l4, -l11, l8))
        left = gmsh.model.geo.add_plane_surface([cl], tag=boundaries["Left"]) 
        
        cl = gmsh.model.geo.add_curve_loop((l1, l10, -l5, l9))
        front = gmsh.model.geo.add_plane_surface([cl], tag=boundaries["Front"])  
        
        cl = gmsh.model.geo.add_curve_loop((-l3, l12, l7, l11))
        back = gmsh.model.geo.add_plane_surface([cl], tag=boundaries["Back"]) 
        
            
        sloop = gmsh.model.geo.add_surface_loop([front, right, back, top, left, bottom])
        volume = gmsh.model.geo.add_volume([sloop])
        
        gmsh.model.geo.synchronize()
        
        nx, ny, nz = elementRes

        gmsh.model.mesh.set_transfinite_curve(l1, numNodes=nx+1, meshType="Progression", coef=1.0)
        gmsh.model.mesh.set_transfinite_curve(l2, numNodes=ny+1, meshType="Progression", coef=1.0)
        gmsh.model.mesh.set_transfinite_curve(l3, numNodes=nx+1, meshType="Progression", coef=1.0)
        gmsh.model.mesh.set_transfinite_curve(l4, numNodes=ny+1, meshType="Progression", coef=1.0)
        gmsh.model.mesh.set_transfinite_curve(l5, numNodes=nx+1, meshType="Progression", coef=1.0)
        gmsh.model.mesh.set_transfinite_curve(l6, numNodes=ny+1, meshType="Progression", coef=1.0)
        gmsh.model.mesh.set_transfinite_curve(l7, numNodes=nx+1, meshType="Progression", coef=1.0)
        gmsh.model.mesh.set_transfinite_curve(l8, numNodes=ny+1, meshType="Progression", coef=1.0)
        gmsh.model.mesh.set_transfinite_curve(l9, numNodes=nz+1, meshType="Progression", coef=1.0)
        gmsh.model.mesh.set_transfinite_curve(l10, numNodes=nz+1, meshType="Progression", coef=1.0)
        gmsh.model.mesh.set_transfinite_curve(l11, numNodes=nz+1, meshType="Progression", coef=1.0)
        gmsh.model.mesh.set_transfinite_curve(l12, numNodes=nz+1, meshType="Progression", coef=1.0)

        gmsh.model.mesh.set_transfinite_surface(tag=bottom, arrangement="Left", cornerTags=[p1,p2,p4,p3])
        gmsh.model.mesh.set_transfinite_surface(tag=top, arrangement="Left", cornerTags=[p5,p6,p8,p7])
        gmsh.model.mesh.set_transfinite_surface(tag=front, arrangement="Left", cornerTags=[p1,p2,p6,p5])
        gmsh.model.mesh.set_transfinite_surface(tag=back, arrangement="Left", cornerTags=[p3,p4,p8,p7])
        gmsh.model.mesh.set_transfinite_surface(tag=right, arrangement="Left", cornerTags=[p2,p6,p8,p4])
        gmsh.model.mesh.set_transfinite_surface(tag=left, arrangement="Left", cornerTags=[p5,p1,p3,p7])

        gmsh.model.mesh.set_recombine(2, front)
        gmsh.model.mesh.set_recombine(2, back)
        gmsh.model.mesh.set_recombine(2, bottom)
        gmsh.model.mesh.set_recombine(2, top)
        gmsh.model.mesh.set_recombine(2, right)
        gmsh.model.mesh.set_recombine(2, left)
        
        gmsh.model.mesh.set_transfinite_volume(volume, cornerTags=[p1, p2, p4, p3, p5, p6, p8, p7])        
        
        # Add Physical groups
        for name, tag in boundaries.items():
            gmsh.model.add_physical_group(2, [tag] , tag)
            gmsh.model.set_physical_name(2, tag, name)

        gmsh.model.addPhysicalGroup(3, [volume], volume)
        gmsh.model.setPhysicalName(3, volume, "Elements")
        
    # Generate Mesh
    with tempfile.NamedTemporaryFile(mode="w", suffix=".msh") as fp:
        gmsh.model.mesh.generate(dim) 
        gmsh.write(fp.name)
        gmsh.finalize()
        plex = PETSc.DMPlex().createFromFile(fp.name)
    
    for name, tag in boundaries.items():
        plex.createLabel(name)
        label = plex.getLabel(name)
        indexSet = plex.getStratumIS("Face Sets", tag)
        if indexSet:
            label.insertIS(indexSet, 1)
        else:
            plex.removeLabel(name)

    plex.removeLabel("Face Sets")     

    return Mesh(plex, simplex=False)


def SphericalShell(radius_outer, radius_inner, cell_size):

    boundaries = {
     "Lower": 1,
     "Upper": 2}
    
    import gmsh
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("Sphere")

    ball1_tag = gmsh.model.occ.addSphere(0, 0, 0, radius_outer)

    if radius_inner > 0.0:
        ball2_tag = gmsh.model.occ.addSphere(0, 0, 0, radius_inner)
        gmsh.model.occ.cut([(3, ball1_tag)], [(3, ball2_tag)], removeObject=True, removeTool=True)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cell_size)
    gmsh.model.occ.synchronize()

    innerSurface, outerSurface = gmsh.model.getEntities(2)
    volume = gmsh.model.getEntities(3)[0]
    gmsh.model.addPhysicalGroup(innerSurface[0], [innerSurface[1]], boundaries["Lower"])
    gmsh.model.setPhysicalName(innerSurface[1], boundaries["Lower"], "Lower")
    gmsh.model.addPhysicalGroup(outerSurface[0], [outerSurface[1]], boundaries["Upper"])
    gmsh.model.setPhysicalName(outerSurface[1], boundaries["Upper"], "Upper")
    gmsh.model.addPhysicalGroup(volume[0], [volume[1]], volume[1])
    gmsh.model.setPhysicalName(volume[1], volume[1], "Elements")

    gmsh.model.occ.synchronize()
    
    # Generate Mesh
    with tempfile.NamedTemporaryFile(mode="w", suffix=".msh") as fp:
        gmsh.model.mesh.generate(3) 
        gmsh.write(fp.name)
        gmsh.finalize()
        plex = PETSc.DMPlex().createFromFile(fp.name)
    
    for name, tag in boundaries.items():
        plex.createLabel(name)
        label = plex.getLabel(name)
        indexSet = plex.getStratumIS("Face Sets", tag)
        if indexSet:
            label.insertIS(indexSet, 1)
        else:
            plex.removeLabel(name)

    plex.removeLabel("Face Sets")     

    return Mesh(plex, simplex=True)


def Annulus(radius_outer, radius_inner, cell_size):

    boundaries = {
     "Lower": 1,
     "Upper": 2}

    import gmsh
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("Annulus")

    p1 = gmsh.model.geo.add_point(0.0,0.0,0.0, meshSize=cell_size)
    p2 = gmsh.model.geo.add_point(radius_inner, 0.0, 0.0, meshSize=cell_size)
    p3 = gmsh.model.geo.add_point(-radius_inner, 0.0, 0.0, meshSize=cell_size)

    c1 = gmsh.model.geo.add_circle_arc(p2, p1, p3)
    c2 = gmsh.model.geo.add_circle_arc(p3, p1, p2)

    cl1 = gmsh.model.geo.add_curve_loop([c1, c2], tag=boundaries["Lower"])
    
    p4 = gmsh.model.geo.add_point(radius_outer, 0.0, 0.0, meshSize=cell_size)
    p5 = gmsh.model.geo.add_point(-radius_outer, 0.0, 0.0, meshSize=cell_size)

    c3 = gmsh.model.geo.add_circle_arc(p4, p1, p5)
    c4 = gmsh.model.geo.add_circle_arc(p5, p1, p4)

    cl2 = gmsh.model.geo.add_curve_loop([c3, c4], tag=boundaries["Upper"])
    
    s = gmsh.model.geo.add_plane_surface([cl2, cl1])        
    gmsh.model.geo.synchronize()
    
    gmsh.model.addPhysicalGroup(1, [c1, c2], boundaries["Lower"])
    gmsh.model.setPhysicalName(1, boundaries["Lower"], "Lower")
    gmsh.model.addPhysicalGroup(1, [c3, c4], boundaries["Upper"])
    gmsh.model.setPhysicalName(1, boundaries["Lower"], "Upper")
    gmsh.model.addPhysicalGroup(2, [s], s)
    gmsh.model.setPhysicalName(2, s, "Elements")

    # Generate Mesh
    with tempfile.NamedTemporaryFile(mode="w", suffix=".msh") as fp:
        gmsh.model.mesh.generate(2) 
        gmsh.write(fp.name)
        gmsh.finalize()
        plex = PETSc.DMPlex().createFromFile(fp.name)
    
    for name, tag in boundaries.items():
        plex.createLabel(name)
        label = plex.getLabel(name)
        indexSet = plex.getStratumIS("Face Sets", tag)
        if indexSet:
            label.insertIS(indexSet, 1)
        else:
            plex.removeLabel(name)

    plex.removeLabel("Face Sets")     

    return Mesh(plex, simplex=True)