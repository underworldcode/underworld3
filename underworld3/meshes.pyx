# cython: profile=False
from typing import Optional, Tuple
from tempfile import gettempdir
import os

import  numpy
import  numpy as np
cimport numpy as np

from mpi4py import MPI
from petsc4py import PETSc

include "./petsc_extras.pxi"

import underworld3
import underworld3 as uw 
from   underworld3 import _api_tools
from   underworld3.mesh import MeshClass
import underworld3.timing as timing


tmpdir = gettempdir()

class MeshFromGmshFile(MeshClass):

    @timing.routine_timer_decorator
    def __init__(self,
                 filename      :str,
                 verbose       :bool  = False,
                 degree        :int = 1
                ):
        """
        This is a generic mesh class for which users will provide 
        the mesh as a gmsh (.msh) file.

            - the file pointed to by filename needs to be a .msh file 
            - labels are extracted from the gmsh file "physical labels"
            - Note that the PETSc gmsh reader does not honour membership of multiple 
              physical groups as indicated in a gmsh file - only the first one is used 

        """
        self.verbose = verbose
        self.filename = filename
        self.degree = degree

        options = PETSc.Options()
        self.dm = PETSc.DMPlex().createFromFile(self.filename, interpolate=True)

        part = self.dm.getPartitioner()
        part.setFromOptions()
        self.dm.distribute()
        self.dm.setFromOptions()

        import gmsh
        
        gmsh.initialize()
        
        if not self.verbose:
            gmsh.option.setNumber("General.Verbosity", 0)     
        
        gmsh.open(self.filename)

        # Extract Physical groups from the gmsh file
        physical_groups = {}
        for dim, tag in gmsh.model.get_physical_groups():
            name = gmsh.model.get_physical_name(dim, tag)
            
            physical_groups[name] = tag
            self.dm.createLabel(name)
            label = self.dm.getLabel(name)
            
            for elem in ["Face Sets", "Vertex Sets"]:
                indexSet = self.dm.getStratumIS(elem, tag)
                if indexSet:
                    label.insertIS(indexSet, 1)
                indexSet.destroy()

        if self.verbose:
            self.dm.view()

        self.simplex = any(x in gmsh.model.mesh.get_element_types() for x in [2, 4])

        gmsh.finalize()

        self.vtk = self._convert2vtk(self.filename) 

        super().__init__(simplex=self.simplex, degree=self.degree)

    @staticmethod
    def _convert2vtk(filename):
        import gmsh
        gmsh_filename = filename
        vtk_filename = filename.split(".")[0] + ".vtk"

        gmsh.initialize()
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.open(gmsh_filename)
        gmsh.write(vtk_filename) 
        gmsh.finalize()
        return vtk_filename


class UnstructuredBox(MeshFromGmshFile):
  
    @timing.routine_timer_decorator
    def __init__(self, 
                minCoords    :Tuple = (0., 0.),
                maxCoords    :Tuple = (1.0, 1.0),
                cell_size    :float = 0.1,
                regular      :bool  = True,
                ):
        """
        Generates a 2 or 3-dimensional box mesh.

        Parameters
        ----------
        elementRes:
            Tuple specifying number of elements in each axis direction.
        minCoord:
            Optional. Tuple specifying minimum mesh location.
        maxCoord:
            Optional. Tuple specifying maximum mesh location.
        simplex:
            If `True`, simplex elements are used, and otherwise quad or 
            hex elements. 
        """
        
        self.cell_size = cell_size
        self.regular = regular

        self.minCoords = minCoords
        self.maxCoords = maxCoords

        self.filename = os.path.join(tmpdir, "unstructuredboxmesh_" + str(hash(self)) + ".msh")
        self._build_gmsh_file(self.cell_size, 
                              self.minCoords, 
                              self.maxCoords, 
                              self.regular,
                              self.filename)
        
        super().__init__(filename=self.filename)

    @staticmethod
    def _build_gmsh_file(cell_size, minCoords, maxCoords, regular, filename):
    
        import gmsh

        gmsh.initialize()
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("Box")
        
        # Create Box Geometry
        dim = len(minCoords)

        if dim == 2:
            
            xmin, ymin = minCoords
            xmax, ymax = maxCoords
            
            p1 = gmsh.model.geo.add_point(xmin,ymin,0., meshSize=cell_size, tag=1)
            p2 = gmsh.model.geo.add_point(xmax,ymin,0., meshSize=cell_size, tag=2)
            p3 = gmsh.model.geo.add_point(xmin,ymax,0., meshSize=cell_size, tag=3)
            p4 = gmsh.model.geo.add_point(xmax,ymax,0., meshSize=cell_size, tag=4)

            l1 = gmsh.model.geo.add_line(p1, p2, tag=1)
            l2 = gmsh.model.geo.add_line(p2, p4, tag=2)
            l3 = gmsh.model.geo.add_line(p4, p3, tag=3)
            l4 = gmsh.model.geo.add_line(p3, p1, tag=4)

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

            gmsh.model.addPhysicalGroup(0, [p1, p2], l1)
            gmsh.model.setPhysicalName(0, l1, "Bottom")
            gmsh.model.addPhysicalGroup(0, [p3, p4], l3)
            gmsh.model.setPhysicalName(0, l3, "Top")

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
            front = gmsh.model.geo.add_plane_surface([cl])
            cl = gmsh.model.geo.add_curve_loop((l5, l6, l7, l8))
            back = gmsh.model.geo.add_plane_surface([cl]) 
            cl = gmsh.model.geo.add_curve_loop((l1, l10, -l5, l9))
            bottom = gmsh.model.geo.add_plane_surface([cl])  
            cl = gmsh.model.geo.add_curve_loop((-l3, l12, l7, l11))
            top = gmsh.model.geo.add_plane_surface([cl]) 
            cl = gmsh.model.geo.add_curve_loop((l10, l6, -l12, -l2))
            right = gmsh.model.geo.add_plane_surface([cl]) 
            cl = gmsh.model.geo.add_curve_loop((l9, -l4, -l11, l8))
            left = gmsh.model.geo.add_plane_surface([cl]) 
                
            sloop = gmsh.model.geo.add_surface_loop([front, right, back, top, left, bottom])
            volume = gmsh.model.geo.add_volume([sloop])
            
            gmsh.model.geo.synchronize()
            
            # Add Physical groups
            gmsh.model.add_physical_group(2, [front] , front)
            gmsh.model.set_physical_name(2, front, "Front")
            gmsh.model.add_physical_group(2, [back] , back)
            gmsh.model.set_physical_name(2, back, "Back")
            gmsh.model.add_physical_group(2, [bottom] , bottom)
            gmsh.model.set_physical_name(2, bottom, "Bottom")
            gmsh.model.add_physical_group(2, [top] , top)
            gmsh.model.set_physical_name(2, top, "Top")
            gmsh.model.add_physical_group(2, [right] , right)
            gmsh.model.set_physical_name(2, right, "Right")
            gmsh.model.add_physical_group(2, [left] , left)
            gmsh.model.set_physical_name(2, left, "Left")
            
        # Generate Mesh
        gmsh.model.mesh.generate(dim) 
        gmsh.write(filename)
        gmsh.finalize()


class StructuredBox(MeshFromGmshFile):
  
    @timing.routine_timer_decorator
    def __init__(self, 
                elementRes   :Tuple = (16, 16), 
                minCoords    :Tuple = (0.,0.),
                maxCoords    :Tuple = (1.0, 1.0),
                ):
        """
        Generates a 2 or 3-dimensional box mesh.

        Parameters
        ----------
        elementRes:
            Tuple specifying number of elements in each axis direction.
        minCoord:
            Optional. Tuple specifying minimum mesh location.
        maxCoord:
            Optional. Tuple specifying maximum mesh location.
        simplex:
            If `True`, simplex elements are used, and otherwise quad or 
            hex elements. 
        """
        
        self.elementRes = elementRes

        self.minCoords = minCoords
        self.maxCoords = maxCoords

        self.filename = os.path.join(tmpdir, "structuredboxmesh_" + str(hash(self)) + ".msh")
        self._build_gmsh_file(self.elementRes,
                              self.minCoords, 
                              self.maxCoords, 
                              self.filename)
        
        super().__init__(filename=self.filename)

    @staticmethod
    def _build_gmsh_file(elementRes, minCoords, maxCoords, filename):
    
        import gmsh

        gmsh.initialize()
        gmsh.option.setNumber("Mesh.SaveAll", 1)
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

            l1 = gmsh.model.geo.add_line(p1, p2, tag=1)
            l2 = gmsh.model.geo.add_line(p2, p4, tag=2)
            l3 = gmsh.model.geo.add_line(p4, p3, tag=3)
            l4 = gmsh.model.geo.add_line(p3, p1, tag=4)

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

            gmsh.model.addPhysicalGroup(0, [p1, p2], l1)
            gmsh.model.setPhysicalName(0, l1, "Bottom")
            gmsh.model.addPhysicalGroup(0, [p3, p4], l3)
            gmsh.model.setPhysicalName(0, l3, "Top")

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
            front = gmsh.model.geo.add_plane_surface([cl])
            cl = gmsh.model.geo.add_curve_loop((l5, l6, l7, l8))
            back = gmsh.model.geo.add_plane_surface([cl]) 
            cl = gmsh.model.geo.add_curve_loop((l1, l10, -l5, l9))
            bottom = gmsh.model.geo.add_plane_surface([cl])  
            cl = gmsh.model.geo.add_curve_loop((-l3, l12, l7, l11))
            top = gmsh.model.geo.add_plane_surface([cl]) 
            cl = gmsh.model.geo.add_curve_loop((l10, l6, -l12, -l2))
            right = gmsh.model.geo.add_plane_surface([cl]) 
            cl = gmsh.model.geo.add_curve_loop((l9, -l4, -l11, l8))
            left = gmsh.model.geo.add_plane_surface([cl]) 
                
            sloop = gmsh.model.geo.add_surface_loop([front, right, back, top, left, bottom])
            volume = gmsh.model.geo.add_volume([sloop])
            
            gmsh.model.geo.synchronize()
            
            # Add Physical groups
            gmsh.model.add_physical_group(2, [front] , front)
            gmsh.model.set_physical_name(2, front, "Front")
            gmsh.model.add_physical_group(2, [back] , back)
            gmsh.model.set_physical_name(2, back, "Back")
            gmsh.model.add_physical_group(2, [bottom] , bottom)
            gmsh.model.set_physical_name(2, bottom, "Bottom")
            gmsh.model.add_physical_group(2, [top] , top)
            gmsh.model.set_physical_name(2, top, "Top")
            gmsh.model.add_physical_group(2, [right] , right)
            gmsh.model.set_physical_name(2, right, "Right")
            gmsh.model.add_physical_group(2, [left] , left)
            gmsh.model.set_physical_name(2, left, "Left")
            
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

            gmsh.model.mesh.set_transfinite_surface(tag=front, arrangement="Left", cornerTags=[p1,p2,p4,p3])
            gmsh.model.mesh.set_transfinite_surface(tag=back, arrangement="Left", cornerTags=[p5,p6,p8,p7])
            gmsh.model.mesh.set_transfinite_surface(tag=bottom, arrangement="Left", cornerTags=[p1,p2,p6,p5])
            gmsh.model.mesh.set_transfinite_surface(tag=top, arrangement="Left", cornerTags=[p3,p4,p8,p7])
            gmsh.model.mesh.set_transfinite_surface(tag=right, arrangement="Left", cornerTags=[p2,p6,p8,p4])
            gmsh.model.mesh.set_transfinite_surface(tag=left, arrangement="Left", cornerTags=[p5,p1,p3,p7])

            gmsh.model.mesh.set_recombine(2, front)
            gmsh.model.mesh.set_recombine(2, back)
            gmsh.model.mesh.set_recombine(2, bottom)
            gmsh.model.mesh.set_recombine(2, top)
            gmsh.model.mesh.set_recombine(2, right)
            gmsh.model.mesh.set_recombine(2, left)

            gmsh.model.mesh.set_transfinite_volume(volume, cornerTags=[p1, p2, p4, p3, p5, p6, p8, p7])        
            
        # Generate Mesh
        gmsh.model.mesh.generate(dim) 
        gmsh.write(filename)
        gmsh.finalize()


class Sphere(MeshFromGmshFile):
  
    @timing.routine_timer_decorator
    def __init__(self, 
                radius_outer     :float = 1.0,
                radius_inner     :float = 0.0,
                cell_size        :float = 0.1,
                ):
        """
        This class generates a spherical shell, or a full sphere
        where the inner radius is zero.
        Parameters
        ----------
        radius_outer :
            The outer radius for the spherical shell.
        radius_inner :
            The inner radius for the spherical shell. If this is set to 
            zero, a full sphere is generated.
        cell_size :
            The target cell size for the final mesh. Mesh refinements will occur to achieve this target 
            resolution.
        """
        
        if radius_inner >= radius_outer:
            raise ValueError("`radius_inner` must be smaller than `radius_outer`.")  

        self.radius_outer = radius_outer
        self.radius_inner = radius_inner
        self.cell_size = cell_size

        self.filename = os.path.join(tmpdir, "spheremesh_" + str(hash(self)) + ".msh")
        self._build_gmsh_file(self.radius_outer, 
                              self.radius_inner, 
                              self.cell_size, 
                              self.filename)
        
        super().__init__(filename=self.filename)

    @staticmethod
    def _build_gmsh_file(radius_outer, radius_inner, cell_size, filename):

        import gmsh
        
        gmsh.initialize()
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("Sphere")

        ball1_tag = gmsh.model.occ.addSphere(0, 0, 0, radius_outer)

        if radius_inner > 0.0:
            ball2_tag = gmsh.model.occ.addSphere(0, 0, 0, radius_inner)
            gmsh.model.occ.cut([(3, ball1_tag)], [(3, ball2_tag)], removeObject=True, removeTool=True)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cell_size)
        gmsh.model.occ.synchronize()

        innerSurface, outerSurface = gmsh.model.getEntities(2)
        gmsh.model.addPhysicalGroup(innerSurface[0], [innerSurface[1]], innerSurface[1])
        gmsh.model.setPhysicalName(innerSurface[1], innerSurface[1], "Lower")
        gmsh.model.addPhysicalGroup(outerSurface[0], [outerSurface[1]], outerSurface[1])
        gmsh.model.setPhysicalName(outerSurface[1], outerSurface[1], "Upper")

        gmsh.model.occ.synchronize()
        
        gmsh.model.mesh.generate(3)
        gmsh.write(filename)
        gmsh.finalize()


class Annulus(MeshFromGmshFile):
  
    @timing.routine_timer_decorator
    def __init__(self, 
                radius_outer     :float = 1.0,
                radius_inner     :float = 0.2,
                cell_size        :float = 0.1,
                ):
        """
        """
        
        if radius_inner >= radius_outer:
            raise ValueError("`radius_inner` must be smaller than `radius_outer`.")  

        self.radius_outer = radius_outer
        self.radius_inner = radius_inner
        self.cell_size = cell_size

        self.filename = os.path.join(tmpdir, "annulus_" + str(hash(self)) + ".msh")
        self._build_gmsh_file(self.radius_outer, 
                              self.radius_inner, 
                              self.cell_size, 
                              self.filename)
        
        super().__init__(filename=self.filename)

    @staticmethod
    def _build_gmsh_file(radius_outer, radius_inner, cell_size, filename):

        import gmsh
        
        gmsh.initialize()
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("Annulus")

        p0 = gmsh.model.geo.add_point(0,0,0, meshSize=cell_size)
        p1 = gmsh.model.geo.add_point(radius_inner, 0., 0, meshSize=cell_size)
        p2 = gmsh.model.geo.add_point(-radius_inner, 0, 0, meshSize=cell_size)

        c1 = gmsh.model.geo.add_circle_arc(p1, p0, p2)
        c2 = gmsh.model.geo.add_circle_arc(p2, p0, p1)

        cl1 = gmsh.model.geo.add_curve_loop([c1, c2])
        
        gmsh.model.addPhysicalGroup(1, [c1, c2], cl1)
        gmsh.model.setPhysicalName(1, cl1, "Lower")

        p0 = gmsh.model.geo.add_point(0,0,0, meshSize=cell_size)
        p1 = gmsh.model.geo.add_point(radius_outer, 0., 0, meshSize=cell_size)
        p2 = gmsh.model.geo.add_point(-radius_outer, 0, 0, meshSize=cell_size)

        c1 = gmsh.model.geo.add_circle_arc(p1, p0, p2)
        c2 = gmsh.model.geo.add_circle_arc(p2, p0, p1)
 
        cl2 = gmsh.model.geo.add_curve_loop([c1, c2])

        s = gmsh.model.geo.add_plane_surface([cl1, cl2])

        gmsh.model.addPhysicalGroup(1, [c1, c2], cl2)
        gmsh.model.setPhysicalName(1, cl2, "Upper")

        gmsh.model.geo.synchronize()

        gmsh.model.mesh.generate(2)
        gmsh.write(filename)
        gmsh.finalize()