# cython: profile=False
from typing import Optional, Tuple, Union
from collections import namedtuple

from enum import Enum

import math

import  numpy
import  numpy as np
cimport numpy as np

from mpi4py import MPI
from petsc4py import PETSc

import meshio

include "./petsc_extras.pxi"

import underworld3
import underworld3 as uw 
from   underworld3 import _api_tools
from   underworld3.mesh import MeshClass
import underworld3.timing as timing


class Box(MeshClass):
    @timing.routine_timer_decorator
    def __init__(self, 
                elementRes   :Optional[Tuple[  int,  int,  int]] = (16, 16), 
                minCoords    :Optional[Tuple[float,float,float]] = None,
                maxCoords    :Optional[Tuple[float,float,float]] = None,
                simplex      :Optional[bool]                     = False,
                degree       :Optional[int]                      = 1
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
        interpolate=False
        options = PETSc.Options()
        options["dm_plex_separate_marker"] = None
        self.elementRes = elementRes
        if minCoords==None : minCoords=len(elementRes)*(0.,)
        self.minCoords = minCoords
        if maxCoords==None : maxCoords=len(elementRes)*(1.,)
        self.maxCoords = maxCoords
        
        # self.dm = PETSc.DMPlex().createBoxMesh(
        #     elementRes, 
        #     lower=minCoords, 
        #     upper=maxCoords,
        #     simplex=simplex)
        # part = self.dm.getPartitioner()
        # part.setFromOptions()
        # self.dm.distribute()
        # self.dm.setFromOptions()

        super().__init__(simplex=simplex, degree=degree)

    def create_box_mesh(elementRes=(16, 16), minCoords=(0., 0.), maxCoords=(1., 1.), simplex=False):
    
        import gmsh
        
        lc = 0.1
        
        gmsh.initialize()
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("Box")
        
        # Create Box Geometry
        dim = len(minCoords)

        if dim == 2:
            
            xmin, ymin = minCoords
            xmax, ymax = maxCoords
            
            p1 = gmsh.model.geo.add_point(xmin,ymin,0., meshSize=lc, tag=1)
            p2 = gmsh.model.geo.add_point(xmax,ymin,0., meshSize=lc, tag=2)
            p3 = gmsh.model.geo.add_point(xmin,ymax,0., meshSize=lc, tag=3)
            p4 = gmsh.model.geo.add_point(xmax,ymax,0., meshSize=lc, tag=4)

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

            if not simplex:
                
                nx, ny = elementRes

                gmsh.model.mesh.set_transfinite_curve(tag=l1, numNodes=nx+1, meshType="Progression", coef=1.0)
                gmsh.model.mesh.set_transfinite_curve(tag=l2, numNodes=ny+1, meshType="Progression", coef=1.0)
                gmsh.model.mesh.set_transfinite_curve(tag=l3, numNodes=nx+1, meshType="Progression", coef=1.0)
                gmsh.model.mesh.set_transfinite_curve(tag=l4, numNodes=ny+1, meshType="Progression", coef=1.0)
                gmsh.model.mesh.set_transfinite_surface(tag=surface, arrangement="Left", cornerTags=[p1,p2,p3,p4])
                gmsh.model.mesh.set_recombine(2, surface) 
                
        if dim == 3:
            
            xmin, ymin, zmin = minCoords
            xmax, ymax, zmax = maxCoords
            
            p1 = gmsh.model.geo.add_point(xmin,ymin,zmin, meshSize=lc)
            p2 = gmsh.model.geo.add_point(xmax,ymin,zmin, meshSize=lc)
            p3 = gmsh.model.geo.add_point(xmin,ymax,zmin, meshSize=lc)
            p4 = gmsh.model.geo.add_point(xmax,ymax,zmin, meshSize=lc)
            p5 = gmsh.model.geo.add_point(xmin,ymin,zmax, meshSize=lc)
            p6 = gmsh.model.geo.add_point(xmax,ymin,zmax, meshSize=lc)
            p7 = gmsh.model.geo.add_point(xmin,ymax,zmax, meshSize=lc)
            p8 = gmsh.model.geo.add_point(xmax,ymax,zmax, meshSize=lc)    

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
            
            if not simplex:
                
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
        gmsh.write("test.vtk")
        gmsh.finalize()


class MeshFromGmshFile(MeshClass):

    # Data structures for tracking gmsh labels
    physical_label_group = namedtuple('LabelGroup', ('name', 'labels') )

    @timing.routine_timer_decorator
    def __init__(self,
                 dim           :int,
                 filename      :str,
                 label_groups  :Optional[list] = [],
                 cell_size     :Optional[float] = None,
                 refinements   :Optional[int]   = 0,
                 simplex       :Optional[bool] = True,  # Not sure if this will be useful
                 degree        :Optional[int]    =1,
                 verbose       :Optional[bool]  =False
                ):
        """
        This is a generic mesh class for which users will provide 
        the mesh as a gmsh (.msh) file.

            - dim, simpformatlex not inferred from the file at this point 

            - the file pointed to by filename needs to be a .msh file 
            - labels are extracted from the gmsh file "physical labels"

            DISABLED:
            - groups are named collections of labels (or aliases)
                    sides = LabelGroup("Sides", ["Left", "Right"]) 
                    centre_alias = LabelGroup("Center", ["Centre"])

            - Note that the PETSc gmsh reader does not honour membership of multiple 
              physical groups as indicated in a gmsh file - only the first one is used 
            - Note 2, that the use of aliases causes the current mesh save routines to hang.

        """

        self.verbose = verbose

        if cell_size and (refinements>0):
            raise ValueError("You should either provide a `cell_size`, or a `refinements` count, but not both.")

        self.cell_size = cell_size
        self.refinements = refinements

        options = PETSc.Options()
        # options["dm_plex_separate_marker"] = None # this is never used and flags errors for mpirun

        self.dm =  PETSc.DMPlex().createFromFile(filename)

        # if the gmsh file contains periodic boundaries, we also are supposed to call this: DMLocalizeCoordinates()

        self.meshio = meshio.read(filename)

        part = self.dm.getPartitioner()
        part.setFromOptions()
        self.dm.distribute()
        self.dm.setFromOptions()

        try: 
            self.dm.markBoundaryFaces("All_dm_boundaries", value=1)
        except:
            pass

        ## Face Sets are boundaries defined by element surfaces (1d or 2d entities)
        ## Vertex Sets are discrete points 
        ## pygmsh interlaces their labelling so, we have to try each one.

        # Code to generate labels and label groups assuming the gmsh physical labels

        label_dict = self.meshio.field_data

        ## Face Sets are boundaries defined by element surfaces (1d or 2d entities)
        ## Vertex Sets are discrete points 
        ## pygmsh interlaces their labelling so, we have to try each one.

        for l in label_dict:
            self.dm.createLabel(str(l).encode('utf8'))
            label = self.dm.getLabel(str(l).encode('utf8'))
            
            indexSet = self.dm.getStratumIS("Face Sets", label_dict[l][0])
            if not indexSet: # try the other one 
                indexSet = self.dm.getStratumIS("Vertex Sets", label_dict[l][0])
            
            if indexSet:
                label.insertIS(indexSet, 1)
                
            indexSet.destroy()
            

        ## Groups
        """
        for g in label_groups:
            self.dm.createLabel(str(g.name).encode('utf8'))
            label = self.dm.getLabel(str(g.name).encode('utf8'))
            
            for l in label_dict:
                print("Looking for {} in {}".format(l,g.labels))
                if l in g.labels:
                    indexSet = self.dm.getStratumIS("Face Sets", label_dict[l][0])
                    if not indexSet: # try the other one 
                        indexSet = self.dm.getStratumIS("Vertex Sets", label_dict[l][0])

                    if indexSet:
                        label.insertIS(indexSet, 1)

                    indexSet.destroy()
        """
        # See note above - this is disabled since it crashes the mesh writing routines. 
        label_groups = []

        # Provide these to the mesh for boundary conditions
        self.labels =  ["All_dm_boundaries"] + [ l for l in label_dict ] + [ g.name for g in label_groups ]

        if self.verbose:
            self.dm.view()

        # The mesh may over-ride this ... 

        import vtk

        if simplex:
            if dim==2:
                self._elementType = vtk.VTK_TRIANGLE
            else:
                self._elementType = vtk.VTK_TETRA
        else:
            if dim==2:
                self._elementType = vtk.VTK_QUAD
            else:
                self._elementType = vtk.VTK_HEXAHEDRON        

        super().__init__(simplex=simplex, degree=degree)


class SphericalShell(MeshFromGmshFile):

    @timing.routine_timer_decorator
    def __init__(self,
                 dim              :Optional[  int] =2,
                 radius_outer     :Optional[float] =1.0,
                 radius_inner     :Optional[float] =0.5,
                 cell_size        :Optional[float] =0.05,
                 cell_size_upper  :Optional[float] =None,
                 cell_size_lower  :Optional[float] =None,
                 degree           :Optional[int]   =1,
                 centre_point     :Optional[bool]  =True,
                 verbose          :Optional[bool]  =False
        ):

        """
        This class generates a spherical shell, or a full sphere
        where the inner radius is zero.

        Parameters
        ----------
        dim :
            The mesh dimensionality.
        radius_outer :
            The outer radius for the spherical shell.
        radius_inner :
            The inner radius for the spherical shell. If this is set to 
            zero, a full sphere is generated.
        cell_size :
            The target cell size for the final mesh. Mesh refinements will occur to achieve this target 
            resolution.
        """

        if radius_inner>=radius_outer:
            raise ValueError("`radius_inner` must be smaller than `radius_outer`.")
        self.pygmesh = None
        groups = [] 

        # Only root proc generates pygmesh, then it's distributed.
        if MPI.COMM_WORLD.rank==0:

            csize_local = cell_size

            if cell_size_upper is None:
                cell_size_upper = cell_size

            if cell_size_lower is None:
                cell_size_lower = cell_size

            import pygmsh
            # Generate local mesh.
            with pygmsh.geo.Geometry() as geom:
                geom.characteristic_length_max = csize_local

                if dim==2:
                    if radius_inner > 0.0:
                        inner  = geom.add_circle((0.0,0.0,0.0),radius_inner, make_surface=False, mesh_size=cell_size_lower)
                        domain = geom.add_circle((0.0,0.0,0.0), radius_outer, mesh_size=cell_size_upper, holes=[inner])
                        geom.add_physical(inner.curve_loop.curves,  label="Lower")
                        geom.add_physical(domain.curve_loop.curves, label="Upper")
                        geom.add_physical(domain.plane_surface, label="Elements")
                    else:
                        
                        domain = geom.add_circle((0.0,0.0,0.0), radius_outer, mesh_size=cell_size_upper)
                        
                        if centre_point:
                            centre = geom.add_point((0.0,0.0,0.0), mesh_size=cell_size_lower)
                            geom.in_surface(centre, domain.plane_surface)
                            geom.add_physical(centre, label="Centre")
                            centre_alias = self.physical_label_group("Center", ["Centre"])
                            groups.append(centre_alias)

                        geom.add_physical(domain.curve_loop.curves, label="Upper")
                        geom.add_physical(domain.plane_surface, label="Elements")



                else:
                    if radius_inner > 0.0:
                        inner  = geom.add_ball((0.0,0.0,0.0),radius_inner, with_volume=False, mesh_size=cell_size_lower)
                        domain = geom.add_ball((0.0,0.0,0.0), radius_outer, mesh_size=cell_size_upper, holes=[inner.surface_loop])
                        geom.add_physical(inner.surface_loop.surfaces,  label="Lower")
                        geom.add_physical(domain.surface_loop.surfaces, label="Upper")
                        geom.add_physical(domain.volume, label="Elements")

                    else:
                        centre = geom.add_point((0.0,0.0,0.0), mesh_size=cell_size_lower)
                        domain = geom.add_ball((0.0,0.0,0.0), radius_outer, mesh_size=cell_size_upper)  
                        geom.in_volume(centre, domain.volume)
                        geom.add_physical(centre,  label="Centre")
                        geom.add_physical(domain.surface_loop.surfaces, label="Upper")
                        geom.add_physical(domain.volume, label="Elements")  

                        centre_alias = self.physical_label_group("Center", ["Centre"])
                        groups.append(centre_alias)
                 
     


                geom.generate_mesh(verbose=verbose)

                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".msh") as tfile:
                    geom.save_geometry(tfile.name)
                    geom.save_geometry("ignore_ball_mesh_geom.msh")
                    # Can save vtk file here if required ... or not
                    geom.save_geometry("ignore_ball_mesh_geom.vtk")
                    self.meshio = meshio.read(tfile.name)
                    self.meshio.remove_lower_dimensional_cells()

                # The following is an example of setting a callback for variable resolution.
                # geom.set_mesh_size_callback(
                #     lambda dim, tag, x, y, z: 0.15*abs(1.-sqrt(x ** 2 + y ** 2 + z ** 2)) + 0.15
                # )


        super().__init__(dim, filename="ignore_ball_mesh_geom.msh", label_groups=[groups], 
                              cell_size=cell_size, simplex=True, degree=degree, verbose=verbose)


        import vtk

        if dim == 2:
            self._elementType = vtk.VTK_TRIANGLE
        else:
            self._elementType = vtk.VTK_TETRA

        return

    

# The following does not work correctly as the transfinite volume is not correctly 
# meshed ... / cannot be generated consistently with these surface descriptions. 


class StructuredCubeSphericalCap(MeshFromMeshIO):

    @timing.routine_timer_decorator
    def __init__(self,
                elementRes     :Optional[Tuple[int,  int,  int]]  = (16, 16, 8), 
                angles         :Optional[Tuple[float, float]] = (0.7853981633974483, 0.7853981633974483), # pi/4
                radius_outer   :Optional[float] =1.0,
                radius_inner   :Optional[float] =0.5,
                simplex        :Optional[bool] = False, 
                degree         :Optional[int]  =1,
                cell_size      :Optional[float] =1.0
                ):

        """
        This class generates a structured spherical cap based on a deformed cube

        Parameters
        ----------
        elementRes: 
            Elements in the (NS, EW, R) direction 
        angles:
            Angle subtended at the equator, central meridian for this cube-spherical-cap. 
            Should be less than pi/2 for respectable element distortion.
        radius_outer :
            The outer radius for the spherical shell.
        radius_inner :
            The inner radius for the spherical shell. If this is set to 
            zero, a full sphere is generated.
        cell_size :
            The target cell size for the final mesh. Mesh refinements will occur to achieve this target 
            resolution.
        """

        if radius_inner>=radius_outer:
            raise ValueError("`radius_inner` must be smaller than `radius_outer`.")
            
        import pygmsh

        self.pygmesh = None

        # Only root proc generates pygmesh, then it's distributed.

        if MPI.COMM_WORLD.rank==0:  
            hex_box = StructuredCubeSphericalCap.build_pygmsh(elementRes, angles, radius_outer, radius_inner, simplex)

        super().__init__(3, hex_box, cell_size, simplex=simplex, degree=degree)

        self.meshio = hex_box

        import vtk

        if simplex:
            self._elementType = vtk.VTK_TETRA
        else:
            self._elementType = vtk.VTK_HEXAHEDRON

        self.elementRes = elementRes

        # Is the most useful definition ?
        self.minCoords = (-angles[0]/2.0, -angles[1]/2.0, radius_inner)
        self.maxCoords = ( angles[0]/2.0,  angles[1]/2.0, radius_outer)

        return

    def build_pygmsh(
                elementRes, 
                angles,
                radius_outer,
                radius_inner,
                simplex, 
                ):

            import pygmsh 

            minCoords = (-1.0,-1.0,radius_inner)
            maxCoords = ( 1.0, 1.0,radius_outer)

            xx = maxCoords[0]-minCoords[0]
            yy = maxCoords[1]-minCoords[1]
            zz = maxCoords[2]-minCoords[2]

            x_sep=(maxCoords[0] - minCoords[0])/elementRes[0]

            theta = angles[0]
            phi = angles[1]

            with pygmsh.geo.Geometry() as geom:
                points = [geom.add_point([x, minCoords[1], minCoords[2]], x_sep) for x in [minCoords[0], maxCoords[0]]]
                line = geom.add_line(*points)

                _, rectangle, _ = geom.extrude(line, translation_axis=[0.0, maxCoords[1]-minCoords[1], 0.0], 
                                               num_layers=elementRes[1], recombine=(not simplex))

                geom.extrude(
                        rectangle,
                        translation_axis=[0.0, 0.0, maxCoords[2]-minCoords[2]],
                        num_layers=elementRes[2],
                        recombine=(not simplex),
                    )
                    
                hex_box = geom.generate_mesh()
                hex_box.remove_lower_dimensional_cells()

                # Now adjust the point locations
                # first make a pyramid that subtends the correct angle at each level
                
                hex_box.points[:,0] *= hex_box.points[:,2] * np.tan(theta/2) 
                hex_box.points[:,1] *= hex_box.points[:,2] * np.tan(phi/2) 
        
                # second, adjust the distance so each layer forms a spherical cap 
                
                targetR = hex_box.points[:,2]
                actualR = np.sqrt(hex_box.points[:,0]**2 + hex_box.points[:,1]**2 + hex_box.points[:,2]**2)

                hex_box.points[:,0] *= (targetR / actualR)
                hex_box.points[:,1] *= (targetR / actualR)
                hex_box.points[:,2] *= (targetR / actualR)
                            
                # finalise geom context

            return hex_box
## 


class StructuredCubeSphereBallMesh(MeshFromGmshFile):
    @timing.routine_timer_decorator
    def __init__(self,
                dim            :Optional[int] = 2,
                elementRes     :Tuple[int,  int]  = 8,
                radius_outer   :Optional[float] = 1.0,
                cell_size      :Optional[float] = 1e30,
                simplex        :Optional[bool]  = False, 
                degree         :Optional[int]   = 1,
                verbose        :Optional[bool]  = False
                ):

        """
        This class generates a structured solid spherical ball based on the cubed sphere

        Parameters
        ----------
        dim:
        elementRes: 
            Elements in the R direction 
        radius_outer :
            The outer radius for the spherical shell.
        simplex: 
            Tets (True) or Hexes (False)
        cell_size :
            The target cell size for the final mesh. Mesh refinements will occur to achieve this target 
            resolution.
        """
        
        import pygmsh
        self.meshio = None
        self.verbose = verbose

        # Really this should be "Labels for the mesh not boundaries"

        class Boundary(Enum):
            ALL_BOUNDARIES = 0
            CENTRE = 1
            LOWER  = 1
            TOP    = 2
            UPPER  = 2

        ## We should pass the boundary definitions to the mesh constructor to be sure
        ## that we use consistent values for the labels

        # Really this should be "Labels for the mesh not boundaries"

        class Boundary(Enum):
            ALL_BOUNDARIES = 1
            CENTRE = 10
            TOP    = 20
            UPPER  = 20

        ## We should pass the boundary definitions to the mesh constructor to be sure
        ## that we use consistent values for the labels

        # Only root proc generates pygmesh, then it's distributed.
        if MPI.COMM_WORLD.rank==0:  
            if dim == 2:
                cs_hex_box, filename = StructuredCubeSphereBallMesh.build_pygmsh_2D(elementRes, radius_outer, simplex=simplex)
            else: 
                cs_hex_box, filename = StructuredCubeSphereBallMesh.build_pygmsh_3D(elementRes, radius_outer, simplex=simplex)
        else:
            cs_hex_box = None

        super().__init__(dim, filename=filename, bound_markers=Boundary, 
                              cell_size=cell_size, simplex=simplex, degree=degree, verbose=verbose)


        self.meshio  = cs_hex_box

        import vtk

        if simplex:
            if dim==2:
                self._elementType = vtk.VTK_TRIANGLE
            else:
                self._elementType = vtk.VTK_TETRA
        else:
            if dim==2:
                self._elementType = vtk.VTK_QUAD
            else:
                self._elementType = vtk.VTK_HEXAHEDRON        

        self.elementRes = elementRes

        # Is the most useful definition
        self.minCoords = (0.0,)
        self.maxCoords = (radius_outer,)

        return

    def build_pygmsh_2D(
            elementRes     :Optional[int]  = 16, 
            radius_outer   :Optional[float] =1.0,
            simplex        :Optional[bool]  =False
            ):

        import meshio
        import gmsh

        gmsh.initialize()
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.model.add("squared")

        lc = 0.0 * radius_outer / (elementRes+1)
        ro = radius_outer
        r2 = radius_outer / np.sqrt(2)
        res = elementRes*2+1

        gmsh.model.geo.addPoint(0.0,0.0,0.0, lc, 1)
        
        gmsh.model.geo.addPoint( r2, r2, 0.0, lc, 10)
        gmsh.model.geo.addPoint(-r2, r2, 0.0, lc, 11)
        gmsh.model.geo.addPoint(-r2,-r2, 0.0, lc, 12)
        gmsh.model.geo.addPoint( r2,-r2, 0.0, lc, 13)

        gmsh.model.geo.add_circle_arc(10, 1, 11, tag=100)
        gmsh.model.geo.add_circle_arc(11, 1, 12, tag=101)      
        gmsh.model.geo.add_circle_arc(12, 1, 13, tag=102)      
        gmsh.model.geo.add_circle_arc(13, 1, 10, tag=103)      

        gmsh.model.geo.addCurveLoop([100, 101, 102, 103], 10000, reorient=True)
        gmsh.model.geo.add_surface_filling([10000], 10101)
        
        gmsh.model.geo.mesh.set_transfinite_curve(100, res, meshType="Progression")
        gmsh.model.geo.mesh.set_transfinite_curve(101, res, meshType="Progression")
        gmsh.model.geo.mesh.set_transfinite_curve(102, res, meshType="Progression")
        gmsh.model.geo.mesh.set_transfinite_curve(103, res, meshType="Progression")

        gmsh.model.geo.mesh.setTransfiniteSurface(10101)

        if not simplex:
            gmsh.model.geo.mesh.setRecombine(2, 10101)


        gmsh.model.geo.synchronize()

        centreMarker, upperMarker = 1, 2



        #gmsh.model.add_physical_group(1, [100], outerMarker+1) # temp - to test the bc settings
        #gmsh.model.add_physical_group(1, [101], outerMarker+2)
        #gmsh.model.add_physical_group(1, [102], outerMarker+3)
        #gmsh.model.add_physical_group(1, [103], outerMarker+4)
        gmsh.model.add_physical_group(1, [100, 101, 102, 103], upperMarker)
        
        # Vertex groups (0d)
        gmsh.model.add_physical_group(0, [1], centreMarker)

        # Shove everything (else) in the garbage dump group because the 
        # Option setting above does not seem to work on (my) version of gmsh

        for d in range(0,3):
            e = gmsh.model.getEntities(d)
            gmsh.model.add_physical_group(d, [t for i,t in e], 9999)

        gmsh.model.geo.remove_all_duplicates()
        gmsh.model.mesh.generate(dim=2)
        gmsh.model.mesh.removeDuplicateNodes()

        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".msh") as tfile:
            gmsh.write(tfile.name)
            gmsh.write("ignore_squared_disk.msh")
            gmsh.write("ignore_squared_disk.vtk")
            squared_disk_mesh = meshio.read(tfile.name)
            squared_disk_mesh.remove_lower_dimensional_cells()

        gmsh.finalize()

        return squared_disk_mesh, "ignore_squared_disk.msh"
        

    def build_pygmsh_3D(
                elementRes     :Optional[int]  = 16, 
                radius_outer   :Optional[float] =1.0,
                simplex        :Optional[bool]  =False
                ):

            import meshio
            import gmsh

            gmsh.initialize()
            gmsh.model.add("cubed")
            gmsh.option.setNumber("Mesh.SaveAll", 1)

            lc = 0.001 * radius_outer / (elementRes+1)

            r2 = radius_outer / np.sqrt(3)
            r0 = 0.5 * radius_outer / np.sqrt(3)

            res = elementRes+1

            gmsh.model.geo.addPoint(0.001,0.001,0.001,0.1, 1)

            # The 8 corners of the cubes

            gmsh.model.geo.addPoint(-r2, -r2, -r2, lc, 100)
            gmsh.model.geo.addPoint( r2, -r2, -r2, lc, 101)
            gmsh.model.geo.addPoint( r2,  r2, -r2, lc, 102)
            gmsh.model.geo.addPoint(-r2,  r2, -r2, lc, 103)
            gmsh.model.geo.addPoint(-r2, -r2,  r2, lc, 104)
            gmsh.model.geo.addPoint( r2, -r2,  r2, lc, 105)
            gmsh.model.geo.addPoint( r2,  r2,  r2, lc, 106)
            gmsh.model.geo.addPoint(-r2,  r2,  r2, lc, 107)

            # The 12 edges of the cube2

            gmsh.model.geo.add_circle_arc(100,1,101, 1000)
            gmsh.model.geo.add_circle_arc(101,1,102, 1001)
            gmsh.model.geo.add_circle_arc(102,1,103, 1002)
            gmsh.model.geo.add_circle_arc(103,1,100, 1003)

            gmsh.model.geo.add_circle_arc(101,1,105, 1004)
            gmsh.model.geo.add_circle_arc(102,1,106, 1005)
            gmsh.model.geo.add_circle_arc(103,1,107, 1006)
            gmsh.model.geo.add_circle_arc(100,1,104, 1007)

            gmsh.model.geo.add_circle_arc(104,1,105, 1008)
            gmsh.model.geo.add_circle_arc(105,1,106, 1009)
            gmsh.model.geo.add_circle_arc(106,1,107, 1010)
            gmsh.model.geo.add_circle_arc(107,1,104, 1011)

            ## These should all be transfinite lines

            for i in range(1000, 1012):
                gmsh.model.geo.mesh.set_transfinite_curve(i, res)

            # The 6 faces of the cube2

            gmsh.model.geo.addCurveLoop([1000, 1004, 1008, 1007], 10000, reorient=True)
            gmsh.model.geo.addCurveLoop([1001, 1005, 1009, 1004], 10001, reorient=True)
            gmsh.model.geo.addCurveLoop([1002, 1006, 1010, 1005], 10002, reorient=True)
            gmsh.model.geo.addCurveLoop([1003, 1007, 1011, 1006], 10003, reorient=True)
            gmsh.model.geo.addCurveLoop([1000, 1003, 1002, 1001], 10004, reorient=True)
            gmsh.model.geo.addCurveLoop([1008, 1009, 1010, 1011], 10005, reorient=True)

            gmsh.model.geo.add_surface_filling([10000], 10101, sphereCenterTag=1)
            gmsh.model.geo.add_surface_filling([10001], 10102, sphereCenterTag=1)
            gmsh.model.geo.add_surface_filling([10002], 10103, sphereCenterTag=1)
            gmsh.model.geo.add_surface_filling([10003], 10104, sphereCenterTag=1)
            gmsh.model.geo.add_surface_filling([10004], 10105, sphereCenterTag=1)
            gmsh.model.geo.add_surface_filling([10005], 10106, sphereCenterTag=1)

            gmsh.model.geo.synchronize()

            for i in range(10101, 10107):
                gmsh.model.geo.mesh.setTransfiniteSurface(i, "Left")
                if not simplex:
                    gmsh.model.geo.mesh.setRecombine(2, i)

            gmsh.model.geo.synchronize()

            # outer surface / inner_surface
            gmsh.model.geo.add_surface_loop([10101, 10102, 10103, 10104, 10105, 10106], 10111)
            gmsh.model.geo.add_volume([10111], 100001)

            gmsh.model.geo.synchronize()

            gmsh.model.mesh.set_transfinite_volume(100001)
            if not simplex:
                gmsh.model.geo.mesh.setRecombine(3, 100001)

            centreMarker, outerMarker = 10, 20
            gmsh.model.add_physical_group(0, [1], centreMarker)
            gmsh.model.add_physical_group(2, [i for i in range(10101, 10107)], outerMarker)

            # Shove everything in the garbage dump group because the 
            # Option setting above does not seem to work on (my) version of gmsh

            for d in range(0,4):
                e = gmsh.model.getEntities(d)
                gmsh.model.add_physical_group(d, [t for i,t in e], 9999)

            gmsh.model.geo.remove_all_duplicates()
            gmsh.model.mesh.generate(dim=3)
            gmsh.model.mesh.removeDuplicateNodes()

            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".msh") as tfile:
                gmsh.write(tfile.name)
                gmsh.write("ignore_cubedsphereball.msh")
                gmsh.write("ignore_cubedsphereball.vtk")
                cubed_sphere_ball_mesh = meshio.read(tfile.name)
                cubed_sphere_ball_mesh.remove_lower_dimensional_cells()
                
            gmsh.finalize()

            return cubed_sphere_ball_mesh


# Replace this one with Romain's CS code


class StructuredCubeSphereShellMesh(MeshFromMeshIO):

    @timing.routine_timer_decorator
    def __init__(self,
                elementRes     :Tuple[int,  int]  = (16, 8), 
                radius_outer   :Optional[float] = 1.0,
                radius_inner   :Optional[float] = 0.5,
                cell_size      :Optional[float] = 1e30,
                simplex        :Optional[bool] = False, 
                degree       :Optional[int]    = 1
                ):

        """
        This class generates a structured spherical shell based on the cubed sphere 

        Parameters
        ----------
        elementRes: 
            Elements in the (NS & EW , R) direction 
        radius_outer :
            The outer radius for the spherical shell.
        radius_inner :
            The inner radius for the spherical shell. If this is set to 
            zero, a full sphere is generated.
        simplex: 
            Tets (True) or Hexes (False)
        cell_size :
            The target cell size for the final mesh. Mesh refinements will occur to achieve this target 
            resolution.
        """

        if radius_inner>=radius_outer:
            raise ValueError("`radius_inner` must be smaller than `radius_outer`.")
            
        import pygmsh

        self.meshio = None

        # Only root proc generates pygmesh, then it's distributed.
        if MPI.COMM_WORLD.rank==0:  

            cs_hex_box = StructuredCubeSphereShellMesh.build_pygmsh(elementRes, radius_outer, radius_inner, simplex=simplex)

        super().__init__(3, cs_hex_box, cell_size, simplex=simplex, degree=degree)

        self.meshio  = cs_hex_box

        import vtk

        if simplex:
            self._elementType = vtk.VTK_TETRA
        else:
            self._elementType = vtk.VTK_HEXAHEDRON        

        self.elementRes = elementRes

        # Is the most useful definition
        self.minCoords = (radius_inner,)
        self.maxCoords = (radius_outer,)

        return

    def build_pygmsh(
                elementRes     :Tuple[int,  int]  = (16, 8), 
                radius_outer   :Optional[float] =1.0,
                radius_inner   :Optional[float] =0.5,
                simplex        :Optional[bool]  =False
                ):


            import pygmsh 
            import meshio

            l = 0.0

            inner_radius = radius_inner
            outer_radius = radius_outer
            nodes = elementRes[0]+1  # resolution of the cube laterally
            layers= elementRes[1]

            with pygmsh.geo.Geometry() as geom:
                cpoint = geom.add_point([0.0,0.0,0.0], l)
                
                genpt = [0,0,0,0,0,0,0,0]

                # 8 corners of the cube 
                
                r2 = 1.0 / np.sqrt(3.0) # Generate a unit sphere
                
                genpt[0] = geom.add_point([ -r2, -r2, -r2],  l)
                genpt[1] = geom.add_point([  r2, -r2, -r2],  l)
                genpt[2] = geom.add_point([  r2,  r2, -r2],  l)
                genpt[3] = geom.add_point([ -r2,  r2, -r2],  l)
                genpt[4] = geom.add_point([ -r2, -r2,  r2],  l)
                genpt[5] = geom.add_point([  r2, -r2,  r2],  l)
                genpt[6] = geom.add_point([  r2,  r2,  r2],  l)
                genpt[7] = geom.add_point([ -r2,  r2,  r2],  l)

              
                # 12 edges of the cube
                
                b_circ00 = geom.add_circle_arc(genpt[0], cpoint, genpt[1])
                b_circ01 = geom.add_circle_arc(genpt[1], cpoint, genpt[2])
                b_circ02 = geom.add_circle_arc(genpt[2], cpoint, genpt[3])
                b_circ03 = geom.add_circle_arc(genpt[0], cpoint, genpt[3])

                b_circ04 = geom.add_circle_arc(genpt[1], cpoint, genpt[5])
                b_circ05 = geom.add_circle_arc(genpt[2], cpoint, genpt[6])
                b_circ06 = geom.add_circle_arc(genpt[3], cpoint, genpt[7])
                b_circ07 = geom.add_circle_arc(genpt[0], cpoint, genpt[4])

                b_circ08 = geom.add_circle_arc(genpt[4], cpoint, genpt[5])
                b_circ09 = geom.add_circle_arc(genpt[5], cpoint, genpt[6])
                b_circ10 = geom.add_circle_arc(genpt[6], cpoint, genpt[7])
                b_circ11 = geom.add_circle_arc(genpt[4], cpoint, genpt[7])

                for arc in [b_circ00, b_circ01, b_circ02, b_circ03,
                            b_circ04, b_circ05, b_circ06, b_circ07,
                            b_circ08, b_circ09, b_circ10, b_circ11 ]:
                    
                        geom.set_transfinite_curve(arc, num_nodes=nodes, 
                                                mesh_type="Progression", coeff=1.0)

                # 6 Cube faces



                
                face00_loop = geom.add_curve_loop([b_circ00, b_circ04, -b_circ08, -b_circ07])
                face00 = geom.add_surface(face00_loop) 
                geom.set_transfinite_surface(face00, arrangement="Left",
                                            corner_pts = [genpt[0], genpt[1], genpt[5], genpt[4]])   


                face01_loop = geom.add_curve_loop([-b_circ01, b_circ05, b_circ09, -b_circ04])
                face01 = geom.add_surface(face01_loop) 
                geom.set_transfinite_surface(face01, arrangement="Left",
                                            corner_pts = [genpt[1], genpt[2], genpt[6], genpt[5]])   


                face02_loop = geom.add_curve_loop([b_circ02, b_circ06, -b_circ10, -b_circ05])
                face02 = geom.add_surface(face02_loop) 
                geom.set_transfinite_surface(face02, arrangement="Left",
                                            corner_pts = [genpt[2], genpt[3], genpt[7], genpt[6]])   


                face03_loop = geom.add_curve_loop([-b_circ03, b_circ07, b_circ11, -b_circ06])
                face03 = geom.add_surface(face03_loop) 
                geom.set_transfinite_surface(face03, arrangement="Left",
                                            corner_pts = [genpt[3], genpt[0], genpt[4], genpt[7]])   


                face04_loop = geom.add_curve_loop([-b_circ00, b_circ03, -b_circ02, -b_circ01])
                face04 = geom.add_surface(face04_loop) 
                geom.set_transfinite_surface(face04, arrangement="Left",
                                            corner_pts = [genpt[1], genpt[0], genpt[3], genpt[2]])   


                face05_loop = geom.add_curve_loop([b_circ08, b_circ09,  b_circ10, b_circ11])
                face05 = geom.add_surface(face05_loop) 
                geom.set_transfinite_surface(face05, arrangement="Left",
                                            corner_pts = [genpt[4], genpt[5], genpt[6], genpt[7]])   


                geom.set_recombined_surfaces([face00, face01, face02, face03, face04, face05])
                shell = geom.add_surface_loop([face00, face01, face02, face03, face04, face05])
                    
                two_D_cubed_sphere = geom.generate_mesh(dim=2, verbose=False)
                two_D_cubed_sphere.remove_orphaned_nodes()
                two_D_cubed_sphere.remove_lower_dimensional_cells()

            ## Now stack the 2D objects to make a 3d shell 
                
            cells = two_D_cubed_sphere.cells[0].data - 1
            cells_per_layer = cells.shape[0]
            mesh_points = two_D_cubed_sphere.points[1:,:]
            points_per_layer = mesh_points.shape[0]

            cells_layer = np.empty((cells_per_layer, 8), dtype=int)
            cells_layer[:, 0:4] = cells[:,:]
            cells_layer[:, 4:8] = cells[:,:] + points_per_layer

            # stack this layer multiple times

            cells_3D = np.empty((layers, cells_per_layer, 8), dtype=int) 

            for i in range(0,layers):
                cells_3D[i,:,:] = cells_layer[:,:] + i * points_per_layer

            mesh_cells_3D = cells_3D.reshape(-1,8)

            ## Point locations

            radii = np.linspace(inner_radius, outer_radius, layers+1)

            mesh_points_3D = np.empty(((layers+1)*points_per_layer, 3))

            for i in range(0, layers+1):
                mesh_points_3D[i*points_per_layer:(i+1)*points_per_layer] = mesh_points * radii[i]
                
            cubed_sphere_pygmsh = meshio.Mesh(mesh_points_3D, [("hexahedron", mesh_cells_3D)])

            # tetrahedral version (subdivide all hexes into 6 tets)

            if simplex:
                cells = cubed_sphere_pygmsh.cells[0][1]
 
                t1 = cells[:,[3,0,1,5]]
                t2 = cells[:,[3,2,1,5]]
                t3 = cells[:,[3,2,6,5]]
                t4 = cells[:,[3,7,6,5]]
                t5 = cells[:,[3,7,4,5]]
                t6 = cells[:,[3,0,4,5]]

                tcells = np.vstack([t1,t2,t3,t4,t5,t6])

                tet_cubed_sphere_pygmsh = meshio.Mesh(mesh_points_3D, [("tetra", tcells)])
                tet_cubed_sphere_pygmsh.remove_lower_dimensional_cells()

                return tet_cubed_sphere_pygmsh

            else:
                return cubed_sphere_pygmsh    

