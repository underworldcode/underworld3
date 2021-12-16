# cython: profile=False
from typing import Optional, Tuple, Union
from collections import namedtuple

from enum import Enum

import math

import  numpy
import  numpy as np
cimport numpy as np
# import  sympy
# import  sympy.vector

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
                elementRes   :Optional[Tuple[  int,  int,  int]] =(16, 16), 
                minCoords    :Optional[Tuple[float,float,float]] =None,
                maxCoords    :Optional[Tuple[float,float,float]] =None,
                simplex      :Optional[bool]                     =False,
                degree       :Optional[int]                      =1
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
        self.dm = PETSc.DMPlex().createBoxMesh(
            elementRes, 
            lower=minCoords, 
            upper=maxCoords,
            simplex=simplex)
        part = self.dm.getPartitioner()
        part.setFromOptions()
        self.dm.distribute()
        self.dm.setFromOptions()

        # bcs
        from enum import Enum        
        if len(elementRes) == 2:
            class Boundary2D(Enum):
                BOTTOM = 1
                RIGHT  = 2
                TOP    = 3
                LEFT   = 4
            self.boundary = Boundary2D
        else:
            class Boundary3D(Enum):
                BOTTOM = 1
                TOP    = 2
                FRONT  = 3
                BACK   = 4
                RIGHT  = 5
                LEFT   = 6
            self.boundary = Boundary3D

        # self.dm.view()
        # create boundary sets
        for val in self.boundary:
            boundary_set = self.dm.getStratumIS("marker",val.value)        # get the set
            self.dm.createLabel(str(val).encode('utf8'))                   # create the label
            boundary_label = self.dm.getLabel(str(val).encode('utf8'))     # get label
            # Without this check, we have failures at this point in parallel. 
            # Further investigation required. JM.
            if boundary_set:
                boundary_label.insertIS(boundary_set, 1) # add set to label with value 1

        super().__init__(simplex=simplex, degree=degree)


class MeshFromGmshFile(MeshClass):

    # Data structures for tracking gmsh labels
    physical_label_group = namedtuple('Group', ('name', 'labels') )

    @timing.routine_timer_decorator
    def __init__(self,
                 dim           :int,
                 filename      :str,
                 label_groups  :Optional[list] = [],
                 cell_size     :Optional[float] = None,
                 refinements   :Optional[int]   = 0,
                 simplex       :Optional[bool] = True,  # Not sure if this will be useful
                 degree        :Optional[int]    =1
                ):
        """
        This is a generic mesh class for which users will provide 
        the mesh as a gmsh (.msh) file.

            - dim, simplex not inferred from the file at this point 

            - the file pointed to by filename needs to be a .msh file 
            - bound_markers is an Enum that identifies the markers used by gmsh physical objects
            - etc etc
        """

        if cell_size and (refinements>0):
            raise ValueError("You should either provide a `cell_size`, or a `refinements` count, but not both.")

        self.cell_size = cell_size
        self.refinements = refinements

        options = PETSc.Options()
        options["dm_plex_separate_marker"] = None

        self.dm =  PETSc.DMPlex().createFromFile(filename)
        self.meshio = meshio.read(filename)


        part = self.dm.getPartitioner()
        part.setFromOptions()
        self.dm.distribute()
        self.dm.setFromOptions()

        try: 
            self.dm.markBoundaryFaces("Boundary.ALL_BOUNDARIES", value=self.boundary.ALL_BOUNDARIES.value)
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


        # Provide these to the mesh for boundary conditions
        self.labels =  ["Boundary.ALL_BOUNDARIES"] + [ l for l in label_dict ] + [ g.name for g in label_groups ]

        self.dm.view()
        super().__init__(simplex=simplex, degree=degree)



class MeshFromCellList(MeshClass):
    @timing.routine_timer_decorator
    def __init__(self,
                 dim         :int,
                 cells       :numpy.ndarray,
                 coords      :numpy.ndarray,
                 cell_size   :Optional[float] = None,
                 refinements :Optional[int]   = 0,
                 simplex      :Optional[bool] = True,                
                 degree       :Optional[int]  =1


                ):
        """
        This is a generic mesh class for which users will provide 
        the specifying mesh cells and coordinates.

        This method wraps to the PETSc `DMPlexCreateFromCellListPetsc` routine.

        Only the root process needs to provide the `cells` and `coords` arrays. It
        will then be distributed to other processes in the communication group.

        ?? Should we try to add mesh labels in here - LM ??

        Parameters
        ----------
        dim :
            The mesh dimensionality.
        cells :
            An array of integers of size [numCells,numCorners] specifying the vertices for each cell.
        coords :
            An array of floats of size [numVertices,dim] specifying the coordinates of each vertex.
        cell_size :
            The target cell size for the final mesh. Mesh refinements will occur to achieve this target 
            resolution. If this is specified, `refinements` should not be specified. 
        refinements :
            The number of mesh refinements that should be performed. If this is specified, `cell_size` 
            should not be specified. 
        simplex:
            If `True`, simplex elements are assumed (default is True)
        """

        if cell_size and (refinements>0):
            raise ValueError("You should either provide a `cell_size`, or a `refinements` count, but not both.")
        self.cell_size = cell_size
        self.refinements = refinements

        options = PETSc.Options()
        # options.setValue("dm_refine", self.refinements)
        from . import mesh_utils
        from mpi4py import MPI
        cdef DM dm  = mesh_utils._from_cell_list(dim, cells, coords, MPI.COMM_WORLD)
        self.dm = dm

        from enum import Enum
        class Boundary(Enum):
            ALL_BOUNDARIES = 1
        self.boundary = Boundary

        bound = self.boundary.ALL_BOUNDARIES
        self.dm.markBoundaryFaces(str(bound).encode('utf8'),bound.value)

        part = self.dm.getPartitioner()
        part.setFromOptions()
        self.dm.distribute()
        self.dm.setFromOptions()

        if cell_size:
            while True:
                # Use a factor of 2 here as PETSc returns distance from centroid to edge
                min_rad = 2.*self.get_min_radius()
                if min_rad > 1.5*cell_size:
                    if MPI.COMM_WORLD.rank==0:
                        print(f"Mesh cell size ({min_rad:.3E}) larger than requested ({cell_size:.3E}). Refining.")
                    self.dm = self.dm.refine()
                    self._min_radius = None
                else:
                    break

        for val in range(refinements):
            if MPI.COMM_WORLD.rank==0:
                print(f"Mesh refinement {val+1} of {refinements}.")
            self.dm = self.dm.refine()
            self._min_radius = None

        if MPI.COMM_WORLD.rank==0:
            print(f"Generated mesh minimum cell size: {2.*self.get_min_radius():.3E}. Requested size: {cell_size:.3E}.")

        self.dm.view()
        super().__init__(simplex=simplex, degree=degree)

class MeshFromMeshIO(MeshFromCellList):
    @timing.routine_timer_decorator
    def __init__(self,
                 dim         :int,
                 meshio      :"MeshIO",
                 cell_size   :Optional[float] =None,
                 refinements :Optional[int]   = 0,
                 simplex      :Optional[bool] = True,
                 degree       :Optional[int]                      =1

                 ):
        """
        This is a generic mesh class for which users will provide 
        the specifying mesh data as a `MeshIO` object.

        Only the root process needs to provide the MeshIO object. It
        will then be distributed to other processes in the communication group.

        Parameters
        ----------
        dim :
            The mesh dimensionality.
        meshio :
            The MeshIO object specifying the mesh.
        cell_size :
            The target cell size for the final mesh. Mesh refinements will occur to achieve this target 
            resolution. If this is specified, `refinements` should not be specified. 
        refinements :
            The number of mesh refinements that should be performed. If this is specified, `cell_size` 
            should not be specified. 
        simplex:
            If `True`, simplex elements are assumed (default is True)
        """

        if dim not in (2,3):
            raise ValueError(f"`dim` must be 2 or 3. You have passed in dim={dim}.")

        # if remove_lower_dimensional_cells is called on the mesh, then the cell list 
        # for our regular meshes should be of length 1 (all hexes, all tets, all quads and so on)

        cells = coords = None
        if MPI.COMM_WORLD.rank==0:
            cells  = meshio.cells[0][1]
            coords = meshio.points[:,0:dim]
  
        super().__init__(dim, cells, coords, cell_size, refinements, simplex=simplex, degree=degree)

    def _get_local_cell_size(self,
                             dim        : int,
                             domain_vol : float,
                             cell_size  : float) -> int:
        """
        This method attempts to determine an appropriate resolution for
        the generated *local* gmsh object.

        This will consequently be refined in parallel to achieve the user's 
        requested cell_size.

        Parameters
        ----------
        dim :
            Mesh dimensionality.
        domain_vol : 
            The expected domain volume (or area) for the generated
            mesh.
        cell_size :
            Typical cell characteristic length.

        Returns
        -------
        The suggested local cell size.

        """
        def cell_count(cell_size):
            # approx size of tri/tet
            cell_area  = 0.5 if (dim==2) else 1/(6.*math.sqrt(2.))
            cell_area *= cell_size**dim
            return int(domain_vol/cell_area)
        # Set max local at some nominal number.
        # This may well need to be different in 2 or 3-d.
        max_local_cells = 48**3  
        csize_local = cell_size
        #Keep doubling cell size until we can generate reasonably sized local mesh.
        while cell_count(csize_local)>max_local_cells:
            csize_local *= 1.1
        return csize_local 


# pygmesh generator for Hex (/ Quad)-based structured, box mesh

# Note boundary labels are needed (cf PETSc box mesh above)

class Hex_Box(MeshFromMeshIO):
    @timing.routine_timer_decorator
    def __init__(self,
                dim          :Optional[  int] = 2,
                elementRes   :Tuple[int,  int,  int]    = (16, 16, 0), 
                minCoords    :Optional[Tuple[float,float,float]] =None,
                maxCoords    :Optional[Tuple[float,float,float]] =None,
                cell_size    :Optional[float] =0.05,
                degree       :Optional[int]                      =1

                ):
        """
        This class generates a Cartesian box of regular hexahedral (3D) or quadrilateral (2D) 
        elements. 

        Parameters
        ----------
        dim :
            The mesh dimensionality.
        elementRes:
            Tuple specifying number of elements in each axis direction.
        minCoord:
            Optional. Tuple specifying minimum mesh location.
        maxCoord:
            Optional. Tuple specifying maximum mesh location.
        cell_size :
            The target cell size for the final mesh. Mesh refinements will occur to achieve this target 
            resolution. If this is specified, `refinements` should not be specified. 
        """

        if minCoords==None: 
            minCoords=len(elementRes)*(0.,)

        if maxCoords==None: 
            maxCoords=len(elementRes)*(1.,)

        self.pygmesh = None
        # Only root proc generates pygmesh, then it's distributed.

        if MPI.COMM_WORLD.rank==0:
            mesh = Hex_Box.build_pygmsh( dim, elementRes, minCoords, maxCoords )

        super().__init__(dim, mesh, cell_size, simplex=False, degree=degree)

        self.pygmesh = mesh
        self.elementRes = elementRes
        self.minCoords = minCoords
        self.maxCoords = maxCoords

        import vtk

        if dim == 2:
            self._elementType = vtk.VTK_QUAD
        else:
            self._elementType = vtk.VTK_HEXAHEDRON

        return

    def build_pygmsh(
                dim          :Optional[  int] = 2,
                elementRes   :Optional[Tuple[int,  int,  int]]    = (16, 16, 0), 
                minCoords    :Optional[Tuple[float,float,float]] =None,
                maxCoords    :Optional[Tuple[float,float,float]] =None 
            ):

        x_sep=(maxCoords[0] - minCoords[0])/elementRes[0]

        import pygmsh
        with pygmsh.geo.Geometry() as geom:
            points = [geom.add_point([x, minCoords[1], minCoords[2]], x_sep) for x in [minCoords[0], maxCoords[0]]]
            line = geom.add_line(*points)

            _, rectangle, _ = geom.extrude(
                line, translation_axis=[0.0, maxCoords[1]-minCoords[1], 0.0], num_layers=elementRes[1], recombine=True
            )

            # It would make sense to validate that elementRes[2] > 0 if dim==3

            if dim == 3:
                geom.extrude(
                    rectangle,
                    translation_axis=[0.0, 0.0, maxCoords[2]-minCoords[2]],
                    num_layers=elementRes[2],
                    recombine=True,
                )
                
            quad_hex_box = geom.generate_mesh()
            quad_hex_box.remove_lower_dimensional_cells()

        return quad_hex_box



# pygmesh generator for Tet/Tri-based structured, box mesh
# Note boundary labels are needed (cf PETSc box mesh above)

class Simplex_Box(MeshFromMeshIO):
    @timing.routine_timer_decorator
    def __init__(self,
                dim          :Optional[  int] = 2,
                elementRes   :Tuple[int,  int,  int]    = (16, 16, 0), 
                minCoords    :Optional[Tuple[float,float,float]] =None,
                maxCoords    :Optional[Tuple[float,float,float]] =None,
                cell_size    :Optional[float] =0.05,
                degree       :Optional[int]  =1
                ):
        """
        This class generates a box with gmsh

        Parameters
        ----------
        dim :
            The mesh dimensionality.
        elementRes:
            Tuple specifying number of elements in each axis direction.
        minCoord:
            Optional. Tuple specifying minimum mesh location.
        maxCoord:
            Optional. Tuple specifying maximum mesh location.
        cell_size :
            The target cell size for the final mesh. Mesh refinements will occur to achieve this target 
            resolution.   
        """

        if minCoords==None: 
            minCoords=len(elementRes)*(0.,)

        if maxCoords==None: 
            maxCoords=len(elementRes)*(1.,)

        self.pygmesh = None

        # Only root proc generates pygmesh, then it's distributed.
        if MPI.COMM_WORLD.rank==0:
            mesh = Simplex_Box.build_pygmsh(dim, elementRes, minCoords, maxCoords)
    
        super().__init__(dim, mesh, cell_size, degree=degree, simplex=True)

        self.elementRes = elementRes
        self.minCoords = minCoords
        self.maxCoords = maxCoords
        self.pygmesh = mesh

        import vtk

        if dim == 2:
            self._elementType = vtk.VTK_TRIANGLE
        else:
            self._elementType = vtk.VTK_TETRA

        return


    def build_pygmsh(
                dim          :Optional[  int] = 2,
                elementRes   :Optional[Tuple[int,  int,  int]]    = (16, 16, 0), 
                minCoords    :Optional[Tuple[float,float,float]] =None,
                maxCoords    :Optional[Tuple[float,float,float]] =None,
                cell_size    :Optional[float] =1.0):
 
        xx = maxCoords[0]-minCoords[0]
        yy = maxCoords[1]-minCoords[1]
        zz = maxCoords[2]-minCoords[2]

        import pygmsh
        with pygmsh.occ.Geometry() as geom:
            p = geom.add_point(minCoords, 1)
            _, l, _ = geom.extrude(p, [xx, 0, 0], num_layers=elementRes[0])
            _, s, _ = geom.extrude(l, [0, yy, 0], num_layers=elementRes[1])
            if elementRes[2] > 0:
                geom.extrude(s, [0, 0, zz], num_layers=elementRes[2])
                
            structured_tri_rect = geom.generate_mesh()   
            structured_tri_rect.remove_lower_dimensional_cells()

        return structured_tri_rect


# pygmesh generator for Tet/Tri-based structured, box mesh
# Note boundary labels are needed (cf PETSc box mesh above)

class Unstructured_Simplex_Box(MeshFromMeshIO):
    @timing.routine_timer_decorator
    def __init__(self,
                dim          :Optional[  int] = 2,
                minCoords    :Optional[Tuple[float,float,float]] =None,
                maxCoords    :Optional[Tuple[float,float,float]] =None,
                coarse_cell_size: Optional[float]=0.1, 
                global_cell_size:Optional[float]=0.05,
                degree       :Optional[int]        =1

                ):
        """
        This class generates a box

        Parameters
        ----------
        dim :
            The mesh dimensionality.
        minCoord:
            Optional. Tuple specifying minimum mesh location. 
        maxCoord:
            Optional. Tuple specifying maximum mesh location.
        coarse_cell_size :
            The target cell size for the unstructured template mesh that will later be refined for 
            solving the unknowns. 
        global_cell_size :
            The target cell size for the final mesh. Mesh refinements will occur to achieve this target 
            resolution. 
        """

        if minCoords==None: 
            minCoords=(0.0,0.0,0.0)

        if len(minCoords) == 2:
            minCoords = (minCoords[0], minCoords[1], 0.0)

        if maxCoords==None: 
            maxCoords=(1.0,1.0,1.0) 

        if len(maxCoords) == 2:
            maxCoords = (maxCoords[0], maxCoords[1], 0.0)


        self.pygmesh = None
        # Only root proc generates pygmesh, then it's distributed.


        if MPI.COMM_WORLD.rank==0:
            unstructured_simplex_box = Unstructured_Simplex_Box.build_pygmsh(dim, 
                            minCoords, maxCoords, 
                            coarse_cell_size,
                            global_cell_size)

        super().__init__(dim, unstructured_simplex_box, global_cell_size, simplex=True, degree=degree)

        self.pygmesh = unstructured_simplex_box
        self.meshio = unstructured_simplex_box
        self.elementRes = None
        self.minCoords = minCoords
        self.maxCoords = maxCoords

        import vtk

        if dim == 2:
            self._elementType = vtk.VTK_TRIANGLE
        else:
            self._elementType = vtk.VTK_TETRA

        return

    def build_pygmsh(
                     dim, 
                     minCoords, maxCoords, 
                     coarse_cell_size,  
                     global_cell_size ):


        xx = maxCoords[0]-minCoords[0]
        yy = maxCoords[1]-minCoords[1]
        zz = maxCoords[2]-minCoords[2]

        import pygmsh
        with pygmsh.occ.Geometry() as geom:
            if dim == 2:
                # args: corner point (3-tuple), width, height, corner-roundness ... 
                box = geom.add_rectangle(minCoords,xx,yy,0.0, mesh_size=coarse_cell_size)
            else:
                # args: corner point (3-tuple), size (3-tuple) ... 
                box = geom.add_box(minCoords,(xx,yy,zz), mesh_size=coarse_cell_size)

            unstructured_simplex_box = geom.generate_mesh()  
            unstructured_simplex_box.remove_lower_dimensional_cells()

        return unstructured_simplex_box

class SphericalShell(MeshFromGmshFile):


    @timing.routine_timer_decorator
    def __init__(self,
                 dim              :Optional[  int] =2,
                 radius_outer     :Optional[float] =1.0,
                 radius_inner     :Optional[float] =0.5,
                 cell_size        :Optional[float] =0.05,
                 cell_size_upper  :Optional[float] =None,
                 cell_size_lower  :Optional[float] =None,
                 degree           :Optional[int]   =2,
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
                        centre = geom.add_point((0.0,0.0,0.0), mesh_size=cell_size_lower)
                        domain = geom.add_circle((0.0,0.0,0.0), radius_outer, mesh_size=cell_size_upper)
                        geom.in_surface(centre, domain.plane_surface)
                        geom.add_physical(centre, label="Centre")
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
     

                groups = [] # no groups !

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
        super().__init__(dim, filename="ignore_ball_mesh_geom.msh", label_groups=groups, 
                              cell_size=cell_size, simplex=True, degree=degree)


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
                degree         :Optional[int]  =2,
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
                simplex        :Optional[bool] = False, 
                degree         :Optional[int]  = 2
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
                              cell_size=cell_size, simplex=simplex, degree=degree)


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
                degree       :Optional[int]    =2
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

