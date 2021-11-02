# cython: profile=False
from libc.stdlib cimport malloc, free
from petsc4py.PETSc cimport DM, PetscDM, DS, PetscDS, FE, PetscFE, Vec, PetscVec, IS, PetscIS, PetscSF, MPI_Comm, PetscObject, Mat, PetscMat, GetCommDefault, PetscViewer
from .petsc_types cimport PetscInt, PetscReal, PetscScalar, PetscErrorCode, PetscBool, DMBoundaryConditionType, PetscDSResidualFn, PetscDSJacobianFn
from petsc4py import PETSc
from .petsc_gen_xdmf import generateXdmf
import contextlib
import numpy as np
cimport numpy as np
import sympy
import sympy.vector
import underworld3 as uw 
import underworld3
import numpy
from underworld3 import _api_tools
from mpi4py import MPI
import underworld3.timing as timing
import math
from typing import Optional, Tuple

ctypedef enum PetscBool:
    PETSC_FALSE
    PETSC_TRUE

cdef extern from "petsc.h" nogil:
    PetscErrorCode DMPlexCreateBallMesh(MPI_Comm, PetscInt, PetscReal, PetscDM*)
    PetscErrorCode DMPlexComputeGeometryFVM( PetscDM dm, PetscVec *cellgeom, PetscVec *facegeom)
    PetscErrorCode DMPlexGetMinRadius(PetscDM dm, PetscReal *minradius)
    PetscErrorCode VecDestroy(PetscVec *v)
    PetscErrorCode DMDestroy(PetscDM *dm)
    PetscErrorCode DMCreateSubDM(PetscDM, PetscInt, const PetscInt *, PetscIS *, PetscDM *)
    PetscErrorCode DMProjectCoordinates(PetscDM dm, PetscFE disc)
    PetscErrorCode MatInterpolate(PetscMat A, PetscVec x, PetscVec y)
    PetscErrorCode DMCompositeGetLocalISs(PetscDM dm,PetscIS **isets)
    PetscErrorCode DMPlexExtrude(PetscDM idm, PetscInt layers, PetscReal height, PetscBool orderHeight, const PetscReal extNormal[], PetscBool interpolate, PetscDM* dm)
    # PetscErrorCode PetscViewerHDF5PushTimestepping(PetscViewer viewer)
    MPI_Comm MPI_COMM_SELF

cdef CHKERRQ(PetscErrorCode ierr):
    cdef int interr = <int>ierr
    if ierr != 0: raise RuntimeError(f"PETSc error code '{interr}' was encountered.\nhttps://www.mcs.anl.gov/petsc/petsc-current/include/petscerror.h.html")


class MeshClass(_api_tools.Stateful):
    @timing.routine_timer_decorator
    def __init__(self, simplex, *args,**kwargs):
        self.isSimplex = simplex

        # Set sympy constructs
        from sympy.vector import CoordSys3D
        self._N = CoordSys3D("N")
        # Tidy some of this printing. Note that this
        # only changes the user interface printing, and 
        # switches out for simpler `BaseScalar` representations.
        # For example, we now have "x" instead of "x_N". This makes
        # Jupyter rendered Latex easier to digest, but depending on 
        # how we end up using Sympy coordinate systems it may be 
        # desirable to bring back the more verbose version.
        self._N.x._latex_form=r"\mathrm{x}"
        self._N.y._latex_form=r"\mathrm{y}"
        self._N.z._latex_form=r"\mathrm{z}"

        # dictionary for variables
        import weakref
        self._vars = weakref.WeakValueDictionary()

        self._accessed = False
        self._stale_lvec = True
        self._lvec = None

        # dictionary for variable coordinate arrays
        self._coord_array = {}
        # let's go ahead and do an initial projection from linear (the default) 
        # to linear. this really is a nothing operation, but a 
        # side effect of this operation is that coordinate DM DMField is 
        # converted to the required `PetscFE` type. this may become necessary
        # later where we call the interpolation routines to project from the linear
        # mesh coordinates to other mesh coordinates. 
        options = PETSc.Options()
        options.setValue("meshproj_petscspace_degree", 1) 
        cdmfe = PETSc.FE().createDefault(self.dim, self.dim, self.isSimplex, 1,"meshproj_", PETSc.COMM_WORLD)
        cdef FE c_fe = cdmfe
        cdef DM c_dm = self.dm
        ierr = DMProjectCoordinates( c_dm.dm, c_fe.fe ); CHKERRQ(ierr)
        # now set copy of linear array into dictionary
        arr = self.dm.getCoordinatesLocal().array
        self._coord_array[(self.isSimplex,1)] = arr.reshape(-1, self.dim).copy()
        self._index = None

        self._elementType = None

        super().__init__()

    @timing.routine_timer_decorator
    def update_lvec(self):
        """
        This method creates and/or updates the mesh variable local vector. 
        If the local vector is already up to date, this method will do nothing.
        """
        cdef DM dm = self.dm
        if self._stale_lvec:
            if not self._lvec:
                self.dm.clearDS()
                self.dm.createDS()
                # create the local vector (memory chunk) and attach to original dm
                self._lvec = self.dm.createLocalVec()
            # push avar arrays into the parent dm array
            a_global = self.dm.getGlobalVec()
            names, isets, dms = self.dm.createFieldDecomposition()
            with self.access():
                # traverse subdms, taking user generated data in the subdm
                # local vec, pushing it into a global sub vec 
                for var,subiset,subdm in zip(self.vars.values(),isets,dms):
                    lvec = var.vec
                    subvec = a_global.getSubVector(subiset)
                    subdm.localToGlobal(lvec,subvec, addv=False)
                    a_global.restoreSubVector(subiset,subvec)

            self.dm.globalToLocal(a_global,self._lvec)
            self.dm.restoreGlobalVec(a_global)
            self._stale_lvec = False

    @property
    def lvec(self) -> PETSc.Vec:
        """
        Returns a local Petsc vector containing the flattened array 
        of all the mesh variables.
        """
        if self._stale_lvec:
            raise RuntimeError("Mesh `lvec` needs to be updated using the update_lvec()` method.")
        return self._lvec

    def __del__(self):
        if hasattr(self, "_lvec") and self._lvec:
            self._lvec.destroy()

    def access(self, *writeable_vars:"MeshVariable"):
        """
        This context manager makes the underlying mesh variables data available to
        the user. The data should be accessed via the variables `data` handle. 

        As default, all data is read-only. To enable writeable data, the user should
        specify which variable they wish to modify.

        Parameters
        ----------
        writeable_vars
            The variables for which data write access is required.

        Example
        -------
        >>> import underworld3 as uw
        >>> someMesh = uw.mesh.FeMesh_Cartesian()
        >>> with someMesh.deform_mesh():
        ...     someMesh.data[0] = [0.1,0.1]
        >>> someMesh.data[0]
        array([ 0.1,  0.1])
        """

        import time
        uw.timing._incrementDepth()
        stime = time.time()

        self._accessed = True
        deaccess_list = []
        for var in self.vars.values():
            # if already accessed within higher level context manager, continue.
            if var._is_accessed == True:
                continue
            # set flag so variable status can be known elsewhere
            var._is_accessed = True
            # add to de-access list to rewind this later
            deaccess_list.append(var)
            # create & set vec
            var._set_vec(available=True)
            # grab numpy object, setting read only if necessary
            var._data = var.vec.array.reshape( -1, var.num_components )
            if var not in writeable_vars:
                var._old_data_flag = var._data.flags.writeable
                var._data.flags.writeable = False
            else:
                # increment variable state
                var._increment()

        class exit_manager:
            def __init__(self,mesh): self.mesh = mesh
            def __enter__(self): pass
            def __exit__(self,*args):
                cdef DM subdm
                cdef DM dm
                cdef PetscInt fields
                for var in self.mesh.vars.values():
                    # only de-access variables we have set access for.
                    if var not in deaccess_list:
                        continue
                    # set this back, although possibly not required.
                    if var not in writeable_vars:
                        var._data.flags.writeable = var._old_data_flag
                    # perform sync for any modified vars.
                    if var in writeable_vars:
                        subdm = PETSc.DM()
                        dm = self.mesh.dm
                        fields = var.field_id
                        ierr = DMCreateSubDM(dm.dm, 1, &fields, NULL, &subdm.dm);CHKERRQ(ierr)
                        # sync ghost values
                        subdm.localToGlobal(var.vec,var._gvec, addv=False)
                        subdm.globalToLocal(var._gvec,var.vec, addv=False)
                        ierr = DMDestroy(&subdm.dm);CHKERRQ(ierr)
                        self.mesh._stale_lvec = True
                    var._data = None
                    var._set_vec(available=False)
                    var._is_accessed = False
                uw.timing._decrementDepth()
                uw.timing.log_result(time.time()-stime, "Mesh.access",1)
        return exit_manager(self)


    @property
    def N(self) -> sympy.vector.CoordSys3D:
        """
        The mesh coordinate system.
        """
        return self._N

    @property
    def r(self) -> Tuple[sympy.vector.BaseScalar]:
        """
        The tuple of base scalar objects (N.x,N.y,N.z) for the mesh. 
        """
        return self._N.base_scalars()[0:self.dim]

    @property
    def rvec(self) -> sympy.vector.Vector:
        """
        The r vector, `r = N.x*N.i + N.y*N.j [+ N.z*N.k]`.
        """
        N = self.N
        rvecguy = N.x*N.i + N.y*N.j
        if self.dim==3:
            rvecguy += N.z*N.k
        return rvecguy

    @property
    def data(self) -> numpy.ndarray:
        """
        The array of mesh element vertex coordinates.
        """
        # get flat array
        arr = self.dm.getCoordinatesLocal().array
        return arr.reshape(-1, self.dim)

    @property
    def dim(self) -> int:
        """ 
        The mesh dimensionality.
        """
        return self.dm.getDimension()


    @property
    def elementType(self) -> int:
        """
        The (vtk) element type classification for the mesh.
        Will be set to None if it is not meaningful to set one
        value fo the whole mesh
        """
        return self._elementType

    @timing.routine_timer_decorator
    def save(self, filename : str,
                   index    : Optional[int] = None):
        """
        Save mesh data to the specified file. 

        Users will generally create this file, and then
        append mesh variable data to it via the variable
        `save` method.

        Parameters
        ----------
        filename :
            The filename for the mesh checkpoint file. 
        index :
            Not yet implemented. An optional index which might 
            correspond to the timestep (for example). 

        """
        viewer = PETSc.ViewerHDF5().create(filename, "w", comm=MPI.COMM_WORLD)
        # cdef PetscViewer cviewer = ....
        if index:
            raise RuntimeError("Recording `index` not currently supported")
            ## JM:To enable timestep recording, the following needs to be called.
            ## I'm unsure if the corresponding xdmf functionality is enabled via 
            ## the PETSc xdmf script.
            # PetscViewerHDF5PushTimestepping(cviewer)
            # viewer.setTimestep(index)
        viewer(self.dm)
    
    def generate_xdmf(self, filename:str):
        """
        This method generates an xdmf schema for the specified file.

        The filename of the generated file will be the same as the hdf5 file
        but with the `xmf` extension. 

        Parameters
        ----------
        filename :
            File name of the checkpointed hdf5 file for which the
            xdmf schema will be written.
        """
        generateXdmf(filename)

    @property
    def vars(self):
        """
        A list of variables recorded on the mesh.
        """
        return self._vars

    def _get_coords_for_var(self, var):
        """
        This function returns the vertex array for the 
        provided variable. If the array does not already exist, 
        it is first created and then returned.
        """
        key = (self.isSimplex,var.degree) 
        # if array already created, return. 
        if key in self._coord_array:
            return self._coord_array[key]
        # otherwise create and return
        cdmOld = self.dm.getCoordinateDM()
        cdmNew = cdmOld.clone()
        options = PETSc.Options()
        options.setValue("coordinterp_petscspace_degree", var.degree) 
        cdmfe = PETSc.FE().createDefault(self.dim, self.dim, self.isSimplex, var.degree, "coordinterp_", PETSc.COMM_WORLD)
        cdmNew.setField(0,cdmfe)
        cdmNew.createDS()
        (matInterp, vecScale) = cdmOld.createInterpolation(cdmNew)
        vecScale.destroy() # not needed
        coordsOld = self.dm.getCoordinates()
        coordsNewG = cdmNew.getGlobalVec()
        coordsNewL = cdmNew.getLocalVec()
        cdef Mat c_matInterp = matInterp
        cdef Vec c_coordsOld = coordsOld
        cdef Vec c_coordsNewG = coordsNewG
        ierr = MatInterpolate(c_matInterp.mat, c_coordsOld.vec, c_coordsNewG.vec); CHKERRQ(ierr)
        cdmNew.globalToLocal(coordsNewG,coordsNewL)
        arr = coordsNewL.array
        # reshape and grab copy
        arrcopy = arr.reshape(-1,self.dim).copy()
        # record into coord array
        self._coord_array[key] = arrcopy
        # clean up
        cdmNew.restoreLocalVec(coordsNewL)
        cdmNew.restoreGlobalVec(coordsNewG)
        cdmNew.destroy()
        cdmfe.destroy()
        # return
        return arrcopy
    
    def get_closest_cells(self, coords: numpy.ndarray) -> numpy.ndarray:
        """
        This method uses a kd-tree algorithm to find the closest
        cells to the provided coords. For a regular mesh, this should 
        be exactly the owning cell, but if the mesh is deformed, this 
        is not guaranteed. 

        Parameters:
        -----------
        coords:
            An array of the coordinates for which we wish to determine the
            closest cells. This should be a 2-dimensional array of 
            shape (n_coords,dim).

        Returns:
        --------
        closest_cells:
            An array of indices representing the cells closest to the provided
            coordinates. This will be a 1-dimensional array of 
            shape (n_coords).
        """
        # Create index if required
        if not self._index:
            # Get cell centroids. 
            # Note that if necessary, we might like to do something
            # like index on Gauss points instead of centroids. This should
            # give better results for deformed mesh. 
            elstart,elend = self.dm.getHeightStratum(0)
            centroids = np.empty((elend,self.dim))
            for index in range(elend):
                centroids[index] = self.dm.computeCellGeometryFVM(index)[1]
            self._index = uw.algorithms.KDTree(centroids)
            self._index.build_index()

        closest_cells, dist, found = self._index.find_closest_point(coords)

        if not np.allclose(found,True):
            raise RuntimeError("An error was encountered attempting to find the closest cells to the provided coordinates.")
        
        return closest_cells

    def get_min_radius(self) -> double:
        """
        This method returns the minimum distance from any cell centroid to a face.
        It wraps to the PETSc `DMPlexGetMinRadius` routine. 
        """

        cdef PetscVec cellgeom = NULL
        cdef PetscVec facegeom = NULL
        cdef DM dm = self.dm
        cdef double minradius
        if (not hasattr(self,"_min_radius")) or (self._min_radius==None):
            DMPlexComputeGeometryFVM(dm.dm,&cellgeom,&facegeom)
            DMPlexGetMinRadius(dm.dm,&minradius)
            self._min_radius = minradius
            VecDestroy(&cellgeom)
            VecDestroy(&facegeom)
        return self._min_radius

    def mesh2pyvista(self, 
                     elementType: Optional[int]=None
                     ):
        """
        Returns a (vtk) pyvista.UnstructuredGrid object for visualisation of the mesh
        skeleton. Note that this will normally use the elementType that is on the mesh
        (HEX, QUAD) but, for data visualisation on higher-order elements, the actual 
        order of the element is needed.
        
        ToDo: check for pyvista installation

        ToDo: parallel safety

        ToDo: use vtk instead of pyvista 
        """

        import vtk
        import pyvista as pv 

        if elementType is None:
            elementType = self.elementType

        # vtk defines all meshes using 3D coordinate arrays
        if self.dim == 2: 
            coords = self.data
            vtk_coords = np.zeros((coords.shape[0], 3))
            vtk_coords[:,0:2] = coords[:,:]
        else:
            vtk_coords = self.data

        cells = self.mesh_dm_cells()
        cell_type = np.empty(cells.shape[0], dtype=np.int64)
        cell_type[:] = elementType

        vtk_cells = np.empty((cells.shape[0], cells.shape[1]+1), dtype=np.int64)
        vtk_cells[:,0] = cells.shape[1]
        vtk_cells[:,1:] = cells[:,:]

        pyvtk_unstructured_grid = pv.UnstructuredGrid(vtk_cells, cell_type, vtk_coords)

        return pyvtk_unstructured_grid

    def mesh_dm_coords(self):
        cdim = self.dm.getCoordinateDim()
        #lcoords = self.dm.getCoordinatesLocal().array.reshape(-1,cdim)
        coords = self.dm.getCoordinates().array.reshape(-1,cdim)
        return coords

    def mesh_dm_edges(self):
        # coords = mesh_coords(mesh)
        # import pdb; pdb.set_trace()
        starti,endi = self.dm.getDepthStratum(1)
        #Offset of the node indices (level 0)
        coffset = self.dm.getDepthStratum(0)[0]
        edgesize = self.dm.getConeSize(starti)
        edges = np.zeros((endi-starti,edgesize), dtype=np.uint32)
        for c in range(starti, endi):
            edges[c-starti,:] = self.dm.getCone(c) - coffset

        #edges -= edges.min() #Why the offset?
        #print(edges)
        #print(edges.min(), edges.max(), coords.shape)
        return edges

    def mesh_dm_faces(self):
        #Faces / 2d cells
        coords = self.mesh_dm_coords()
        #cdim = mesh.dm.getCoordinateDim()

        #Index range in mesh.dm of level 2
        starti,endi = self.dm.getDepthStratum(2)
        #Offset of the node indices (level 0)
        coffset = self.dm.getDepthStratum(0)[0]
        FACES=(endi-starti)
        facesize = self.mesh_dm_facesize() # Face elements 3(tri) or 4(quad)
        faces = np.zeros((FACES,facesize), dtype=np.uint32)
        for c in range(starti, endi):
            point_closure = self.dm.getTransitiveClosure(c)[0]
            faces[c-starti,:] = point_closure[-facesize:] - coffset
        return faces

    def mesh_dm_facesize(self):
        return self.dm.getConeSize(self.dm.getDepthStratum(2)[0]) #Face elements 3(tri) or 4(quad)

    def mesh_dm_cellsize(self):
        depth = self.dm.getDepth()
        if depth < 3:
            return self.mesh_dm_facesize() #Cells are faces
        return self.dm.getConeSize(self.dm.getDepthStratum(3)[0])  #Cell elements 4(tet) or 6(cuboid)

    def mesh_dm_info(mesh):
        depth = mesh.dm.getDepth()
        sz = mesh.dm.getChart()
        print('getChart (index range)', sz, 'getDepth', depth)
        for i in range(depth+1):
            starti,endi = mesh.dm.getDepthStratum(i)
            conesize = mesh.dm.getConeSize(starti)
            print(i, "range: [", starti, endi, "] coneSize", conesize)
        return

    def mesh_dm_cells(self):

        depth = self.dm.getDepth()

        if depth < 3:
            return self.mesh_dm_faces()
    
            coords = self.mesh_dm_coords()

            #Index range in mesh.dm of level 2
            starti,endi = self.dm.getDepthStratum(2)
            #Offset of the node indices (level 0)

            coffset = self.dm.getDepthStratum(0)[0]
            FACES=(endi-starti)
            facesize = self.mesh_dm_facesize() # Face elements 3(tri) or 4(quad)
            faces = np.zeros((FACES,facesize), dtype=np.uint32)
            for c in range(starti, endi):
                point_closure = self.dm.getTransitiveClosure(c)[0]
                faces[c-starti,:] = point_closure[-facesize:] - coffset

            return faces

        #Index range in mesh.dm of level 3
        starti,endi = self.dm.getDepthStratum(3)
        #Offset of the node indices (level 0)
        coffset = self.dm.getDepthStratum(0)[0]
        CELLS=(endi-starti)
        facesize = self.mesh_dm_facesize() # Face elements 3(tri) or 4(quad)
        cellsize = self.mesh_dm_cellsize() # Cell elements 4(tet) or 6(cuboid)
        FACES = CELLS * cellsize

        if cellsize == 4:
            cell_corners = 4
        else:
            cell_corners = 8
        
        # List of faces (vertex indices)
        faces = np.zeros((FACES,facesize), dtype=np.uint32)
        #print("CELLSIZE:", cellsize, "FACESIZE:",facesize, "SHAPE:",faces.shape)
        
        cell_vertices = np.empty((CELLS, cell_corners), dtype=np.int64)
        
        for c in range(CELLS):
            # The "cone" is the list of face indices for this cell
            cone = self.dm.getCone(c+starti)
            #print("CONE",cone)
            #Iterate through each face element of the cone
            facepoints = np.empty((cellsize, facesize), dtype=np.int64)
            for co in range(cellsize):
                #This contains the face vertex indices in correct order at the end
                point_closure = self.dm.getTransitiveClosure(cone[co])[0]
                faces[cellsize*c + co,:] = point_closure[-facesize:] - coffset
                facepoints[co, :] = point_closure[-facesize:] - coffset
            
            # Unique vertex values only, but preserve ordering
            unique,indices = np.unique(facepoints.flatten(), return_index=True)
            cell_vertices[c,:] = facepoints.flatten()[np.sort(indices)]
                      
        return cell_vertices

class Box(MeshClass):
    @timing.routine_timer_decorator
    def __init__(self, 
                elementRes   :         Tuple[int,  int,  int]    =(16, 16), 
                minCoords    :Optional[Tuple[float,float,float]] =None,
                maxCoords    :Optional[Tuple[float,float,float]] =None,
                simplex      :Optional[bool]                     =False
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
        if "dm_plex_hash_location" in options: del options["dm_plex_hash_location"]
        if "dm_plex_hash_box_nijk" in options: del options["dm_plex_hash_box_nijk"]
        if len(elementRes)==2:
            options["dm_plex_hash_location"] = None
            options["dm_plex_hash_box_nijk"] = max(elementRes)
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
            boundary_label = self.dm.getLabel(str(val).encode('utf8')) # get label
            # Without this check, we have failures at this point in parallel. 
            # Further investigation required. JM.
            if boundary_set:
                boundary_label.insertIS(boundary_set, 1) # add set to label with value 1

        super().__init__(simplex=simplex)





# JM: I don't think this class is required any longer
# and the pygmsh version should instead be used. I'll
# leave this implementation here for now.
# class Sphere(MeshClass):
#     """
#     Generates a 2 or 3-dimensional box mesh.
#     """
#     @timing.routine_timer_decorator
#     def __init__(self, 
#                 refinements=4, 
#                 radius=1.):

#         self.refinements = refinements
#         self.radius = radius

#         options = PETSc.Options()
#         options.setValue("bd_dm_refine", self.refinements)

#         cdef DM dm = PETSc.DMPlex()
#         cdef MPI_Comm ccomm = GetCommDefault()
#         cdef PetscInt cdim = 3
#         cdef PetscReal cradius = self.radius
#         DMPlexCreateBallMesh(ccomm, cdim, cradius, &dm.dm)
#         self.dm = dm


#         part = self.dm.getPartitioner()
#         part.setFromOptions()
#         self.dm.distribute()
#         self.dm.setFromOptions()

#         from enum import Enum        
#         class Boundary(Enum):
#             OUTER = 1
        
#         self.boundary = Boundary

#         self.dm.view()        
        
#         super().__init__(simplex=True)

class MeshFromCellList(MeshClass):
    @timing.routine_timer_decorator
    def __init__(self,
                 dim         :int,
                 cells       :numpy.ndarray,
                 coords      :numpy.ndarray,
                 cell_size   :Optional[float] = None,
                 refinements :Optional[int]   = 0,
                 simplex      :Optional[bool] = True
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
        super().__init__(simplex=simplex)

class MeshFromMeshIO(MeshFromCellList):
    @timing.routine_timer_decorator
    def __init__(self,
                 dim         :int,
                 meshio      :"MeshIO",
                 cell_size   :Optional[float] =None,
                 refinements :Optional[int]   = 0,
                 simplex      :Optional[bool] = True
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
        cells = coords = None
        if MPI.COMM_WORLD.rank==0:
            cells  = meshio.cells[dim-1][1]
            coords = meshio.points[:,0:dim]
  
        super().__init__(dim, cells, coords, cell_size, refinements, simplex=simplex)

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
                cell_size    :Optional[float] =0.05):
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
                self.pygmesh = quad_hex_box

        super().__init__(dim,self.pygmesh, cell_size, simplex=False)

        self.elementRes = elementRes
        self.minCoords = minCoords
        self.maxCoords = maxCoords

        import vtk

        if dim == 2:
            self._elementType = vtk.VTK_QUAD
        else:
            self._elementType = vtk.VTK_HEXAHEDRON

        return


# pygmesh generator for Tet/Tri-based structured, box mesh
# Note boundary labels are needed (cf PETSc box mesh above)

class Simplex_Box(MeshFromMeshIO):
    @timing.routine_timer_decorator
    def __init__(self,
                dim          :Optional[  int] = 2,
                elementRes   :Tuple[int,  int,  int]    = (16, 16, 0), 
                minCoords    :Optional[Tuple[float,float,float]] =None,
                maxCoords    :Optional[Tuple[float,float,float]] =None,
                cell_size    :Optional[float] =0.05):
        """
        This class generates a spherical shell, or a full sphere
        where the inner radius is zero.

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
                self.pygmesh = structured_tri_rect

        super().__init__(dim, self.pygmesh, cell_size)

        self.elementRes = elementRes
        self.minCoords = minCoords
        self.maxCoords = maxCoords

        import vtk

        if dim == 2:
            self._elementType = vtk.VTK_TRIANGLE
        else:
            self._elementType = vtk.VTK_TETRA

        return

# pygmesh generator for Tet/Tri-based structured, box mesh
# Note boundary labels are needed (cf PETSc box mesh above)

class Unstructured_Simplex_Box(MeshFromMeshIO):
    @timing.routine_timer_decorator
    def __init__(self,
                dim          :Optional[  int] = 2,
                minCoords    :Optional[Tuple[float,float,float]] =None,
                maxCoords    :Optional[Tuple[float,float,float]] =None,
                coarse_cell_size: Optional[float]=0.1, 
                global_cell_size:Optional[float]=0.05
                ):
        """
        This class generates a spherical shell, or a full sphere
        where the inner radius is zero.

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

        super().__init__(dim, unstructured_simplex_box, global_cell_size)

        self.pygmesh = unstructured_simplex_box
        self.elementRes = None
        self.minCoords = minCoords
        self.maxCoords = maxCoords

        import vtk

        if dim == 2:
            self._elementType = vtk.VTK_TRIANGLE
        else:
            self._elementType = vtk.VTK_TETRA

        return



class SphericalShell(MeshFromMeshIO):

    @timing.routine_timer_decorator
    def __init__(self,
                 dim            :Optional[  int] =2,
                 radius_outer   :Optional[float] =1.0,
                 radius_inner   :Optional[float] =0.5,
                 cell_size      :Optional[float] =0.05):
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
            if   dim==2:
                domain_area = math.pi*(radius_outer**2 - radius_inner**2)
                # Add factor of 2 to `cell_size`. This is a fudge factor to bring
                # the typical generated cell_size from the pygmsh mesh closer to what
                # petsc reports.
                csize_local = self._get_local_cell_size(2,domain_area,2.*cell_size)
            elif dim==3:
                domain_vol = 4./3.*math.pi*(radius_outer**3 - radius_inner**3)
                # Add factor of 4 to `cell_size`. This is a fudge factor to bring
                # the typical generated cell_size from the pygmsh mesh closer to what
                # petsc reports.
                csize_local = self._get_local_cell_size(3,domain_vol,4.*cell_size)
            else:
                raise ValueError("`dim` must be in [2,3].")

            import pygmsh
            # Generate local mesh.
            with pygmsh.occ.Geometry() as geom:
                geom.characteristic_length_max = csize_local
                if dim==2:
                    ndimspherefunc = geom.add_disk
                else:
                    ndimspherefunc = geom.add_ball
                ball_outer = ndimspherefunc([0.0,]*dim, radius_outer)
                if radius_inner > 0.:
                    ball_inner = ndimspherefunc([0.0,]*dim, radius_inner)
                    geom.boolean_difference(ball_outer,ball_inner)
                # The following is an example of setting a callback for variable resolution.
                # geom.set_mesh_size_callback(
                #     lambda dim, tag, x, y, z: 0.15*abs(1.-sqrt(x ** 2 + y ** 2 + z ** 2)) + 0.15
                # )
                self.pygmesh = geom.generate_mesh()

        super().__init__(dim, self.pygmesh, cell_size)

        import vtk

        if dim == 2:
            self._elementType = vtk.VTK_TRIANGLE
        else:
            self._elementType = vtk.VTK_TETRA

        return


# The following does not work correctly as the transfinite volume is not correctly 
# meshed ... / cannot be generated consistently with these surface descriptions. 


'''
class StructuredCubeSphericalCap(MeshFromMeshIO):

    @timing.routine_timer_decorator
    def __init__(self,
                elementRes   :Tuple[int,  int,  int]  = (16, 16, 8), 
                angles       :Tuple[float, float] = (0.7853981633974483, 0.7853981633974483), # pi/4
                radius_outer   :Optional[float] =1.0,
                radius_inner   :Optional[float] =0.5,
                simplex        :Optional[bool] =False, 
                cell_size      :Optional[float] =0.1
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

            def spherical_cap_mesh(elementsNS, elementsEW, elementsR, 
                                   radius_i, radius_o, 
                                   theta, phi, simplex=False):
                """
                Spherical cap pygmsh routines
                """            

                with pygmsh.geo.Geometry() as geom:
                    
                    cpoint = geom.add_point([0.0,0.0,0.0], 1)

                    def cube_corners(R, theta, phi):

                        X = R * np.cos(theta/2.0)*np.cos(phi/2.0)
                        Y = R * np.sin(theta/2.0)*np.cos(phi/2.0)
                        Z = R * np.sin(phi/2.0)
                        
                        genpoint = [0,0,0,0]

                        genpoint[0] = geom.add_point([  X,  Y,  Z], 1)  
                        genpoint[1] = geom.add_point([  X,  Y, -Z], 1)
                        genpoint[2] = geom.add_point([  X, -Y,  Z], 1)
                        genpoint[3] = geom.add_point([  X, -Y, -Z], 1)

                        return genpoint

                    def build_sph_cap_surface(genpoint, a,b,c,d):

                        b_circ = [None, None, None, None]

                        b_circ[0] = geom.add_circle_arc(genpoint[a], cpoint, genpoint[b])
                        b_circ[1] = geom.add_circle_arc(genpoint[b], cpoint, genpoint[c])
                        b_circ[2] = geom.add_circle_arc(genpoint[c], cpoint, genpoint[d])
                        b_circ[3] = geom.add_circle_arc(genpoint[d], cpoint, genpoint[a])

                        geom.set_transfinite_curve(b_circ[0], num_nodes=elementsEW, mesh_type="Progression", coeff=1.0)
                        geom.set_transfinite_curve(b_circ[1], num_nodes=elementsNS, mesh_type="Progression", coeff=1.0)
                        
                        geom.set_transfinite_curve(b_circ[2], num_nodes=elementsEW, mesh_type="Progression", coeff=1.0)
                        geom.set_transfinite_curve(b_circ[3], num_nodes=elementsNS, mesh_type="Progression", coeff=1.0)

                        wedge_edge = geom.add_curve_loop(b_circ)
                        this_wedge = geom.add_surface(wedge_edge) 

                        geom.set_transfinite_surface(this_wedge, arrangement="1", corner_pts=[genpoint[a], genpoint[b], genpoint[c], genpoint[d]])

                        return b_circ, this_wedge

                    genpoint_i = cube_corners(radius_i, theta, phi)
                    genpoint_o = cube_corners(radius_o, theta, phi)

                    cap_outlines_o, cap1o = build_sph_cap_surface(genpoint_o,0,1,3,2)
                    cap_outlines_i, cap1i = build_sph_cap_surface(genpoint_i,0,1,3,2)

                    # 4 lines defining the radial edges

                    lineNE = geom.add_line(genpoint_i[0], genpoint_o[0])
                    lineSE = geom.add_line(genpoint_i[1], genpoint_o[1])
                    lineNW = geom.add_line(genpoint_i[2], genpoint_o[2])
                    lineSW = geom.add_line(genpoint_i[3], genpoint_o[3])

                    geom.set_transfinite_curve(lineNE, num_nodes=elementsR, mesh_type="Progression", coeff=1.0)
                    geom.set_transfinite_curve(lineSE, num_nodes=elementsR, mesh_type="Progression", coeff=1.0)
                    geom.set_transfinite_curve(lineSW, num_nodes=elementsR, mesh_type="Progression", coeff=1.0)
                    geom.set_transfinite_curve(lineNW, num_nodes=elementsR, mesh_type="Progression", coeff=1.0)

                    faceE_outline = geom.add_curve_loop([lineNE, cap_outlines_o[0], -lineSE, -cap_outlines_i[0]  ])
                    faceE = geom.add_surface(faceE_outline)
                    geom.set_transfinite_surface(faceE, arrangement="1", corner_pts=[genpoint_i[0], genpoint_i[1], genpoint_o[0], genpoint_o[1]])

                    faceW_outline = geom.add_curve_loop([lineNW, -cap_outlines_o[2], -lineSW, cap_outlines_i[2]  ])
                    faceW = geom.add_surface(faceW_outline)
                    geom.set_transfinite_surface(faceW, arrangement="1", corner_pts=[genpoint_i[2], genpoint_i[3], genpoint_o[2], genpoint_o[3]])

                    faceN_outline = geom.add_curve_loop([lineNE, -cap_outlines_o[3], -lineNW, cap_outlines_i[3]  ])
                    faceN = geom.add_surface(faceN_outline)
                    geom.set_transfinite_surface(faceN, arrangement="1", corner_pts=[genpoint_i[0], genpoint_i[2], genpoint_o[0], genpoint_o[2]])

                    faceS_outline = geom.add_curve_loop([lineSE, cap_outlines_o[1], -lineSW,  -cap_outlines_i[1] ])
                    faceS = geom.add_surface(faceS_outline)
                    geom.set_transfinite_surface(faceS, arrangement="1", corner_pts=[genpoint_i[1], genpoint_i[3], genpoint_o[1], genpoint_o[3]])

                    shell = geom.add_surface_loop([cap1o, cap1i, faceE, faceS, faceW, faceN])  
                    geom.add_volume(shell)

                    if not simplex:
                        geom.set_recombined_surfaces([cap1i, cap1o])
                        geom.set_recombined_surfaces([faceE, faceW, faceN, faceS])

                    spherical_cap = geom.generate_mesh(dim=3, verbose=False)
                    
                    return spherical_cap

        self.pygmesh = spherical_cap_mesh(elementsNS=elementRes[0], 
                                          elementsEW=elementRes[1],
                                          elementsR=elementRes[2], 
                                          radius_i=radius_inner, 
                                          radius_o=radius_outer, 
                                          theta=angles[0], 
                                          phi=angles[1], 
                                          simplex=simplex)

        super().__init__(3, self.pygmesh, cell_size)

        import vtk

        if simplex:
            self._elementType = vtk.VTK_TETRA
        else:
            self._elementType = vtk.VTK_HEXAHEDRON

        self.elementRes = elementRes
        self.minCoords = (-angles[0]/2.0, -angles[1]/2.0)
        self.maxCoords = ( angles[0]/2.0,  angles[1]/2.0)

  

        return
'''

class MeshVariable(_api_tools.Stateful):
    @timing.routine_timer_decorator
    def __init__(self, name                             : str, 
                       mesh                             : "underworld.mesh.MeshClass", 
                       num_components                   : int, 
                       vtype                            : Optional["underworld.VarType"] = None, 
                       degree                           : int =1 ):
        """
        The MeshVariable class generates a variable supported by a finite element mesh.

        To set / read nodal values, use the numpy interface via the 'data' property.

        Parameters
        ----------
        name :
            A textual name for this variable.
        mesh :
            The supporting underworld mesh.
        num_components :
            The number of components this variable has.
            For example, scalars will have `num_components=1`, 
            while a 2d vector would have `num_components=2`.
        vtype :
            Optional. The underworld variable type for this variable.
            If not defined it will be inferred from `num_components`
            if possible.
        degree :
            The polynomial degree for this variable.

        """

        self._lvec = None
        self._gvec = None
        self._data = None
        self._is_accessed = False

        if mesh._accessed:
            raise RuntimeError("It is not possible to add new variables to a mesh after existing variables have been accessed.")
        if name in mesh.vars.keys():
            raise ValueError("Variable with name {} already exists on mesh.".format(name))
        self.name = name

        if vtype==None:
            if   num_components==1:
                vtype=uw.VarType.SCALAR
            elif num_components==mesh.dim:
                vtype=uw.VarType.VECTOR
            else:
                raise ValueError("Unable to infer variable type from `num_components`. Please explicitly set the `vtype` parameter.")
        if not isinstance(vtype, uw.VarType):
            raise ValueError("'vtype' must be an instance of 'Variable_Type', for example `underworld.VarType.SCALAR`.")
        self.vtype = vtype
        self.mesh = mesh
        self.num_components = num_components

        options = PETSc.Options()
        options.setValue(f"{name}_petscspace_degree", degree)
        self.degree = degree

        self.petsc_fe = PETSc.FE().createDefault(self.mesh.dm.getDimension(), num_components, self.mesh.isSimplex, PETSc.DEFAULT, name+"_", PETSc.COMM_WORLD)

        self.field_id = self.mesh.dm.getNumFields()
        self.mesh.dm.setField(self.field_id,self.petsc_fe)

        # create associated sympy function
        from underworld3.function import UnderworldFunction
        if   vtype==uw.VarType.SCALAR:
            self._fn = UnderworldFunction(name,self,vtype)(*self.mesh.r)
        elif vtype==uw.VarType.VECTOR:
            if num_components!=mesh.dim:
                raise ValueError("For 'VarType.VECTOR' types 'num_components' must equal 'mesh.dim'.")
            from sympy.vector import VectorZero
            self._fn = VectorZero()
            for comp in range(num_components):
                subfn = UnderworldFunction(name,self,vtype,comp)(*self.mesh.r)
                self._fn += subfn*self.mesh.N.base_vectors()[comp]
        super().__init__()

        self.mesh.vars[name] = self



    @timing.routine_timer_decorator
    def save(self, filename : str,
                   name     : Optional[str] = None,
                   index    : Optional[int] = None):
        """
        Append variable data to the specified mesh 
        checkpoint file. The file must already exist.

        Parameters
        ----------
        filename :
            The filename of the mesh checkpoint file. It
            must already exist.
        name :
            Textual name for dataset. In particular, this
            will be used for XDMF generation. If not 
            provided, the variable name will be used. 
        index :
            Not currently supported. An optional index which 
            might correspond to the timestep (for example).
        """
        viewer = PETSc.ViewerHDF5().create(filename, "a", comm=MPI.COMM_WORLD)
        # cdef PetscViewer cviewer = ....
        if index:
            raise RuntimeError("Recording `index` not currently supported")
            ## JM:To enable timestep recording, the following needs to be called.
            ## I'm unsure if the corresponding xdmf functionality is enabled via 
            ## the PETSc xdmf script.
            # PetscViewerHDF5PushTimestepping(cviewer)
            # viewer.setTimestep(index)
        if name:
            oldname = self._gvec.getName()
            self._gvec.setName(name)
        viewer(self._gvec)
        if name: self._gvec.setName(oldname)

    @property
    def fn(self) -> sympy.Basic:
        """
        The handle to the function view of this variable.
        """
        return self._fn

    def _set_vec(self, available):
        cdef DM subdm = PETSc.DM()
        cdef DM dm = self.mesh.dm
        cdef PetscInt fields = self.field_id
        if self._lvec==None:
            # Create a subdm for this variable.
            # This allows us to generate a local vectors.
            ierr = DMCreateSubDM(dm.dm, 1, &fields, NULL, &subdm.dm);CHKERRQ(ierr)
            self._lvec  = subdm.createLocalVector()
            self._lvec.zeroEntries()       # not sure if required, but to be sure. 
            self._gvec  = subdm.createGlobalVector()
            self._gvec.setName(self.name)  # This is set for checkpointing. 
            self._gvec.zeroEntries()
            ierr = DMDestroy(&subdm.dm);CHKERRQ(ierr)
        self._available = available

    def __del__(self):
        if self._lvec:
            self._lvec.destroy()
        if self._gvec:
            self._gvec.destroy()

    @property
    def vec(self) -> PETSc.Vec:
        """
        The corresponding PETSc local vector for this variable.
        """
        if not self._available:
            raise RuntimeError("Vector must be accessed via the mesh `access()` context manager.")
        return self._lvec

    @property
    def data(self) -> numpy.ndarray:
        """
        Numpy proxy array to underlying variable data.
        Note that the returned array is a proxy for all the *local* nodal
        data, and is provided as 1d list. 

        For both read and write, this array can only be accessed via the
        mesh `access()` context manager.
        """
        if self._data is None:
            raise RuntimeError("Data must be accessed via the mesh `access()` context manager.")
        return self._data

    def min(self) -> float:
        """
        The global variable minimum value.
        """
        if not self._lvec:
            raise RuntimeError("It doesn't appear that any data has been set as of yet.")
        return self._gvec.min()

    def max(self) -> float:
        """
        The global variable maximum value.
        """
        if not self._lvec:
            raise RuntimeError("It doesn't appear that any data has been set as of yet.")
        return self._gvec.max()

    @property
    def coords(self) -> numpy.ndarray:
        """
        The array of variable vertex coordinates. 
        """
        return self.mesh._get_coords_for_var(self)