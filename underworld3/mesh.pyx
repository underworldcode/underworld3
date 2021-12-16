# cython: profile=False
from typing import Optional, Tuple, Union
from collections import namedtuple

from enum import Enum

import math

import numpy
import numpy as np
cimport numpy as np
import sympy
import sympy.vector

from mpi4py import MPI
from petsc4py import PETSc

import meshio

include "./petsc_extras.pxi"

import underworld3
import underworld3 as uw 
from underworld3 import _api_tools
import underworld3.timing as timing

class MeshClass(_api_tools.Stateful):
    @timing.routine_timer_decorator
    def __init__(self, simplex, degree=1, *args,**kwargs):
        self.isSimplex = simplex

        # Enable hashing
        options = PETSc.Options()
        options["dm_plex_hash_location"] = 0
        # Need some tweaks for <3.16.
        petsc_version_minor = PETSc.Sys().getVersion()[1]
        if petsc_version_minor < 16:
            # Let's use 3.16 default heuristics to set hashing grid size.
            cStart,cEnd = self.dm.getHeightStratum(0)
            options["dm_plex_hash_box_nijk"] = max(2, math.floor( (cEnd - cStart)**(1.0/self.dim) * 0.8) )
            # However, if we're in 3d and petsc <3.16, no bueno :-(
            if self.dim==3:
                options["dm_plex_hash_location"] = 0
                options.delValue("dm_plex_hash_box_nijk")
        self.dm.setFromOptions()

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
        self._N.i._latex_form=r"\mathbf{\hat{i}}"
        self._N.j._latex_form=r"\mathbf{\hat{j}}"
        self._N.k._latex_form=r"\mathbf{\hat{k}}"

        # dictionary for variables
        import weakref
        self._vars = weakref.WeakValueDictionary()

     
        self._accessed = False
        self._stale_lvec = True
        self._lvec = None
        self.petsc_fe = None

        self._elementType = None

        self.degree = degree
        self.nuke_coords_and_rebuild()

        # A private work array
        self._work_MeshVar = MeshVariable('work_array_1', self,  1, degree=3 ) 


        super().__init__()

    def nuke_coords_and_rebuild(self):

        # This is a reversion to the old version (3.15 compatible which seems to work)

        self._coord_array = {}

        # let's go ahead and do an initial projection from linear (the default) 
        # to linear. this really is a nothing operation, but a 
        # side effect of this operation is that coordinate DM DMField is 
        # converted to the required `PetscFE` type. this may become necessary
        # later where we call the interpolation routines to project from the linear
        # mesh coordinates to other mesh coordinates. 

        ## LM  - I put in the option to specify the default coordinate interpolation degree 
        ## LM  - which seems sensible given linear interpolation seems likely to be a problem
        ## LM  - for spherical meshes. However, I am not sure about this because it means that
        ## LM  - the mesh coords and the size of the nodal array are different. This might break 
        ## LM  - stuff so I will leave the default at 1

        options = PETSc.Options()
        options.setValue("meshproj_petscspace_degree", self.degree) 
        cdmfe = PETSc.FE().createDefault(self.dim, self.dim, self.isSimplex,
                                                    self.degree,  "meshproj_", PETSc.COMM_WORLD)
    
        self.petsc_fe = cdmfe
        cdef FE c_fe = cdmfe
        cdef DM c_dm = self.dm
        ierr = DMProjectCoordinates( c_dm.dm, c_fe.fe ); CHKERRQ(ierr)

        # now set copy of this array into dictionary

        arr = self.dm.getCoordinatesLocal().array
        self._coord_array[(self.isSimplex,self.degree)] = arr.reshape(-1, self.dim).copy()

        # invalidate the cell-search k-d tree 
        self._index = None

        return


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

    @timing.routine_timer_decorator
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
            from underworld3.swarm import Swarm
            # Create a temp swarm which we'll use to populate particles
            # at gauss points. These will then be used as basis for 
            # kd-tree indexing back to owning cells.
            tempSwarm = Swarm(self)
            # 4^dim pop is used. This number may need to be considered
            # more carefully, or possibly should be coded to be set dynamically. 
            tempSwarm.populate(fill_param=4)
            with tempSwarm.access():
                # Build index on particle coords
                self._indexCoords = tempSwarm.particle_coordinates.data.copy()
                self._index = uw.algorithms.KDTree(self._indexCoords)
                self._index.build_index()
                # Grab mapping back to cell_ids. 
                # Note that this is the numpy array that we eventually return from this 
                # method. As such, we take measures to ensure that we use `np.int64` here 
                # because we cast from this type in  `_function.evaluate` to construct 
                # the PETSc cell-sf datasets, and if instead a `np.int32` is used it 
                # will cause bugs that are difficult to find.
                self._indexMap = np.array(tempSwarm.particle_cellid.data[:,0], dtype=np.int64)

        closest_points, dist, found = self._index.find_closest_point(coords)

        if not np.allclose(found,True):
            raise RuntimeError("An error was encountered attempting to find the closest cells to the provided coordinates.")

        return self._indexMap[closest_points]

 
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


    def stats(self, uw_function):
        """
        Returns various norms on the mesh for the provided function. 
          - size
          - mean
          - min
          - max
          - sum
          - L2 norm
          - rms

          NOTE: this currently assumes scalar variables !
        """    

#       This uses a private work MeshVariable and the various norms defined there but
#       could either be simplified to just use petsc vectors, or extended to 
#       compute integrals over the elements which is in line with uw1 and uw2 

        from petsc4py.PETSc import NormType 

        tmp = self._work_MeshVar

        with self.access(tmp):
            tmp.data[...] = uw.function.evaluate(uw_function, tmp.coords).reshape(-1,1)
        
        vsize =  self._work_MeshVar._gvec.getSize()
        vmean  = tmp.mean()
        vmax   = tmp.max()[1]
        vmin   = tmp.min()[1]
        vsum   = tmp.sum()
        vnorm2 = tmp.norm(NormType.NORM_2)
        vrms   = vnorm2 / np.sqrt(vsize)

        return vsize, vmean, vmin, vmax, vsum, vnorm2, vrms


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

    def mesh_dm_info(self):
        depth = self.dm.getDepth()
        sz = self.dm.getChart()
        print('getChart (index range)', sz, 'getDepth', depth)
        for i in range(depth+1):
            starti,endi = self.dm.getDepthStratum(i)
            conesize = self.dm.getConeSize(starti)
            print(i, "range: [", starti, endi, "] coneSize", conesize)
        return

    def mesh_dm_cells(self):

        depth = self.dm.getDepth()

        if depth < 3:
            return self.mesh_dm_faces()


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
            # This allows us to generate a local vector.
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

    def min(self) -> Union[float , tuple]:
        """
        The global variable minimum value.
        """
        if not self._lvec:
            raise RuntimeError("It doesn't appear that any data has been set.")

        if self.num_components == 1:
            return self._gvec.min()
        else:
            cpts = []
            for i in range(0,self.num_components):
                cpts.append(self._gvec.strideMin(i)[1])

            return tuple(cpts)

    def max(self) -> Union[float , tuple]:
        """
        The global variable maximum value.
        """
        if not self._lvec:
            raise RuntimeError("It doesn't appear that any data has been set.")

        if self.num_components == 1:
            return self._gvec.max()
        else:
            cpts = []
            for i in range(0,self.num_components):
                cpts.append(self._gvec.strideMax(i)[1])

            return tuple(cpts)

    def sum(self) -> Union[float , tuple]:
        """
        The global variable sum value.
        """
        if not self._lvec:
            raise RuntimeError("It doesn't appear that any data has been set.")

        if self.num_components == 1:
            return self._gvec.sum()
        else:
            cpts = []
            for i in range(0,self.num_components):
                cpts.append(self._gvec.strideSum(i))

            return tuple(cpts)

    def norm(self, norm_type) -> Union[float , tuple]:
        """
        The global variable maximum value.
        """
        if not self._lvec:
            raise RuntimeError("It doesn't appear that any data has been set.")

        if self.num_components == 1:
            return self._gvec.norm(norm_type)
        else:
            cpts = []
            for i in range(0,self.num_components):
                cpts.append(self._gvec.strideNorm(i, norm_type))

            return tuple(cpts)


    def mean(self) -> Union[float , tuple]:
        """
        The global variable maximum value.
        """
        if not self._lvec:
            raise RuntimeError("It doesn't appear that any data has been set.")

        if self.num_components == 1:
            vecsize = self._gvec.getSize()
            return self._gvec.sum() / vecsize
        else:
            vecsize = self._gvec.getSize() / self.num_components
            cpts = []
            for i in range(0,self.num_components):
                cpts.append(self._gvec.strideSum(i)/vecsize)

            return tuple(cpts)

    def stats(self):
        """
        The equivalent of mesh.stats but using the native coordinates for this variable
        Not set up for vector variables so we just skip that for now.

        Returns various norms on the mesh using the native mesh discretisation for this
        variable. It is a wrapper on the various _gvec stats routines for the variable.

          - size
          - mean
          - min
          - max
          - sum
          - L2 norm
          - rms
        """    

        if self.num_components > 1:
            raise NotImplementedError('stats not available for multi-component variables')

#       This uses a private work MeshVariable and the various norms defined there but
#       could either be simplified to just use petsc vectors, or extended to 
#       compute integrals over the elements which is in line with uw1 and uw2 

        from petsc4py.PETSc import NormType 

        vsize =  self._gvec.getSize()
        vmean  = self.mean()
        vmax   = self.max()[1]
        vmin   = self.min()[1]
        vsum   = self.sum()
        vnorm2 = self.norm(NormType.NORM_2)
        vrms   = vnorm2 / np.sqrt(vsize)

        return vsize, vmean, vmin, vmax, vsum, vnorm2, vrms

    @property
    def coords(self) -> numpy.ndarray:
        """
        The array of variable vertex coordinates. 
        """
        return self.mesh._get_coords_for_var(self)