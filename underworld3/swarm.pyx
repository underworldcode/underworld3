from typing import Optional, Tuple
import contextlib

import numpy as np
import petsc4py.PETSc as PETSc
from mpi4py import MPI

import underworld3 as uw
from underworld3 import _api_tools
import underworld3.timing as timing

include "./petsc_extras.pxi"

cdef extern from "petsc.h" nogil:
    PetscErrorCode DMCreateMassMatrix(PetscDM dac, PetscDM daf, PetscMat *mat)
    PetscErrorCode DMSwarmDestroyGlobalVectorFromField(PetscDM dm, const char fieldname[], PetscVec *vec)

cdef inline object str2bytes(object s, char *p[]):
    if s is None:
        p[0] = NULL
        return None
    if not isinstance(s, bytes):
        s = s.encode()
    p[0] = <char*>(<char*>s)
    return s

comm = MPI.COMM_WORLD

from enum import Enum
class SwarmType(Enum):
    DMSWARM_PIC = 1

class SwarmPICLayout(Enum):
    """
    Particle population fill type:

    SwarmPICLayout.REGULAR     defines points on a regular ijk mesh. Supported by simplex cell types only.
    SwarmPICLayout.GAUSS       defines points using an npoint Gauss-Legendre tensor product quadrature rule.
    SwarmPICLayout.SUBDIVISION defines points on the centroid of a sub-divided reference cell.
    """
    REGULAR     = 0
    GAUSS       = 1
    SUBDIVISION = 2


class SwarmVariable(_api_tools.Stateful):
    @timing.routine_timer_decorator
    def __init__(self, name, swarm, num_components, 
                 vtype=None, dtype=float, proxy_degree=2, _register=True, _proxy=True, _nn_proxy=False):

        if name in swarm.vars.keys():
            raise ValueError("Variable with name {} already exists on swarm.".format(name))

        self.name = name
        self.swarm = swarm
        self.num_components = num_components
        if   (dtype==float) or (dtype=="float") or (dtype==dtype,np.float64):
            self.dtype = float
            petsc_type = PETSc.ScalarType
        elif (dtype==int)   or (dtype=="int")   or (dtype==np.int32):
            self.dtype = int
            petsc_type = PETSc.IntType
        else:
            raise TypeError(f"Provided dtype={dtype} is not supported. Supported types are 'int' and 'float'.")
        if _register:
            self.swarm.dm.registerField(self.name, self.num_components, dtype=petsc_type)
        self._data = None
        # add to swarms dict
        swarm.vars[name] = self
        self._is_accessed = False

        # create proxy variable
        self._meshVar = None
        if _proxy:
            self._meshVar = uw.mesh.MeshVariable(name, self.swarm.mesh, num_components, vtype, degree=proxy_degree)

        self._register = _register
        self._proxy = _proxy
        self._nn_proxy = _nn_proxy

        super().__init__()

    def _update(self):
        """
        This method updates the proxy mesh variable for the current 
        swarm & particle variable state.

        Here is how it works:

            1) for each particle, create a distance-weighted average on the node data
            2) check to see which nodes have zero weight / zero contribution and replace with nearest particle value

        Todo: caching the k-d trees etc for the proxy-mesh-variable nodal points
        Todo: some form of global fall-back for when there are no particles on a processor 

        """

        # if not proxied, nothing to do. return.
        if not self._meshVar:
            return
       
        # 1 - Average particles to nodes with distance weighted average

        kd = uw.algorithms.KDTree(self._meshVar.coords)
        kd.build_index()

        with self.swarm.access():
            n,d,b = kd.find_closest_point(self.swarm.data)
   
            node_values  = np.zeros((self._meshVar.coords.shape[0],self.num_components))
            w = np.zeros(self._meshVar.coords.shape[0]) 

            if not self._nn_proxy:
                for i in range(self.data.shape[0]):
                    if b[i]:
                        node_values[n[i],:] += self.data[i,:] / (1.0e-16+d[i])
                        w[n[i]] += 1.0 / (1.0e-16+d[i])

                node_values[np.where(w > 0.0)[0],:] /= w[np.where(w > 0.0)[0]].reshape(-1,1)

        # 2 - set NN vals on mesh var where w == 0.0 
         
        p_nnmap = self.swarm._get_map(self)

        with self.swarm.mesh.access(self._meshVar), self.swarm.access():
            self._meshVar.data[...] = node_values[...]
            self._meshVar.data[np.where(w==0.0),:] = self.data[p_nnmap[np.where(w==0.0)],:]
        
        return      

    @timing.routine_timer_decorator
    def project_from(self, meshvar):
        # use method found in 
        # /tmp/petsc-build/petsc/src/dm/impls/swarm/tests/ex2.c
        # to project from fields to particles

        self.swarm.mesh.dm.clearDS()
        self.swarm.mesh.dm.createDS()

        cdef DM meshvardm = PETSc.DM()
        cdef DM meshdm = meshvar.mesh.dm 
        cdef PetscInt fields = meshvar.field_id
        ierr = DMCreateSubDM(meshdm.dm, 1, &fields, NULL, &meshvardm.dm); CHKERRQ(ierr)

#   ierr = KSPCreate(comm, &ksp);CHKERRQ(ierr);
#   ierr = KSPSetOptionsPrefix(ksp, "ftop_");CHKERRQ(ierr);
#   ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
        ksp = PETSc.KSP().create()
        ksp.setOptionsPrefix("swarm_project_from_")
        options = PETSc.Options()
        options.setValue("swarm_project_from_ksp_type", "lsqr")
        options.setValue("swarm_project_from_ksp_rtol", 1e-17)
        options.setValue("swarm_project_from_pc_type" , "none")
        ksp.setFromOptions()


#   ierr = DMGetGlobalVector(dm, &fhat);CHKERRQ(ierr);
#   ierr = DMGetGlobalVector(dm, &rhs);CHKERRQ(ierr);
        rhs = meshvardm.getGlobalVec()

#   ierr = DMCreateMassMatrix(sw, dm, &M_p);CHKERRQ(ierr);
#   ierr = MatViewFromOptions(M_p, NULL, "-M_p_view");CHKERRQ(ierr);
        cdef DM swarmdm = self.swarm
        cdef Mat M_p = PETSc.Mat()
        ierr = DMCreateMassMatrix(swarmdm.dm, meshvardm.dm, &M_p.mat); CHKERRQ(ierr)

#   /* make particle weight vector */
#   ierr = DMSwarmCreateGlobalVectorFromField(sw, "w_q", &f);CHKERRQ(ierr);
        f = self.swarm.createGlobalVectorFromField(self.name)

#   /* create matrix RHS vector, in this case the FEM field fhat with the coefficients vector #alpha */
#   ierr = PetscObjectSetName((PetscObject) rhs,"rhs");CHKERRQ(ierr);
#   ierr = VecViewFromOptions(rhs, NULL, "-rhs_view");CHKERRQ(ierr);
#   ierr = DMCreateMatrix(dm, &M);CHKERRQ(ierr);
#   ierr = DMPlexSNESComputeJacobianFEM(dm, fhat, M, M, user);CHKERRQ(ierr);
#   ierr = MatViewFromOptions(M, NULL, "-M_view");CHKERRQ(ierr);
#   ierr = MatMultTranspose(M, fhat, rhs);CHKERRQ(ierr);
#   if (user->useBlockDiagPrec) {ierr = DMSwarmCreateMassMatrixSquare(sw, dm, &PM_p);CHKERRQ(ierr);}
#   else                        {ierr = PetscObjectReference((PetscObject) M_p);CHKERRQ(ierr); PM_p = M_p;}
        cdef Mat M = PETSc.Mat()
        ierr = DMCreateMassMatrix(meshvardm.dm, meshvardm.dm, &M.mat); CHKERRQ(ierr)
        with meshvar.mesh.access():
            M.multTranspose(meshvar.vec_global,rhs)
    

#   ierr = KSPSetOperators(ksp, M_p, PM_p);CHKERRQ(ierr);
#   ierr = KSPSolveTranspose(ksp, rhs, f);CHKERRQ(ierr);
#   ierr = PetscObjectSetName((PetscObject) fhat,"fhat");CHKERRQ(ierr);
#   ierr = VecViewFromOptions(fhat, NULL, "-fhat_view");CHKERRQ(ierr);
        ksp.setOperators(M_p, M_p)
        ksp.solveTranspose(rhs,f)

#   ierr = DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &f);CHKERRQ(ierr);
        # self.swarm.destroyGlobalVectorFromField(self.name)  # this appears to be broken in petsc4py
        cdef Vec cf = f
        cdef char *cval = NULL
        fieldname = str2bytes(self.name, &cval)

        DMSwarmDestroyGlobalVectorFromField(swarmdm.dm, cval, &cf.vec)
        meshvardm.restoreGlobalVec(rhs)
        meshvardm.destroy()
        ksp.destroy()
        M.destroy()
        M_p.destroy()

    @property
    def data(self):
        if self._data is None:
            raise RuntimeError("Data must be accessed via the swarm `access()` context manager.")
        return self._data

    @property
    def fn(self):
        return self._meshVar.fn


#@typechecked
class Swarm(_api_tools.Stateful):
    @timing.routine_timer_decorator
    def __init__(self, mesh):
        self.mesh = mesh
        self.dim = mesh.dim
        self.cdim = mesh.cdim
        self.dm = PETSc.DMSwarm().create()
        self.dm.setDimension(self.dim)
        self.dm.setType(SwarmType.DMSWARM_PIC.value)
        self.dm.setCellDM(mesh.dm)
        self._data = None

        # dictionary for variables
        import weakref
        self._vars = weakref.WeakValueDictionary()

        # add variable to handle particle coords
        self._coord_var = SwarmVariable("DMSwarmPIC_coor", self, self.cdim, dtype=float, _register=False, _proxy=False)

        # add variable to handle particle cell id
        self._cellid_var = SwarmVariable("DMSwarm_cellid", self, 1, dtype=int, _register=False, _proxy=False)

        self._index = None
        self._nnmapdict = {}

        super().__init__()

    @property
    def data(self):
        return self.particle_coordinates.data

    @property
    def particle_coordinates(self):
        return self._coord_var

    @property
    def particle_cellid(self):
        return self._cellid_var

    @timing.routine_timer_decorator
    def populate(self, 
                 fill_param :Optional[int]            =3, 
                 layout     :Optional[SwarmPICLayout] =None):
        """
        Populate the swarm with particles throughout the domain.

        """ + SwarmPICLayout.__doc__ + """

        When using SwarmPICLayout.REGULAR,     `fill_param` defines the number of points in each spatial direction.
        When using SwarmPICLayout.GAUSS,       `fill_param` defines the number of quadrature points in each spatial direction.
        When using SwarmPICLayout.SUBDIVISION, `fill_param` defines the number times the reference cell is sub-divided.

        Parameters
        ----------
        fill_param:
            Parameter determining the particle count per cell for the given layout.
        layout:
            Type of layout to use. Defaults to `SwarmPICLayout.REGULAR` for mesh objects with simplex
            type cells, and `SwarmPICLayout.GAUSS` otherwise.
        """

        self.fill_param = fill_param

        """
        Currently (2021.11.15) supported by PETSc release 3.16.x
 
        When using a DMPLEX the following case are supported:
              (i) DMSWARMPIC_LAYOUT_REGULAR: 2D (triangle),
             (ii) DMSWARMPIC_LAYOUT_GAUSS: 2D and 3D provided the cell is a tri/tet or a quad/hex,
            (iii) DMSWARMPIC_LAYOUT_SUBDIVISION: 2D and 3D for quad/hex and 2D tri.

        So this means, simplex mesh in 3D only supports GAUSS 

        """

        
        if layout==None:
            if self.mesh.isSimplex==True and self.dim == 2:
                layout=SwarmPICLayout.REGULAR
            else:
                layout=SwarmPICLayout.GAUSS

        if not isinstance(layout, SwarmPICLayout):
            raise ValueError("'layout' must be an instance of 'SwarmPICLayout'")
        
        self.layout = layout
        self.dm.finalizeFieldRegister()

        ## Commenting this out for now. 
        ## Code seems to operate fine without it, and the 
        ## existing values are wrong. It should be something like
        ## `(elend-elstart)*fill_param^dim` for quads, and around
        ## half that for simplices, depending on layout.
        # elstart,elend = self.mesh.dm.getHeightStratum(0)
        # self.dm.setLocalSizes((elend-elstart) * fill_param, 0)

        self.dm.insertPointUsingCellDM(self.layout.value, fill_param)
        return # self # LM: Is there any reason to return self ?

    @timing.routine_timer_decorator
    def add_variable(self, name, num_components=1, dtype=float, proxy_degree=2, _nn_proxy=False):
        return SwarmVariable(name, self, num_components, dtype=dtype, proxy_degree=proxy_degree, _nn_proxy=_nn_proxy)

    @property
    def vars(self):
        return self._vars

    def access(self, *writeable_vars:SwarmVariable):
        """
        This context manager makes the underlying swarm variables data available to
        the user. The data should be accessed via the variables `data` handle. 

        As default, all data is read-only. To enable writeable data, the user should
        specify which variable they wish to modify.

        At the conclusion of the users context managed block, numerous further operations
        will be automatically executed. This includes swarm parallel migration routines
        where the swarm's `particle_coordinates` variable has been modified. The swarm
        variable proxy mesh variables will also be updated for modifed swarm variables. 

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

        deaccess_list = []
        for var in self.vars.values():
            # if already accessed within higher level context manager, continue.
            if var._is_accessed == True:
                continue
            # set flag so variable status can be known elsewhere
            var._is_accessed = True
            # add to de-access list to rewind this later
            deaccess_list.append(var)
            # grab numpy object, setting read only if necessary
            var._data = self.dm.getField(var.name).reshape( (-1, var.num_components) )
            if var not in writeable_vars:
                var._old_data_flag = var._data.flags.writeable
                var._data.flags.writeable = False
            else:
                # increment variable state
                var._increment()
        # if particles moving, update swarm state
        if self.particle_coordinates in writeable_vars:
            self._increment()

        # Create a class which specifies the required context
        # manager hooks (`__enter__`, `__exit__`).
        class exit_manager:
            def __init__(self,swarm): self.em_swarm = swarm 
            def __enter__(self): pass
            def __exit__(self, *args):
                for var in self.em_swarm.vars.values():
                    # only de-access variables we have set access for.
                    if var not in deaccess_list:
                        continue
                    # set this back, although possibly not required.
                    if var not in writeable_vars:
                        var._data.flags.writeable = var._old_data_flag
                    var._data = None
                    self.em_swarm.dm.restoreField(var.name)
                    var._is_accessed = False
                # do particle migration if coords changes
                if self.em_swarm.particle_coordinates in writeable_vars:
                    # let's use the mesh index to update the particles owning cells.
                    # note that the `petsc4py` interface is more convenient here as the 
                    # `SwarmVariable.data` interface is controlled by the context manager
                    # that we are currently within, and it is therefore too easy to  
                    # get things wrong that way.
                    cellid = self.em_swarm.dm.getField("DMSwarm_cellid")
                    coords = self.em_swarm.dm.getField("DMSwarmPIC_coor").reshape( (-1, self.em_swarm.dim) )
                    cellid[:] = self.em_swarm.mesh.get_closest_cells(coords)
                    self.em_swarm.dm.restoreField("DMSwarmPIC_coor")
                    self.em_swarm.dm.restoreField("DMSwarm_cellid")
                    # now migrate.
                    self.em_swarm.dm.migrate(remove_sent_points=True)
                    # void these things too
                    self.em_swarm._index = None
                    self.em_swarm._nnmapdict = {}
                # do var updates
                for var in self.em_swarm.vars.values():
                    # if swarm migrated, update all. 
                    # if var updated, update var.
                    if (self.em_swarm.particle_coordinates in writeable_vars) or \
                       (var                                in writeable_vars) :
                        var._update()

                uw.timing._decrementDepth()
                uw.timing.log_result(time.time()-stime, "Swarm.access",1)
        return exit_manager(self)

    def _get_map(self,var):
        # generate tree if not avaiable
        if not self._index:
            with self.access():
                self._index = uw.algorithms.KDTree(self.data)

        # get or generate map
        meshvar_coords = var._meshVar.coords
        # we can't use numpy arrays directly as keys in python dicts, so 
        # we'll use `xxhash` to generate a hash of array. 
        # this shouldn't be an issue performance wise but we should test to be 
        # sufficiently confident of this. 
        import xxhash
        h = xxhash.xxh64()
        h.update(meshvar_coords)
        digest = h.intdigest()
        if digest not in self._nnmapdict:
            self._nnmapdict[digest] = self._index.find_closest_point(meshvar_coords)[0]
        return self._nnmapdict[digest]

