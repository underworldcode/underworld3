from petsc4py.PETSc cimport DM, PetscDM, DS, PetscDS, Vec, PetscVec, PetscSF, IS, PetscIS, Quad, PetscQuadrature, FE, PetscFE, Mat, PetscMat
from .petsc_types cimport PetscInt, PetscReal, PetscScalar, PetscErrorCode, PetscBool
import petsc4py.PETSc as PETSc
from .petsc_gen_xdmf import generateXdmf
from mpi4py import MPI
import contextlib
from typeguard import check_argument_types, check_return_type
from typeguard import typechecked
import underworld3 as uw
import numpy as np
from underworld3 import _api_tools
import underworld3.timing as timing

cdef extern from "petsc.h" nogil:
    PetscErrorCode DMCreateMassMatrix(PetscDM dac, PetscDM daf, PetscMat *mat)
    PetscErrorCode DMCreateSubDM(PetscDM, PetscInt, const PetscInt *, PetscIS *, PetscDM *)
    PetscErrorCode DMSwarmDestroyGlobalVectorFromField(PetscDM dm, const char fieldname[], PetscVec *vec)

cdef CHKERRQ(PetscErrorCode ierr):
    cdef int interr = <int>ierr
    if ierr != 0: raise RuntimeError(f"PETSc error code '{interr}' was encountered.\nhttps://www.mcs.anl.gov/petsc/petsc-current/include/petscerror.h.html")

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
    DMSWARMPIC_LAYOUT_GAUSS = 1

class SwarmVariable(_api_tools.Stateful):
    @timing.routine_timer_decorator
    def __init__(self, name, swarm, num_components, vtype=None, dtype=float, proxy_order=2, _register=True, _proxy=True):

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
            self.swarm.registerField(self.name, self.num_components, dtype=petsc_type)
        self._data = None
        # add to swarms dict
        swarm.vars[name] = self
        self._is_accessed = False

        # create proxy variable
        self._meshVar = None
        if _proxy:
            self._meshVar = uw.mesh.MeshVariable(name, self.swarm.mesh, num_components, vtype, degree=proxy_order)

        super().__init__()

    def _update(self):
        """
        This method updates the proxy mesh variable for the current 
        swarm & particle variable state.
        """
        # if not proxied, nothing to do. return.
        if not self._meshVar:
            return
        nnmap = self.swarm._get_map(self)
        # set NN vals on mesh var
        with self.swarm.mesh.access(self._meshVar), self.swarm.access():
            self._meshVar.data[:,:] = self.data[nnmap,:]

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
class Swarm(PETSc.DMSwarm,_api_tools.Stateful):
    @timing.routine_timer_decorator
    def __init__(self, mesh):
        self.mesh = mesh
        self.dim = mesh.dim
        self.dm = Swarm.create(self)
        self.dm.setDimension(self.dim)
        self.dm.setType(SwarmType.DMSWARM_PIC.value)
        self.dm.setCellDM(mesh.dm)
        self._data = None

        # dictionary for variables
        import weakref
        self._vars = weakref.WeakValueDictionary()

        # add variable to handle particle coords
        self._coord_var = SwarmVariable("DMSwarmPIC_coor", self, self.dim, dtype=float, _register=False, _proxy=False)

        self._kdtree = None
        self._nnmapdict = {}

        super().__init__()

    @property
    def particle_coordinates(self):
        return self._coord_var

    @timing.routine_timer_decorator
    def populate(self, ppcell=25, layout=SwarmPICLayout.DMSWARMPIC_LAYOUT_GAUSS):
        
        self.ppcell = ppcell
        
        if not isinstance(layout, SwarmPICLayout):
            raise ValueError("'layout' must be an instance of 'SwarmPICLayout'")
        
        self.layout = layout
        
        elstart,elend = self.mesh.dm.getHeightStratum(0)
        self.dm.finalizeFieldRegister()
        self.dm.setLocalSizes((elend-elstart) * ppcell, 0)
        self.dm.insertPointUsingCellDM(self.layout.value, ppcell)
        return self

    @timing.routine_timer_decorator
    def add_variable(self, name, num_components=1, dtype=float):
        var = SwarmVariable(name, self, num_components, dtype=dtype)

        return var

    @timing.routine_timer_decorator
    def save(self, filename):
        self.dm.viewXDMF(filename)
    
    @property
    def vars(self):
        return self._vars

    @contextlib.contextmanager
    def access(self, *writeable_vars:SwarmVariable):
        """
        This context manager makes the underlying swarm variables data available to
        the user. The data should be accessed via the variables `data` handle. 

        As default, all data is read-only. To enable writeable data, the user should
        specify which variable they wish to modify.

        At the conclusion of the users context managed block, numerous further operations
        will be automatically executed. This includes swarm parallel migration routines
        where the swarm's `particle_coordinate` variable has been modified. The swarm
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
            var._data = self.getField(var.name).reshape( (-1, var.num_components) )
            if var not in writeable_vars:
                var._old_data_flag = var._data.flags.writeable
                var._data.flags.writeable = False
            else:
                # increment variable state
                var._increment()
        # if particles moving, update swarm state
        if self.particle_coordinates in writeable_vars:
            self._increment()


        try:
            yield
        except:
            raise
        finally:
            for var in self.vars.values():
                # only de-access variables we have set access for.
                if var not in deaccess_list:
                    continue
                # set this back, although possibly not required.
                if var not in writeable_vars:
                    var._data.flags.writeable = var._old_data_flag
                var._data = None
                self.restoreField(var.name)
                var._is_accessed = False
            # do particle migration if coords changes
            if self.particle_coordinates in writeable_vars:
                self.migrate(remove_sent_points=True)
                # void these things too
                self._kdtree = None
                self._nnmapdict = {}
            # do var updates
            for var in self.vars.values():
                # if swarm migrated, update all. 
                # if var updated, update var.
                if (self.particle_coordinates in writeable_vars) or \
                   (var                       in writeable_vars) :
                    var._update()

    def _get_map(self,var):
        # generate tree if not avaiable
        if not self._kdtree:
            from scipy import spatial
            with self.access():
                self._kdtree = spatial.KDTree(self.particle_coordinates.data)

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
            self._nnmapdict[digest] = self._kdtree.query(meshvar_coords)[1]
        return self._nnmapdict[digest]

