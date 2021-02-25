from petsc4py.PETSc cimport DM, PetscDM, DS, PetscDS, Vec, PetscVec, PetscSF, IS, PetscIS, Quad, PetscQuadrature, FE, PetscFE, Mat, PetscMat
from ..petsc_types cimport PetscInt, PetscReal, PetscScalar, PetscErrorCode, PetscBool, DMBoundaryConditionType, PetscDSResidualFn, PetscDSJacobianFn
from ..petsc_types cimport PtrContainer
import underworld3 as uw
import sympy
from sympy import sympify
from sympy.vector import gradient, divergence
from .._jitextension import getext, diff_fn1_wrt_fn2

# TODO
# gil v nogil 
# ctypeds DMBoundaryConditionType etc.. is there a cleaner way? 
cdef extern from "petsc.h" nogil:
    PetscErrorCode DMCreateSubDM(PetscDM, PetscInt, const PetscInt *, PetscIS *, PetscDM *)
    PetscErrorCode DMPlexSetMigrationSF( PetscDM, PetscSF )
    PetscErrorCode DMPlexGetMigrationSF( PetscDM, PetscSF*)

cdef CHKERRQ(PetscErrorCode ierr):
    cdef int interr = <int>ierr
    if ierr != 0: raise RuntimeError(f"PETSc error code '{interr}' was encountered.\nhttps://www.mcs.anl.gov/petsc/petsc-current/include/petscerror.h.html")


cdef extern from "petsc.h":
    PetscErrorCode PetscDSAddBoundary( PetscDS, DMBoundaryConditionType, const char[], const char[], PetscInt, PetscInt, const PetscInt *, void (*)(), void (*)(), PetscInt, const PetscInt *, void *)

cdef extern from "petsc.h" nogil:
    PetscErrorCode PetscDSSetResidual( PetscDS, PetscInt, PetscDSResidualFn, PetscDSResidualFn )
    PetscErrorCode PetscDSSetJacobian( PetscDS, PetscInt, PetscInt, PetscDSJacobianFn, PetscDSJacobianFn, PetscDSJacobianFn, PetscDSJacobianFn)
    PetscErrorCode PetscDSSetJacobianPreconditioner( PetscDS, PetscInt, PetscInt, PetscDSJacobianFn, PetscDSJacobianFn, PetscDSJacobianFn, PetscDSJacobianFn)
    PetscErrorCode DMPlexSetSNESLocalFEM( PetscDM, void *, void *, void *)
    PetscErrorCode DMPlexSNESComputeBoundaryFEM( PetscDM, void *, void *)
    PetscErrorCode PetscFEGetQuadrature(PetscFE fem, PetscQuadrature *q)
    PetscErrorCode PetscFESetQuadrature(PetscFE fem, PetscQuadrature q)

from petsc4py import PETSc
    
class Stokes:

    def __init__(self, mesh, u_degree=2, p_degree=None):

        self.mesh = mesh
        self.dm   = mesh.dm.clone()

        if p_degree==None:
            p_degree = u_degree - 1

        # create public velocity/pressure variables
        self._u = uw.mesh.MeshVariable( mesh=mesh, num_components=mesh.dim, name="u", vtype=uw.mesh.VarType.VECTOR, degree=u_degree )
        self._p = uw.mesh.MeshVariable( mesh=mesh, num_components=1,        name="p", vtype=uw.mesh.VarType.SCALAR, degree=p_degree )

        # create private variables
        options = PETSc.Options()
        options.setValue("uprivate_petscspace_degree", u_degree) # for private variables
        self.petsc_fe_u = PETSc.FE().createDefault(self.dm.getDimension(), mesh.dim, mesh.isSimplex, u_degree,"uprivate_", PETSc.COMM_WORLD)
        self.petsc_fe_u_id = self.dm.getNumFields()
        self.dm.setField( self.petsc_fe_u_id, self.petsc_fe_u )
        options.setValue("pprivate_petscspace_degree", p_degree)
        self.petsc_fe_p = PETSc.FE().createDefault(self.dm.getDimension(),        1, mesh.isSimplex, u_degree,"pprivate_", PETSc.COMM_WORLD)
        self.petsc_fe_p_id = self.dm.getNumFields()
        self.dm.setField( self.petsc_fe_p_id, self.petsc_fe_p)

        # Set pressure to use velocity's quadrature object.
        # I'm not sure if this is necessary actually as we 
        # set them to have the same degree quadrature above. 
        cdef PetscQuadrature u_quad
        cdef FE c_fe = self.petsc_fe_u
        ierr = PetscFEGetQuadrature(c_fe.fe, &u_quad); CHKERRQ(ierr)
        # set pressure quad here
        c_fe = self.petsc_fe_p
        ierr = PetscFESetQuadrature(c_fe.fe,u_quad); CHKERRQ(ierr)

        self.viscosity = 1.
        self.bodyforce = (0.,0.)

        self.bcs = []

        # Construct strainrate tensor for future usage.
        # Grab gradients, and let's switch out to sympy.Matrix notation
        # immediately as it is probably cleaner for this.
        N = mesh.N
        grad_u_x = gradient(self.u.fn.dot(N.i)).to_matrix(N)
        grad_u_y = gradient(self.u.fn.dot(N.j)).to_matrix(N)
        grad_u_z = gradient(self.u.fn.dot(N.k)).to_matrix(N)
        grad_u = sympy.Matrix((grad_u_x.T,grad_u_y.T,grad_u_z.T))
        self._strainrate = 1/2 * (grad_u + grad_u.T)[0:mesh.dim,0:mesh.dim].as_immutable()  # needs to be made immuate so it can be hashed later
        
        # this attrib records if we need to re-setup
        self.is_setup = False
        super().__init__()

    @property
    def u(self):
        return self._u
    @property
    def p(self):
        return self._p
    @property
    def strainrate(self):
        return self._strainrate
    @property
    def stress_deviator(self):
        return 2*self.viscosity*self.strainrate
    @property
    def stress(self):
        return self.stress_deviator - sympy.eye(self.mesh.dim)*self.p.fn
    @property
    def div_u(self):
        return divergence(self.u.fn)

    @property
    def viscosity(self):
        return self._viscosity
    @viscosity.setter
    def viscosity(self, value):
        self.is_setup = False
        symval = sympify(value)
        if isinstance(symval, sympy.vector.Vector):
            raise RuntimeError("Viscosity appears to be a vector quantity. Scalars are required.")
        if isinstance(symval, sympy.Matrix):
            raise RuntimeError("Viscosity appears to be a matrix quantity. Scalars are required.")
        self._viscosity = symval

    @property
    def bodyforce(self):
        return self._bodyforce
    @bodyforce.setter
    def bodyforce(self, value):
        self.is_setup = False
        symval = sympify(value)
        # if not isinstance(symval, sympy.vector.Vector):
        #     raise RuntimeError("Body force term must be a vector quantity.")
        self._bodyforce = symval

    
    def add_dirichlet_bc(self, fn, boundaries, components):
        # switch to numpy arrays
        # ndmin arg forces an array to be generated even
        # where comps/indices is a single value.
        self.is_setup = False
        import numpy as np
        components = np.array(components, dtype=np.int32, ndmin=1)
        boundaries = np.array(boundaries, dtype=object,   ndmin=1)
        from collections import namedtuple
        BC = namedtuple('BC', ['components', 'fn', 'boundaries'])
        self.bcs.append(BC(components,sympify(fn),boundaries))

    def _setup_terms(self):
        N = self.mesh.N

        # residual terms
        fns_residual = []
        self._u_f0 = -self.bodyforce
        fns_residual.append(self._u_f0)
        self._u_f1 = self.stress
        fns_residual.append(self._u_f1)
        self._p_f0 = self.div_u
        fns_residual.append(self._p_f0)

        ## jacobian terms
        # needs to be checked!
        dim = self.mesh.dim
        N = self.mesh.N
        fns_jacobian = []

        ### uu terms
        ##  linear part
        # note that g3 is effectively a block
        # but it seems that petsc expects a version
        # that is transposed (in the block sense) relative
        # to what i'd expect. need to ask matt about this. JM.
        g3 = sympy.eye(dim**2)
        for i in range(dim):
            for j in range(i,dim):
                row = i*dim+i
                col = j*dim+j
                g3[row,col] += 1
                if row != col:
                    g3[col,row] += 1
        self._uu_g3 = (g3*self.viscosity).as_immutable()
        # fns_jacobian.append(self._uu_g3) add this guy below

        ## velocity dependant part
        # build derivatives with respect to velocities (v_x,v_y,v_z)
        d_mu_d_u_x = dim*[None,]
        for i in range(dim):
            d_mu_d_u_x[i] = diff_fn1_wrt_fn2(self.viscosity,self.u.fn.dot(N.base_vectors()[i]))
        # now construct required matrix. build submats first. 
        # NOTE: We're flipping the i/k ordering to obtain the blockwise transpose, as possibly 
        #       expected by PETSc
        # TODO: This needs to be checked!
        rows = []
        for k in range(dim):
            row = []
            for i in range(dim):
                # 2 * d_mu_d_u_x_{k} * \dot\eps * e_{i}  
                row.append(2*d_mu_d_u_x[k]*self.strainrate[:,i])
            rows.append(row)
        self._uu_g2 = sympy.Matrix(rows).as_immutable()  # construct full matrix from sub matrices
        fns_jacobian.append(self._uu_g2)

        ## velocity gradient dependant part
        # build derivatives with respect to velocity gradients (u_x_dx,u_x_dy,u_x_dz,u_y_dx,u_y_dy,u_y_dz,u_z_dx,u_z_dy,u_z_dz)
        d_mu_d_u_i_dk = dim*[None,]
        for i in range(dim):
            lst = dim*[None,]
            grad_u_i = gradient(self.u.fn.dot(N.base_vectors()[i]))  # u_i_dx, u_i_dy, u_i_dz
            for j in range(dim):
                lst[j] = diff_fn1_wrt_fn2(self.viscosity,grad_u_i.dot(N.base_vectors()[j]))
            d_mu_d_u_i_dk[i] = sympy.Matrix(lst) # generate Mat for future machination
        # now construct required matrix. build submats first. 
        # NOTE: We're flipping the i/k ordering to obtain the blockwise transpose, as possibly 
        #       expected by PETSc
        # TODO: This needs to be checked!
        rows = []
        for k in range(dim):
            row = []
            for i in range(dim):
                # 2 * \dot\eps * e_{i} * d_mu_d_u_i_k   
                row.append(2*self.strainrate[:,i]*d_mu_d_u_i_dk[k].T)
            rows.append(row)
        self._uu_g3 += sympy.Matrix(rows).as_immutable()  # construct full matrix from sub matrices
        fns_jacobian.append(self._uu_g3)

        ## pressure dependant part
        # get derivative with respect to pressure
        d_mu_d_p = diff_fn1_wrt_fn2(self.viscosity,self.p.fn)
        self._up_g2 = (2*d_mu_d_p*self.strainrate).as_immutable()
        # add linear in pressure part
        self._up_g2 += -sympy.eye(dim).as_immutable()
        fns_jacobian.append(self._up_g2)

        ## pressure gradient dependant part
        # build derivatives with respect to pressure gradients (p_dx, p_dy, p_dz)
        grad_p = gradient(self.p.fn)
        lst = dim*[None,]
        for i in range(dim):
            lst[i] = diff_fn1_wrt_fn2(self.viscosity,grad_p.dot(N.base_vectors()[i]))
        d_mu_d_p_dx = sympy.Matrix(lst) # generate Mat for future machination
        # now construct required matrix. build submats first. 
        # NOTE: We're flipping the i/k ordering to obtain the blockwise transpose, as possibly 
        #       expected by PETSc
        # TODO: This needs to be checked!
        rows = []
        for k in range(dim):
            row = []
            for i in range(dim):
                # 2 * \dot\eps * e_{i} * d_mu_d_p_dx   
                row.append(2*self.strainrate[:,i]*d_mu_d_p_dx.T)
            rows.append(row)
        self._up_g3 = sympy.Matrix(rows).as_immutable()  # construct full matrix from sub matrices
        fns_jacobian.append(self._up_g3) 

        # pu terms
        self._pu_g1 = sympy.eye(dim).as_immutable()
        fns_jacobian.append(self._pu_g1)

        # pp term
        self._pp_g0 = 1/self.viscosity
        fns_jacobian.append(self._pp_g0)

        # generate JIT code.
        # first, we must specify the primary fields.
        # these are fields for which the corresponding sympy functions 
        # should be replaced with the primary (instead of auxiliary) petsc 
        # field value arrays. in this instance, we want to switch out 
        # `self.u` and `self.p` for their primary field 
        # petsc equivalents. without specifying this list, 
        # the aux field equivalents will be used instead, which 
        # will give incorrect results for non-linear problems.
        # note also that the order here is important.
        prim_field_list = [self.u, self.p]
        cdef PtrContainer ext = getext(self.mesh, tuple(fns_residual), tuple(fns_jacobian), [x[1] for x in self.bcs], primary_field_list=prim_field_list)
        # create indexes so that we don't rely on indices that can change
        i_res = {}
        for index,fn in enumerate(fns_residual):
            i_res[fn] = index
        i_jac = {}
        for index,fn in enumerate(fns_jacobian):
            i_jac[fn] = index

        # set functions 
        self.dm.createDS()
        cdef DS ds = self.dm.getDS()
        PetscDSSetResidual(ds.ds, 0, ext.fns_residual[i_res[self._u_f0]], ext.fns_residual[i_res[self._u_f1]])
        PetscDSSetResidual(ds.ds, 1, ext.fns_residual[i_res[self._p_f0]],                                NULL)
        # TODO: check if there's a significant performance overhead in passing in 
        # identically `zero` pointwise functions instead of setting to `NULL`
        PetscDSSetJacobian(              ds.ds, 0, 0,                                 NULL,                                 NULL, ext.fns_jacobian[i_jac[self._uu_g2]], ext.fns_jacobian[i_jac[self._uu_g3]])
        PetscDSSetJacobian(              ds.ds, 0, 1,                                 NULL,                                 NULL, ext.fns_jacobian[i_jac[self._up_g2]], ext.fns_jacobian[i_jac[self._up_g3]])
        PetscDSSetJacobian(              ds.ds, 1, 0,                                 NULL, ext.fns_jacobian[i_jac[self._pu_g1]],                                 NULL,                                 NULL)
        PetscDSSetJacobianPreconditioner(ds.ds, 0, 0,                                 NULL,                                 NULL, ext.fns_jacobian[i_jac[self._uu_g2]], ext.fns_jacobian[i_jac[self._uu_g3]])
        PetscDSSetJacobianPreconditioner(ds.ds, 0, 1,                                 NULL,                                 NULL, ext.fns_jacobian[i_jac[self._up_g2]], ext.fns_jacobian[i_jac[self._up_g3]])
        PetscDSSetJacobianPreconditioner(ds.ds, 1, 0,                                 NULL, ext.fns_jacobian[i_jac[self._pu_g1]],                                 NULL,                                 NULL)
        PetscDSSetJacobianPreconditioner(ds.ds, 1, 1, ext.fns_jacobian[i_jac[self._pp_g0]],                                 NULL,                                 NULL,                                 NULL)

        cdef int ind=1
        cdef int [::1] comps_view  # for numpy memory view
        for index,bc in enumerate(self.bcs):
            comps_view = bc.components
            for boundary in bc.boundaries:
                # use type 5 bc for `DM_BC_ESSENTIAL_FIELD` enum
                PetscDSAddBoundary(ds.ds, 5, NULL, str(boundary).encode('utf8'), 0, comps_view.shape[0], <const PetscInt *> &comps_view[0], <void (*)()>ext.fns_bcs[index], NULL, 1, <const PetscInt *> &ind, NULL)
        self.dm.setUp()

        self.dm.createClosureIndex(None)
        self.snes = PETSc.SNES().create(PETSc.COMM_WORLD)
        self.snes.setDM(self.dm)
        self.snes.setFromOptions()
        cdef DM dm = self.dm
        DMPlexSetSNESLocalFEM(dm.dm, NULL, NULL, NULL)

        self.is_setup = True

    def solve(self, 
              zero_init_guess: bool =True, 
              _force_setup:    bool =False ):
        """
        Generates solution to constructed system.

        Params
        ------
        zero_init_guess:
            If `True`, a zero initial guess will be used for the 
            system solution. Otherwise, the current values of `self.u` 
            and `self.p` will be used.
        """
        if (not self.is_setup) or _force_setup:
            self._setup_terms()

        gvec = self.dm.getGlobalVec()
        lvec = self.dm.getLocalVec()

        if not zero_init_guess:
            with self.mesh.access():
                # TODO: Check if this approach is valid generally. 
                #       Alternative is to grab dm->subdm IS (from `DMCreateSubDM`), 
                #       use that for full_vec->sub_vec mapping (global), followed
                #       with global->local.
                p_len = len(self.p.vec.array)
                lvec.array[p_len:     ] = self.u.vec.array[:]
                lvec.array[     :p_len] = self.p.vec.array[:]
                self.dm.localToGlobal(lvec,gvec)
        else:
            gvec.array[:] = 0.

        # Set all quadratures to velocity quadrature.
        # Note that this needs to be done before the call to 
        # `getInterlacedLocalVariableVec()`, as `createDS` 
        # TODO: Check if we need to unwind these quadratures. 
        #       I believe the PETSc reference counting should 
        #       do the job, but I'm not 100% sure.
        cdef PetscQuadrature u_quad
        cdef FE c_fe = self.petsc_fe_u
        ierr = PetscFEGetQuadrature(c_fe.fe, &u_quad); CHKERRQ(ierr)
        for fe in [var.petsc_fe for var in self.mesh.vars.values()]:
            c_fe = fe
            ierr = PetscFESetQuadrature(c_fe.fe,u_quad); CHKERRQ(ierr)        

        # Call `createDS()` on aux dm. This is necessary after the 
        # quadratures are set above, as it generates the tablatures 
        # from the quadratures (among other things no doubt). 
        # TODO: What does createDS do?
        # TODO: What are the implications of calling this every solve.
        self.mesh.dm.clearDS()
        self.mesh.dm.createDS()

        a_local = self.mesh.getInterlacedLocalVariableVec()
        self.dm.compose("A", a_local)
        # TODO required? - attach the aux dm to the original dm
        self.dm.compose("dmAux", self.mesh.dm)

        # solve
        self.snes.solve(None,gvec)
        self.dm.globalToLocal(gvec,lvec)
        # add back boundaries.. 
        cdef Vec clvec = lvec
        cdef DM dm = self.dm
        DMPlexSNESComputeBoundaryFEM(dm.dm, <void*>clvec.vec, NULL)
        self.mesh.restoreInterlacedLocalVariableVec()

        # This comment relates to previous implementation, but I'll leave it 
        # here for now as something to be aware of / investigate further.
        
        # # create SubDMs now to isolate velocity/pressure variables.
        # # this is currently problematic as the calls to DMCreateSubDM
        # # need to be executed after the solve above, as otherwise the 
        # # results are altered/corrupted. it's unclear to me why this is
        # # the case.  the migrationsf doesn't change anything.
        # # also, the SubDMs and associated vectors are potentially leaking
        # # due to our petsc4py/cython hacks below. 
        with self.mesh.access(self.u,self.p):
            # TODO: Check if this approach is valid generally. 
            #       Alternative is to grab dm->subdm IS (from `DMCreateSubDM`), 
            #       use that for full_vec->sub_vec mapping (global), followed
            #       with global->local.
            p_len = len(self.p.vec.array)
            self.u.vec.array[:] = lvec.array[p_len:     ]
            self.p.vec.array[:] = lvec.array[     :p_len]

        self.dm.restoreLocalVec(lvec)
        self.dm.restoreGlobalVec(gvec)

    def dt(self):
        """
        Calculates an appropriate advective timestep for the given 
        mesh and velocity configuration.
        """
        # we'll want to do this on an element by element basis 
        # for more general mesh

        # first let's extract a max global velocity magnitude
        import math
        with self.mesh.access():
            vel = self.u.data
            magvel_squared = vel[:,0]**2 + vel[:,1]**2
            max_magvel = math.sqrt(magvel_squared.max())
        
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        max_magvel_glob = comm.allreduce( max_magvel, op=MPI.MAX)

        min_dx = self.mesh.min_radius
        return min_dx/max_magvel_glob
