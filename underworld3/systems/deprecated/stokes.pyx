from typing import Optional

import sympy
from sympy import sympify
from sympy.vector import gradient, divergence

from petsc4py import PETSc

import underworld3 
import underworld3 as uw
from .._jitextension import getext, diff_fn1_wrt_fn2
import underworld3.timing as timing

include "../petsc_extras.pxi"

class Stokes:

    instances = 0   # count how many of these there are in order to create unique private mesh variable ids

    @timing.routine_timer_decorator
    def __init__(self, 
                 mesh          : underworld3.mesh.Mesh, 
                 velocityField : Optional[underworld3.mesh.MeshVariable] =None,
                 pressureField : Optional[underworld3.mesh.MeshVariable] =None,
                 u_degree      : Optional[int]                           =2, 
                 p_degree      : Optional[int]                           =None,
                 solver_name   : Optional[str]                           ="stokes_",
                 verbose       : Optional[str]                           =False,
                 penalty       : Optional[float]                         = 0.0,
                 _Ppre_fn      = None
                  ):
        """
        This class provides functionality for a discrete representation
        of the Stokes flow equations.

        Specifically, the class uses a mixed finite element implementation to
        construct a system of linear equations which may then be solved.

        The strong form of the given boundary value problem, for :math:`f`,
        :math:`g` and :math:`h` given, is

        .. math::
            \\begin{align}
            \\sigma_{ij,j} + f_i =& \\: 0  & \\text{ in }  \\Omega \\\\
            u_{k,k} =& \\: 0  & \\text{ in }  \\Omega \\\\
            u_i =& \\: g_i & \\text{ on }  \\Gamma_{g_i} \\\\
            \\sigma_{ij}n_j =& \\: h_i & \\text{ on }  \\Gamma_{h_i} \\\\
            \\end{align}

        where,

        * :math:`\\sigma_{i,j}` is the stress tensor
        * :math:`u_i` is the velocity,
        * :math:`p`   is the pressure,
        * :math:`f_i` is a body force,
        * :math:`g_i` are the velocity boundary conditions (DirichletCondition)
        * :math:`h_i` are the traction boundary conditions (NeumannCondition).

        The problem boundary, :math:`\\Gamma`,
        admits the decompositions :math:`\\Gamma=\\Gamma_{g_i}\\cup\\Gamma_{h_i}` where
        :math:`\\emptyset=\\Gamma_{g_i}\\cap\\Gamma_{h_i}`. The equivalent weak form is:

        .. math::
            \\int_{\Omega} w_{(i,j)} \\sigma_{ij} \\, d \\Omega = \\int_{\\Omega} w_i \\, f_i \\, d\\Omega + \sum_{j=1}^{n_{sd}} \\int_{\\Gamma_{h_j}} w_i \\, h_i \\,  d \\Gamma

        where we must find :math:`u` which satisfies the above for all :math:`w`
        in some variational space.

        Parameters
        ----------
        mesh : 
            The mesh object which forms the basis for problem discretisation,
            domain specification, and parallel decomposition.
        velocityField :
            Optional. Variable used to record system velocity. If not provided,
            it will be generated and will be available via the `u` stokes object property.
        pressureField :
            Optional. Variable used to record system pressure. If not provided,
            it will be generated and will be available via the `p` stokes object property.
            If provided, it is up to the user to ensure that it is of appropriate order
            relative to the provided velocity variable (usually one order lower degree).
        u_degree :
            Optional. The polynomial degree for the velocity field elements.
        p_degree :
            Optional. The polynomial degree for the pressure field elements. 
            If provided, it is up to the user to ensure that it is of appropriate order
            relative to the provided velocitxy variable (usually one order lower degree).
            If not provided, it will be set to one order lower degree than the velocity field.
        solver_name :
            Optional. The petsc options prefix for the SNES solve. This is important to provide
            a name space when multiples solvers are constructed that may have different SNES options.
            For example, if you name the solver "stokes", the SNES options such as `snes_rtol` become `stokes_snes_rtol`.
            The default is blank, and an underscore will be added to the end of the solver name if not already present.
 
        Notes
        -----
        Constructor must be called by collectively all processes.

        """       


        Stokes.instances += 1

        self.mesh = mesh
        self.verbose = verbose

        if (velocityField is None) ^ (pressureField is None):
            raise ValueError("You must provided *both* `pressureField` and `velocityField`, or neither, but not one or the other.")
        
        # I expect the following to break for anyone who wants to name their solver _stokes__ etc etc (LM)

        if solver_name != "" and not solver_name.endswith("_"):
            self.petsc_options_prefix = solver_name+"_"
        else:
            self.petsc_options_prefix = solver_name

        self.petsc_options = PETSc.Options(self.petsc_options_prefix)

        # Here we can set some defaults for this set of KSP / SNES solvers
        # self.petsc_options["snes_type"] = "newtonls"
        self.petsc_options["ksp_rtol"] = 1.0e-3
        self.petsc_options["ksp_monitor"] = None
        # self.petsc_options["ksp_type"] = "fgmres"
        # self.petsc_options["pre_type"] = "gamg"
        self.petsc_options["snes_converged_reason"] = None
        self.petsc_options["snes_monitor_short"] = None
        # self.petsc_options["snes_view"] = None
        self.petsc_options["snes_rtol"] = 1.0e-3
        self.petsc_options["pc_type"] = "fieldsplit"
        self.petsc_options["pc_fieldsplit_type"] = "schur"
        self.petsc_options["pc_fieldsplit_schur_factorization_type"] = "full"
        self.petsc_options["pc_fieldsplit_schur_precondition"] = "a11"
        self.petsc_options["fieldsplit_velocity_ksp_type"] = "fgmres"
        self.petsc_options["fieldsplit_velocity_ksp_rtol"] = 1.0e-4
        self.petsc_options["fieldsplit_velocity_pc_type"]  = "gamg"
        self.petsc_options["fieldsplit_pressure_ksp_rtol"] = 3.e-4
        self.petsc_options["fieldsplit_pressure_pc_type"] = "gamg" 


        if not velocityField:
            if p_degree==None:
                p_degree = u_degree - 1

            # create public velocity/pressure variables
            self._u = uw.mesh.MeshVariable( mesh=mesh, num_components=mesh.dim, name="u", vtype=uw.VarType.VECTOR, degree=u_degree )
            self._p = uw.mesh.MeshVariable( mesh=mesh, num_components=1,        name="p", vtype=uw.VarType.SCALAR, degree=p_degree )
        else:
            self._u = velocityField
            self._p = pressureField

        # Create this dict
        self.fields = {}
        self.fields["pressure"] = self.p
        self.fields["velocity"] = self.u

        # Some other setup 

        self.mesh._equation_systems_register.append(self)

        # Build the DM / FE structures (should be done on remeshing, which is usually handled by the mesh register above)

        self._build_dm_and_mesh_discretisation()
        self._rebuild_after_mesh_update = self._build_dm_and_mesh_discretisation

        self.viscosity = 1.
        self.bodyforce = (0.,0.)
        self._Ppre_fn = _Ppre_fn
        self._penalty =  penalty

        self.bcs = []

        # Construct strainrate tensor for future usage.
        # Grab gradients, and let's switch out to sympy.Matrix notation
        # immediately as it is probably cleaner for this.
        N = mesh.N
        grad_u_x = gradient(self.u.fn.dot(N.i)).to_matrix(N)
        grad_u_y = gradient(self.u.fn.dot(N.j)).to_matrix(N)
        grad_u_z = gradient(self.u.fn.dot(N.k)).to_matrix(N)
        grad_u = sympy.Matrix((grad_u_x.T,grad_u_y.T,grad_u_z.T))
        grad_p = gradient(self.p.fn).to_matrix(N)

        ## sympy.Array 
        self._U = sympy.Array((self._u.fn.to_matrix(self.mesh.N))[0:self.mesh.dim])
        self._P = sympy.Array([self._p.fn])
        self._X = sympy.Array(self.mesh.r)
        self._L = sympy.derive_by_array(self._U, self._X).transpose()
        self._G = sympy.derive_by_array(self._P, self._X)
        self._E = (self._L + self._L.transpose())/2
        self._Einv2 = sympy.sqrt((self._E.tomatrix()**2).trace()) # scalar 2nd invariant

        self._V = sympy.Matrix([self.u.fn.dot(N.i), self.u.fn.dot(N.j), self.u.fn.dot(N.k)])
        self._L1 = grad_u 
        self._G1 = grad_p

        self._strainrate = 1/2 * (grad_u + grad_u.T)[0:mesh.dim,0:mesh.dim].as_immutable()  # needs to be made immutable so it can be hashed later
        self._strainrate_inv2 = sympy.sqrt((self._strainrate**2).trace())

        # this attrib records if we need to re-setup
        self.is_setup = False
        super().__init__()

    def _build_dm_and_mesh_discretisation(self):

        """
        Most of what is in the init phase that is not called by _setup_terms()

        """
        
        mesh = self.mesh
        u_degree = self.u.degree
        p_degree = self.p.degree

        self.dm   = mesh.dm.clone()

        options = PETSc.Options()
        options.setValue("uprivate_petscspace_degree", u_degree) # for private variables
        self.petsc_fe_u = PETSc.FE().createDefault(mesh.dim, mesh.dim, mesh.isSimplex, u_degree,"uprivate_", PETSc.COMM_WORLD)
        self.petsc_fe_u.setName("velocity")
        self.petsc_fe_u_id = self.dm.getNumFields()  ## can we avoid re-numbering ?
        self.dm.setField( self.petsc_fe_u_id, self.petsc_fe_u )

        options.setValue("pprivate_petscspace_degree", p_degree)
        self.petsc_fe_p = PETSc.FE().createDefault(mesh.dim,        1, mesh.isSimplex, u_degree,"pprivate_", PETSc.COMM_WORLD)
        self.petsc_fe_p.setName("pressure")
        self.petsc_fe_p_id = self.dm.getNumFields()
        self.dm.setField( self.petsc_fe_p_id, self.petsc_fe_p)

        # Set pressure to use velocity's quadrature object.
        # I'm not sure if this is necessary actually as we 
        # set them to have the same degree quadrature above. 
        u_quad = self.petsc_fe_u.getQuadrature()
        # set pressure quad here
        self.petsc_fe_p.setQuadrature(u_quad)

        self.is_setup = False

        return

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


    @timing.routine_timer_decorator
    def add_dirichlet_bc(self, fn, boundaries, components):
        # switch to numpy arrays
        # ndmin arg forces an array to be generated even
        # where comps/indices is a single value.
        self.is_setup = False
        import numpy as np
        components = np.array(components, dtype=np.int32, ndmin=1)
        boundaries = np.array(boundaries, dtype=object,   ndmin=1)
        from collections import namedtuple
        BC = namedtuple('BC', ['components', 'fn', 'boundaries', 'type'])
        self.bcs.append(BC(components,sympify(fn),boundaries,'dirichlet'))

    @timing.routine_timer_decorator
    def add_neumann_bc(self, fn, boundaries, components):
        # switch to numpy arrays
        # ndmin arg forces an array to be generated even
        # where comps/indices is a single value.
        self.is_setup = False
        import numpy as np
        components = np.array(components, dtype=np.int32, ndmin=1)
        boundaries = np.array(boundaries, dtype=object,   ndmin=1)
        from collections import namedtuple
        BC = namedtuple('BC', ['components', 'fn', 'boundaries', 'type'])
        self.bcs.append(BC(components,sympify(fn),boundaries,'neumann'))


    @timing.routine_timer_decorator
    def _setup_problem_description(self):

        dim = self.mesh.dim
        N = self.mesh.N

        # residual terms 

        # terms that become part of the weighted integral
        self._u_f0 = -self.bodyforce + self.penalty * (self.div_u)**2 * (N.i +  N.j + N.k)

        # Integration by parts into the stiffness matrix
        self._u_f1 = self.stress # + self._u.fn.dot(self._u.fn) * sympy.eye(dim)  # sympy.eye(dim) * self.penalty * self.div_u

        # forces in the constraint (pressure) equations
        self._p_f0 = self.div_u

        return 

    @timing.routine_timer_decorator
    def _setup_terms(self):
        dim = self.mesh.dim
        N = self.mesh.N

        # residual terms
        self._setup_problem_description()
        fns_residual = [self._u_f0, self._u_f1, self._p_f0]

        ## jacobian terms

        fns_jacobian = []

        # uu terms  
        
        ## J_uu_00 block - G0 which is d f_0_i / d u_j

        g0 = sympy.Matrix.zeros(dim,dim)
        for i in range(dim):
            for j in range(dim):
                # g0[i,j] = diff_fn1_wrt_fn2(self._u_f0.dot(N.base_vectors()[i]),self.u.fn.dot(N.base_vectors()[j]))
                g0[i,j] = sympy.diff(self._u_f0.dot(N.base_vectors()[i]), self._V[j])
                
        self._uu_g0 = g0.as_immutable()
        fns_jacobian.append(self._uu_g0)

        ## J_uu_01 block - G1 which is d f_0_i / d L_kl  (Non linear RHS with gradients)

        g1 = sympy.Matrix.zeros(dim,dim**2)
        for l in range(0,dim):
            for k in range(0,dim):
                for i in range(0,dim):
                    jj = l + 2*k
                    g1[i,jj] = sympy.diff(self._u_f0.dot(N.base_vectors()[i]), self._L1[k,l])
        
        self._uu_g1 = g1.as_immutable()  # construct full matrix from sub matrices
        fns_jacobian.append(self._uu_g1)

        ### This term is zero for stokes / navier-stokes

        ## J_uu_10 block - G2 which is d f_1_ij / d u_k

        g2 = sympy.Matrix.zeros(dim**2,dim)
        for k in range(0,dim):
            for i in range(0,dim):
                for j in range(0,dim):
                    ii = i + 2*j 
                    g2[ii,k] = sympy.diff(self._u_f1[i,j], self._V[k])
 
        self._uu_g2 = g2.as_immutable()  # construct full matrix from sub matrices
        fns_jacobian.append(self._uu_g2)

        ## J_uu_11 block - G2 which is d f_1_ij / d L_kl

        ## velocity gradient dependant part
        # build derivatives with respect to velocity gradients (u_x_dx,u_x_dy,u_x_dz,u_y_dx,u_y_dy,u_y_dz,u_z_dx,u_z_dy,u_z_dz)
        # As long as we express everything in terms of _L terms, then sympy
        # can construct the derivatives wrt _L correctly here

        g3 = sympy.Matrix.zeros(dim**2,dim**2)
        for k in range(0,dim):
            for l in range(0,dim):
                for i in range(0,dim):
                    for j in range(0,dim):
                        ii = i + dim*k 
                        jj = j + dim*l
                        g3[ii,jj] = sympy.diff(self._u_f1[i,j], self._L1[k,l])

        self._uu_g3 = g3.as_immutable()  
        fns_jacobian.append(self._uu_g3)

        # pressure dependant part of velocity block  d f_0_i d_p 
        # ZERO

        # pressure-gradient dependant part of velocity block  d f_0_i d_grad p_j  
        # ZERO

        # pressure-dependent terms in f1  d f_1_ij, dp

        g2 = sympy.Matrix.zeros(dim,dim)
        for i in range(0,dim):
            for j in range(0,dim):
                g2[i,j] = sympy.diff(self._u_f1[i,j], self.p.fn) 

        self._up_g2 = g2.as_immutable()
        fns_jacobian.append(self._up_g2)

        # pressure-gradient dependent terms in f1  d f_1_ij, d grad p_k

        g3 = sympy.Matrix.zeros(dim**2,dim)
        for k in range(0,dim):
            for i in range(0,dim):
                for j in range(0,dim):
                    ii = i + 2*k 
                    g3[ii,j] = sympy.diff(self._u_f1[i,j], self._G1[k])
 
        self._up_g3 = g3.as_immutable()  # construct full matrix from sub matrices
        fns_jacobian.append(self._up_g3)

        # pu terms  
        # THE ONLY RHS term in the pressure is div_u  (i.e. pf_0 = div u)
        # d pf_0 d_Lij = identity matrix

        g1 = sympy.Matrix.zeros(dim,dim)
        for i in range(0,dim):
            for j in range(0,dim):
                g1[i,j] = sympy.diff(self._p_f0, self._L1[i,j])

        self._pu_g1 = g1.as_immutable()
        # OR # self._pu_g1 = sympy.eye(dim).as_immutable()
        fns_jacobian.append(self._pu_g1)

        # pp term
        if self._Ppre_fn is None:
            self._pp_g0 = 1/(self.viscosity)
        else:
            self._pp_g0 = self._Ppre_fn

        fns_jacobian.append(self._pp_g0)


        ## Alternative ... using sympy ARRAY which should generalize well
        ## but has some issues with the change in ordering in petsc v. sympy.
        ## so we will leave both here to compare across a range of problems.

        dim = self.mesh.dim

        F0 = sympy.Array((self._u_f0.to_matrix(self.mesh.N))[0:dim])
        F1 = sympy.Array(self._u_f1)
        FP0 = sympy.Array(self._p_f0).reshape(1)

        G0 = sympy.derive_by_array(F0, self._U)
        G1 = sympy.derive_by_array(F0, self._L)
        G2 = sympy.derive_by_array(F1, self._U)
        G3 = sympy.derive_by_array(F1, self._L)

        # reorganise indices

        # -> ij
        self._uu_G0 = G0.as_immutable()

        # -> i.kl
        self._uu_G1 = sympy.permutedims(G1, (2,1,0)).reshape(dim,dim*dim).as_immutable()

        # -> ij.l
        self._uu_G2 = sympy.permutedims(G2,(2,1,0)).reshape(dim*dim,dim).as_immutable()

        ## -> ij.kl
        self._uu_G3 = sympy.permutedims(G3, (3,1,2,0)).reshape(dim*dim,dim*dim).as_immutable()

        # U/P block (permutations ??)

        self._up_G0 = sympy.derive_by_array(F0, self._P).reshape(dim).as_immutable()
        self._up_G1 = sympy.derive_by_array(F0, self._G).reshape(dim,dim).as_immutable()
        self._up_G2 = sympy.derive_by_array(F1, self._P).reshape(dim,dim).as_immutable()
        self._up_G3 = sympy.derive_by_array(F1, self._G).reshape(dim*dim,dim).as_immutable()


        # P/U block (permutations ??)

        # PU0 = sympy.derive_by_array(FP0, self._U).reshape(dim).as_immutable()
        self._pu_G1 = sympy.derive_by_array(FP0, self._L).reshape(dim,dim).as_immutable()
        # PU2 = sympy.derive_by_array(FP1, self._P).reshape(dim,dim).as_immutable()
        # PU3 = sympy.derive_by_array(FP1, self._G).reshape(dim,dim*2).as_immutable()

        ## PP block is a preconditioner term, not auto-constructed


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
        PetscDSSetJacobian(              ds.ds, 0, 0, ext.fns_jacobian[i_jac[self._uu_g0]], ext.fns_jacobian[i_jac[self._uu_g1]], ext.fns_jacobian[i_jac[self._uu_g2]], ext.fns_jacobian[i_jac[self._uu_g3]])
        PetscDSSetJacobian(              ds.ds, 0, 1,                                 NULL,                                 NULL, ext.fns_jacobian[i_jac[self._up_g2]], ext.fns_jacobian[i_jac[self._up_g3]])
        PetscDSSetJacobian(              ds.ds, 1, 0,                                 NULL, ext.fns_jacobian[i_jac[self._pu_g1]],                                 NULL,                                 NULL)
        PetscDSSetJacobianPreconditioner(ds.ds, 0, 0, ext.fns_jacobian[i_jac[self._uu_g0]], ext.fns_jacobian[i_jac[self._uu_g1]], ext.fns_jacobian[i_jac[self._uu_g2]], ext.fns_jacobian[i_jac[self._uu_g3]])
        PetscDSSetJacobianPreconditioner(ds.ds, 0, 1,                                 NULL,                                 NULL, ext.fns_jacobian[i_jac[self._up_g2]], ext.fns_jacobian[i_jac[self._up_g3]])
        PetscDSSetJacobianPreconditioner(ds.ds, 1, 0,                                 NULL, ext.fns_jacobian[i_jac[self._pu_g1]],                                 NULL,                                 NULL)
        PetscDSSetJacobianPreconditioner(ds.ds, 1, 1, ext.fns_jacobian[i_jac[self._pp_g0]],                                 NULL,                                 NULL,                                 NULL)

        cdef int ind=1
        cdef int [::1] comps_view  # for numpy memory view
        cdef DM cdm = self.dm

        for index,bc in enumerate(self.bcs):
            comps_view = bc.components
            for boundary in bc.boundaries:
                if self.verbose:
                    print("Setting bc {} ({})".format(index, bc.type))
                    print(" - components: {}".format(bc.components))
                    print(" - boundary:   {}".format(bc.boundaries))
                    print(" - fn:         {} ".format(bc.fn))
                # use type 5 bc for `DM_BC_ESSENTIAL_FIELD` enum
                # use type 6 bc for `DM_BC_NATURAL_FIELD` enum  (is this implemented for non-zero values ?)
                if bc.type == 'neumann':
                    bc_type = 6
                else:
                    bc_type = 5

                PetscDSAddBoundary_UW(cdm.dm, bc_type, str(boundary).encode('utf8'), str(boundary).encode('utf8'), 0, comps_view.shape[0], <const PetscInt *> &comps_view[0], <void (*)()>ext.fns_bcs[index], NULL, 1, <const PetscInt *> &ind, NULL)  
        
        self.dm.setUp()

        self.dm.createClosureIndex(None)
        self.snes = PETSc.SNES().create(PETSc.COMM_WORLD)
        self.snes.setDM(self.dm)
        self.snes.setOptionsPrefix(self.petsc_options_prefix)
        self.snes.setFromOptions()
        cdef DM dm = self.dm
        DMPlexSetSNESLocalFEM(dm.dm, NULL, NULL, NULL)

        # Setup subdms here too.
        # These will be used to copy back/forth SNES solutions
        # into user facing variables.
        
        names, isets, dms = self.dm.createFieldDecomposition()
        self._subdict = {}
        for index,name in enumerate(names):
            self._subdict[name] = (isets[index],dms[index])

        self.is_setup = True

    @timing.routine_timer_decorator
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

        if not zero_init_guess:
            with self.mesh.access():
                for name,var in self.fields.items():
                    sgvec = gvec.getSubVector(self._subdict[name][0])  # Get global subvec off solution gvec.
                    sdm   = self._subdict[name][1]                     # Get subdm corresponding to field
                    sdm.localToGlobal(var.vec,sgvec)                   # Copy variable data into gvec
        else:
            gvec.array[:] = 0.

        # Set all quadratures to velocity quadrature.
        quad = self.petsc_fe_u.getQuadrature()
        for fe in [var.petsc_fe for var in self.mesh.vars.values()]:
            fe.setQuadrature(quad)

        # Call `createDS()` on aux dm. This is necessary after the 
        # quadratures are set above, as it generates the tablatures 
        # from the quadratures (among other things no doubt). 
        # TODO: What does createDS do?
        # TODO: What are the implications of calling this every solve.

        self.mesh.dm.clearDS()
        self.mesh.dm.createDS()

        self.mesh.update_lvec()
        cdef DM dm = self.dm
        cdef Vec cmesh_lvec
        # PETSc == 3.16 introduced an explicit interface 
        # for setting the aux-vector which we'll use when available.
        cmesh_lvec = self.mesh.lvec
        ierr = DMSetAuxiliaryVec_UW(dm.dm, NULL, 0, 0, cmesh_lvec.vec); CHKERRQ(ierr)

        # solve
        self.snes.solve(None,gvec)

        cdef Vec clvec
        cdef DM csdm
        # Copy solution back into user facing variables
        with self.mesh.access(self.p,self.u):
            for name,var in self.fields.items():
                sgvec = gvec.getSubVector(self._subdict[name][0])  # Get global subvec off solution gvec.
                sdm   = self._subdict[name][1]                     # Get subdm corresponding to field.
                lvec = sdm.getLocalVec()                           # Get a local vector to push data into.
                sdm.globalToLocal(sgvec,lvec)                      # Do global to local into lvec
                # Put in boundaries values.
                # Note that `DMPlexSNESComputeBoundaryFEM()` seems to need to use an lvec
                # derived from the sub-dm (as opposed to the var.vec local vector), else 
                # failures can occur. 
                clvec = lvec
                csdm = sdm
                ierr = DMPlexSNESComputeBoundaryFEM(csdm.dm, <void*>clvec.vec, NULL); CHKERRQ(ierr)
                # Now copy into the user vec.
                var.vec.array[:] = lvec.array[:]
                sdm.restoreLocalVec(lvec)

        self.dm.restoreGlobalVec(gvec)

    @timing.routine_timer_decorator
    def estimate_dt(self):
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
            if self.mesh.dim ==3:
                magvel_squared += vel[:,2]**2 

            max_magvel = math.sqrt(magvel_squared.max())
        
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        max_magvel_glob = comm.allreduce( max_magvel, op=MPI.MAX)

        min_dx = self.mesh.get_min_radius()
        return min_dx/max_magvel_glob


