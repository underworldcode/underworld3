from typing import Optional

import sympy
from sympy import sympify
from sympy.vector import gradient, divergence
import numpy as np

from petsc4py import PETSc

import underworld3 
import underworld3 as uw
from .._jitextension import getext, diff_fn1_wrt_fn2
import underworld3.timing as timing
import mpi4py

include "../petsc_extras.pxi"

class NavierStokes:

    instances = 0   # count how many of these there are in order to create unique private mesh variable ids

    @timing.routine_timer_decorator
    def __init__(self, 
                 mesh          : underworld3.mesh.MeshClass, 
                 velocityField : Optional[underworld3.mesh.MeshVariable] =None,
                 pressureField : Optional[underworld3.mesh.MeshVariable] =None,
                 u_degree      : Optional[int]                           =2, 
                 p_degree      : Optional[int]                           =None,
                 rho           : Optional[float]                         =0.0,
                 theta         : Optional[float]                         =0.5,
                 solver_name   : Optional[str]                           ="navier_stokes_",
                 verbose       : Optional[str]                           =False,
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
            \\sigma_{ij,j} + f_i =& \\: 0  &  \\text{ in }  \\Omega \\\\
            u_{k,k} =& \\: 0      & \\ text{ in }  \\Omega \\\\
            u_i =& \\: g_i        & \\text{ on }   \\Gamma_{g_i} \\\\
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

        NavierStokes.instances += 1

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
        self.petsc_options.delValue("ksp_monitor")
        self.petsc_options["snes_converged_reason"] = None
        self.petsc_options["snes_monitor_short"] = None
        # self.petsc_options["snes_view"] = None

        # The jacobians are not currently correct for the NS problem, so we will 
        # default to quasi-newton 

        self.petsc_options["snes_type"]="qn"
        self.petsc_options["snes_qn_type"]="lbfgs"
        # self.petsc_options["snes_qn_linesearch_type"]="basic"
        self.petsc_options["snes_qn_monitor"]=None
        self.petsc_options["snes_qn_scale_type"]="scalar"
        self.petsc_options["snes_qn_restart_type"]="periodic"
        self.petsc_options["snes_qn_m"]=25

        self.petsc_options["snes_rtol"] = 1.0e-3
        self.petsc_options["pc_type"] = "fieldsplit"
        self.petsc_options["pc_fieldsplit_type"] = "schur"
        self.petsc_options["pc_fieldsplit_schur_factorization_type"] = "full"
        self.petsc_options["pc_fieldsplit_schur_precondition"] = "a11"
        # self.petsc_options["fieldsplit_velocity_ksp_monitor"] = None
        self.petsc_options["fieldsplit_velocity_ksp_type"] = "fgmres"
        self.petsc_options["fieldsplit_velocity_ksp_rtol"] = 1.0e-4
        self.petsc_options["fieldsplit_velocity_pc_type"]  = "gamg"
        self.petsc_options["fieldsplit_pressure_ksp_rtol"] = 3.0e-4
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

        # Parameters
        self.delta_t = 0.001
        self.theta = theta
        self.rho = rho
        self.penalty = 1.0

        # Build the DM / FE structures (should be done on remeshing, which is usually handled by the mesh register above)

        # this attrib records if we need to re-setup
        self.is_setup = False
        
        self._build_dm_and_mesh_discretisation()
        self._rebuild_after_mesh_update = self._build_dm_and_mesh_discretisation


        # Some other setup 

        self.mesh._equation_systems_register.append(self)

        # Add the nodal point swarm which we'll use to track the characteristics

        # There seems to be an issue with points launched from proc. boundaries
        # and managing the deletion of points, so a small perturbation to the coordinate
        # might fix this.

        # perturbation = self.mesh.get_min_radius() * 1.0e-5 * np.random.random(self._u.coords.shape)

        nswarm = uw.swarm.Swarm(self.mesh)
        nU1 = uw.swarm.SwarmVariable("ns_Ustar_{}".format(self.instances), nswarm, nswarm.dim)
        nP1 = uw.swarm.SwarmVariable("ns_Pstar_{}".format(self.instances), nswarm, 1)
        nX0 = uw.swarm.SwarmVariable("ns_X0_{}".format(self.instances), nswarm, nswarm.dim, _proxy=False)
                 
        nswarm.dm.finalizeFieldRegister()
        nswarm.dm.addNPoints(self._u.coords.shape[0]+1) # why + 1 ? That's the number of spots actually allocated
        cellid = nswarm.dm.getField("DMSwarm_cellid")
        coords = nswarm.dm.getField("DMSwarmPIC_coor").reshape( (-1, nswarm.dim) )
        coords[...] = self._u.coords[...] 
        cellid[:] = self.mesh.get_closest_cells(coords)
        centroid_coords = self.mesh._centroids[cellid]

        # Move slightly within the chosen cell
        shift = 1.0e-2 * self.mesh.get_min_radius()
        coords[...] = (1.0 - shift) * coords[...] + shift * centroid_coords[...]

        nswarm.dm.restoreField("DMSwarmPIC_coor")
        nswarm.dm.restoreField("DMSwarm_cellid")
        nswarm.dm.migrate(remove_sent_points=True)

        self._nswarm  = nswarm
        self._u_star  = nU1
        self._p_star  = nP1
        self._X0      = nX0

        self.viscosity = 1.
        self.bodyforce = (0.,0.)
        self._Ppre_fn = _Ppre_fn

        self.bcs = []

        # Construct strainrate tensor for future usage.
        # Grab gradients, and let's switch out to sympy.Matrix notation
        # immediately as it is probably cleaner for this.
        N = mesh.N
        grad_u_x = gradient(self.u.fn.dot(N.i)).to_matrix(N)
        grad_u_y = gradient(self.u.fn.dot(N.j)).to_matrix(N)
        grad_u_z = gradient(self.u.fn.dot(N.k)).to_matrix(N)
        grad_u = sympy.Matrix((grad_u_x.T,grad_u_y.T,grad_u_z.T))
        self._strainrate = 1/2 * (grad_u + grad_u.T)[0:mesh.dim,0:mesh.dim].as_immutable()  # needs to be made immutable so it can be hashed later
        
        grad_u_star_x = gradient(self.u_star.fn.dot(N.i)).to_matrix(N)
        grad_u_star_y = gradient(self.u_star.fn.dot(N.j)).to_matrix(N)
        grad_u_star_z = gradient(self.u_star.fn.dot(N.k)).to_matrix(N)
        grad_u_star = sympy.Matrix((grad_u_star_x.T,grad_u_star_y.T,grad_u_star_z.T))
        self._strainrate_star = 1/2 * (grad_u_star + grad_u_star.T)[0:mesh.dim,0:mesh.dim].as_immutable()  # needs to be made immutable so it can be hashed later

        self._penalty_term = (sympy.eye(self.mesh.dim)*self.div_u).as_immutable()

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
        cdef PetscQuadrature u_quad
        cdef FE c_fe = self.petsc_fe_u
        ierr = PetscFEGetQuadrature(c_fe.fe, &u_quad); CHKERRQ(ierr)
        # set pressure quad here
        c_fe = self.petsc_fe_p
        ierr = PetscFESetQuadrature(c_fe.fe,u_quad); CHKERRQ(ierr)

        self.is_setup = False

        return

    @property
    def u(self):
        return self._u
    @property
    def u_star(self):
        return self._u_star
    @property
    def p(self):
        return self._p
    @property
    def p_star(self):
        return self._p_star
    @property
    def strainrate(self):
        return self._strainrate
    @property
    def strainrate_star(self):
        return self._strainrate_star
    @property
    def stress_deviator(self):
        return 2*self.viscosity*self.strainrate
    @property
    def stress_deviator_star(self):
        return 2*self.viscosity*self.strainrate_star
    @property
    def stress(self):
        return self.stress_deviator - sympy.eye(self.mesh.dim)*self.p.fn
    @property
    def stress_star(self):
        return self.stress_deviator_star - sympy.eye(self.mesh.dim)*self.p.fn  # keep pressure term simple
    @property
    def div_u(self):
        return divergence(self.u.fn)
    @property
    def div_u_star(self):
        return divergence(self.u_star.fn)
    @property
    def delta_t(self):
        return self._delta_t
    @delta_t.setter
    def delta_t(self, value):
        self.is_setup = False
        self._delta_t = sympify(value)
    @property
    def rho(self):
        return self._rho
    @rho.setter
    def rho(self, value):
        self.is_setup = False
        self._rho = sympify(value)
    @property
    def penalty(self):
        return self._penalty
    @penalty.setter
    def penalty(self, value):
        self.is_setup = False
        self._penalty = sympify(value)
    @property
    def theta(self):
        return self._theta
    @theta.setter
    def theta(self, value):
        self.is_setup = False
        self._theta = sympify(value)
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
    def _setup_terms(self):
        N = self.mesh.N

        # residual terms
        fns_residual = []
        self._u_f0 = -self.bodyforce + self.rho * (self.u.fn - self._u_star.fn) / self.delta_t
        fns_residual.append(self._u_f0)
        self._u_f1 = self.stress * self.theta + self.stress_star * (1.0-self.theta) + self.penalty * self._penalty_term
        fns_residual.append(self._u_f1)
        self._p_f0 = self.div_u * self.theta + self.div_u_star * (1.0-self.theta)
        fns_residual.append(self._p_f0)

        ## jacobian terms
        # needs to be checked!
        dim = self.mesh.dim
        N = self.mesh.N
        fns_jacobian = []

        strain_rate_ave_dt = self.theta*self.strainrate + (1.0-self.theta) * self.strainrate_star
        # strain_rate_ave_dt = self.strainrate

        ### uu terms

        g0 = sympy.eye(dim)
        for i in range(dim):
            for j in range(dim):
                g0[i,j] = diff_fn1_wrt_fn2(self._u_f0.dot(N.base_vectors()[i]),self.u.fn.dot(N.base_vectors()[j]))

        self._uu_g0 = g0.as_immutable()
        fns_jacobian.append(self._uu_g0)

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
                row.append(2*d_mu_d_u_x[k]*strain_rate_ave_dt[:,i])
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
                row.append(2*strain_rate_ave_dt[:,i]*d_mu_d_u_i_dk[k].T)
            rows.append(row)
        self._uu_g3 += sympy.Matrix(rows).as_immutable()  # construct full matrix from sub matrices
        fns_jacobian.append(self._uu_g3)

        ## pressure dependant part
        # get derivative with respect to pressure
        d_mu_d_p = diff_fn1_wrt_fn2(self.viscosity,self.p.fn)
        self._up_g2 = (2 * d_mu_d_p * strain_rate_ave_dt).as_immutable()
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

        # pp term
        if self._Ppre_fn is None:
            self._pp_g0 = 1/(self.viscosity + self.rho / self.delta_t)
        else:
            self._pp_g0 = self._Ppre_fn

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
        PetscDSSetJacobian(              ds.ds, 0, 0, ext.fns_jacobian[i_jac[self._uu_g0]],                                 NULL, ext.fns_jacobian[i_jac[self._uu_g2]], ext.fns_jacobian[i_jac[self._uu_g3]])
        # PetscDSSetJacobian(              ds.ds, 0, 0,                                 NULL,                                 NULL, ext.fns_jacobian[i_jac[self._uu_g2]], ext.fns_jacobian[i_jac[self._uu_g3]])
        PetscDSSetJacobian(              ds.ds, 0, 1,                                 NULL,                                 NULL, ext.fns_jacobian[i_jac[self._up_g2]], ext.fns_jacobian[i_jac[self._up_g3]])
        PetscDSSetJacobian(              ds.ds, 1, 0,                                 NULL, ext.fns_jacobian[i_jac[self._pu_g1]],                                 NULL,                                 NULL)
        PetscDSSetJacobianPreconditioner(ds.ds, 0, 0, ext.fns_jacobian[i_jac[self._uu_g0]],                                 NULL, ext.fns_jacobian[i_jac[self._uu_g2]], ext.fns_jacobian[i_jac[self._uu_g3]])
        PetscDSSetJacobianPreconditioner(ds.ds, 0, 1,                                 NULL,                                 NULL, ext.fns_jacobian[i_jac[self._up_g2]], ext.fns_jacobian[i_jac[self._up_g3]])
        PetscDSSetJacobianPreconditioner(ds.ds, 1, 0,                                 NULL, ext.fns_jacobian[i_jac[self._pu_g1]],                                 NULL,                                 NULL)
        PetscDSSetJacobianPreconditioner(ds.ds, 1, 1, ext.fns_jacobian[i_jac[self._pp_g0]],                                 NULL,                                 NULL,                                 NULL)

        cdef int ind=1
        cdef int [::1] comps_view  # for numpy memory view
        cdef DM cdm = self.dm

        for index,bc in enumerate(self.bcs):
            comps_view = bc.components
            for boundary in bc.boundaries:
                if self.verbose and mpi4py.MPI.COMM_WORLD.rank==0:
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
              timestep       : float = 1.0e-32,
              coords         : np.ndarray = None,
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

        if timestep != self.delta_t:
            self.delta_t = timestep    # this will force an initialisation because the functions need to be updated

        if (not self.is_setup) or _force_setup:
            self._setup_terms()

        # mid pt update scheme by default, but it is possible to supply
        # coords to over-ride this (e.g. rigid body rotation example)

        # placeholder definitions can be removed later
        nswarm = self._nswarm
        u_soln = self._u
        p_soln = self._p
        nX0 = self._X0
        nU1 = self._u_star
        nP1 = self._p_star
        delta_t = timestep

        with nswarm.access(nX0):
            nX0.data[...] = nswarm.data[...]

        with self.mesh.access():
            n_points = u_soln.data.shape[0]


        if coords is None: # Mid point method to find launch points (T*)

        # limitations of the swarm data structures make it difficult to use mid point 
        # integration, so instead we take several small steps ... 

            numerical_dt = self.estimate_dt()

            if delta_t > 0.25 * numerical_dt:
                substeps = int(np.floor(delta_t / (0.25*numerical_dt))) + 1
                if self.verbose and mpi4py.MPI.COMM_WORLD.rank==0:
                    print("substeps = {}".format(substeps))
            else:
                substeps = 1
           
            for substep in range(0, substeps):
                sub_delta_t = timestep / substeps

                with nswarm.access(nswarm.particle_coordinates):
                    v_at_Vpts = uw.function.evaluate(u_soln.fn, nswarm.data).reshape(-1,self.mesh.dim)
                    nswarm.data[...] -= sub_delta_t * v_at_Vpts

                with nswarm.access():
                    if nswarm.data.shape[0] > 1.05 * n_points: #  and self.verbose:
                        print("Swarm points {} = {} ({})".format(mpi4py.MPI.COMM_WORLD.rank,nswarm.data.shape[0], n_points), flush=True)


#               # May not work if points go outside the boundary ... 
#               with nswarm.access(nswarm.particle_coordinates):
#                   v_at_Vpts_half_dt = uw.function.evaluate(u_soln.fn, nswarm.data).reshape(-1,self.mesh.dim)
#                   nswarm.data[...] -= sub_delta_t * (v_at_Vpts_half_dt - 0.5 * v_at_Vpts )
       

        else:  # launch points (T*) provided by omniscience user
            with nswarm.access(nswarm.particle_coordinates):
                nswarm.data[...] = coords[...]

        with nswarm.access(nU1, nP1):
            nU1.data[...] = uw.function.evaluate(u_soln.fn, nswarm.data).reshape(-1,self.mesh.dim)
            nP1.data[...] = uw.function.evaluate(p_soln.fn, nswarm.data).reshape(-1,1)

        # restore coords 
        with nswarm.access(nswarm.particle_coordinates):
            nswarm.data[...] = nX0.data[...]

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

        self.mesh.update_lvec()
        cdef DM dm = self.dm
        cdef Vec cmesh_lvec
        # PETSc == 3.16 introduced an explicit interface 
        # for setting the aux-vector which we'll use when available.
        cmesh_lvec = self.mesh.lvec
        ierr = DMSetAuxiliaryVec(dm.dm, NULL, 0, cmesh_lvec.vec); CHKERRQ(ierr)

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
        max_magvel_glob = comm.allreduce( max_magvel, op=MPI.MAX) + 1.0e-32

        min_dx = self.mesh.get_min_radius()
        return min_dx/max_magvel_glob
