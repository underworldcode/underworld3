import sympy
from sympy import sympify
import numpy as np

from typing import Optional, Callable, Union

import underworld3 as uw
from underworld3 import VarType

import underworld3.timing as timing
from underworld3.utilities._api_tools import uw_object

from petsc4py import PETSc

## We need a pure Eulerian one of these too

# class Eulerian(uw_object):
# etc etc...


class Symbolic(uw_object):
    r"""
    Symbolic History Manager:

    This class manages the update of a variable ψ across timesteps.
    The history operator stores ψ over several timesteps (given by 'order')
    so that it can compute backward differentiation (BDF) or Adams–Moulton expressions.

    The history operator is defined as follows:
    $$\quad \psi_p^{t-n\Delta t} \leftarrow \psi_p^{t-(n-1)\Delta t}\quad$$
    $$\quad \psi_p^{t-(n-1)\Delta t} \leftarrow \psi_p^{t-(n-2)\Delta t} \cdots\quad$$
    $$\quad \psi_p^{t-\Delta t} \leftarrow \psi_p^{t}$$


    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        psi_fn: sympy.Basic,  # a sympy expression for ψ; can be scalar or matrix
        theta: Optional[float] = 0.5,
        varsymbol: Optional[str] = r"\psi",
        verbose: Optional[bool] = False,
        bcs=[],
        order: int = 1,
        smoothing: float = 0.0,
    ):
        super().__init__()
        self.theta = theta
        self.bcs = bcs
        self.verbose = verbose
        self.smoothing = smoothing
        self.order = order

        # Ensure psi_fn is a sympy Matrix.
        if not isinstance(psi_fn, sympy.Matrix):
            try:
                psi_fn = sympy.Matrix(psi_fn)
            except Exception:
                psi_fn = sympy.Matrix([[psi_fn]])
        self._psi_fn = psi_fn  # stored with its native shape
        self._shape = psi_fn.shape  # capture the shape

        # Set the display symbol for psi_fn and for the history variable.
        self._psi_fn_symbol = varsymbol  # e.g. "\psi"
        self._psi_star_symbol = varsymbol + r"^\ast"  # e.g. "\psi^\ast"

        # Create the history list: each element is a Matrix of shape _shape.
        self.psi_star = [sympy.zeros(*self._shape) for _ in range(order)]
        self.initiate_history_fn()
        return

    @property
    def psi_fn(self):
        return self._psi_fn

    @psi_fn.setter
    def psi_fn(self, new_fn):
        if not isinstance(new_fn, sympy.Matrix):
            try:
                new_fn = sympy.Matrix(new_fn)
            except Exception:
                new_fn = sympy.Matrix([[new_fn]])
        # Optionally, one could check for matching shape; here we update both.
        self._psi_fn = new_fn
        self._shape = new_fn.shape
        return

    def _object_viewer(self):
        from IPython.display import Latex, display

        # Display the primary variable
        display(Latex(rf"$\quad {self._psi_fn_symbol} = {sympy.latex(self._psi_fn)}$"))
        # Display the history variable using the different symbol.
        history_latex = ", ".join([sympy.latex(elem) for elem in self.psi_star])
        display(
            Latex(rf"$\quad {self._psi_star_symbol} = \left[{history_latex}\right]$")
        )

    def update_history_fn(self):
        # Update the first history element with a copy of the current ψ.
        self.psi_star[0] = self.psi_fn.copy()

    def initiate_history_fn(self):
        self.update_history_fn()
        # Propagate the initial history to all history steps.
        for i in range(1, self.order):
            self.psi_star[i] = self.psi_star[0].copy()
        return

    def update(
        self,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        self.update_pre_solve(evalf, verbose)
        return

    def update_pre_solve(
        self,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        # Default: no action.
        return

    def update_post_solve(
        self,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        if verbose:
            print(f"Updating history for ψ = {self.psi_fn}", flush=True)

        # Shift history: copy each element down the chain.
        for i in range(self.order - 1, 0, -1):
            self.psi_star[i] = self.psi_star[i - 1].copy()
        self.update_history_fn()
        return

    def bdf(self, order: Optional[int] = None):
        r"""Compute the backward differentiation approximation of the time-derivative of ψ.
        For order 1: bdf ≡ ψ - psi_star[0]
        """
        if order is None:
            order = self.order
        else:
            order = max(1, min(self.order, order))

        with sympy.core.evaluate(False):
            if order == 1:
                bdf0 = self.psi_fn - self.psi_star[0]
            elif order == 2:
                bdf0 = 3 * self.psi_fn / 2 - 2 * self.psi_star[0] + self.psi_star[1] / 2
            elif order == 3:
                bdf0 = (
                    11 * self.psi_fn / 6
                    - 3 * self.psi_star[0]
                    + (3 * self.psi_star[1]) / 2
                    - self.psi_star[2] / 3
                )
        return bdf0

    def adams_moulton_flux(self, order: Optional[int] = None):
        if order is None:
            order = self.order
        else:
            order = max(1, min(self.order, order))

        with sympy.core.evaluate(False):
            if order == 1:
                am = self.theta * self.psi_fn + (1.0 - self.theta) * self.psi_star[0]
            elif order == 2:
                am = (5 * self.psi_fn + 8 * self.psi_star[0] - self.psi_star[1]) / 12
            elif order == 3:
                am = (
                    9 * self.psi_fn
                    + 19 * self.psi_star[0]
                    - 5 * self.psi_star[1]
                    + self.psi_star[2]
                ) / 24
        return am


class Eulerian(uw_object):
    r"""Eulerian  (mesh based) History Manager:
    This manages the update of a variable, $\psi$ on the mesh across timesteps.
    $$\quad \psi_p^{t-n\Delta t} \leftarrow \psi_p^{t-(n-1)\Delta t}\quad$$
    $$\quad \psi_p^{t-(n-1)\Delta t} \leftarrow \psi_p^{t-(n-2)\Delta t} \cdots\quad$$
    $$\quad \psi_p^{t-\Delta t} \leftarrow \psi_p^{t}$$
    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        psi_fn: Union[
            uw.discretisation.MeshVariable, sympy.Basic
        ],  # sympy function or mesh variable
        vtype: uw.VarType,
        degree: int,
        continuous: bool,
        evalf: Optional[bool] = False,
        theta: Optional[float] = 0.5,
        varsymbol: Optional[str] = r"u",
        verbose: Optional[bool] = False,
        bcs=[],
        order=1,
        smoothing=0.0,
    ):
        super().__init__()

        self.mesh = mesh
        self.theta = theta
        self.bcs = bcs
        self.verbose = verbose
        self.degree = degree
        self.vtype = vtype
        self.continuous = continuous
        self.smoothing = smoothing
        self.evalf = evalf

        # meshVariables are required for:
        #
        # u(t) - evaluation of u_fn at the current time
        # u*(t) - u_* evaluated from

        # psi is evaluated/stored at `order` timesteps. We can't
        # be sure if psi is a meshVariable or a function to be evaluated
        # psi_star is reaching back through each evaluation and has to be a
        # meshVariable (storage)

        if isinstance(psi_fn, uw.discretisation._MeshVariable):
            self._psi_fn = psi_fn.sym  ### get symbolic form of the meshvariable
            self._psi_meshVar = psi_fn
        else:
            self._psi_fn = psi_fn  ### already in symbolic form
            self._psi_meshVar = None

        self.order = order

        psi_star = []
        self.psi_star = psi_star

        for i in range(order):
            self.psi_star.append(
                uw.discretisation.MeshVariable(
                    f"psi_star_Eulerian_{self.instance_number}_{i}",
                    self.mesh,
                    vtype=vtype,
                    degree=degree,
                    continuous=continuous,
                    varsymbol=rf"{varsymbol}^{{ {'*'*(i+1)} }}",
                )
            )

        # print('initiating history fn', flush=True)
        ### Initiate first history value in chain
        self.initiate_history_fn()

        return

    @property
    def psi_fn(self):
        return self._psi_fn

    @psi_fn.setter
    def psi_fn(self, new_fn):
        self._psi_fn = new_fn
        # self._psi_star_projection_solver.uw_function = self.psi_fn
        return

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        ## feedback on this instance
        # display(Latex(r"$\quad\psi = $ " + self.psi._repr_latex_()))
        # display(
        #     Latex(
        #         r"$\quad\Delta t_{\textrm{phys}} = $ "
        #         + sympy.sympify(self.dt_physical)._repr_latex_()
        #     )
        # )
        display(Latex(rf"$\quad$History steps = {self.order}"))

    def _setup_projections(self):
        ### using this to store terms that can't be evaluated (e.g. derivatives)
        # The projection operator for mapping derivative values to the mesh - needs to be different for each variable type, unfortunately ...
        if self.vtype == uw.VarType.SCALAR:
            self._psi_star_projection_solver = uw.systems.solvers.SNES_Projection(
                self.mesh, self.psi_star[0], verbose=False
            )
        elif self.vtype == uw.VarType.VECTOR:
            self._psi_star_projection_solver = (
                uw.systems.solvers.SNES_Vector_Projection(
                    self.mesh, self.psi_star[0], verbose=False
                )
            )
        elif self.vtype == uw.VarType.SYM_TENSOR or self.vtype == uw.VarType.TENSOR:
            self._WorkVar = uw.discretisation.MeshVariable(
                f"W_star_Eulerian_{self.instance_number}",
                self.mesh,
                vtype=uw.VarType.SCALAR,
                degree=self.degree,
                continuous=self.continuous,
                varsymbol=r"W^{*}",
            )
            self._psi_star_projection_solver = (
                uw.systems.solvers.SNES_Tensor_Projection(
                    self.mesh, self.psi_star[0], self._WorkVar, verbose=False
                )
            )

        self._psi_star_projection_solver.uw_function = self.psi_fn
        self._psi_star_projection_solver.bcs = self.bcs
        self._psi_star_projection_solver.smoothing = self.smoothing

    def update_history_fn(self):
        ### update first value in history chain
        ### avoids projecting if function can be evaluated
        try:
            with self.mesh.access(self.psi_star[0]):
                try:
                    with self.mesh.access(self._psi_meshVar):
                        self.psi_star[0].data[...] = self._psi_meshVar.data[...]
                    # print('copying data', flush=True)
                except:
                    # if self.evalf:
                    #     self.psi_star[0].data[...] = uw.function.evalf(
                    #         self.psi_fn, self.psi_star[0].coords
                    #     ).reshape(-1, max(self.psi_fn.shape))
                    #     # print('evalf data', flush=True)
                    # else:
                    self.psi_star[0].data[...] = uw.function.evaluate(
                        self.psi_fn,
                        self.psi_star[0].coords,
                        evalf=self.evalf,
                    ).reshape(-1, max(self.psi_fn.shape))
                    # print('evaluate data', flush=True)

        except:
            self._setup_projections()
            self._psi_star_projection_solver.solve()
            # print('projecting data', flush=True)

    def initiate_history_fn(self):
        self.update_history_fn()

        ### set up all history terms to the initial values
        for i in range(self.order - 1, 0, -1):
            with self.mesh.access(self.psi_star[i]):
                self.psi_star[i].data[...] = self.psi_star[0].data[...]

        return

    def update(
        self,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        self.update_pre_solve(evalf, verbose)
        return

    def update_pre_solve(
        self,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        return

    def update_post_solve(
        self,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        # if average_over_dt:
        #     phi = min(1.0, dt / self.dt_physical)
        # else:
        #     phi = 1.0

        if verbose and uw.mpi.rank == 0:
            print(f"Update {self.psi_fn}", flush=True)

        ### copy values down the chain
        for i in range(self.order - 1, 0, -1):
            with self.mesh.access(self.psi_star[i]):
                self.psi_star[i].data[...] = self.psi_star[i - 1].data[...]

        ### update the history fn
        self.update_history_fn()
        # ### update the first value in the chain
        # if self.evaluateable:
        #     with self.mesh.access(self.psi_star[0]):
        #         try:
        #             with self.mesh.access(self._psi_meshVar):
        #                 self.psi_star[0].data[...] = self._psi_meshVar.data[...]
        #         except:
        #             if self.evalf:
        #                 self.psi_star[0].data[...] = uw.function.evalf(self.psi_fn, self.psi_star[0].coords).reshape(-1, max(self.psi_fn.shape))
        #             else:
        #                 self.psi_star[0].data[...] = uw.function.evaluate(self.psi_fn, self.psi_star[0].coords).reshape(-1, max(self.psi_fn.shape))
        # else:
        #     self._psi_star_projection_solver.uw_function = self.psi_fn
        #     self._psi_star_projection_solver.solve()

        return

    def bdf(self, order=None):
        r"""Backwards differentiation form for calculating DuDt
        Note that you will need `bdf` / $\delta t$ in computing derivatives"""

        if order is None:
            order = self.order
        else:
            order = max(1, min(self.order, order))

        with sympy.core.evaluate(False):
            if order == 1:
                bdf0 = self.psi_fn - self.psi_star[0].sym

            elif order == 2:
                bdf0 = (
                    3 * self.psi_fn / 2
                    - 2 * self.psi_star[0].sym
                    + self.psi_star[1].sym / 2
                )

            elif order == 3:
                bdf0 = (
                    11 * self.psi_fn / 6
                    - 3 * self.psi_star[0].sym
                    + 3 * self.psi_star[1].sym / 2
                    - self.psi_star[2].sym / 3
                )

        return bdf0

    def adams_moulton_flux(self, order=None):
        if order is None:
            order = self.order
        else:
            order = max(1, min(self.order, order))

        with sympy.core.evaluate(False):
            if order == 1:
                am = (self.psi_fn + self.psi_star[0].sym) / 2
                # am = self.theta*self.psi_fn + ((1.-self.theta)*self.psi_star[0].sym)

            elif order == 2:
                am = (
                    5 * self.psi_fn + 8 * self.psi_star[0].sym - self.psi_star[1].sym
                ) / 12

            elif order == 3:
                am = (
                    9 * self.psi_fn
                    + 19 * self.psi_star[0].sym
                    - 5 * self.psi_star[1].sym
                    + self.psi_star[2].sym
                ) / 24

        return am


class SemiLagrangian(uw_object):
    r"""
    # Nodal-Swarm  Semi-Lagrangian History Manager:

    This manages the semi-Lagrangian update of a Mesh Variable, $\psi$, on the mesh across timesteps.
    $$\quad \psi_p^{t-n\Delta t} \leftarrow \psi_p^{t-(n-1)\Delta t}\quad$$
    $$\quad \psi_p^{t-(n-1)\Delta t} \leftarrow \psi_p^{t-(n-2)\Delta t} \cdots\quad$$
    $$\quad \psi_p^{t-\Delta t} \leftarrow \psi_p^{t}$$
    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        psi_fn: sympy.Function,
        V_fn: sympy.Function,
        vtype: uw.VarType,
        degree: int,
        continuous: bool,
        swarm_degree: Optional[int] = None,
        swarm_continuous: Optional[bool] = None,
        varsymbol: Optional[str] = None,
        verbose: Optional[bool] = False,
        bcs=[],
        order=1,
        smoothing=0.0,
        preserve_moments=False,
    ):
        super().__init__()

        self.mesh = mesh
        self.bcs = bcs
        self.verbose = verbose
        self.degree = degree
        self.continuous = continuous
        self._psi_fn = psi_fn
        self.V_fn = V_fn
        self.order = order
        self.preserve_moments = preserve_moments

        if swarm_degree is None:
            self.swarm_degree = degree
        else:
            self.swarm_degree = swarm_degree

        if swarm_continuous is None:
            self.swarm_continuous = continuous
        else:
            self.swarm_continuous = swarm_continuous

        if varsymbol is None:
            varsymbol = rf"u_{{ [{self.instance_number}] }}"

        # meshVariables are required for:
        #
        # u(t) - evaluation of u_fn at the current time
        # u*(t) - u_* evaluated from

        # psi is evaluated/stored at `order` timesteps. We can't
        # be sure if psi is a meshVariable or a function to be evaluated
        # but psi_star is reaching back through each evaluation and has to be a
        # meshVariable (storage)

        psi_star = []
        self.psi_star = psi_star

        for i in range(order):
            self.psi_star.append(
                uw.discretisation.MeshVariable(
                    f"psi_star_sl_{self.instance_number}_{i}",
                    self.mesh,
                    vtype=vtype,
                    degree=self.degree,
                    continuous=self.continuous,
                    varsymbol=rf"{{ {varsymbol}^{{ {'*'*(i+1)} }} }}",
                )
            )

        # Working variable that has a potentially different discretisation from psi_star
        # We project from this to psi_star and we use this variable to define the
        # advection sample points

        self._workVar = uw.discretisation.MeshVariable(
            f"W_{self.instance_number}_{i}",
            self.mesh,
            vtype=vtype,
            degree=self.swarm_degree,
            continuous=self.swarm_continuous,
            varsymbol=rf"{{ {varsymbol}^\nabla }}",
        )

        # We just need one swarm since this is inherently a sequential operation
        nswarm = uw.swarm.NodalPointUWSwarm(self._workVar, verbose)
        self._nswarm_psi = nswarm

        # The projection operator for mapping swarm values to the mesh - needs to be different for
        # each variable type, unfortunately ...

        if vtype == uw.VarType.SCALAR:
            self._psi_star_projection_solver = uw.systems.solvers.SNES_Projection(
                self.mesh, self.psi_star[0], verbose=False
            )
        elif vtype == uw.VarType.VECTOR:
            self._psi_star_projection_solver = (
                uw.systems.solvers.SNES_Vector_Projection(
                    self.mesh,
                    self.psi_star[0],
                    verbose=False,
                )
            )

        elif vtype == uw.VarType.SYM_TENSOR or vtype == uw.VarType.TENSOR:
            self._WorkVarTP = uw.discretisation.MeshVariable(
                f"W_star_slcn_{self.instance_number}",
                self.mesh,
                vtype=uw.VarType.SCALAR,
                degree=degree,
                continuous=continuous,
                varsymbol=r"W^{*}",
            )
            self._psi_star_projection_solver = (
                uw.systems.solvers.SNES_Tensor_Projection(
                    self.mesh, self.psi_star[0], self._WorkVarTP, verbose=False
                )
            )

        # We should find a way to add natural bcs here
        # (self.Unknowns.u carried as a symbol from solver to solver)

        self._psi_star_projection_solver.uw_function = self._workVar.sym
        self._psi_star_projection_solver.bcs = bcs
        self._psi_star_projection_solver.smoothing = smoothing

        self._smoothing = smoothing

        self.I = uw.maths.Integral(mesh, None)

        return

    @property
    def psi_fn(self):
        return self._psi_fn

    @psi_fn.setter
    def psi_fn(self, new_fn):
        self._psi_fn = new_fn
        self._psi_star_projection_solver.uw_function = self._psi_fn
        return

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        display(Latex(rf"$\quad$History steps = {self.order}"))

    def update(
        self,
        dt: float,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
        dt_physical: Optional = None,
    ):
        self.update_pre_solve(dt, evalf, verbose, dt_physical)
        return

    def update_post_solve(
        self,
        dt: float,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
        dt_physical: Optional[float] = None,
    ):
        return

    def update_pre_solve(
        self,
        dt: float,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
        dt_physical: Optional[float] = None,
    ):

        ## Progress from the oldest part of the history
        # 1. Copy the stored values down the chain

        if dt_physical is not None:
            phi = min(1, dt / dt_physical)
        else:
            phi = sympy.sympify(1)

        for i in range(self.order - 1, 0, -1):
            with self.mesh.access(self.psi_star[i]):
                self.psi_star[i].data[...] = (
                    phi * self.psi_star[i - 1].data[...]
                    + (1 - phi) * self.psi_star[i].data[...]
                )

        # 2. Compute the upstream values from the psi_fn

        # We use the u_star variable as a working value here so we have to work backwards
        # so we don't over-write the history terms

        for i in range(self.order - 1, -1, -1):
            with self._nswarm_psi.access(self._nswarm_psi._X0):
                self._nswarm_psi._X0.data[...] = self._nswarm_psi.data[...]

            # march nodes backwards along characteristics
            self._nswarm_psi.advection(
                self.V_fn,
                -dt,
                order=1,
                corrector=False,
                restore_points_to_domain_func=self.mesh.return_coords_to_bounds,
                evalf=evalf,
                step_limit=False,
                #! substepping: this seems to be too diffusive if left on.
                #! Check the code carefully !
            )

            if i == 0:
                # Recalculate psi_star from psi_fn. If psi_fn containts
                # derivatives, the evaluation will fail and a projection
                # is required instead.

                try:
                    with self.mesh.access(self.psi_star[0]):
                        self.psi_star[0].data[...] = uw.function.evaluate(
                            self.psi_fn,
                            self.psi_star[0].coords,
                            evalf=evalf,
                        )
                except:
                    self._psi_star_projection_solver.uw_function = self.psi_fn
                    self._psi_star_projection_solver.smoothing = 0.0
                    self._psi_star_projection_solver.solve(verbose=verbose)

            # if evalf:
            #     with self._nswarm_psi.access(self._nswarm_psi.swarmVariable):
            #         for d in range(self.psi_star[i].shape[1]):
            #             self._nswarm_psi.swarmVariable.data[:, d] = uw.function.evalf(
            #                 self.psi_star[i].sym[d], self._nswarm_psi.data
            #             )
            # else:
            #

            with self._nswarm_psi.access(self._nswarm_psi.swarmVariable):
                for d in range(self.psi_star[i].shape[1]):
                    self._nswarm_psi.swarmVariable.data[:, d] = uw.function.evaluate(
                        self.psi_star[i].sym[d],
                        self._nswarm_psi.data,
                        evalf=evalf,
                    )

            if self.preserve_moments and self._workVar.num_components == 1:

                self.I.fn = self.psi_star[i].sym[0]
                Imean0 = self.I.evaluate()

                self.I.fn = (self.psi_star[i].sym[0] - Imean0) ** 2
                IL20 = np.sqrt(self.I.evaluate())

                # if uw.mpi.rank == 0:
                #     print(f"Pre advection:  {Imean0}, {IL20}", flush=True)

            # restore coords (will call dm.migrate after context manager releases)
            # We need some modifications to dm.migrate to snapback
            # to original location without substepping

            og_mig_type = uw.function.dm_swarm_get_migrate_type(
                self._nswarm_psi
            )  # get original migrate type
            uw.function.dm_swarm_set_migrate_type(
                self._nswarm_psi, PETSc.DMSwarm.MigrateType.MIGRATE_BASIC
            )

            # change the rank in DMSwarm_rank with the rank before advection
            nR0_field_name = self._nswarm_psi._nR0.name
            nI0_field_name = self._nswarm_psi._nI0.name

            orig_ranks = self._nswarm_psi.dm.getField(nR0_field_name)
            node_ranks = self._nswarm_psi.dm.getField("DMSwarm_rank")

            node_ranks[...] = orig_ranks[...]

            self._nswarm_psi.dm.restoreField(nR0_field_name)
            self._nswarm_psi.dm.restoreField("DMSwarm_rank")

            # will update DMSwarm_cellid, DMSwarmPIC_cooor, etc and call migrate

            with self._nswarm_psi.access(self._nswarm_psi.particle_coordinates):
                self._nswarm_psi.data[...] = self._nswarm_psi._nX0.data[...]

            # reset to original migrate type
            uw.function.dm_swarm_set_migrate_type(self._nswarm_psi, og_mig_type)

            # Push data from swarm back to _workVar.data.
            # Note: particles are removed when sent and added to the
            # end of the swarm when received, so we need to re-order
            # the data when we put it back onto the nodes

            with self._nswarm_psi.access():
                orig_index = self._nswarm_psi._nI0.data.copy().reshape(-1)

                with self.mesh.access(self._workVar):
                    self._workVar.data[orig_index, :] = (
                        self._nswarm_psi.swarmVariable.data[:, :]
                    )

            # Project / Copy from advected swarm to semi-Lagrangian variables.

            if self._workVar.coords.shape == self.psi_star[i].coords.shape:
                with self.mesh.access(self.psi_star[i]):
                    self.psi_star[i].data[...] = self._workVar.data[...]
            else:
                self._psi_star_projection_solver.uw_function = self._workVar.sym
                self._psi_star_projection_solver.smoothing = 0.0
                self._psi_star_projection_solver.solve()

            # Copy data from the projection operator if i!=0
            if i != 0:
                with self.mesh.access(self.psi_star[i]):
                    self.psi_star[i].data[...] = self.psi_star[0].data[...]

            # Optional: Conserve moments for scalar fields
            # (could extend this to other field types but not
            #  sure if this is wanted / warranted at all )

            if self.preserve_moments and self._workVar.num_components == 1:

                self.I.fn = self.psi_star[i].sym[0]
                Imean = self.I.evaluate()

                self.I.fn = (self.psi_star[i].sym[0] - Imean) ** 2
                IL2 = np.sqrt(self.I.evaluate())

                with self.mesh.access(self.psi_star[i]):
                    self.psi_star[i].data[...] += Imean0 - Imean

                self.I.fn = (self.psi_star[i].sym[0] - Imean0) ** 2
                IL2 = np.sqrt(self.I.evaluate())

                with self.mesh.access(self.psi_star[i]):
                    self.psi_star[i].data[...] = (
                        self.psi_star[i].data[...] - Imean0
                    ) * IL20 / IL2 + Imean0

                # self.I.fn = self.psi_star[i].sym[0]
                # Imean = self.I.evaluate()

                # self.I.fn = (self.psi_star[0].sym[0] - Imean) ** 2
                # IL2 = np.sqrt(self.I.evaluate())

                # if uw.mpi.rank == 0:
                #     print(f"Post advection: {Imean}, {IL2}", flush=True)

        return

    def bdf(self, order=None):
        r"""Backwards differentiation form for calculating DuDt
        Note that you will need `bdf` / $\delta t$ in computing derivatives"""

        if order is None:
            order = self.order
        else:
            order = max(1, min(self.order, order))

        with sympy.core.evaluate(True):
            if order == 1:
                bdf0 = self.psi_fn - self.psi_star[0].sym

            elif order == 2:
                bdf0 = (
                    3 * self.psi_fn / 2
                    - 2 * self.psi_star[0].sym
                    + self.psi_star[1].sym / 2
                )

            elif order == 3:
                bdf0 = (
                    11 * self.psi_fn / 6
                    - 3 * self.psi_star[0].sym
                    + 3 * self.psi_star[1].sym / 2
                    - self.psi_star[2].sym / 3
                )

        return bdf0

    def adams_moulton_flux(self, order=None):
        if order is None:
            order = self.order
        else:
            order = max(0, min(self.order, order))

        with sympy.core.evaluate(True):

            if order == 0:
                am = self.psi_fn

            elif order == 1:
                am = (self.psi_fn + self.psi_star[0].sym) / 2

            elif order == 2:
                am = (
                    5 * self.psi_fn + 8 * self.psi_star[0].sym - self.psi_star[1].sym
                ) / 12

            elif order == 3:
                am = (
                    9 * self.psi_fn
                    + 19 * self.psi_star[0].sym
                    - 5 * self.psi_star[1].sym
                    + self.psi_star[2].sym
                ) / 24

        return am


## Consider Deprecating this one - it is the same as the Lagrangian_Swarm but
## sets up the swarm for itself. This does not have a practical use-case - the swarm version
## is slower, more cumbersome, and less stable / accurate. The only reason to use
## it is if there is an existing swarm that we can re-purpose.


class Lagrangian(uw_object):
    r"""Swarm-based Lagrangian History Manager:

    This manages the update of a Lagrangian variable, $\psi$ on the swarm across timesteps.

    $\quad \psi_p^{t-n\Delta t} \leftarrow \psi_p^{t-(n-1)\Delta t}\quad$

    $\quad \psi_p^{t-(n-1)\Delta t} \leftarrow \psi_p^{t-(n-2)\Delta t} \cdots\quad$

    $\quad \psi_p^{t-\Delta t} \leftarrow \psi_p^{t}$
    """

    instances = 0  # count how many of these there are in order to create unique private mesh variable ids

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        psi_fn: sympy.Function,
        V_fn: sympy.Function,
        vtype: uw.VarType,
        degree: int,
        continuous: bool,
        varsymbol: Optional[str] = r"u",
        verbose: Optional[bool] = False,
        bcs=[],
        order=1,
        smoothing=0.0,
        fill_param=3,
    ):
        super().__init__()

        # create a new swarm to manage here
        dudt_swarm = uw.swarm.UWSwarm(mesh)

        self.mesh = mesh
        self.swarm = dudt_swarm
        self.psi_fn = psi_fn
        self.V_fn = V_fn
        self.verbose = verbose
        self.order = order

        psi_star = []
        self.psi_star = psi_star

        for i in range(order):
            print(f"Creating psi_star[{i}]")
            self.psi_star.append(
                uw.swarm.SwarmVariable(
                    f"psi_star_sw_{self.instance_number}_{i}",
                    self.swarm,
                    vtype=vtype,
                    proxy_degree=degree,
                    proxy_continuous=continuous,
                    varsymbol=rf"{varsymbol}^{{ {'*'*(i+1)} }}",
                )
            )

        dudt_swarm.populate(fill_param)

        return

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        ## feedback on this instance
        display(Latex(r"$\quad\psi = $ " + self.psi._repr_latex_()))
        display(
            Latex(
                r"$\quad\Delta t_{\textrm{phys}} = $ "
                + sympy.sympify(self.dt_physical)._repr_latex_()
            )
        )
        display(Latex(rf"$\quad$History steps = {self.order}"))

    ## Note: We may be able to eliminate this
    ## The SL updater and the Lag updater have
    ## different sequencing because of the way they
    ## update the history. It makes more sense for the
    ## full Lagrangian swarm to be updated after the solve
    ## and this means we have to grab the history values first.

    def update(
        self,
        dt: float,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        self.update_post_solve(dt, evalf, verbose)
        return

    def update_pre_solve(
        self,
        dt: float,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        return

    def update_post_solve(
        self,
        dt: float,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        for h in range(self.order - 1):
            i = self.order - (h + 1)

            # copy the information down the chain
            print(f"Lagrange order = {self.order}")
            print(f"Lagrange copying {i-1} to {i}")

            with self.swarm.access(self.psi_star[i]):
                self.psi_star[i].data[...] = self.psi_star[i - 1].data[...]

        # Now update the swarm variable

        # if evalf:
        #     psi_star_0 = self.psi_star[0]
        #     with self.swarm.access(psi_star_0):
        #         for i in range(psi_star_0.shape[0]):
        #             for j in range(psi_star_0.shape[1]):
        #                 updated_psi = uw.function.evalf(
        #                     self.psi_fn[i, j], self.swarm.data
        #                 )
        #                 psi_star_0[i, j].data[:] = updated_psi

        # else:
        psi_star_0 = self.psi_star[0]
        with self.swarm.access(psi_star_0):
            for i in range(psi_star_0.shape[0]):
                for j in range(psi_star_0.shape[1]):
                    updated_psi = uw.function.evaluate(
                        self.psi_fn[i, j],
                        self.swarm.data,
                        evalf=evalf,
                    )
                    psi_star_0[i, j].data[:] = updated_psi

        # Now update the swarm locations

        self.swarm.advection(
            self.V_fn,
            delta_t=dt,
            restore_points_to_domain_func=self.mesh.return_coords_to_bounds,
        )

    def bdf(self, order=None):
        r"""Backwards differentiation form for calculating DuDt
        Note that you will need `bdf` / $\delta t$ in computing derivatives"""

        if order is None:
            order = self.order

        with sympy.core.evaluate(True):
            if order == 0:  # special case - no history term (catch )
                bdf0 = sympy.simpify[0]

            if order == 1:
                bdf0 = self.psi_fn - self.psi_star[0].sym

            elif order == 2:
                bdf0 = (
                    3 * self.psi_fn / 2
                    - 2 * self.psi_star[0].sym
                    + self.psi_star[1].sym / 2
                )

            elif order == 3:
                bdf0 = (
                    11 * self.psi_fn / 6
                    - 3 * self.psi_star[0].sym
                    + 3 * self.psi_star[1].sym / 2
                    - self.psi_star[2].sym / 3
                )

        return bdf0

    def adams_moulton_flux(self, order=None):
        if order is None:
            order = self.order

        with sympy.core.evaluate(True):
            if order == 0:  # Special case - no history term
                am = self.psi_fn

            elif order == 1:
                am = (self.psi_fn + self.psi_star[0].sym) / 2

            elif order == 2:
                am = (
                    5 * self.psi_fn + 8 * self.psi_star[0].sym - self.psi_star[1].sym
                ) / 12

            elif order == 3:
                am = (
                    9 * self.psi_fn
                    + 19 * self.psi_star[0].sym
                    - 5 * self.psi_star[1].sym
                    + self.psi_star[2].sym
                ) / 24

        return am


class Lagrangian_Swarm(uw_object):
    r"""Swarm-based Lagrangian History Manager:
    This manages the update of a Lagrangian variable, $\psi$ on the swarm across timesteps.

    $\quad \psi_p^{t-n\Delta t} \leftarrow \psi_p^{t-(n-1)\Delta t}\quad$

    $\quad \psi_p^{t-(n-1)\Delta t} \leftarrow \psi_p^{t-(n-2)\Delta t} \cdots\quad$

    $\quad \psi_p^{t-\Delta t} \leftarrow \psi_p^{t}$
    """

    instances = 0  # count how many of these there are in order to create unique private mesh variable ids

    @timing.routine_timer_decorator
    def __init__(
        self,
        swarm: uw.swarm.Swarm,
        psi_fn: sympy.Function,
        vtype: uw.VarType,
        degree: int,
        continuous: bool,
        varsymbol: Optional[str] = r"u",
        verbose: Optional[bool] = False,
        bcs=[],
        order=1,
        smoothing=0.0,
        step_averaging=2,
    ):
        super().__init__()

        self.mesh = swarm.mesh
        self.swarm = swarm
        self.psi_fn = psi_fn
        self.verbose = verbose
        self.order = order
        self.step_averaging = step_averaging

        psi_star = []
        self.psi_star = psi_star

        for i in range(order):
            print(f"Creating psi_star[{i}]")
            self.psi_star.append(
                uw.swarm.SwarmVariable(
                    f"psi_star_sw_{self.instance_number}_{i}",
                    self.swarm,
                    vtype=vtype,
                    proxy_degree=degree,
                    proxy_continuous=continuous,
                    varsymbol=rf"{varsymbol}^{{ {'*'*(i+1)} }}",
                )
            )

        return

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        ## feedback on this instance
        display(Latex(r"$\quad\psi = $ " + self.psi._repr_latex_()))
        display(
            Latex(
                r"$\quad\Delta t_{\textrm{phys}} = $ "
                + sympy.sympify(self.dt_physical)._repr_latex_()
            )
        )
        display(Latex(rf"$\quad$History steps = {self.order}"))

    ## Note: We may be able to eliminate this
    ## The SL updater and the Lag updater have
    ## different sequencing because of the way they
    ## update the history. It makes more sense for the
    ## full Lagrangian swarm to be updated after the solve
    ## and this means we have to grab the history values first.

    def update(
        self,
        dt: float,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        self.update_post_solve(dt, evalf, verbose)
        return

    def update_pre_solve(
        self,
        dt: float,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        return

    def update_post_solve(
        self,
        dt: float,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        for h in range(self.order - 1):
            i = self.order - (h + 1)

            # copy the information down the chain
            if verbose:
                print(f"Lagrange swarm order = {self.order}", flush=True)
                print(
                    f"Mesh interpolant order = {self.psi_star[0]._meshVar.degree}",
                    flush=True,
                )
                print(f"Lagrange swarm copying {i-1} to {i}", flush=True)

            with self.swarm.access(self.psi_star[i]):
                self.psi_star[i].data[...] = self.psi_star[i - 1].data[...]

        phi = 1 / self.step_averaging

        # Now update the swarm variable
        # if evalf:
        #     psi_star_0 = self.psi_star[0]
        #     with self.swarm.access(psi_star_0):
        #         for i in range(psi_star_0.shape[0]):
        #             for j in range(psi_star_0.shape[1]):
        #                 updated_psi = uw.function.evalf(
        #                     self.psi_fn[i, j], self.swarm.data
        #                 )
        #                 psi_star_0[i, j].data[:] = (
        #                     phi * updated_psi + (1 - phi) * psi_star_0[i, j].data[:]
        #                 )
        # else:
        #
        psi_star_0 = self.psi_star[0]
        with self.swarm.access(psi_star_0):
            for i in range(psi_star_0.shape[0]):
                for j in range(psi_star_0.shape[1]):
                    updated_psi = uw.function.evaluate(
                        self.psi_fn[i, j],
                        self.swarm.data,
                        evalf=evalf,
                    )
                    psi_star_0[i, j].data[:] = (
                        phi * updated_psi + (1 - phi) * psi_star_0[i, j].data[:]
                    )

        return

    def bdf(self, order=None):
        r"""Backwards differentiation form for calculating DuDt
        Note that you will need `bdf` / $\delta t$ in computing derivatives"""

        if order is None:
            order = self.order
        else:
            order = max(1, min(self.order, order))

        with sympy.core.evaluate(False):
            if order <= 1:
                bdf0 = self.psi_fn - self.psi_star[0].sym

            elif order == 2:
                bdf0 = (
                    3 * self.psi_fn / 2
                    - 2 * self.psi_star[0].sym
                    + self.psi_star[1].sym / 2
                )

            elif order == 3:
                bdf0 = (
                    11 * self.psi_fn / 6
                    - 3 * self.psi_star[0].sym
                    + 3 * self.psi_star[1].sym / 2
                    - self.psi_star[2].sym / 3
                )

            bdf0 /= self.step_averaging

        # This is actually calculated over several steps so scaling is required
        return bdf0

    def adams_moulton_flux(self, order=None):
        if order is None:
            order = self.order
        else:
            order = max(1, min(self.order, order))

        with sympy.core.evaluate(False):
            if order == 1:
                am = (self.psi_fn + self.psi_star[0].sym) / 2

            elif order == 2:
                am = (
                    5 * self.psi_fn + 8 * self.psi_star[0].sym - self.psi_star[1].sym
                ) / 12

            elif order == 3:
                am = (
                    9 * self.psi_fn
                    + 19 * self.psi_star[0].sym
                    - 5 * self.psi_star[1].sym
                    + self.psi_star[2].sym
                ) / 24

        return am
