r"""Streamline-upwind Petrov-Galerkin scalar transport."""

from typing import Optional
import math
from dataclasses import dataclass

import numpy as np
import sympy
from petsc4py import PETSc

import underworld3 as uw
import underworld3.timing as timing
from underworld3.function import expression
from underworld3.systems.ddt import Eulerian as Eulerian_DDt
from underworld3.systems.solvers import SNES_Diffusion, _centroid_velocities_nd
from underworld3.checkpoint.state import SnapshottableState


@dataclass
class AdvDiffusionSUPGState(SnapshottableState):
    """Snapshot metadata not carried by SUPG mesh variables."""

    time_integrator: str = "bdf"
    rate_initialised: bool = False


class SNES_AdvectionDiffusionSUPG(SNES_Diffusion):
    r"""Implicit scalar advection-diffusion with SUPG stabilization.

    The scalar residual is

    .. math::

       R = \frac{\mathrm{BDF}(T)}{\Delta t}
           + \mathbf{u}\cdot\nabla T - f,

    with pointwise residual terms

    .. math::

       F_0 = R, \qquad
       \mathbf{F}_1 = \boldsymbol{\kappa}\nabla T
                       + \tau\mathbf{u}R.

    The second contribution to :math:`\mathbf{F}_1` produces
    :math:`\tau(\mathbf{u}\cdot\nabla w)R` in the weak form. Diffusion
    remains standard Galerkin.

    Parameters
    ----------
    mesh : Mesh
        Computational mesh.
    u_Field : MeshVariable
        Scalar field being transported.
    V_fn : MeshVariable or sympy Matrix
        Prescribed advection velocity. It is frozen during each solve.
    order : int, default=1
        Eulerian BDF history order.
    theta : float, default=1.0
        Flux integration parameter. Only fully implicit fluxes are currently
        supported.
    tau : scalar expression, optional
        User-provided stabilization parameter. When omitted, a transient
        isotropic parameter is computed from a cell-constant streamline
        length, local velocity, diffusivity, and timestep.
    tau_model : {"generic", "citcoms"}, optional
        Automatic stabilization model. ``generic`` uses the optimal 1-D
        coth(Pe)-1/Pe relation with a transient scale. ``citcoms`` uses the
        clipped steady relation on simplex streamline lengths. This option
        does not change the implicit BDF time integrator.
    time_integrator : {"bdf", "citcoms"}, default="bdf"
        ``bdf`` uses the implicit Eulerian BDF solver. ``citcoms`` uses a
        positive P1 row-sum mass, gamma=0.5 predictor, and two fixed residual
        corrections. The latter is restricted to continuous P1 fields.
    temperature_rate_field : MeshVariable, optional
        Stored temperature derivative for the CitcomS predictor-corrector.
        Supplying this field gives production checkpoint files a stable name.
    DuDt, DFDt : optional
        Existing history operators. A supplied ``DuDt`` must be Eulerian and
        must not contain a velocity, because advection is represented in R.

    Notes
    -----
    Automatic tau currently supports volume simplex meshes and scalar
    isotropic diffusivity. Supply ``tau`` explicitly for other meshes or
    constitutive models. The default ``bdf`` integrator is implicit. The
    ``citcoms`` integrator provides the continuous-P1 row-lumped
    predictor-corrector used by CitcomS-style mantle-convection benchmarks.
    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        u_Field: uw.discretisation.MeshVariable,
        V_fn,
        order: int = 1,
        theta: float = 1.0,
        tau=None,
        tau_model: Optional[str] = None,
        time_integrator: str = "bdf",
        adv_gamma: float = 0.5,
        corrector_steps: int = 2,
        temperature_rate_field: Optional[uw.discretisation.MeshVariable] = None,
        evalf: Optional[bool] = False,
        verbose: bool = False,
        DuDt: Optional[Eulerian_DDt] = None,
        DFDt=None,
    ):
        if float(theta) != 1.0:
            raise ValueError("AdvDiffusionSUPG currently requires theta=1.0.")
        if time_integrator not in ("bdf", "citcoms"):
            raise ValueError("time_integrator must be 'bdf' or 'citcoms'.")
        if tau_model is None:
            tau_model = "citcoms" if time_integrator == "citcoms" else "generic"
        if tau_model not in ("generic", "citcoms"):
            raise ValueError("tau_model must be 'generic' or 'citcoms'.")
        if DuDt is not None and not isinstance(DuDt, Eulerian_DDt):
            raise TypeError("AdvDiffusionSUPG requires an Eulerian DuDt operator.")
        if DuDt is not None and DuDt.V_fn is not None:
            raise ValueError(
                "DuDt.V_fn must be None; AdvDiffusionSUPG includes advection "
                "in its strong residual."
            )
        if time_integrator == "citcoms":
            if u_Field.degree != 1 or not u_Field.continuous:
                raise ValueError(
                    "The CitcomS predictor-corrector requires continuous P1 "
                    "temperature."
                )
            if DuDt is not None or DFDt is not None:
                raise ValueError(
                    "The CitcomS predictor-corrector manages its own derivative "
                    "state; do not supply DuDt or DFDt."
                )
            if not 0.0 < float(adv_gamma) <= 1.0:
                raise ValueError("adv_gamma must be in (0, 1].")
            if int(corrector_steps) < 1:
                raise ValueError("corrector_steps must be positive.")

        super().__init__(
            mesh,
            u_Field,
            order=order,
            theta=theta,
            evalf=evalf,
            verbose=verbose,
            DuDt=DuDt,
            DFDt=DFDt,
        )

        self.V_fn = V_fn
        self.tau_model = tau_model
        self.time_integrator = time_integrator
        self.adv_gamma = float(adv_gamma)
        self.corrector_steps = int(corrector_steps)
        self._automatic_tau = tau is None
        self._supg_h = None
        self._supg_tau = None
        self._temperature_rate = None
        self._lumped_mass = None
        self._lumped_mass_mesh_version = None
        self._citcoms_work_vectors = None
        self._citcoms_work_mesh_version = None
        self._diffusion_dt_cache = None
        self._rate_initialised = False

        if self.time_integrator == "citcoms":
            if temperature_rate_field is not None:
                if (
                    temperature_rate_field.mesh is not mesh
                    or temperature_rate_field.degree != 1
                    or not temperature_rate_field.continuous
                    or temperature_rate_field.num_components != 1
                ):
                    raise ValueError(
                        "temperature_rate_field must be a continuous scalar P1 "
                        "variable on the solver mesh."
                    )
                self._temperature_rate = temperature_rate_field
            else:
                self._temperature_rate = uw.discretisation.MeshVariable(
                    f"_supg_dTdt_{self.instance_number}",
                    mesh,
                    1,
                    degree=1,
                    continuous=True,
                )

        uw.get_default_model()._register_state_bearer(self)

        if self._automatic_tau:
            suffix = self.instance_number
            self._supg_h = uw.discretisation.MeshVariable(
                f"_supg_h_{suffix}", mesh, 1, degree=0, continuous=False
            )
            self._supg_tau = uw.discretisation.MeshVariable(
                f"_supg_tau_{suffix}", mesh, 1, degree=0, continuous=False
            )
            self._tau = self._supg_tau.sym[0]
        else:
            self._tau = sympy.sympify(tau)

    @property
    def V_fn(self):
        """Advection velocity expression."""
        return self._V_fn

    @V_fn.setter
    def V_fn(self, value):
        self.is_setup = False
        self._V_fn = (
            value.sym if isinstance(value, uw.discretisation.MeshVariable) else value
        )

    @property
    def tau(self):
        """SUPG stabilization parameter used in the residual."""
        return self._tau

    @property
    def temperature_rate(self):
        """Stored derivative used by the CitcomS predictor-corrector."""
        return self._temperature_rate

    @property
    def state(self):
        """Return predictor-corrector initialization metadata."""
        return AdvDiffusionSUPGState(
            time_integrator=self.time_integrator,
            rate_initialised=self._rate_initialised,
        )

    @state.setter
    def state(self, state):
        if not isinstance(state, AdvDiffusionSUPGState):
            raise TypeError("AdvDiffusionSUPG state has the wrong type.")
        if state.time_integrator != self.time_integrator:
            raise ValueError("AdvDiffusionSUPG time integrator changed since snapshot.")
        self._rate_initialised = bool(state.rate_initialised)

    def _strong_transport_residual(self):
        gradient = self.mesh.vector.gradient(self.u.sym)
        advection = sympy.Matrix((self.V_fn.dot(gradient),))
        if self.time_integrator == "citcoms":
            time_derivative = self._temperature_rate.sym
        else:
            time_derivative = self.DuDt.bdf() / self.delta_t
        return time_derivative + advection - self.f

    def _simplex_data(self):
        """Return local simplex connectivity, basis gradients, and volumes."""
        from underworld3.meshing.smoothing import _tet_cells, _tri_cells

        cells = (
            _tri_cells(self.mesh.dm)
            if self.mesh.dim == 2
            else _tet_cells(self.mesh.dm) if self.mesh.dim == 3 else None
        )
        if cells is None or self.mesh.dim != self.mesh.cdim:
            raise NotImplementedError(
                "Automatic SUPG operations require a 2-D or 3-D volume " "simplex mesh."
            )

        coords = np.asarray(self.mesh.X.coords)
        cell_coords = coords[cells]
        edges = cell_coords[:, 1:, :] - cell_coords[:, :1, :]
        try:
            inverse_edges = np.linalg.inv(edges)
        except np.linalg.LinAlgError as error:
            raise RuntimeError("Cannot operate on a singular simplex.") from error

        gradients = np.empty_like(cell_coords)
        gradients[:, 1:, :] = np.transpose(inverse_edges, (0, 2, 1))
        gradients[:, 0, :] = -gradients[:, 1:, :].sum(axis=1)
        volumes = np.abs(np.linalg.det(edges)) / math.factorial(self.mesh.dim)
        return cells, gradients, volumes

    def _cell_diffusivity(self, cell_count):
        """Evaluate non-negative scalar diffusivity at cell centroids."""
        diffusivity_expr = sympy.sympify(self.constitutive_model.K)
        if isinstance(diffusivity_expr, sympy.MatrixBase):
            raise NotImplementedError(
                "Automatic SUPG operations require scalar isotropic "
                "diffusivity; supply tau explicitly for tensor diffusivity."
            )
        diffusivity = uw.function.evaluate(diffusivity_expr, self.mesh._centroids)
        if hasattr(diffusivity, "units") and diffusivity.units is not None:
            diffusivity = uw.non_dimensionalise(diffusivity)
        elif hasattr(diffusivity, "magnitude"):
            diffusivity = diffusivity.magnitude
        diffusivity = np.asarray(diffusivity, dtype=float).reshape(-1)
        if diffusivity.size == 1:
            diffusivity = np.full(cell_count, diffusivity.item())
        if diffusivity.shape != (cell_count,):
            raise ValueError("Diffusivity must evaluate to one scalar per cell.")
        if np.any(diffusivity < 0.0):
            raise ValueError("SUPG diffusivity must be non-negative.")
        return diffusivity

    @property
    def F0(self):
        """Transient-advection-source strong residual."""
        value = expression(
            r"f_0^{SUPG}",
            self._strong_transport_residual(),
            "SUPG transient-advection-source residual",
            _unique_name_generation=True,
        )
        self._f0 = value
        return value

    @property
    def F1(self):
        """Galerkin diffusion flux plus streamline stabilization flux."""
        residual = self._strong_transport_residual()[0]
        diffusion_flux = (
            self.constitutive_model.flux.T
            if self.time_integrator == "citcoms"
            else self.DFDt.adams_moulton_flux()
        )
        value = expression(
            r"\mathbf{F}_1^{SUPG}",
            diffusion_flux + self.tau * self.V_fn * residual,
            "Diffusive and SUPG streamline flux",
            _unique_name_generation=True,
        )
        self._f1 = value
        return value

    def _update_automatic_tau(self):
        """Update local simplex streamline lengths and automatic tau values."""
        if not self._automatic_tau:
            return
        if self.constitutive_model is None:
            raise RuntimeError(
                "Set constitutive_model before solving AdvDiffusionSUPG."
            )

        _, gradients, _ = self._simplex_data()

        velocity = _centroid_velocities_nd(self.V_fn, self.mesh)
        speed = np.linalg.norm(velocity, axis=1)
        directional_rate = np.abs(np.einsum("cad,cd->ca", gradients, velocity)).sum(
            axis=1
        )
        h_stream = np.divide(
            2.0 * speed,
            directional_rate,
            out=np.zeros_like(speed),
            where=directional_rate > 0.0,
        )

        diffusivity = self._cell_diffusivity(speed.size)

        tau_steady = np.zeros_like(speed)
        moving = speed > np.finfo(float).eps
        diffusive = moving & (diffusivity > 0.0)
        nondiffusive = moving & ~diffusive

        if np.any(diffusive):
            pe = speed[diffusive] * h_stream[diffusive] / (2.0 * diffusivity[diffusive])
            xi = np.empty_like(pe)
            small = np.abs(pe) < 1.0e-3
            pe_small = pe[small]
            xi[small] = pe_small / 3.0 - pe_small**3 / 45.0 + 2.0 * pe_small**5 / 945.0
            xi[~small] = 1.0 / np.tanh(pe[~small]) - 1.0 / pe[~small]
            if self.tau_model == "generic":
                tau_steady[diffusive] = (
                    h_stream[diffusive] * xi / (2.0 * speed[diffusive])
                )
            else:
                tau_steady[diffusive] = (
                    h_stream[diffusive]
                    * np.maximum(0.0, 1.0 - 1.0 / pe)
                    / (2.0 * speed[diffusive])
                )
        tau_steady[nondiffusive] = h_stream[nondiffusive] / (2.0 * speed[nondiffusive])

        if self.tau_model == "generic":
            dt = float(self.delta_t.data)
            if dt <= 0.0:
                raise ValueError("AdvDiffusionSUPG requires a positive timestep.")
            tau_values = np.divide(
                1.0,
                np.sqrt(
                    (2.0 / dt) ** 2
                    + np.divide(
                        1.0,
                        tau_steady**2,
                        out=np.full_like(tau_steady, np.inf),
                        where=tau_steady > 0.0,
                    )
                ),
                out=np.zeros_like(tau_steady),
                where=tau_steady > 0.0,
            )
        else:
            tau_values = tau_steady

        if self._supg_h.data.shape[0] != h_stream.size:
            raise RuntimeError("SUPG P0 field and local simplex counts do not match.")
        self._supg_h.data[:, 0] = h_stream
        self._supg_tau.data[:, 0] = tau_values

    def _setup_citcoms_residual(self, verbose=False):
        """Build the reusable residual assembler for predictor-corrector steps."""
        if not self.constitutive_model._solver_is_setup:
            self._needs_function_rewire = True
        self._build(verbose, False, None)
        self.is_setup = True
        self.constitutive_model._solver_is_setup = True

    def _assemble_lumped_mass(self):
        """Assemble positive P1 simplex row-sum masses on free global DOFs."""
        mesh_version = getattr(self.mesh, "_mesh_version", 0)
        if (
            self._lumped_mass is not None
            and self._lumped_mass_mesh_version == mesh_version
        ):
            return self._lumped_mass
        if self._lumped_mass is not None:
            self._lumped_mass.destroy()
            self._lumped_mass = None

        from underworld3.meshing.smoothing import _owned_cell_mask

        cells, _, volumes = self._simplex_data()
        owned = _owned_cell_mask(self.mesh.dm)

        local_mass = self.dm.createLocalVector()
        global_mass = self.dm.createGlobalVector()
        local_mass.set(0.0)
        global_mass.set(0.0)
        section = self.dm.getLocalSection()
        vertex_start, _ = self.mesh.dm.getDepthStratum(0)

        for cell_index in np.flatnonzero(owned):
            contribution = volumes[cell_index] / (self.mesh.dim + 1)
            for vertex_index in cells[cell_index]:
                offset = section.getOffset(vertex_start + int(vertex_index))
                if offset >= 0:
                    local_mass.array[offset] += contribution

        self.dm.localToGlobal(
            local_mass,
            global_mass,
            addv=PETSc.InsertMode.ADD_VALUES,
        )
        local_mass.destroy()
        if global_mass.getLocalSize() and np.any(global_mass.array <= 0.0):
            global_mass.destroy()
            raise RuntimeError("CitcomS P1 lumped mass contains non-positive rows.")

        self._lumped_mass = global_mass
        self._lumped_mass_mesh_version = mesh_version
        return self._lumped_mass

    def _citcoms_vectors(self):
        """Return reusable global vectors for predictor-corrector updates."""
        mesh_version = getattr(self.mesh, "_mesh_version", 0)
        if (
            self._citcoms_work_vectors is not None
            and self._citcoms_work_mesh_version == mesh_version
        ):
            return self._citcoms_work_vectors

        if self._citcoms_work_vectors is not None:
            for vector in self._citcoms_work_vectors:
                vector.destroy()

        solution = self.dm.createGlobalVector()
        residual = solution.duplicate()
        delta_rate = solution.duplicate()
        rate = solution.duplicate()
        self._citcoms_work_vectors = (solution, residual, delta_rate, rate)
        self._citcoms_work_mesh_version = mesh_version
        return self._citcoms_work_vectors

    @timing.routine_timer_decorator
    def estimate_dt(self):
        """Estimate a simplex advection-diffusion timestep.

        The CitcomS-compatible predictor-corrector uses
        ``0.9 * min(1/max(lambda_adv), 2/max(rowsum(abs(M_L^-1 K))))``.
        The same conservative value is also available for the implicit BDF
        path as a resolution-accuracy estimate.
        """
        from mpi4py import MPI
        from underworld3.meshing.smoothing import _owned_cell_mask

        cells, gradients, volumes = self._simplex_data()
        velocity = _centroid_velocities_nd(self.V_fn, self.mesh)
        directional_rate = np.abs(np.einsum("cad,cd->ca", gradients, velocity)).sum(
            axis=1
        )
        local_adv_rate = (
            float(np.max(directional_rate)) if directional_rate.size else 0.0
        )
        adv_rate = uw.mpi.comm.allreduce(local_adv_rate, op=MPI.MAX)
        dt_adv = 1.0 / adv_rate if adv_rate > 0.0 else np.inf

        diffusivity = self._cell_diffusivity(len(cells))
        has_diffusivity = bool(
            uw.mpi.comm.allreduce(
                int(np.any(diffusivity > 0.0)),
                op=MPI.MAX,
            )
        )
        if not has_diffusivity:
            dt_diff = np.inf
        elif self.time_integrator != "citcoms":
            local_diff_rate = (
                float(
                    np.max(
                        2.0
                        * self.mesh.dim
                        * diffusivity
                        / np.maximum(self.mesh._radii**2, np.finfo(float).tiny)
                    )
                )
                if diffusivity.size
                else 0.0
            )
            diff_rate = uw.mpi.comm.allreduce(local_diff_rate, op=MPI.MAX)
            dt_diff = 2.0 / diff_rate
        else:
            self._setup_citcoms_residual()
            mass = self._assemble_lumped_mass()
            diffusion_signature = (
                getattr(self.mesh, "_mesh_version", 0),
                hash(diffusivity.tobytes()),
            )
            local_cache_valid = (
                self._diffusion_dt_cache is not None
                and self._diffusion_dt_cache[0] == diffusion_signature
            )
            cache_valid = bool(
                uw.mpi.comm.allreduce(int(local_cache_valid), op=MPI.MIN)
            )
            if cache_valid:
                dt_diff = self._diffusion_dt_cache[1]
                self.dt_adv = dt_adv
                self.dt_diff = dt_diff
                return 0.9 * min(dt_adv, dt_diff)

            stiffness = self.dm.createMatrix()
            stiffness.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR, False)
            section = self.dm.getLocalSection()
            vertex_start, _ = self.mesh.dm.getDepthStratum(0)
            owned = _owned_cell_mask(self.mesh.dm)

            for cell_index in np.flatnonzero(owned):
                points = [vertex_start + int(index) for index in cells[cell_index]]
                local_dofs = [section.getOffset(point) for point in points]
                element_stiffness = (
                    diffusivity[cell_index]
                    * volumes[cell_index]
                    * gradients[cell_index].dot(gradients[cell_index].T)
                )
                stiffness.setValuesLocal(
                    local_dofs,
                    local_dofs,
                    element_stiffness,
                    addv=PETSc.InsertMode.ADD_VALUES,
                )
            stiffness.assemble()

            row_start, row_end = stiffness.getOwnershipRange()
            local_diff_rate = 0.0
            for row in range(row_start, row_end):
                _, values = stiffness.getRow(row)
                row_sum = float(np.sum(np.abs(values)))
                local_diff_rate = max(
                    local_diff_rate,
                    row_sum / mass.array[row - row_start],
                )
            diff_rate = uw.mpi.comm.allreduce(local_diff_rate, op=MPI.MAX)
            stiffness.destroy()
            dt_diff = 2.0 / diff_rate if diff_rate > 0.0 else np.inf
            self._diffusion_dt_cache = (diffusion_signature, dt_diff)

        self.dt_adv = dt_adv
        self.dt_diff = dt_diff
        return 0.9 * min(dt_adv, dt_diff)

    def _compute_citcoms_residual(self, solution=None, residual=None):
        """Assemble the residual at the current temperature and rate."""
        if solution is None:
            solution = self.dm.createGlobalVector()
        if residual is None:
            residual = solution.duplicate()
        solution.set(0.0)
        self.dm.localToGlobal(self.u.vec, solution, addv=False)
        residual.set(0.0)
        self.mesh.update_lvec()
        self.dm.setAuxiliaryVec(self.mesh.lvec, None)
        self._update_constants()
        self.snes.computeFunction(solution, residual)
        return solution, residual

    def _solve_citcoms(self, timestep, verbose=False):
        """Advance one CitcomS-compatible predictor-corrector timestep."""
        if timestep is None:
            timestep = float(self.delta_t.data)
        self.delta_t = timestep
        dt = float(self.delta_t.data)
        if dt <= 0.0:
            raise ValueError("AdvDiffusionSUPG requires a positive timestep.")

        self._update_automatic_tau()
        self._setup_citcoms_residual(verbose)
        mass = self._assemble_lumped_mass()
        temperature_global, residual, delta_rate, rate_global = self._citcoms_vectors()

        if not self._rate_initialised:
            self._temperature_rate.data[:, 0] = 0.0
            self._compute_citcoms_residual(temperature_global, residual)
            delta_rate.pointwiseDivide(residual, mass)
            delta_rate.scale(-1.0)
            self._temperature_rate.vec.set(0.0)
            self.dm.globalToLocal(delta_rate, self._temperature_rate.vec)
            self.mesh._stale_lvec = True
            self._rate_initialised = True

        self.u.data[:, 0] += (
            (1.0 - self.adv_gamma) * dt * self._temperature_rate.data[:, 0]
        )
        self._temperature_rate.data[:, 0] = 0.0
        self.mesh._stale_lvec = True

        from underworld3.cython.petsc_discretisation import (
            petsc_dm_insert_boundary_values,
        )

        for _ in range(self.corrector_steps):
            self._compute_citcoms_residual(temperature_global, residual)
            delta_rate.pointwiseDivide(residual, mass)
            delta_rate.scale(-1.0)

            rate_global.set(0.0)
            self.dm.localToGlobal(self._temperature_rate.vec, rate_global, addv=False)
            rate_global.axpy(1.0, delta_rate)
            temperature_global.axpy(self.adv_gamma * dt, delta_rate)

            self._temperature_rate.vec.set(0.0)
            self.u.vec.set(0.0)
            self.dm.globalToLocal(rate_global, self._temperature_rate.vec)
            self.dm.globalToLocal(temperature_global, self.u.vec)
            petsc_dm_insert_boundary_values(self.dm, self.u.vec)
            self.mesh._stale_lvec = True

        self.is_setup = True
        self.constitutive_model._solver_is_setup = True
        return

    @timing.routine_timer_decorator
    def solve(
        self,
        zero_init_guess: bool = None,
        timestep: float = None,
        evalf: bool = False,
        _force_setup: bool = False,
        verbose: bool = False,
        divergence_retries: int = 0,
    ):
        """Update automatic stabilization and solve one implicit timestep."""
        if self.time_integrator == "citcoms":
            return self._solve_citcoms(timestep, verbose=verbose)
        if timestep is not None:
            self.delta_t = timestep
        self._update_automatic_tau()
        return super().solve(
            zero_init_guess=zero_init_guess,
            timestep=timestep,
            evalf=evalf,
            _force_setup=_force_setup,
            verbose=verbose,
            divergence_retries=divergence_retries,
        )
