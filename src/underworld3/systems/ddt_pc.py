"""Rate-based Eulerian SUPG integration, internal implementation for :mod:`ddt`."""

import math
import weakref
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sympy
from petsc4py import PETSc

import underworld3 as uw
import underworld3.timing as timing
from underworld3.checkpoint.state import SnapshottableState


@dataclass
class DDtEulerianSUPGPCState(SnapshottableState):
    """Rate metadata; rate and stabilization DOFs travel with the mesh."""

    _schema_version: int = 1
    method: str = "citcoms"
    rate_var_name: str = ""
    rate_initialised: bool = False
    dt: Optional[float] = None
    adv_gamma: float = 0.5
    corrector_steps: int = 2
    corrector_rtol: float = 1.0e-10
    corrector_atol: float = 1.0e-12
    max_corrector_steps: int = 100
    supg_weight: float = 1.0
    tau_override: Optional[str] = None
    last_corrector_iterations: int = 0
    last_corrector_residual: float = np.inf
    corrector_target: float = np.inf


class _EulerianSUPGPCMethods:
    """PC storage and execution; the public class supplies the DDt base."""

    def __init__(
        self, mesh, psi_fn, V_fn, *, method="citcoms",
        temperature_rate_field=None, adv_gamma=0.5, corrector_steps=2,
        corrector_rtol=1.0e-10, corrector_atol=1.0e-12,
        max_corrector_steps=100, tau=None, supg_weight=1.0,
    ):
        from underworld3.systems.ddt import _as_row_vector, _UWexpression

        if method not in ("citcoms", "pc_converged"):
            raise ValueError("method must be 'citcoms' or 'pc_converged'.")
        if not isinstance(psi_fn, uw.discretisation.MeshVariable):
            raise TypeError("EulerianSUPGPC requires a temperature MeshVariable.")
        if (psi_fn.mesh is not mesh or psi_fn.num_components != 1
                or psi_fn.degree != 1 or not psi_fn.continuous):
            raise ValueError("Predictor-corrector transport requires a continuous scalar P1 field on mesh.")
        if mesh.dim != mesh.cdim or mesh.dim not in (2, 3):
            raise NotImplementedError("Predictor-corrector transport requires a 2-D or 3-D volume mesh.")
        if not 0.0 < float(adv_gamma) <= 1.0:
            raise ValueError("adv_gamma must be in (0, 1].")
        if int(corrector_steps) != corrector_steps or corrector_steps < 1:
            raise ValueError("corrector_steps must be a positive integer.")
        if temperature_rate_field is not None and (
            temperature_rate_field is psi_fn
            or temperature_rate_field.mesh is not mesh
            or temperature_rate_field.degree != 1
            or not temperature_rate_field.continuous
            or temperature_rate_field.num_components != 1
        ):
            raise ValueError("temperature_rate_field must be a separate continuous scalar P1 variable on mesh.")
        if method == "pc_converged":
            if float(adv_gamma) != 0.5:
                raise ValueError("pc_converged requires adv_gamma=0.5 for second-order time accuracy.")
            if corrector_steps != 2:
                raise ValueError("corrector_steps configures fixed CitcomS corrections only.")
            if not np.isfinite(float(corrector_rtol)) or float(corrector_rtol) <= 0.0:
                raise ValueError("corrector_rtol must be finite and positive.")
            if not np.isfinite(float(corrector_atol)) or float(corrector_atol) < 0.0:
                raise ValueError("corrector_atol must be finite and non-negative.")
            if int(max_corrector_steps) != max_corrector_steps or max_corrector_steps < 1:
                raise ValueError("max_corrector_steps must be a positive integer.")
        elif (corrector_rtol != 1.0e-10 or corrector_atol != 1.0e-12
              or max_corrector_steps != 100):
            raise ValueError("corrector tolerances and max_corrector_steps configure pc_converged only.")

        super().__init__()
        self.mesh = mesh
        self._psi_meshVar = psi_fn
        self.psi_fn = psi_fn.sym
        self.V_fn = _as_row_vector(V_fn, mesh.dim)
        self.method = method
        self.order = 1
        self.psi_star = []
        self._init_history_tracking(1)
        self.adv_gamma = float(adv_gamma)
        self.corrector_steps = int(corrector_steps)
        self.corrector_rtol = float(corrector_rtol)
        self.corrector_atol = float(corrector_atol)
        self.max_corrector_steps = int(max_corrector_steps)
        self.last_corrector_iterations = 0
        self.last_corrector_residual = np.inf
        self.corrector_target = np.inf
        self._rate_initialised = False
        self._solver_ref = None
        self._assembly_dm = None
        tag = self.instance_number
        self._supg_weight = _UWexpression(
            rf"w^{{SUPG}}_{{{tag}}}", float(supg_weight), "SUPG term weight",
            _unique_name_generation=True)
        self._tau_override = None if tau is None else sympy.sympify(tau)
        if isinstance(self._tau_override, sympy.MatrixBase):
            raise ValueError("tau must be a scalar expression.")
        self._automatic_tau = tau is None
        self._temperature_rate = temperature_rate_field
        if self._temperature_rate is None:
            self._temperature_rate = uw.discretisation.MeshVariable(
                f"_supg_dTdt_{tag}", mesh, 1, degree=1, continuous=True)
        self._supg_h = None
        self._supg_tau = None
        if self._automatic_tau:
            self._supg_h = uw.discretisation.MeshVariable(
                f"_supg_h_{tag}", mesh, 1, degree=0, continuous=False)
            self._supg_tau = uw.discretisation.MeshVariable(
                f"_supg_tau_{tag}", mesh, 1, degree=0, continuous=False)
        self._lumped_mass = None
        self._lumped_mass_mesh_version = None
        self._citcoms_work_vectors = None
        self._citcoms_work_mesh_version = None
        self._simplex_data_cache = None
        self._simplex_data_mesh_version = None
        self._directional_rate_work = None
        self._directional_rate_mesh_version = None
        self._diffusion_dt_cache = None
        self._register_with_default_model()

    @property
    def V_fn(self):
        """Advecting velocity as a row matrix."""
        return self._V_fn

    @V_fn.setter
    def V_fn(self, value):
        from underworld3.systems.ddt import _as_row_vector

        self._V_fn = _as_row_vector(value, self.mesh.dim)
        solver_ref = getattr(self, "_solver_ref", None)
        solver = None if solver_ref is None else solver_ref()
        if solver is not None:
            solver._needs_function_rewire = True
            solver.is_setup = False

    @property
    def integrator(self):
        """Rate integration method, not a BDF history depth."""
        return self.method

    @property
    def theta(self):
        """Spatial residuals are evaluated at the current iterate."""
        return 1.0

    @theta.setter
    def theta(self, value):
        if float(value) != 1.0:
            raise ValueError("Predictor-corrector transport uses adv_gamma, not theta.")

    @property
    def temperature_rate(self):
        """Persistent derivative field used by the next predictor."""
        return self._temperature_rate

    @property
    def supg_weight(self):
        """Runtime multiplier of the stabilization term."""
        return float(self._supg_weight.sym)

    @supg_weight.setter
    def supg_weight(self, value):
        self._supg_weight.sym = float(value)

    def states(self):
        return [sympy.Matrix(self.psi_fn)]

    def spatial_weights(self):
        return [sympy.Integer(1)]

    def time_derivative(self):
        return sympy.Matrix(self._temperature_rate.sym)

    def advecting_velocity(self, level=0):
        return self.V_fn

    def advection(self):
        from underworld3.systems.ddt import EulerianSUPG

        return EulerianSUPG._convective(self, self.V_fn, self.psi_fn)

    def tau(self):
        """Steady CitcomS tau, or the supplied scalar override."""
        value = self._tau_override if self._tau_override is not None else self._supg_tau.sym[0]
        return self._supg_weight * value

    def stabilisation_flux(self, R):
        from underworld3.systems.ddt import EulerianSUPG

        return EulerianSUPG.stabilisation_flux(self, R)

    def update_pre_solve(self, dt, evalf=False, verbose=False):
        raise NotImplementedError("EulerianSUPGPC needs a solver supporting the transport execution hook.")

    def update_post_solve(self, dt, evalf=False, verbose=False):
        raise NotImplementedError("EulerianSUPGPC commits its rate through the transport execution hook.")

    def _bind_transport_solver(self, solver):
        if solver.mesh is not self.mesh or solver.u is not self._psi_meshVar:
            raise ValueError("EulerianSUPGPC must track the solver's temperature field on its mesh.")
        if self._solver_ref is not None and self._solver_ref() not in (None, solver):
            raise ValueError("EulerianSUPGPC is already attached to another solver.")
        self._solver_ref = weakref.ref(solver)

    def _solver(self):
        solver = None if self._solver_ref is None else self._solver_ref()
        if solver is None:
            raise RuntimeError("Attach EulerianSUPGPC to AdvDiffusion with DuDt= before using solver services.")
        return solver

    def _invalidate_assembly_cache(self):
        if self._lumped_mass is not None:
            self._lumped_mass.destroy()
        if self._citcoms_work_vectors is not None:
            for vector in self._citcoms_work_vectors:
                vector.destroy()
        self._lumped_mass = None
        self._citcoms_work_vectors = None
        self._diffusion_dt_cache = None
        self._assembly_dm = None

    def _setup_citcoms_residual(self, verbose=False):
        solver = self._solver()
        solver._prepare_transport_residual(verbose)
        if self._assembly_dm is not solver.dm:
            self._invalidate_assembly_cache()
            self._assembly_dm = solver.dm

    def _compute_citcoms_residual(self, solution=None, residual=None):
        """Assemble at the current temperature/rate; caller owns returned vectors."""
        solver = self._solver()
        if solution is None:
            solution = solver.dm.createGlobalVector()
        if residual is None:
            residual = solution.duplicate()
        solver._compute_transport_residual(solution, residual)
        return solution, residual

    def _solve_transport(self, solver, dt, *, zero_init_guess=None,
                         verbose=False, divergence_retries=0):
        self._bind_transport_solver(solver)
        if zero_init_guess or divergence_retries:
            raise ValueError("Rate-based transport does not support zero_init_guess or SNES divergence retries.")
        self._solve_predictor_corrector(dt, verbose=verbose)

    def estimate_dt(self, fraction=0.02, basis=None,
                    direction_aware=False, percentile=0.0):
        """Return the CitcomS stability timestep, dimensionalised when applicable."""
        from underworld3.systems.solvers import _dimensionalise_dt

        if basis not in (None, "stability"):
            raise ValueError("Predictor-corrector transport requires basis='stability'.")
        if fraction != 0.02 or direction_aware or percentile != 0.0:
            raise ValueError("Predictor-corrector transport uses its fixed 0.9 stability factor and directional simplex length.")
        return _dimensionalise_dt(self._estimate_citcoms_dt())

    def _estimate_transport_dt(self, **kwargs):
        return self.estimate_dt(**kwargs)

    @property
    def state(self):
        return DDtEulerianSUPGPCState(
            method=self.method, rate_var_name=self.temperature_rate.clean_name,
            rate_initialised=self._rate_initialised, dt=self._dt,
            adv_gamma=self.adv_gamma, corrector_steps=self.corrector_steps,
            corrector_rtol=self.corrector_rtol, corrector_atol=self.corrector_atol,
            max_corrector_steps=self.max_corrector_steps,
            supg_weight=self.supg_weight,
            tau_override=None if self._tau_override is None else str(self._tau_override),
            last_corrector_iterations=self.last_corrector_iterations,
            last_corrector_residual=self.last_corrector_residual,
            corrector_target=self.corrector_target,
        )

    @state.setter
    def state(self, state):
        if not isinstance(state, DDtEulerianSUPGPCState):
            raise TypeError("EulerianSUPGPC state has the wrong type.")
        current = self.state
        for name in ("_schema_version", "method", "rate_var_name", "adv_gamma",
                     "corrector_steps", "corrector_rtol", "corrector_atol",
                     "max_corrector_steps", "tau_override"):
            if getattr(state, name) != getattr(current, name):
                raise ValueError(f"EulerianSUPGPC {name} changed since snapshot.")
        self._dt = state.dt
        self._rate_initialised = bool(state.rate_initialised)
        self.supg_weight = state.supg_weight
        self.last_corrector_iterations = state.last_corrector_iterations
        self.last_corrector_residual = state.last_corrector_residual
        self.corrector_target = state.corrector_target
        self._invalidate_assembly_cache()
        self._simplex_data_cache = None
        self._directional_rate_work = None
        self.mesh._stale_lvec = True

    def _simplex_data(self):
        """Return local simplex data; validate the layout collectively on rebuild."""
        from underworld3.meshing.smoothing import _tet_cells, _tri_cells

        mesh_version = getattr(self.mesh, "_mesh_version", 0)
        if (
            self._simplex_data_cache is not None
            and self._simplex_data_mesh_version == mesh_version
        ):
            return self._simplex_data_cache

        cells = (
            _tri_cells(self.mesh.dm)
            if self.mesh.dim == 2
            else _tet_cells(self.mesh.dm) if self.mesh.dim == 3 else None
        )
        cell_start, cell_end = self.mesh.dm.getHeightStratum(0)
        invalid = (
            (uw.mpi.rank, cell_end - cell_start, self.mesh.dim, self.mesh.cdim)
            if cells is None or self.mesh.dim != self.mesh.cdim else None
        )
        invalid_ranks = [item for item in uw.mpi.comm.allgather(invalid) if item is not None]
        if invalid_ranks:
            raise NotImplementedError(
                "Automatic CitcomS operations require a non-empty 2-D or 3-D "
                "volume simplex partition on every rank. Unsupported local "
                f"layouts (rank, cells, dim, cdim): {invalid_ranks}."
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
        self._simplex_data_cache = (cells, gradients, volumes)
        self._simplex_data_mesh_version = mesh_version
        return self._simplex_data_cache

    def _streamline_directional_rate(self, gradients, velocity):
        """Return ``sum_a |u.grad(N_a)|`` using reusable cell work arrays."""
        mesh_version = getattr(self.mesh, "_mesh_version", 0)
        cell_count = velocity.shape[0]
        if (
            self._directional_rate_work is None
            or self._directional_rate_mesh_version != mesh_version
            or self._directional_rate_work[0].shape != (cell_count,)
        ):
            self._directional_rate_work = (
                np.empty(cell_count, dtype=float),
                np.empty(cell_count, dtype=float),
            )
            self._directional_rate_mesh_version = mesh_version

        directional_rate, projection = self._directional_rate_work
        directional_rate.fill(0.0)
        for basis_index in range(gradients.shape[1]):
            np.einsum(
                "cd,cd->c",
                gradients[:, basis_index, :],
                velocity,
                out=projection,
            )
            np.abs(projection, out=projection)
            np.add(directional_rate, projection, out=directional_rate)
        return directional_rate

    def _cell_diffusivity(self, cell_count):
        """Evaluate non-negative scalar diffusivity at cell centroids."""
        diffusivity_expr = sympy.sympify(self._solver()._scalar_diffusivity())
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

    def _update_automatic_tau(self):
        """Update local simplex streamline lengths and automatic tau values."""
        if not self._automatic_tau:
            return

        _, gradients, _ = self._simplex_data()

        from underworld3.systems.solvers import _centroid_velocities_nd

        velocity = _centroid_velocities_nd(self.V_fn, self.mesh)
        speed = np.linalg.norm(velocity, axis=1)
        directional_rate = self._streamline_directional_rate(gradients, velocity)
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
            tau_steady[diffusive] = (
                h_stream[diffusive]
                * np.maximum(0.0, 1.0 - 1.0 / pe)
                / (2.0 * speed[diffusive])
            )
        tau_steady[nondiffusive] = h_stream[nondiffusive] / (2.0 * speed[nondiffusive])

        tau_values = tau_steady

        if self._supg_h.array.shape[0] != h_stream.size:
            raise RuntimeError("SUPG P0 field and local simplex counts do not match.")
        self._supg_h.array[:, 0, 0] = h_stream
        self._supg_tau.array[:, 0, 0] = tau_values

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

        local_mass = self._solver().dm.createLocalVector()
        global_mass = self._solver().dm.createGlobalVector()
        local_mass.set(0.0)
        global_mass.set(0.0)
        section = self._solver().dm.getLocalSection()
        vertex_start, _ = self.mesh.dm.getDepthStratum(0)

        for cell_index in np.flatnonzero(owned):
            contribution = volumes[cell_index] / (self.mesh.dim + 1)
            for vertex_index in cells[cell_index]:
                offset = section.getOffset(vertex_start + int(vertex_index))
                if offset >= 0:
                    local_mass.array[offset] += contribution

        self._solver().dm.localToGlobal(
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

        solution = self._solver().dm.createGlobalVector()
        residual = solution.duplicate()
        delta_rate = solution.duplicate()
        rate = solution.duplicate()
        self._citcoms_work_vectors = (solution, residual, delta_rate, rate)
        self._citcoms_work_mesh_version = mesh_version
        return self._citcoms_work_vectors

    @timing.routine_timer_decorator
    def _estimate_citcoms_dt(self):
        """Estimate a simplex advection-diffusion timestep.

        The predictor-corrector modes use
        ``0.9 * min(1/max(lambda_adv), 2/max(rowsum(abs(M_L^-1 K))))``.
        Generic implicit transport retains its separate Eulerian estimator.
        """
        from underworld3.systems.solvers import _centroid_velocities_nd
        from mpi4py import MPI
        from underworld3.meshing.smoothing import _owned_cell_mask

        cells, gradients, volumes = self._simplex_data()
        velocity = _centroid_velocities_nd(self.V_fn, self.mesh)
        directional_rate = self._streamline_directional_rate(gradients, velocity)
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

            stiffness = self._solver().dm.createMatrix()
            stiffness.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR, False)
            section = self._solver().dm.getLocalSection()
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

    def _apply_pc_correction(
        self,
        temperature_global,
        residual,
        delta_rate,
        rate_global,
        mass,
        dt,
        *,
        advance_temperature,
    ):
        """Apply one lumped-preconditioned correction to rate and temperature."""
        delta_rate.pointwiseDivide(residual, mass)
        delta_rate.scale(-1.0)
        rate_global.set(0.0)
        self._solver().dm.localToGlobal(self._temperature_rate.vec, rate_global, addv=False)
        rate_global.axpy(1.0, delta_rate)
        if advance_temperature:
            temperature_global.axpy(self.adv_gamma * dt, delta_rate)

        self._temperature_rate.vec.set(0.0)
        self._solver().dm.globalToLocal(rate_global, self._temperature_rate.vec)
        if advance_temperature:
            from underworld3.cython.petsc_discretisation import (
                petsc_dm_insert_boundary_values,
            )

            self._psi_meshVar.vec.set(0.0)
            self._solver().dm.globalToLocal(temperature_global, self._psi_meshVar.vec)
            petsc_dm_insert_boundary_values(self._solver().dm, self._psi_meshVar.vec)
        self.mesh._stale_lvec = True

    def _converge_pc_residual(
        self,
        temperature_global,
        residual,
        delta_rate,
        rate_global,
        mass,
        dt,
        *,
        advance_temperature,
    ):
        """Iterate the predictor-corrector residual to its configured tolerance."""
        initial_norm = None
        for corrections in range(self.max_corrector_steps + 1):
            self._compute_citcoms_residual(temperature_global, residual)
            residual_norm = float(residual.norm(PETSc.NormType.NORM_2))
            if not np.isfinite(residual_norm):
                raise RuntimeError("pc_converged produced a non-finite residual norm.")
            if initial_norm is None:
                initial_norm = residual_norm
                self.corrector_target = max(
                    self.corrector_atol,
                    self.corrector_rtol * initial_norm,
                )
            self.last_corrector_iterations = corrections
            self.last_corrector_residual = residual_norm
            if residual_norm <= self.corrector_target:
                return
            if corrections == self.max_corrector_steps:
                break
            self._apply_pc_correction(
                temperature_global,
                residual,
                delta_rate,
                rate_global,
                mass,
                dt,
                advance_temperature=advance_temperature,
            )
        raise RuntimeError(
            "pc_converged did not reach its predictor-corrector residual "
            f"tolerance after {self.max_corrector_steps} corrections: "
            f"residual={self.last_corrector_residual:.6e}, "
            f"target={self.corrector_target:.6e}."
        )

    def _solve_predictor_corrector(self, timestep, verbose=False):
        """Advance one fixed or residual-converged predictor-corrector step."""
        from underworld3.systems.solvers import _invalidate_solution_cache

        dt = self._dt if timestep is None else float(timestep)
        if dt is None or not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("EulerianSUPGPC requires a finite positive timestep.")
        self._dt = dt

        self._update_automatic_tau()
        self._setup_citcoms_residual(verbose)
        mass = self._assemble_lumped_mass()
        temperature_global, residual, delta_rate, rate_global = self._citcoms_vectors()

        if not self._rate_initialised:
            self._temperature_rate.array[:, 0, 0] = 0.0
            if self.method == "pc_converged":
                self.mesh._stale_lvec = True
                self._converge_pc_residual(
                    temperature_global,
                    residual,
                    delta_rate,
                    rate_global,
                    mass,
                    dt,
                    advance_temperature=False,
                )
            else:
                self._compute_citcoms_residual(temperature_global, residual)
                delta_rate.pointwiseDivide(residual, mass)
                delta_rate.scale(-1.0)
                self._temperature_rate.vec.set(0.0)
                self._solver().dm.globalToLocal(delta_rate, self._temperature_rate.vec)
                self.mesh._stale_lvec = True
            self._rate_initialised = True

        self._psi_meshVar.array[:, 0, 0] += (
            (1.0 - self.adv_gamma) * dt * self._temperature_rate.array[:, 0, 0]
        )
        self._temperature_rate.array[:, 0, 0] = 0.0
        self.mesh._stale_lvec = True

        if self.method == "pc_converged":
            from underworld3.cython.petsc_discretisation import (
                petsc_dm_insert_boundary_values,
            )

            petsc_dm_insert_boundary_values(self._solver().dm, self._psi_meshVar.vec)
            self.mesh._stale_lvec = True
            self._converge_pc_residual(
                temperature_global,
                residual,
                delta_rate,
                rate_global,
                mass,
                dt,
                advance_temperature=True,
            )
        else:
            for _ in range(self.corrector_steps):
                self._compute_citcoms_residual(temperature_global, residual)
                self._apply_pc_correction(
                    temperature_global,
                    residual,
                    delta_rate,
                    rate_global,
                    mass,
                    dt,
                    advance_temperature=True,
                )

        _invalidate_solution_cache(self._psi_meshVar)
        _invalidate_solution_cache(self._temperature_rate)
        return
