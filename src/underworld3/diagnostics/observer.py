"""Public, opt-in observation of the original spherical Stokes equations.

Observer arguments borrow native PETSc objects for the duration of one solve.
Observers may inspect them and retain copies, but must not mutate or destroy
operators, RHS vectors, solver settings, or iterates. All callbacks are collective.
No operator copies or extra residual evaluations occur without an observer.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager, ExitStack
import hashlib
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import underworld3 as uw
from underworld3.utilities._cylindrical_gap import _layout, _scatter_values
from underworld3.utilities import _spherical_stokes_gap as native


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


class SphericalSolveState:
    """Borrowed operator/field layout, with native gauge and residual operations.

    Constructed by the solver. ``environment`` is also accepted by analytical
    oracles and legacy replay adapters; application collectors use the methods
    below rather than reconstructing gauge, block, or field-conversion formulas.
    """
    def __init__(self, environment: Mapping[str, object]) -> None:
        self._environment = environment
        self._lift_integral = None

    def __getitem__(self, key: str) -> object:
        """Read-only compatibility for historical experimental recorder adapters."""
        return self._environment[key]

    @property
    def rhs(self) -> PETSc.Vec:
        return self._environment['rhs']

    @property
    def lift_integral(self) -> float:
        e = self._environment
        if self._lift_integral is None:
            self._lift_integral = float(e['pressure_jump_integral'])+e['mpi'].allreduce(float(e['lift_gap']), op=MPI.SUM)
        return self._lift_integral

    def restore_gauge(self, vector: PETSc.Vec) -> None:
        e = self._environment
        vector.axpy(-(e['pmean'].dot(vector)+self.lift_integral)/e['pmean'].sum(), e['pressure_mode'])
        if e['components'] == 1:
            gram = np.array([[weight.dot(mode) for mode in e['modes'][1:]] for weight in e['rotation_weights']])
            coefficients = np.linalg.solve(gram, [weight.dot(vector) for weight in e['rotation_weights']])
            for coefficient, mode in zip(coefficients, e['modes'][1:], strict=True):
                vector.axpy(-coefficient, mode)

    def gauged_copy(self, vector: PETSc.Vec) -> PETSc.Vec:
        result = vector.copy()
        self.restore_gauge(result)
        return result

    def build_iterate(self, ksp: PETSc.KSP) -> PETSc.Vec:
        result = self.rhs.duplicate()
        ksp.buildSolution(result)
        return result

    @contextmanager
    def _residual_vector(self, vector: PETSc.Vec) -> Iterator[PETSc.Vec]:
        work = self.rhs.duplicate()
        try:
            self._environment['matrix'].mult(vector, work)
            work.axpy(-1, self.rhs)
            yield work
        finally:
            work.destroy()

    def residual_norms(self, vector: PETSc.Vec) -> dict[str, float]:
        with self._residual_vector(vector) as work:
            scale, actual = self.rhs.norm(), work.norm()
            relative = actual/max(scale, np.finfo(float).tiny)
            if not np.isfinite(relative):
                raise ArithmeticError('Nonfinite original spherical Stokes residual')
            return dict(original_rhs_norm=scale, original_residual_norm=actual,
                        original_relative_residual=relative)

    def residual(self, vector: PETSc.Vec) -> dict[str, object]:
        e = self._environment
        with self._residual_vector(vector) as work:
            scale, actual = self.rhs.norm(), work.norm()
            denominator = max(scale, np.finfo(float).tiny)
            relative = actual/denominator
            if not np.isfinite(relative):
                raise ArithmeticError('Nonfinite original spherical Stokes residual')
            blocks = {}
            lo, hi = work.getOwnershipRange()
            nu, npres = e['nu'], e['npres']
            for name, first, last in (('velocity',0,nu),('pressure',nu,nu+npres),('multiplier',nu+npres,work.getSize())):
                indices = PETSc.IS().createGeneral(np.arange(max(lo,first), min(hi,last), dtype=PETSc.IntType), comm=work.comm)
                block = work.getSubVector(indices)
                try:
                    norm = block.norm()
                    blocks[name] = dict(residual_norm=norm, residual_over_full_rhs=norm/denominator)
                finally:
                    work.restoreSubVector(indices, block)
                    indices.destroy()
            return dict(original_rhs_norm=scale, original_residual_norm=actual,
                        original_relative_residual=relative, blocks=blocks,
                        physical_pressure_integral=e['pmean'].dot(vector)+self.lift_integral,
                        rotation_integrals=[weight.dot(vector) for weight in e['rotation_weights']])

    def layout_metadata(self, vector: PETSc.Vec) -> dict[str, object]:
        e = self._environment
        return dict(field_sizes=dict(velocity=e['nu'], pressure=e['npres'], boundary_nodes=e['nboundary'], components=e['components']),
                    mixed_vector_sizes=list(vector.getSizes()),
                    layout_hashes={name:array_sha256(e[name]) for name in ('umap','pmap','compressed','coordinates')})

    def assign_fields(self, solution: PETSc.Vec) -> tuple[uw.discretisation.MeshVariable, uw.discretisation.MeshVariable]:
        e = self._environment
        velocity, pressure, mesh, stokes = (e[name] for name in ('velocity','pressure','mesh','stokes'))
        umap, pmap, compressed, coordinates = (e[name] for name in ('umap','pmap','compressed','coordinates'))
        nu, npres, components = (e[name] for name in ('nu','npres','components'))
        radius_inner, radius_outer = e['radius_inner'], e['radius_outer']
        velocity.array[:, 0, :] = _scatter_values(solution, umap).reshape(-1, 3)
        pressure.array[:, 0, 0] = _scatter_values(solution, pmap+nu)
        normal_values = np.zeros(len(coordinates)); tangent_values = np.zeros((len(coordinates), 3))
        rows = np.flatnonzero(compressed >= 0)
        lids = nu+npres+components*compressed[rows, None]+np.arange(components)
        coefficients = _scatter_values(solution, lids.ravel()).reshape(-1, components)
        distance = np.linalg.norm(coordinates[rows], axis=1)
        is_inner = np.abs(distance-radius_inner) < np.abs(distance-radius_outer)
        signs = np.where(is_inner, -1, 1)
        radii = np.where(is_inner, radius_inner, radius_outer)
        outward = signs[:, None]*coordinates[rows]/distance[:, None]
        if components == 1:
            normal_values[rows] = coefficients[:, 0]
        else:
            physical = coefficients*radii[:, None]/distance[:, None]
            normal_values[rows] = np.einsum("ed,ed->e", physical, outward)
            tangent_values[rows] = physical-normal_values[rows, None]*outward
        if not hasattr(stokes, "_spherical_gap_normal_reaction"):
            stokes._spherical_gap_normal_reaction = uw.discretisation.MeshVariable(
                f"SphericalNormalReaction_{stokes.instance_number}", mesh, 1, degree=2,
            )
            stokes._spherical_gap_tangential_reaction = uw.discretisation.MeshVariable(
                f"SphericalTangentialReaction_{stokes.instance_number}", mesh, 3, degree=2,
            )
        normal_field = stokes._spherical_gap_normal_reaction
        tangent_field = stokes._spherical_gap_tangential_reaction
        normal_field.array[:, 0, 0] = normal_values
        tangent_field.array[:, 0, :] = tangent_values
        for field in (velocity, pressure, normal_field, tangent_field):
            field._sync_lvec_to_gvec()
        mesh._stale_lvec = True
        return normal_field, tangent_field


class SphericalStokesObserver:
    """Optional hooks; subclass only the observations an application needs.

    Source-integrity bookkeeping is owned here, so collectors need no private
    solver access. Overridden constructors should call ``super().__init__()``.
    """
    def __init__(self):
        self._capture_source()

    def _capture_source(self) -> None:
        self._source_function = native._solve_constraints
        path = Path(native.__file__)
        self._solver_source = dict(native_source_path=str(path),
                                   native_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    @property
    def solver_source(self) -> dict[str, str]:
        return dict(self._solver_source)

    def __enter__(self):
        if not hasattr(self, '_solver_source'):
            self._capture_source()
        return self

    def __exit__(self, kind, value, traceback):
        unchanged = (native._solve_constraints is self._source_function
                     and hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
                         == self._solver_source['native_source_sha256'])
        if not unchanged:
            raise RuntimeError('Native spherical solver changed during observation')
        return False

    def background(self, matrix, velocity_size, pressure_size):
        pass

    def physical(self, matrix, velocity_size, pressure_size, pressure_mode, rotations, rotation_weights, walls, components):
        pass

    def phase(self, name: str, snapshot: Mapping[str, object]) -> None:
        pass

    def pc_started(self, application: int) -> None:
        pass

    def pc_finished(self, application: int) -> None:
        pass

    def solve_started(self, ksp: PETSc.KSP, state: SphericalSolveState) -> None:
        pass

    def iteration(self, ksp: PETSc.KSP, iteration: int, norm: float, state: SphericalSolveState) -> None:
        pass

    def solve_finished(self, ksp: PETSc.KSP, solution: PETSc.Vec, state: SphericalSolveState) -> None:
        pass

    def finalized(self, result):
        pass


@contextmanager
def observe_spherical_stokes(stokes: object, observer: SphericalStokesObserver) -> Iterator[SphericalStokesObserver]:
    """Register one scoped observer without replacing native solver functions."""
    existed = hasattr(stokes, '_spherical_gap_observer')
    previous = getattr(stokes, '_spherical_gap_observer', None)
    if previous is not None:
        raise RuntimeError('A spherical Stokes observer is already registered')
    stokes._spherical_gap_observer = observer
    try:
        with observer:
            yield observer
    finally:
        if existed:
            stokes._spherical_gap_observer = previous
        else:
            del stokes._spherical_gap_observer


def solve_with_observer(ksp: PETSc.KSP, solve_rhs: PETSc.Vec, solution: PETSc.Vec, state: SphericalSolveState, observer: SphericalStokesObserver) -> None:
    """Run the unchanged KSP, removing only this observer's Python monitor.

    PETSc ASCII/native monitors and pre-existing Python monitors remain installed.
    The petsc4py monitor context is a list of (callback,args,kwargs) registrations;
    removing our exact registration preserves the other entries and C monitors.
    """
    observer.solve_started(ksp, state)
    def monitor(solver, iteration, norm):
        observer.iteration(solver, iteration, norm, state)
    ksp.setMonitor(monitor)
    registration = ksp.getMonitor()[-1]
    try:
        answer = ksp.solve(solve_rhs, solution)
        observer.solve_finished(ksp, solution, state)
        return answer
    finally:
        entries = ksp.getMonitor()
        entries[:] = [entry for entry in entries if entry is not registration]


def prepare_spherical_replay_fields(mesh: uw.discretisation.Mesh, velocity: uw.discretisation.MeshVariable, pressure: uw.discretisation.MeshVariable, radius_inner: float, radius_outer: float, components: int) -> dict[str, object]:
    """Build field maps only, without a mixed operator, KSP, or gauge change."""
    stokes = uw.systems.Stokes(mesh, velocityField=velocity, pressureField=pressure)
    umap, nu = _layout(velocity)
    pmap, npres = _layout(pressure)
    vertices = np.asarray(mesh._get_coords_for_basis(1, True))[mesh._cell_node_indices(1, True)]
    rows = native._canonical_rows(mesh, velocity, vertices)
    owned = np.flatnonzero(mesh._get_owned_cells_mask())
    walls = [native._wall_faces(mesh, vertices, rows, owned, name, radius)
             for name, radius in (("Lower", radius_inner), ("Upper", radius_outer))]
    with ExitStack() as cleanup:
        compressed, nboundary = native._compressed_boundary_ids(umap.reshape(-1, 3)[:, 0]//3,
            nu//3, walls, mesh.dm.comm, cleanup)
    return dict(mesh=mesh, stokes=stokes, velocity=velocity, pressure=pressure,
                umap=umap, pmap=pmap, compressed=compressed, coordinates=np.asarray(velocity.coords),
                nu=nu, npres=npres, nboundary=nboundary, components=components,
                radius_inner=radius_inner, radius_outer=radius_outer)


def assign_spherical_fields(vector: PETSc.Vec, layout: Mapping[str, object]) -> tuple[uw.discretisation.MeshVariable, uw.discretisation.MeshVariable]:
    """Explicit field-only replay, preserving the native conversion semantics."""
    return SphericalSolveState(layout).assign_fields(vector)
