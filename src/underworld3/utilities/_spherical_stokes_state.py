"""Spherical solver gauge, field finalization and explicit field replay."""
from __future__ import annotations
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, ExitStack
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import underworld3 as uw
from underworld3.utilities._field_layout import _layout, _scatter_values, array_sha256
from underworld3.utilities._stokes_state import StokesSolveState

class SphericalSolveState(StokesSolveState):
    """Borrowed operator/field layout, with native gauge and residual operations.

    Constructed by the solver. ``environment`` is also accepted by analytical
    oracles and legacy replay adapters; application collectors use the methods
    below rather than reconstructing gauge, block, or field-conversion formulas.
    """




    def restore_gauge(self, vector: PETSc.Vec) -> None:
        e = self._environment
        vector.axpy(-(e['pmean'].dot(vector)+self.lift_integral)/e['pmean'].sum(), e['pressure_mode'])
        if e['components'] == 1:
            gram = np.array([[weight.dot(mode) for mode in e['modes'][1:]] for weight in e['rotation_weights']])
            coefficients = np.linalg.solve(gram, [weight.dot(vector) for weight in e['rotation_weights']])
            for coefficient, mode in zip(coefficients, e['modes'][1:], strict=True):
                vector.axpy(-coefficient, mode)






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


def prepare_spherical_replay_fields(mesh: uw.discretisation.Mesh, velocity: uw.discretisation.MeshVariable, pressure: uw.discretisation.MeshVariable, radius_inner: float, radius_outer: float, components: int) -> dict[str, object]:
    """Build field maps only, without a mixed operator, KSP, or gauge change."""
    from underworld3.utilities import _spherical_stokes_gap as native
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

