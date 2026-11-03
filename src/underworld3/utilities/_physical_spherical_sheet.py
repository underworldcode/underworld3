"""True-sphere sheet quadrature assembled in actual containing tetrahedra.

Owned fitted facets partition the physical spherical surface by radial
projection. Quadrature packets are routed through exact-containment proposals;
P2 evaluation uses the containing volume cell, never extrapolation of
a facet owner's polynomial. The returned vector is a positive residual load.
"""

from contextlib import ExitStack
from typing import NamedTuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import underworld3 as uw

from underworld3.maths.spherical_harmonics import (
    harmonic_norm, real_harmonic_values, validate_harmonic,
)
from underworld3.utilities._cylindrical_gap import _layout
from underworld3.utilities._spherical_gap import _spherical_facet_quadrature
from underworld3.utilities._spherical_interface_load import _vertex_coordinates
from underworld3.utilities._tetrahedral_gap import _tetrahedral_p2_basis


class _PhysicalSphericalSheet(NamedTuple):
    residual: object
    checks: dict


def _route_spherical_packets(mesh, points: np.ndarray, weights: np.ndarray):
    """Route each packet once to an exact containing owned cell.

    Only six bounding-box numbers per rank are replicated. Candidate point
    queries use all-to-all traffic; origins select the smallest proposed
    (global cell ID, rank), then send each point and weight exactly once.
    This deliberately replaces Swarm migration's evaluation-tolerance policy.
    """
    dm, comm = mesh.dm, mesh.dm.comm.tompi4py()
    rows = mesh._cell_node_indices(1, True)
    vertices = np.asarray(mesh._get_coords_for_basis(1, True))[rows]
    numbering = np.asarray(dm.getCellNumbering().getIndices(), dtype=np.int64)
    owned = numbering >= 0
    box = np.array([[np.inf]*3, [-np.inf]*3])
    if np.any(owned):
        box = np.stack((vertices[owned].min(axis=(0, 1)), vertices[owned].max(axis=(0, 1))))
    boxes = comm.allgather(box)
    queries = []
    for lower, upper in boxes:
        selected = np.flatnonzero(np.all((points >= lower-1e-12) & (points <= upper+1e-12), axis=1))
        queries.append((selected, points[selected]))
    proposals = []
    for indices, candidate_points in comm.alltoall(queries):
        cells = np.asarray(mesh.get_closest_local_cells(candidate_points, on_boundary=True, tol=0.0), dtype=int)
        eligible = np.flatnonzero(cells >= 0)
        eligible = eligible[owned[cells[eligible]]]
        if len(eligible):
            _, _, bary = _tetrahedral_p2_basis(vertices[cells[eligible]], candidate_points[eligible, None])
            eligible = eligible[bary.min(axis=(1, 2)) >= -1e-10]
        proposals.append((indices[eligible], numbering[cells[eligible]], cells[eligible]))
    chosen = {}
    for rank, (indices, global_cells, local_cells) in enumerate(comm.alltoall(proposals)):
        for index, global_cell, local_cell in zip(indices, global_cells, local_cells, strict=True):
            choice = (int(global_cell), rank, int(local_cell))
            if int(index) not in chosen or choice < chosen[int(index)]:
                chosen[int(index)] = choice
    missing = comm.allgather(len(points)-len(chosen))
    if any(missing):
        raise ValueError(f"Physical spherical packets have no exact containing-cell proposal: {missing}")
    destinations = np.array([chosen[i][1] for i in range(len(points))], dtype=int)
    cell_ids = np.array([chosen[i][2] for i in range(len(points))], dtype=int)
    packets = []
    for rank in range(comm.size):
        selected = destinations == rank
        packets.append((points[selected], weights[selected], cell_ids[selected]))
    received = comm.alltoall(packets)
    moved = comm.allreduce(int(np.count_nonzero(destinations != comm.rank)), op=MPI.SUM)
    return (np.concatenate([packet[0] for packet in received]),
            np.concatenate([packet[1] for packet in received]),
            np.concatenate([packet[2] for packet in received]), moved)


def _owned_spherical_facets(mesh, name: str):
    """Read each globally owned triangular facet exactly once."""
    leaves = None
    if mesh.dm.comm.size > 1:
        _, leaves, _ = mesh.dm.getPointSF().getGraph()
    ghosts = set() if leaves is None else set(map(int, leaves))
    facets = [int(f) for f in mesh._boundary_facets(name) if int(f) not in ghosts]
    triangles = np.array([_vertex_coordinates(mesh, f) for f in facets]).reshape(-1, 3, 3)
    return facets, triangles


def _p2_load_workspace(velocity):
    """Create a field-owned vector and verified canonical continuous-P2 maps."""
    from underworld3.utilities._spherical_stokes_gap import _canonical_rows

    mesh = velocity.mesh
    vertices = np.asarray(mesh._get_coords_for_basis(1, True))[mesh._cell_node_indices(1, True)]
    rows = _canonical_rows(mesh, velocity, vertices)
    mapping, _ = _layout(velocity)
    ids = mapping.reshape(-1, 3)[rows]
    field_is, field_dm = mesh.dm.createSubDM(velocity.field_id)
    try:
        vector = field_dm.createGlobalVec()
        vector.set(0)
    finally:
        field_dm.destroy()
        field_is.destroy()
    return vector, vertices, ids


def _add_p2_point_load(vector, vertices, ids, cells, points, weighted_force):
    """Add known-cell force packets, returning a local containment diagnostic."""
    shape, _, bary = _tetrahedral_p2_basis(vertices[cells], points)
    loads = np.einsum("eqi,eqd->eid", shape, weighted_force)
    vector.setValues(ids[cells].ravel(), loads.ravel(), addv=PETSc.InsertMode.ADD_VALUES)
    return float(bary.min(initial=1))


def _assemble_physical_spherical_sheet(
    velocity, interface: str, radius: float, harmonic_degree: int,
    harmonic_order: int, harmonic_component: str, harmonic_normalization: str,
    density_coefficient: float, quadrature_points: int,
) -> _PhysicalSphericalSheet:
    """Assemble g Y_lm e_r on the physical internal sphere into P2 velocity.

    Call collectively on the world mesh communicator. The caller owns the
    returned PETSc vector. Packet loss, failed containment and weight changes
    are errors; no deleted-particle or owner-cell extrapolation fallback exists.
    Native mixed residual insertion and pressure-block zeros belong to the
    solver adapter, which is deliberately not installed here.
    """
    mesh, dm = velocity.mesh, velocity.mesh.dm
    comm = dm.comm.tompi4py()
    if (mesh.dim != 3 or not mesh.isSimplex or mesh.degree != 1
            or velocity.degree != 2 or not velocity.continuous
            or velocity.num_components != 3):
        raise ValueError("Physical spherical sheets require affine tetrahedra and continuous P2 velocity.")
    if MPI.Comm.Compare(comm, uw.mpi.comm) not in (MPI.IDENT, MPI.CONGRUENT):
        raise ValueError("Physical sheet routing requires the world mesh communicator.")
    if uw.get_default_model().has_units_active():
        raise ValueError("Physical spherical sheet assembly requires nondimensional fields.")
    if interface not in mesh.boundaries.__members__ or not np.isfinite(density_coefficient):
        raise ValueError("Require a named internal interface and finite sheet coefficient.")
    harmonic_degree, harmonic_order, harmonic_component, harmonic_normalization = validate_harmonic(
        harmonic_degree, harmonic_order, harmonic_component, harmonic_normalization,
    )
    if not isinstance(quadrature_points, (int, np.integer)) or quadrature_points < 2:
        raise ValueError("Surface quadrature needs an integer order of at least two.")
    facets, triangles = _owned_spherical_facets(mesh, interface)
    rule = _spherical_facet_quadrature(triangles, radius, quadrature_points, 2)
    points = np.ascontiguousarray(rule["surface_points"].reshape(-1, 3))
    weights = rule["surface_weights"].ravel()
    packets_before = comm.allreduce(len(points), op=MPI.SUM)
    area_before = comm.allreduce(float(weights.sum()), op=MPI.SUM)
    if packets_before == 0:
        raise ValueError("The physical spherical sheet has no owned facets.")

    points, weights, cells, moved = _route_spherical_packets(mesh, points, weights)
    packets_after = comm.allreduce(len(points), op=MPI.SUM)
    area = comm.allreduce(float(weights.sum()), op=MPI.SUM)
    if packets_after != packets_before or not np.isclose(area, area_before, rtol=1e-13, atol=0):
        raise ArithmeticError("Physical sheet routing changed packet count or total quadrature weight.")
    minimum_barycentric = 1.0
    with ExitStack() as cleanup:
        residual, vertices, ids = _p2_load_workspace(velocity)
        cleanup.callback(residual.destroy)
        harmonic_moment = 0.0
        for start in range(0, len(points), 256):
            sl = slice(start, start + 256)
            cell = cells[sl]
            y = real_harmonic_values(points[sl], degree=harmonic_degree, order=harmonic_order,
                                     component=harmonic_component, normalization=harmonic_normalization)
            force = density_coefficient * weights[sl, None] * y[:, None] * points[sl] / radius
            minimum_barycentric = min(minimum_barycentric, _add_p2_point_load(
                residual, vertices, ids, cell, points[sl, None], force[:, None],
            ))
            harmonic_moment += float(np.dot(weights[sl], y*y))
        minimum_barycentric = comm.allreduce(minimum_barycentric, op=MPI.MIN)
        if minimum_barycentric < -1e-10:
            raise ValueError(f"Physical sheet cell locator failed barycentric containment: {minimum_barycentric}")
        residual.assemble()
        harmonic_moment = comm.allreduce(harmonic_moment, op=MPI.SUM)
        expected_moment = radius**2 * harmonic_norm(harmonic_degree, harmonic_order,
                                                   harmonic_component, harmonic_normalization)
        checks = dict(routing="exact_containment_bbox_packets", packets_before=packets_before, packets_after=packets_after,
                      migrated_packets=moved, area=area, expected_area=4*np.pi*radius**2,
                      minimum_barycentric=minimum_barycentric,
                      harmonic_squared_moment=harmonic_moment,
                      harmonic_moment_relative_error=abs(harmonic_moment/expected_moment-1))
        cleanup.pop_all()
    return _PhysicalSphericalSheet(residual=residual, checks=checks)
