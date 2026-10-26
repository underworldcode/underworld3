r"""Prescribed pressure lifting on interface-fitted affine simplex meshes.

Write p=q+p_J and cancel the prescribed normal sheet force [p] n delta_Gamma.
The remaining body force is -grad_h(p_J); the exterior traction correction is
-p_J Gamma. Jump values are interpolated linearly on facets, not inferred from
the unknown flow. Velocity is continuous; this is not a fault-slip formulation.

Chang and Kwak (2009), arXiv:0911.4769, Sections 2--3,
https://arxiv.org/abs/0911.4769, motivate prescribed-jump extraction and numerical
extension. The vertex-star constraint lift below is a UW3 construction, not
their harmonic extension or Crouzeix--Raviart discretisation. At each vertex,
cell-side values satisfy the specified jumps and continuity across all other
facets. Cycle closure detects incompatible tip and junction data. A common
continuous P1 component of the lift is arbitrary and is absorbed by q.

Only interface stars are exchanged, to their vertex owners; no global mesh or
global interface is replicated. Local ghost cells receive the same lift.
"""

from collections import defaultdict
from math import factorial
from typing import NamedTuple

import numpy as np
import sympy as sp
import underworld3 as uw
from mpi4py import MPI

from underworld3.utilities.dm_labels import label_stratum_indices


class _PressureSplit(NamedTuple):
    pressure_jump: object
    pressure_jump_field: object
    regular_force: object
    natural_datum: object
    physical_pressure: object
    pressure_jump_integral: float
    checks: dict


def _sample(mesh, datum, coordinates, components):
    """Sample prescribed data, including empty rank-local coordinate arrays."""
    if callable(datum):
        values = datum(coordinates)
    else:
        if isinstance(datum, uw.discretisation.MeshVariable):
            if datum.mesh is not mesh or not datum.continuous or datum.num_components != components:
                raise ValueError("Jump fields must be continuous and belong to the Stokes mesh.")
            datum = datum.sym[0] if components == 1 else datum.sym
        expr = sp.sympify(datum) if components == 1 else sp.Matrix(datum)
        values = uw.function.evaluate(expr, coordinates, coord_sys=mesh.N)
    array = np.asarray(values, dtype=float)
    if array.size == components:
        array = np.broadcast_to(array.reshape(1, components), (len(coordinates), components))
    array = array.reshape(len(coordinates), components)
    if not np.isfinite(array).all():
        raise ValueError("Prescribed pressure jumps and normals must be finite.")
    return array


def _collective_error(comm, message):
    """Raise on every rank before the next communication phase."""
    errors = comm.allgather(message)
    if any(errors):
        raise ValueError(next(error for error in errors if error))


def _star_values(faces, metadata, tolerance):
    """Solve one vertex's cell-side jump graph, checking closed cycles."""
    graph = defaultdict(list)
    for face, cells in faces.items():
        if len(cells) > 2:
            raise ValueError("Pressure splitting requires manifold simplex facets.")
        if len(cells) == 1:
            if face in metadata:
                raise ValueError("A pressure-jump interface must have cells on both sides.")
            graph[next(iter(cells))]
            continue
        (left, left_centre), (right, right_centre) = cells.items()
        difference = 0.0
        if face in metadata:
            _name, jump, normal, centre = metadata[face]
            a = np.dot(np.asarray(left_centre) - centre, normal)
            b = np.dot(np.asarray(right_centre) - centre, normal)
            if a * b >= 0:
                raise ValueError("The supplied interface normal must separate its two adjacent cells.")
            difference = jump if b > 0 else -jump  # p_right - p_left
        graph[left].append((right, difference))
        graph[right].append((left, -difference))
    values, maximum_residual = {}, 0.0
    for start in graph:
        if start in values:
            continue
        values[start], pending, component = 0.0, [start], []
        while pending:
            cell = pending.pop()
            component.append(cell)
            for neighbour, difference in graph[cell]:
                expected = values[cell] + difference
                if neighbour not in values:
                    values[neighbour] = expected
                    pending.append(neighbour)
                else:
                    residual = abs(values[neighbour] - expected)
                    maximum_residual = max(maximum_residual, residual)
                    if residual > tolerance * max(1.0, abs(expected), abs(values[neighbour])):
                        raise ValueError(
                            "Incompatible prescribed pressure jumps around an interface tip or junction; "
                            "the oriented cycle sum must vanish."
                        )
        mean = np.mean([values[cell] for cell in component])
        for cell in component:
            values[cell] -= mean
    return values, maximum_residual


def _create_pressure_split(stokes, jumps, normals, tolerance):
    """Construct a distributed DG P1 lift and its broken-gradient load."""
    mesh, dm = stokes.mesh, stokes.mesh.dm
    comm = dm.comm.tompi4py()
    if not jumps or set(jumps) != set(normals):
        raise ValueError("Supply nonempty jump and normal mappings with identical interface names.")
    if not stokes.u.continuous or not stokes.p.continuous or stokes.p.degree != 1 or not mesh.isSimplex:
        raise ValueError("Pressure splitting requires continuous P1 pressure on affine simplices.")
    if mesh.degree != 1 or mesh.dim not in (2, 3) or mesh.cdim != mesh.dim:
        raise ValueError("Pressure splitting requires affine 2D triangles or 3D tetrahedra.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite.")
    vstart, vend = dm.getDepthStratum(0)
    cstart, cend = dm.getHeightStratum(0)
    fstart, fend = dm.getHeightStratum(1)
    vertex_ids = np.asarray(dm.getVertexNumbering().getIndices(), dtype=np.int64)
    vertex_ids = np.where(vertex_ids < 0, -vertex_ids - 1, vertex_ids)
    cell_ids = np.asarray(dm.getCellNumbering().getIndices(), dtype=np.int64)
    owned = cell_ids >= 0
    cell_ids = np.where(owned, cell_ids, -cell_ids - 1)
    vertex_numbering = dm.getVertexNumbering().getIndices()
    owner_limits = np.cumsum(comm.allgather(int(np.count_nonzero(vertex_numbering >= 0))))
    coordinate_dm = mesh._basis_coordinate_dm(1, True)
    section = coordinate_dm.getLocalSection()
    coordinates = np.asarray(mesh._get_coords_for_basis(1, True))
    vertex_coords = {v: coordinates[section.getOffset(v) // mesh.cdim] for v in range(vstart, vend)}

    def vertices(point):
        return [int(v) for v in dm.getTransitiveClosure(point)[0] if vstart <= v < vend]

    def vertex_id(vertex):
        return int(vertex_ids[vertex - vstart])

    def vertex_owner(vertex):
        return int(np.searchsorted(owner_limits, vertex_id(vertex), side="right"))

    marked = np.zeros(dm.getChart()[1], dtype=np.int32)
    interface_facets = {}
    for name in sorted(jumps):
        if name not in mesh.boundaries.__members__:
            raise ValueError(f"Unknown mesh interface {name!r}.")
        points = label_stratum_indices(dm.getLabel(name), mesh.boundaries[name].value)
        facets = [int(f) for f in points if fstart <= f < fend]
        interface_facets[name] = facets
        for facet in facets:
            marked[vertices(facet)] = 1
    # A rank can share an interface vertex without holding any interface facet.
    if comm.size > 1:
        sf, roots = dm.getPointSF(), marked.copy()
        sf.reduceBegin(MPI.INT32_T, marked, roots, MPI.MAX)
        sf.reduceEnd(MPI.INT32_T, marked, roots, MPI.MAX)
        sf.bcastBegin(MPI.INT32_T, roots, marked, MPI.MAX)
        sf.bcastEnd(MPI.INT32_T, roots, marked, MPI.MAX)
        np.maximum(marked, roots, out=marked)

    messages = [[] for _ in range(comm.size)]
    for name, facets in interface_facets.items():
        points = sorted({v for f in facets for v in vertices(f)})
        xyz = np.array([vertex_coords[v] for v in points]).reshape(-1, mesh.dim)
        error = None
        try:
            jump_values = _sample(mesh, jumps[name], xyz, 1).ravel()
        except (ValueError, TypeError) as exc:
            error = str(exc)
        _collective_error(comm, error)
        values = dict(zip(points, jump_values))
        centres = np.array([np.mean([vertex_coords[v] for v in vertices(f)], axis=0)
                            for f in facets]).reshape(-1, mesh.dim)
        error = None
        try:
            normal_values = _sample(mesh, normals[name], centres, mesh.dim)
        except (ValueError, TypeError) as exc:
            error = str(exc)
        _collective_error(comm, error)
        for facet, centre, normal in zip(facets, centres, normal_values):
            face = tuple(sorted(vertex_id(v) for v in vertices(facet)))
            for vertex in vertices(facet):
                messages[vertex_owner(vertex)].append(
                    ("jump", vertex_id(vertex), face, name, float(values[vertex]), normal, centre)
                )
    for cell in range(cstart, cend):
        cell_vertices = vertices(cell)
        if len(cell_vertices) != mesh.dim + 1:
            raise ValueError("Pressure splitting requires simplex cells.")
        centre = np.mean([vertex_coords[v] for v in cell_vertices], axis=0)
        cid = int(cell_ids[cell - cstart])
        for vertex in cell_vertices:
            if marked[vertex]:
                messages[vertex_owner(vertex)].append(("request", vertex_id(vertex), cid))
        if not owned[cell - cstart]:
            continue
        for facet in dm.getCone(cell):
            face_vertices = vertices(facet)
            face = tuple(sorted(vertex_id(v) for v in face_vertices))
            for vertex in face_vertices:
                if marked[vertex]:
                    messages[vertex_owner(vertex)].append(
                        ("cell", vertex_id(vertex), face, cid, centre)
                    )

    stars, requests, metadata = defaultdict(lambda: defaultdict(dict)), [], defaultdict(dict)
    error = None
    for rank, records in enumerate(comm.alltoall(messages)):
        for kind, vid, *record in records:
            if kind == "request":
                requests.append((rank, vid, record[0]))
            elif kind == "cell":
                face, cid, centre = record
                stars[vid][face][cid] = centre
            else:
                face, name, jump, normal, centre = record
                previous = metadata[vid].get(face)
                if previous is not None and (previous[0] != name or
                        abs(previous[1] - jump) > tolerance * max(1.0, abs(jump)) or
                        np.dot(previous[2], normal) <= 0):
                    error = "Conflicting jump values or orientations on a shared interface facet."
                metadata[vid][face] = (name, jump, normal, centre)
    offsets, residual, facet_count = {}, 0.0, 0
    try:
        for vid, faces in stars.items():
            offsets[vid], star_residual = _star_values(faces, metadata[vid], tolerance)
            residual = max(residual, star_residual)
            facet_count += sum(vid == min(face) for face in metadata[vid])
    except ValueError as exc:
        error = str(exc)
    _collective_error(comm, error)
    global_facets = comm.allreduce(facet_count, op=MPI.SUM)
    if not global_facets:
        raise ValueError("No interface facets found; the interface must be fitted and labelled.")
    replies = [[] for _ in range(comm.size)]
    for rank, vid, cid in requests:
        replies[rank].append((vid, cid, offsets[vid][cid]))
    local_values = {(vid, cid): value for records in comm.alltoall(replies)
                    for vid, cid, value in records}

    lift = uw.discretisation.MeshVariable("PressureLift", mesh, 1, degree=1, continuous=False)
    force = uw.discretisation.MeshVariable("PressureLiftForce", mesh, mesh.dim,
                                         degree=0, continuous=False)
    lift_rows = mesh._cell_node_indices(1, False).reshape(-1, mesh.dim + 1)
    force_rows = mesh._cell_node_indices(0, False).reshape(-1)
    lift_values, force_values = np.zeros_like(lift.array), np.zeros_like(force.array)
    integral = 0.0
    for cell in range(cstart, cend):
        ci, cid = cell - cstart, int(cell_ids[cell - cstart])
        points = vertices(cell)
        xyz = np.array([vertex_coords[v] for v in points])
        values = np.array([local_values.get((vertex_id(v), cid), 0.0) for v in points])
        coefficients = np.linalg.solve(np.column_stack((np.ones(mesh.dim + 1), xyz)), values)
        rows = lift_rows[ci]
        lift_values[rows, 0, 0] = coefficients[0] + np.asarray(lift.coords)[rows] @ coefficients[1:]
        force_values[force_rows[ci], 0, :] = -coefficients[1:]
        if owned[ci]:
            volume = abs(np.linalg.det(xyz[1:] - xyz[0])) / factorial(mesh.dim)
            integral += volume * values.mean()
    lift.array[...] = lift_values
    force.array[...] = force_values
    integral = float(comm.allreduce(integral, op=MPI.SUM))
    checks = dict(interface_facets=int(global_facets),
                  maximum_jump_residual=float(comm.allreduce(residual, op=MPI.MAX)))
    return _PressureSplit(lift.sym[0], lift, force, -lift.sym[0] * mesh.Gamma,
                          stokes.p.sym[0] + lift.sym[0], integral, checks)
