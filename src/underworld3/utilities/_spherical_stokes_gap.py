r"""Opt-in affine P2/P1 spherical-gap mixed Stokes assembly.

The physical inner cut integrates each tetrahedron's restriction outside the
inner sphere, in both momentum and continuity. Wholly void tetrahedra have
zero physical support. Outer gaps retain their adjacent-cell continuation.
Free slip uses scalar P2 normal multipliers.
Zero slip uses three Cartesian multiplier components with mapped basis
(R/|y|) N_i(y) on each planar facet parameter y. This custom mapping contains
exact sphere normals, preserving the paired pressure/reaction gauge; its
uniform stability is unproven. No polar tangent frame or pressure filtering is
used. Returned reactions are physical values at projected boundary nodes.

The gap extension and weak normal constraint are inspired by Collins et al.,
Gap-SBM (2025), https://arxiv.org/abs/2508.09613, and Urquiza, Garon and Farinas,
JCP 256 (2014), https://doi.org/10.1016/j.jcp.2013.08.045. They do not establish
stability of this three-dimensional mixed space.
The cited Gap-SBM analysis is two-dimensional and does not supply the curtain
terms or mapped Cartesian multipliers used here.

For adjacent cap extensions A/B, let [v]=v_A-v_B and {q}=(q_A+q_B)/2.
With n pointing A to B and cap sign s, the additional continuity form is
+s integral({q} [v].n). The symmetric viscous correction is
-s integral([v].{2 mu epsilon(u) n} + [u].{2 mu epsilon(v) n}).
These cancel the artificial-curtain fluxes for smooth exact fields and retain
all three rigid rotations. These identities do not prove coercivity on ker([B; C]) or an
inf-sup bound. Both require independent measured checks before acceptance.
"""

from contextlib import ExitStack
from itertools import combinations
from underworld3.utilities._spherical_stokes_state import SphericalSolveState
from underworld3.utilities._stokes_observer import solve_with_observer
from typing import NamedTuple
import time

import numpy as np
import sympy as sp
from mpi4py import MPI
from petsc4py import PETSc
import underworld3 as uw

from underworld3.utilities._field_layout import _layout, _scatter_values
from underworld3.utilities._spherical_gap import _spherical_facet_quadrature
from underworld3.utilities._spherical_cut_cell import (
    _inner_cell_relation, _spherical_cut_cell_quadrature, _spherical_cut_cell_surface,
)
from underworld3.utilities._spherical_wall_trace import _inner_wall_cell_facets
from underworld3.utilities._spherical_gap_preconditioner import _SphericalGapNormBlocks
from underworld3.utilities.memprobe import snapshot as memory_snapshot
from underworld3.utilities._spherical_curtain import (
    _spherical_curtain_quadrature, _curtain_stokes_blocks,
)
from underworld3.utilities._tetrahedral_gap import (
    _tetrahedral_p2_basis, _tetrahedron_quadrature, _tetrahedral_stokes_blocks,
)


def _rss_mib(observer, phase):
    snapshot = memory_snapshot(full=False)
    if observer is not None and hasattr(observer, 'phase'):
        observer.phase(phase, snapshot)
    return snapshot['rss_mb']


class _SphericalGapResult(NamedTuple):
    normal_reaction: object
    tangential_reaction: object
    checks: dict


def _canonical_rows(mesh, field, vertices):
    rows = mesh._cell_node_indices(field.degree, field.continuous)
    nodes = vertices
    if field.degree == 2:
        edges = [(vertices[:, i]+vertices[:, j])/2 for i, j in combinations(range(4), 2)]
        nodes = np.concatenate((vertices, np.stack(edges, axis=1)), axis=1)
    distance = np.linalg.norm(np.asarray(field.coords)[rows][:, :, None]-nodes[:, None], axis=-1)
    error = distance.min(axis=1).max(initial=0)
    if mesh.dm.comm.tompi4py().allreduce(error, op=MPI.MAX) > 1e-10:
        raise ValueError("Spherical gap integration requires affine P2/P1 coordinates.")
    return np.take_along_axis(rows, distance.argmin(axis=1), axis=1)


def _coordinate_values(expression, points, components, mesh):
    expression = uw.function.expressions._unwrap_for_compilation(
        expression, keep_constants=False, return_self=False,
    )
    if any(hasattr(type(atom), "meshvar") for atom in expression.atoms(sp.Function)):
        raise ValueError("Gap force and pressure lifting must be coordinate-only expressions.")
    if not expression.free_symbols and not expression.atoms(sp.Function):
        return np.broadcast_to(np.asarray(expression, dtype=float).reshape(components),
                               (*points.shape[:-1], components)).copy()
    if points.size == 0:
        return np.empty((*points.shape[:-1], components))
    values = uw.function.evaluate(expression, np.ascontiguousarray(points.reshape(-1, 3)), coord_sys=mesh.N)
    return np.asarray(values).reshape(*points.shape[:-1], components)


class _SphericalSourceSnapshot(NamedTuple):
    """Cell polynomials prepared before rank-dependent integration batches.

    The geometry must remain fixed while the snapshot is used. Evaluation has
    no mesh, variable or communicator access, including for empty batches.
    """

    degree: int
    vertices: np.ndarray
    coefficients: np.ndarray

    def evaluate(self, points, cells):
        if self.degree == 0:
            return np.broadcast_to(self.coefficients[cells, :1], points.shape).copy()
        shape, _, bary = _tetrahedral_p2_basis(self.vertices[cells], points)
        basis = shape if self.degree == 2 else bary
        return np.einsum("eqi,eid->eqd", basis, self.coefficients[cells])


def _prepare_source_snapshot(expression, vertices, mesh):
    """Collectively snapshot one prescribed DG0/P1/P2 vector field.

    All ranks must call this before any unequal or empty integration batches:
    public variable-array reads synchronize ghost values on every access.
    Coordinate-only expressions return ``None`` and retain their usual path.
    Coefficients are independent of subsequent source-field value changes.
    """
    expression = uw.function.expressions._unwrap_for_compilation(
        expression, keep_constants=False, return_self=False,
    )
    variables = {type(atom).meshvar() for atom in expression.atoms(sp.Function)
                 if hasattr(type(atom), "meshvar")}
    if not variables:
        return None
    if len(variables) != 1:
        raise ValueError("Use one unmodified DG0/P1/P2 vector source field or a coordinate force.")
    field = variables.pop()
    if (field.mesh is not mesh or field.degree not in (0, 1, 2)
            or field.num_components != 3 or tuple(expression) != tuple(field.sym)):
        raise ValueError("Use one unmodified DG0/P1/P2 vector source field on the Stokes mesh.")
    rows = mesh._cell_node_indices(field.degree, field.continuous)
    nodal_values = np.asarray(field.array)[:, 0, :]
    if field.degree == 0:
        coefficients = nodal_values[rows].copy()
    else:
        coordinates = np.asarray(field.coords)
        coefficients = np.empty((*rows.shape, 3), dtype=np.result_type(float, nodal_values.dtype))
        # Keep the nodal design matrices bounded even on production meshes.
        for start in range(0, len(vertices), 256):
            cells = slice(start, start+256)
            shape, _, bary = _tetrahedral_p2_basis(vertices[cells], coordinates[rows[cells]])
            design = shape if field.degree == 2 else bary
            coefficients[cells] = np.linalg.solve(design, nodal_values[rows[cells]])
    coefficients.setflags(write=False)
    return _SphericalSourceSnapshot(field.degree, vertices, coefficients)


def _source_values(expression, vertices, points, cells, mesh):
    """Evaluate one batch; collective for an FE source even if the batch is empty.

    Unequal batch loops must instead collectively prepare one source snapshot,
    then call its NumPy-only ``evaluate`` method from their local loops.
    """
    snapshot = _prepare_source_snapshot(expression, vertices, mesh)
    if snapshot is None:
        return _coordinate_values(expression, points, 3, mesh)
    return snapshot.evaluate(points, cells)


def _wall_faces(mesh, vertices, urows, owned, name, radius):
    if name not in mesh.boundaries.__members__:
        raise ValueError(f"Unknown spherical wall: {name}")
    dm = mesh.dm
    vstart, vend = dm.getDepthStratum(0)
    cstart, _ = dm.getHeightStratum(0)
    coordinates, section = dm.getCoordinatesLocal().array, dm.getCoordinateSection()
    cells, facets, rows, positions = [], [], [], []
    edge_pairs = list(combinations(range(4), 2))
    for cell in owned:
        for face in dm.getCone(int(cell)+cstart):
            if dm.getLabel(name).getValue(face) != mesh.boundaries[name].value:
                continue
            closure = [v for v in dm.getTransitiveClosure(face)[0] if vstart <= v < vend]
            triangle = np.array([coordinates[section.getOffset(v):section.getOffset(v)+3] for v in closure])
            if triangle.shape != (3, 3) or not np.allclose(np.linalg.norm(triangle, axis=1), radius, rtol=1e-10):
                raise ValueError(f"Wall {name} must consist of triangles on radius {radius}.")
            corner = np.linalg.norm(triangle[:, None]-vertices[cell, None], axis=-1).argmin(axis=1)
            trace = list(corner)+[4+edge_pairs.index(tuple(sorted(pair))) for pair in combinations(corner, 2)]
            cells.append(cell)
            facets.append(triangle)
            rows.append(urows[cell, trace])
            positions.append(trace)
    comm = dm.comm.tompi4py()
    if comm.allreduce(len(cells), op=MPI.SUM) == 0:
        raise ValueError(f"No owned spherical wall facets found for {name}.")
    return dict(cells=np.asarray(cells, dtype=int), facets=np.asarray(facets).reshape(-1, 3, 3),
                rows=np.asarray(rows, dtype=int).reshape(-1, 6),
                positions=np.asarray(positions, dtype=int).reshape(-1, 6))


def _compressed_boundary_ids(scalar_ids, size, walls, comm, cleanup):
    """Compress distributed boundary nodes; communicate counts and ghost queries only."""
    marker = PETSc.Vec().createMPI(size, comm=comm)
    cleanup.callback(marker.destroy)
    marker.set(0)
    for wall in walls:
        ids = scalar_ids[wall["rows"]].ravel()
        marker.setValues(ids, np.ones(len(ids)), addv=PETSc.InsertMode.ADD_VALUES)
    marker.assemble()
    selected = marker.array > 0
    mpi = comm.tompi4py()
    count = int(selected.sum())
    offset = mpi.scan(count, op=MPI.SUM)-count
    total = mpi.allreduce(count, op=MPI.SUM)
    marker.array[:] = -1
    marker.array[selected] = offset+np.arange(count)
    ids = _scatter_values(marker, scalar_ids).astype(PETSc.IntType)
    return ids, total


def _curtain_pairs(walls, vertices, uid, pid, mesh):
    """Route each shared boundary edge to one owner without a surface gather."""
    comm = mesh.dm.comm.tompi4py()
    messages = [[] for _ in range(comm.size)]
    for wall in walls:
        for cell, facet, positions in zip(wall["cells"], wall["facets"], wall["positions"], strict=True):
            corners = positions[:3]
            global_vertices = pid[cell, corners]
            facet_key = tuple(sorted(map(int, global_vertices)))
            for pair in combinations(range(3), 2):
                order = sorted(pair, key=lambda i: global_vertices[i])
                edge_key = tuple(int(global_vertices[i]) for i in order)
                destination = ((edge_key[0]*73856093) ^ (edge_key[1]*19349663)) % comm.size
                messages[destination].append((edge_key, facet_key, wall["radius"], wall["sign"],
                    facet[order], facet.mean(axis=0), vertices[cell], uid[cell], pid[cell]))
    groups = {}
    for records in comm.alltoall(messages):
        for record in records:
            groups.setdefault(record[0], []).append(record)
    errors = []
    pairs = []
    for edge, records in groups.items():
        if len(records) != 2 or records[0][1] == records[1][1]:
            errors.append(f"Spherical wall edge {edge} must have exactly two distinct triangles.")
            continue
        a, b = sorted(records, key=lambda item: item[1])
        if a[2:4] != b[2:4] or not np.allclose(a[4], b[4], rtol=0, atol=1e-12):
            errors.append(f"Spherical wall edge {edge} has inconsistent geometry.")
            continue
        pairs.append(dict(radius=a[2], sign=a[3], edge=a[4], center_a=a[5], center_b=b[5],
                          vertices_a=a[6], vertices_b=b[6],
                          uid=np.concatenate((a[7], b[7])).astype(PETSc.IntType),
                          pid=np.concatenate((a[8], b[8])).astype(PETSc.IntType)))
    collective_errors = comm.allgather(errors[:1])
    if any(collective_errors):
        raise ValueError(next(error[0] for error in collective_errors if error))
    return pairs


def _create_matrix(size, uid, pid, walls, curtains, comm):
    graph = PETSc.Mat().create(comm=comm)
    graph.setSizes((size, size)); graph.setType(PETSc.Mat.Type.PREALLOCATOR); graph.setUp()
    for u, p in zip(uid, pid, strict=True):
        graph.setValues(u, u, np.zeros((30, 30)))
        graph.setValues(p, u, np.zeros((4, 30)))
        graph.setValues(u, p, np.zeros((30, 4)))
    for uids, lids in walls:
        for u, l in zip(uids, lids, strict=True):
            graph.setValues(l, u, np.zeros((len(l), 30)))
            graph.setValues(u, l, np.zeros((30, len(l))))
    for curtain in curtains:
        u, p = curtain["uid"], curtain["pid"]
        graph.setValues(u, u, np.zeros((60, 60)))
        graph.setValues(p, u, np.zeros((8, 60)))
        graph.setValues(u, p, np.zeros((60, 8)))
    lo, hi = graph.getOwnershipRange()
    for row in range(lo, hi):
        graph.setValue(row, row, 0)
    graph.assemble()
    matrix = PETSc.Mat().create(comm=comm)
    matrix.setSizes((size, size)); matrix.setType(PETSc.Mat.Type.AIJ)
    graph.preallocatorPreallocate(matrix, fill=True); graph.destroy()
    # The filled graph includes both curtain transposes and every gauge
    # diagonal. A missing connection must fail instead of reallocating silently.
    matrix.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
    matrix.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR, True)
    return matrix


def _insert(matrix, uid, pid, blocks, viscosity, sign):
    for u, p, a, b in zip(uid, pid, blocks["A"], blocks["B"], strict=True):
        matrix.setValues(u, u, np.ascontiguousarray(sign*viscosity*a), addv=PETSc.InsertMode.ADD_VALUES)
        matrix.setValues(p, u, np.ascontiguousarray(sign*b), addv=PETSc.InsertMode.ADD_VALUES)
        matrix.setValues(u, p, np.ascontiguousarray(sign*b.T), addv=PETSc.InsertMode.ADD_VALUES)


def _add(vector, ids, values):
    for index, value in zip(ids, values, strict=True):
        vector.setValues(index, np.asarray(value).ravel(), addv=PETSc.InsertMode.ADD_VALUES)


def _refine_direct_solution(ksp, pinned, pinned_rhs, matrix, rhs,
                            ungauged, physical, restore_gauge):
    """Check the original equations; refine failed direct solves at most five times.

    Corrections reuse the supplied factorisation of the unchanged pinned system.
    The original RHS is read only. A solution already passing the physical
    residual guard is left bitwise unchanged. Residual checks use ungauged
    iterates for the pinned equations and physical-gauge copies for the original
    equations; a physical gauge is not required to satisfy artificial point pins.
    """
    threshold = 1e-8
    pinned_norm, matrix_norm = pinned.norm(), matrix.norm()
    pinned_rhs_norm = max(pinned_rhs.norm(), np.finfo(float).tiny)
    rhs_norm = max(rhs.norm(), np.finfo(float).tiny)
    history = []
    with ExitStack() as cleanup:
        work = rhs.duplicate(); cleanup.callback(work.destroy)

        def measure(candidate, gauged):
            pinned.mult(candidate, work); work.axpy(-1, pinned_rhs)
            pinned_residual = work.norm()
            matrix.mult(gauged, work); work.axpy(-1, rhs)
            full_residual = work.norm()
            record = dict(
                pinned_relative=pinned_residual/pinned_rhs_norm,
                full_relative=full_residual/rhs_norm,
                pinned_backward=pinned_residual/(pinned_norm*candidate.norm()+pinned_rhs_norm),
                full_backward=full_residual/(matrix_norm*gauged.norm()+rhs_norm),
            )
            if not all(np.isfinite(value) for value in record.values()):
                raise ArithmeticError(f"Nonfinite spherical direct residual: {record}")
            history.append(record)
            return record["full_relative"]

        relative = measure(ungauged, physical)
        if relative <= threshold:
            return history
        correction_rhs = rhs.duplicate(); cleanup.callback(correction_rhs.destroy)
        correction = rhs.duplicate(); cleanup.callback(correction.destroy)
        candidate = rhs.duplicate(); cleanup.callback(candidate.destroy)
        for _ in range(5):
            pinned.mult(ungauged, correction_rhs)
            correction_rhs.scale(-1); correction_rhs.axpy(1, pinned_rhs)
            correction.set(0); ksp.solve(correction_rhs, correction)
            if ksp.getConvergedReason() <= 0:
                raise ArithmeticError(f"Spherical cached-LU correction failed: {history}")
            ungauged.axpy(1, correction)
            ungauged.copy(candidate); restore_gauge(candidate)
            improved = measure(ungauged, candidate)
            if improved >= relative and improved > threshold:
                raise ArithmeticError(f"Spherical direct refinement is nonimproving: {history}")
            candidate.copy(physical)
            if improved <= threshold:
                return history
            relative = improved
        raise ArithmeticError(f"Spherical direct refinement exhausted after five corrections: {history}")


def _solve_constraints(stokes, inner_boundary, outer_boundary, radius_inner, radius_outer,
                       viscosity, pressure_jump, pressure_jump_integral, gap_bodyforce,
                       quadrature, linear_solver_options, batch_size, components, volume_force=None,
                       norm_preconditioner_length=None, radial_quadrature=4):
    assembly_started = time.perf_counter()
    mesh, velocity, pressure = stokes.mesh, stokes.u, stokes.p
    use_norm_pc = norm_preconditioner_length is not None
    if use_norm_pc:
        if not np.isfinite(norm_preconditioner_length) or norm_preconditioner_length <= 0:
            raise ValueError("The norm-preconditioner length must be finite and positive.")
        if viscosity != 1:
            raise ValueError("The measured norm preconditioner currently requires viscosity=1.")
        if linear_solver_options.get("ksp_type") != "fgmres" or linear_solver_options.get("pc_type") != "python":
            raise ValueError("Norm preconditioning requires explicit ksp_type=fgmres and pc_type=python.")
    if (mesh.dim != 3 or mesh.cdim != 3 or mesh.degree != 1 or not mesh.isSimplex
            or velocity.degree != 2 or not velocity.continuous
            or pressure.degree != 1 or not pressure.continuous):
        raise ValueError("Spherical gap Stokes requires affine tetrahedra and continuous P2/P1 fields.")
    if uw.is_nondimensional_scaling_active() or uw.get_default_model().has_units_active():
        raise ValueError("Spherical gap Stokes supports nondimensional data only.")
    if (not np.isfinite(pressure_jump_integral) or not 0 < radius_inner < radius_outer
            or not np.isfinite(viscosity) or viscosity <= 0
            or not isinstance(quadrature, int) or quadrature < 3
            or not isinstance(batch_size, int) or batch_size < 1):
        raise ValueError("Require ordered radii, positive viscosity, quadrature>=3 and positive batch size.")
    if (not isinstance(radial_quadrature, (int, np.integer))
            or isinstance(radial_quadrature, bool) or radial_quadrature < 2):
        raise ValueError("radial_quadrature must be an integer at least two.")
    if (stokes.essential_bcs or stokes.natural_bcs or stokes._rotated_freeslip_bcs
            or getattr(stokes, "_block_constraint_bcs", [])):
        raise ValueError("Spherical gap Stokes replaces wall registration on both complete spheres.")
    if type(stokes.constitutive_model) is not uw.constitutive_models.ViscousFlowModel:
        raise ValueError("Spherical gap Stokes supports the linear ViscousFlowModel only.")
    unwrap = uw.function.expressions._unwrap_for_compilation
    mu = unwrap(stokes.constitutive_model.viscosity, keep_constants=False, return_self=False)
    penalty = unwrap(stokes.penalty, keep_constants=False, return_self=False)
    if mu.free_symbols or float(mu) != viscosity or penalty != 0 or stokes.constraints != sp.Matrix((stokes.div_u,)):
        raise ValueError("Require matching constant viscosity, zero penalty and standard continuity.")
    comm = mesh.dm.comm; mpi = comm.tompi4py()
    # Diagnostic observers may copy borrowed operators at these two stages.
    # No extra matrix assembly or storage is incurred by ordinary solves.
    observer = getattr(stokes, "_stokes_observer", None) or getattr(stokes, "_spherical_gap_observer", None)
    resources = {"rss_start_mib": _rss_mib(observer, "rss_start_mib")}
    vertex_rows = mesh._cell_node_indices(1, True)
    vertices = np.asarray(mesh._get_coords_for_basis(1, True))[vertex_rows]
    urows, prows = _canonical_rows(mesh, velocity, vertices), _canonical_rows(mesh, pressure, vertices)
    umap, nu = _layout(velocity); pmap, npres = _layout(pressure)
    uid = umap.reshape(-1, 3)[urows].reshape(-1, 30)
    pid = pmap[prows]+nu
    owned = np.flatnonzero(mesh._get_owned_cells_mask())
    walls = []
    for name, radius, sign in ((inner_boundary, radius_inner, -1), (outer_boundary, radius_outer, 1)):
        wall = _wall_faces(mesh, vertices, urows, owned, name, radius)
        wall.update(radius=radius, sign=sign)
        walls.append(wall)
    # Inner cuts use the actual piecewise cell restriction. Only the outer
    # continuation has artificial radial curtains.
    curtains = _curtain_pairs(walls[1:], vertices, uid, pid, mesh)
    relation = _inner_cell_relation(vertices, radius_inner)
    inner_cells = owned[relation[owned] == 0]
    void_cells = owned[relation[owned] == -1]
    physical_cells = owned[relation[owned] != -1]
    cut_order = max(12, quadrature)
    with ExitStack() as cleanup:
        compressed, nboundary = _compressed_boundary_ids(umap.reshape(-1, 3)[:, 0]//3, nu//3, walls, comm, cleanup)
        size = nu+npres+components*nboundary
        for wall in walls:
            wall["lid"] = (nu+npres+components*compressed[wall["rows"], None]
                           + np.arange(components)).reshape(-1, 6*components).astype(PETSc.IntType)
        # Retain surface rules for the graph and numeric insertion; regenerate
        # volume rules one cell at a time to bound temporary storage.
        inner_surface = []
        candidates = _inner_wall_cell_facets(mesh, walls[0], vertices[inner_cells])
        inner_graph = []
        for cell, (facets, lids) in zip(inner_cells, candidates, strict=True):
            points, weights, trace_lids, trace_values = _spherical_cut_cell_surface(
                vertices[cell], radius_inner, cut_order, facets, lids)
            inner_surface.append((cell, points, weights, trace_lids, trace_values))
            lids = trace_lids
            inner_graph.append((uid[cell:cell+1], np.unique(lids).reshape(1, -1)))
        missing = mpi.allgather(sum(len(item[1]) == 0 for item in inner_surface))
        if any(missing):
            raise ArithmeticError(f"Physical inner-cut cells have no wall quadrature: {missing}")
        # An inner-wall owner need not contain the physical wall. Preallocate
        # every actual cell/multiplier connection, including remote rows.
        resources["geometry_seconds"] = time.perf_counter()-assembly_started
        resources["rss_after_geometry_mib"] = _rss_mib(observer, "rss_after_geometry_mib")
        graph_started = time.perf_counter()
        matrix = _create_matrix(size, uid[owned], pid[owned],
                                [(uid[w["cells"]], w["lid"]) for w in walls[1:]]+inner_graph, curtains, comm)
        cleanup.callback(matrix.destroy)
        resources["graph_seconds"] = time.perf_counter()-graph_started
        graph_baseline = matrix.getInfo(PETSc.Mat.InfoType.LOCAL)
        resources["rss_after_graph_mib"] = _rss_mib(observer, "rss_after_graph_mib")
        numeric_started = time.perf_counter()
        norm_blocks = None
        if use_norm_pc:
            if observer is not None and hasattr(observer, "phase"):
                _rss_mib(observer, "norm_blocks_begin")
            norm_blocks = _SphericalGapNormBlocks(
                velocity, matrix, nu, npres, components*nboundary, components,
                norm_preconditioner_length,
            )
            cleanup.callback(norm_blocks.destroy)
            if observer is not None and hasattr(observer, "phase"):
                _rss_mib(observer, "norm_blocks_end")
        rhs = matrix.createVecRight(); cleanup.callback(rhs.destroy); rhs.set(0)
        if volume_force is not None:
            if (not isinstance(volume_force, PETSc.Vec) or volume_force.getSize() != nu
                    or MPI.Comm.Compare(volume_force.comm.tompi4py(), mpi) not in (MPI.IDENT, MPI.CONGRUENT)):
                raise ValueError("volume_force must be a full-field velocity Vec on the mesh communicator.")
            # Globally assembled source entries are transferred once, including
            # ownership ranges that differ from the larger mixed vector.
            vlo, vhi = volume_force.getOwnershipRange()
            indices = PETSc.IS().createGeneral(np.arange(vlo, vhi, dtype=PETSc.IntType), comm=comm)
            cleanup.callback(indices.destroy)
            scatter = PETSc.Scatter().create(volume_force, indices, rhs, indices)
            cleanup.callback(scatter.destroy)
            scatter.scatter(volume_force, rhs, addv=PETSc.InsertMode.ADD_VALUES,
                            mode=PETSc.ScatterMode.FORWARD)
        pmean = rhs.duplicate(); cleanup.callback(pmean.destroy); pmean.set(0)
        rotation_weights = []
        for _ in range(3):
            vector = rhs.duplicate(); cleanup.callback(vector.destroy); vector.set(0)
            rotation_weights.append(vector)
        force_expr = unwrap(stokes.bodyforce, keep_constants=False, return_self=False)
        has_volume_bodyforce = force_expr.is_zero_matrix is not True
        source_snapshot = _prepare_source_snapshot(force_expr, vertices, mesh) if has_volume_bodyforce else None
        if has_volume_bodyforce:
            fe = PETSc.FE().createDefault(3, 1, True, mesh.qdegree, "sphere_gap_source_")
            cleanup.callback(fe.destroy)
            native_points, native_weights = fe.getQuadrature().getData()
            xi = (np.asarray(native_points).reshape(-1, 3)+1)/2
            native_bary = np.column_stack((1-xi.sum(axis=1), xi))
        for start in range(0, len(owned), batch_size):
            cells = owned[start:start+batch_size]
            points, weights = _tetrahedron_quadrature(vertices[cells], 4)
            blocks = _tetrahedral_stokes_blocks(vertices[cells], points, weights)
            _insert(matrix, uid[cells], pid[cells], blocks, viscosity, 1)
            if norm_blocks is not None:
                # Retain full background norms on supported cells, including
                # cut cells; a wholly void cell has no physical FE support.
                active = relation[cells] != -1
                if active.any():
                    selection = slice(None) if active.all() else active
                    norm_blocks.add_volume(
                        vertices[cells][selection], points[selection], weights[selection],
                        uid[cells][selection], pid[cells][selection], viscosity*blocks["A"][selection],
                    )
            _add(pmean, pid[cells], blocks["pressure_mean"])
            for axis in range(3):
                _add(rotation_weights[axis], uid[cells], blocks["rotation"][..., axis])
            if has_volume_bodyforce:
                points = np.einsum("qi,eid->eqd", native_bary, vertices[cells])
                determinant = np.abs(np.linalg.det(np.swapaxes(vertices[cells, 1:]-vertices[cells, None, 0], 1, 2)))
                weights = determinant[:, None]*np.asarray(native_weights).reshape(1, -1)/8
                shape, _, _ = _tetrahedral_p2_basis(vertices[cells], points)
                force = (_coordinate_values(force_expr, points, 3, mesh)
                         if source_snapshot is None else source_snapshot.evaluate(points, cells))
                _add(rhs, uid[cells], np.einsum("eq,eqi,eqd->eid", weights, shape, force))
        if observer is not None:
            matrix.assemble()
            observer.background(matrix, nu, npres)
        lift_gap = 0.0
        # Replace each intersected or void background-cell integral by its
        # physical restriction. Every block and moment uses the same geometry.
        # Norm preconditioning retains full background norms on supported cells.
        for cell in np.concatenate((inner_cells, void_cells)):
            full_points, full_weights = _tetrahedron_quadrature(vertices[cell:cell+1], 4)
            full = _tetrahedral_stokes_blocks(vertices[cell:cell+1], full_points, full_weights)
            _insert(matrix, uid[cell:cell+1], pid[cell:cell+1], full, viscosity, -1)
            _add(pmean, pid[cell:cell+1], -full["pressure_mean"])
            for axis in range(3):
                _add(rotation_weights[axis], uid[cell:cell+1], -full["rotation"][..., axis])
            rule = _spherical_cut_cell_quadrature(vertices[cell], radius_inner, cut_order, radial_quadrature)
            points, weights = rule["volume_points"][None], rule["volume_weights"][None]
            if weights.size:
                blocks = _tetrahedral_stokes_blocks(vertices[cell:cell+1], points, weights)
                _insert(matrix, uid[cell:cell+1], pid[cell:cell+1], blocks, viscosity, 1)
                _add(pmean, pid[cell:cell+1], blocks["pressure_mean"])
                for axis in range(3):
                    _add(rotation_weights[axis], uid[cell:cell+1], blocks["rotation"][..., axis])
            # Correct the ordinary bodyforce using its actual prescribed cell
            # polynomial. A caller-supplied vector already owns its physical
            # inner-domain integration; its density cannot be inferred here.
            if has_volume_bodyforce:
                native_x = np.einsum("qi,eid->eqd", native_bary, vertices[cell:cell+1])
                determinant = abs(np.linalg.det((vertices[cell, 1:]-vertices[cell, 0]).T))
                native_w = determinant*np.asarray(native_weights).reshape(1, -1)/8
                force = (_coordinate_values(force_expr, native_x, 3, mesh) if source_snapshot is None
                         else source_snapshot.evaluate(native_x, np.array([cell])))
                shape, _, _ = _tetrahedral_p2_basis(vertices[cell:cell+1], native_x)
                _add(rhs, uid[cell:cell+1], -np.einsum("eq,eqi,eqd->eid", native_w, shape, force))
            if weights.size:
                force = (source_snapshot.evaluate(points, np.array([cell]))
                         if source_snapshot is not None
                         else _coordinate_values(force_expr, points, 3, mesh))
                shape, _, _ = _tetrahedral_p2_basis(vertices[cell:cell+1], points)
                _add(rhs, uid[cell:cell+1], np.einsum("eq,eqi,eqd->eid", weights, shape, force))
                lift_gap += np.sum(weights*_coordinate_values(pressure_jump, points, 1, mesh)[..., 0])
            lift_gap -= np.sum(full_weights*_coordinate_values(pressure_jump, full_points, 1, mesh)[..., 0])
        for cell, points, weights, lids, multiplier in inner_surface:
            shape, _, _ = _tetrahedral_p2_basis(vertices[cell:cell+1], points[None])
            normal = -points/radius_inner
            for ids in np.unique(lids, axis=0):
                selected = np.all(lids == ids, axis=1)
                m = multiplier[selected]
                if components == 1:
                    coupling = np.einsum("q,ql,qi,qd->lid", weights[selected], m,
                                         shape[0, selected], normal[selected]).reshape(6, 30)
                else:
                    coupling = np.einsum("q,ql,qi,cd->lcid", weights[selected], m,
                                         shape[0, selected], np.eye(3)).reshape(18, 30)
                matrix.setValues(ids, uid[cell], np.ascontiguousarray(coupling), addv=PETSc.InsertMode.ADD_VALUES)
                matrix.setValues(uid[cell], ids, np.ascontiguousarray(coupling.T), addv=PETSc.InsertMode.ADD_VALUES)
            lift = _coordinate_values(pressure_jump, points[None], 1, mesh)[0, :, 0]
            _add(rhs, uid[cell:cell+1], np.einsum("q,qi,q,qd->id", weights, shape[0], lift, normal)[None])
        # The positive multiplier norm belongs to the original surface space,
        # even though its velocity trace is evaluated on actual cut cells.
        if norm_blocks is not None:
            wall = walls[0]
            for start in range(0, len(wall["cells"]), batch_size):
                sl = slice(start, start+batch_size)
                cells = wall["cells"][sl]
                rule = _spherical_facet_quadrature(wall["facets"][sl], radius_inner, quadrature, radial_quadrature)
                planar, _, _ = _tetrahedral_p2_basis(vertices[cells], rule["planar_points"])
                multiplier = np.take_along_axis(planar, wall["positions"][sl, None], axis=2)
                if components == 3:
                    multiplier *= radius_inner/np.linalg.norm(rule["planar_points"], axis=-1)[..., None]
                norm_blocks.add_surface(rule["surface_weights"], multiplier, wall["lid"][sl])
        for wall in walls[1:]:
            for start in range(0, len(wall["cells"]), batch_size):
                sl = slice(start, start+batch_size)
                cells, sign = wall["cells"][sl], wall["sign"]
                rule = _spherical_facet_quadrature(
                    wall["facets"][sl], wall["radius"], quadrature, radial_quadrature,
                )
                points = rule["gap_points"].reshape(len(cells), -1, 3)
                weights = rule["gap_weights"].reshape(len(cells), -1)
                blocks = _tetrahedral_stokes_blocks(vertices[cells], points, weights)
                _insert(matrix, uid[cells], pid[cells], blocks, viscosity, sign)
                _add(pmean, pid[cells], sign*blocks["pressure_mean"])
                for axis in range(3):
                    _add(rotation_weights[axis], uid[cells], sign*blocks["rotation"][..., axis])
                shape, _, _ = _tetrahedral_p2_basis(vertices[cells], points)
                force = _coordinate_values(gap_bodyforce, points, 3, mesh)
                _add(rhs, uid[cells], sign*np.einsum("eq,eqi,eqd->eid", weights, shape, force))
                lift_gap += sign*np.sum(weights*_coordinate_values(pressure_jump, points, 1, mesh)[..., 0])
                physical = rule["surface_points"]
                shape, _, _ = _tetrahedral_p2_basis(vertices[cells], physical)
                planar, _, _ = _tetrahedral_p2_basis(vertices[cells], rule["planar_points"])
                multiplier = np.take_along_axis(planar, wall["positions"][sl, None], axis=2)
                normal = sign*rule["surface_normals"]
                if components == 1:
                    coupling = np.einsum("eq,eql,eqi,eqd->elid", rule["surface_weights"], multiplier, shape, normal)
                else:
                    multiplier *= wall["radius"]/np.linalg.norm(rule["planar_points"], axis=-1)[..., None]
                    coupling = np.einsum("eq,eql,eqi,cd->elcid", rule["surface_weights"], multiplier, shape, np.eye(3))
                if norm_blocks is not None:
                    norm_blocks.add_surface(rule["surface_weights"], multiplier, wall["lid"][sl])
                coupling = coupling.reshape(-1, 6*components, 30)
                for u, l, c in zip(uid[cells], wall["lid"][sl], coupling, strict=True):
                    matrix.setValues(l, u, np.ascontiguousarray(c), addv=PETSc.InsertMode.ADD_VALUES)
                    matrix.setValues(u, l, np.ascontiguousarray(c.T), addv=PETSc.InsertMode.ADD_VALUES)
                lift = _coordinate_values(pressure_jump, physical, 1, mesh)[..., 0]
                _add(rhs, uid[cells], np.einsum("eq,eqi,eq,eqd->eid", rule["surface_weights"], shape, lift, normal))
        for curtain in curtains:
            rule = _spherical_curtain_quadrature(curtain["edge"], curtain["center_a"],
                                                 curtain["radius"], quadrature, 4)
            if rule["normal"]@curtain["center_b"] <= 0:
                raise ValueError("Paired spherical patches must lie on opposite sides of their curtain.")
            blocks = _curtain_stokes_blocks(curtain["vertices_a"], curtain["vertices_b"],
                                            rule["points"], rule["weights"], rule["normal"], curtain["sign"])
            u, p = curtain["uid"], curtain["pid"]
            matrix.setValues(u, u, np.ascontiguousarray(viscosity*blocks["A"]), addv=PETSc.InsertMode.ADD_VALUES)
            matrix.setValues(p, u, np.ascontiguousarray(blocks["B"]), addv=PETSc.InsertMode.ADD_VALUES)
            matrix.setValues(u, p, np.ascontiguousarray(blocks["B"].T), addv=PETSc.InsertMode.ADD_VALUES)
        matrix.assemble()
        for vector in (rhs, pmean, *rotation_weights):
            vector.assemble()
        # Values on nodes with no physical cell support are undefined. Remove
        # these algebraic unknowns by identity completion, not a wall penalty.
        support = rhs.duplicate(); cleanup.callback(support.destroy); support.set(0)
        supported = np.unique(np.concatenate((uid[physical_cells].ravel(), pid[physical_cells].ravel())))
        support.setValues(supported, np.ones(len(supported)), addv=PETSc.InsertMode.ADD_VALUES)
        support.assemble()
        support_lo, support_hi = support.getOwnershipRange()
        local_rows = np.arange(support_lo, support_hi, dtype=PETSc.IntType)
        unsupported = local_rows[(support.array == 0) & (local_rows < nu+npres)]
        unsupported_load = np.max(np.abs(rhs.array[unsupported-support_lo]), initial=0.)
        unsupported_load = mpi.allreduce(unsupported_load, op=MPI.MAX)
        if unsupported_load > 1e-10*max(rhs.norm(), np.finfo(float).tiny):
            raise ValueError("Prescribed force has entries on nodes with no physical support.")
        matrix.zeroRowsColumns(unsupported, diag=1)
        for vector in (rhs, pmean, *rotation_weights):
            vector.setValues(unsupported, np.zeros(len(unsupported))); vector.assemble()
        resources["numeric_assembly_seconds"] = time.perf_counter()-numeric_started
        resources["rss_after_numeric_assembly_mib"] = _rss_mib(observer, "rss_after_numeric_assembly_mib")
        assembled_info = matrix.getInfo(PETSc.Mat.InfoType.LOCAL)
        graph_checks = {
            "matrix_mallocs_preallocation": mpi.allreduce(int(graph_baseline["mallocs"]), op=MPI.SUM),
            "matrix_mallocs_after_assembly": mpi.allreduce(int(assembled_info["mallocs"]), op=MPI.SUM),
            "matrix_dynamic_allocations": mpi.allreduce(int(assembled_info["mallocs"]-graph_baseline["mallocs"]), op=MPI.SUM),
            "matrix_nz_allocated": mpi.allreduce(int(assembled_info["nz_allocated"]), op=MPI.SUM),
            "matrix_nz_used": mpi.allreduce(int(assembled_info["nz_used"]), op=MPI.SUM),
            "matrix_memory_bytes": mpi.allreduce(assembled_info["memory"], op=MPI.SUM),
        }
        if graph_checks["matrix_dynamic_allocations"] != 0:
            raise ArithmeticError(f"Spherical assembly changed its preallocated graph: {graph_checks}")
        algebraic_started = time.perf_counter()
        pressure_mode = rhs.duplicate(); cleanup.callback(pressure_mode.destroy); pressure_mode.set(0)
        lo, hi = pressure_mode.getOwnershipRange()
        pids = np.arange(max(lo, nu), min(hi, nu+npres), dtype=PETSc.IntType)
        pressure_mode.setValues(pids, np.ones(len(pids)))
        for wall in walls:
            rows = wall["rows"].ravel()
            lids = wall["lid"].reshape(-1, components)
            if components == 1:
                values = np.ones((len(rows), 1))
            else:
                values = wall["sign"]*np.asarray(velocity.coords)[rows]/wall["radius"]
            pressure_mode.setValues(lids.ravel(), values.ravel())
        pressure_mode.assemble()
        coordinates = np.asarray(velocity.coords)
        nodal_rotations = np.stack([np.cross(axis, coordinates) for axis in np.eye(3)], axis=-1).reshape(-1, 3)
        modes = [pressure_mode]
        for axis in range(3):
            vector = rhs.duplicate(); cleanup.callback(vector.destroy); vector.set(0)
            vector.setValues(umap, nodal_rotations[:, axis]); vector.assemble()
            if components == 1:
                modes.append(vector)
        for vector in modes:
            vector.setValues(unsupported, np.zeros(len(unsupported))); vector.assemble()
        work = rhs.duplicate(); cleanup.callback(work.destroy)
        matrix_norm = matrix.norm()
        null_errors = []
        for vector in modes:
            matrix.mult(vector, work)
            null_errors.append(work.norm()/(matrix_norm*vector.norm()))
        if max(null_errors) > 1e-10:
            raise ArithmeticError(f"Spherical pressure/rotation nullspace failed: {null_errors}")
        if not np.isfinite(rhs.norm()):
            raise ArithmeticError("Spherical assembled force is nonfinite.")
        compatibility = [abs(rhs.dot(vector))/vector.norm() for vector in modes]
        if max(compatibility) > 1e-10*max(rhs.norm(), 1):
            raise ValueError(f"Spherical force is incompatible with physical null modes: {compatibility}")
        if observer is not None:
            observer.physical(
                matrix, nu, npres, pressure_mode, modes[1:],
                rotation_weights, walls, components,
            )
        context = None
        removed_relative = 0.0
        physical_space = None
        if use_norm_pc:
            norm_blocks.assemble()
            orthogonal_modes = []
            for original in modes:
                vector = original.copy(); cleanup.callback(vector.destroy)
                for previous in orthogonal_modes:
                    vector.axpy(-previous.dot(vector), previous)
                vector.scale(1/vector.norm())
                orthogonal_modes.append(vector)
            physical_space = PETSc.NullSpace().create(vectors=orthogonal_modes, constant=False, comm=comm)
            cleanup.callback(physical_space.destroy)
            if not physical_space.test(matrix):
                raise ArithmeticError("Physical nullspace fails the original spherical operator test.")
            matrix.setNullSpace(physical_space); matrix.setTransposeNullSpace(physical_space)
            pinned = matrix
            pinned_rhs = rhs.copy(); cleanup.callback(pinned_rhs.destroy)
            physical_space.remove(pinned_rhs)
            removed = pinned_rhs.copy(); cleanup.callback(removed.destroy); removed.axpy(-1, rhs)
            removed_relative = removed.norm()/max(rhs.norm(), np.finfo(float).tiny)
            if removed_relative > 1e-10:
                raise ValueError(f"Force is not roundoff-compatible with physical null modes: {removed_relative}")
            pins = np.empty(0, dtype=PETSc.IntType)
        else:
            pins = [nu]
            if components == 1:
                residual_rows = nodal_rotations.copy()
                for _ in range(3):
                    magnitudes = np.linalg.norm(residual_rows, axis=1)
                    selected = np.arange(len(umap))
                    index = selected[np.argmax(magnitudes[selected])] if len(selected) else None
                    candidate = (float(magnitudes[index]), int(umap[index]), residual_rows[index].copy()) if index is not None else (-1, nu, np.zeros(3))
                    magnitude, pin, direction = max(mpi.allgather(candidate), key=lambda item: (item[0], -item[1]))
                    if magnitude < 1e-12:
                        raise ArithmeticError("Could not select three independent rotation gauges.")
                    pins.append(pin); direction /= magnitude
                    residual_rows -= np.outer(residual_rows@direction, direction)
            pins = np.asarray(pins, dtype=PETSc.IntType)
            pinned = matrix.copy(); cleanup.callback(pinned.destroy); pinned.zeroRowsColumns(pins, diag=1)
            pinned_rhs = rhs.copy(); cleanup.callback(pinned_rhs.destroy)
            local_pins = pins[(pins >= lo) & (pins < hi)]
            pinned_rhs.setValues(local_pins, np.zeros(len(local_pins))); pinned_rhs.assemble()
        resources["algebraic_preparation_seconds"] = time.perf_counter()-algebraic_started
        assembly_seconds = mpi.allreduce(time.perf_counter()-assembly_started, op=MPI.MAX)
        resources["rss_before_setup_mib"] = _rss_mib(observer, "rss_before_setup_mib")
        setup_started = time.perf_counter()
        ksp = PETSc.KSP().create(comm=comm); cleanup.callback(ksp.destroy); ksp.setOperators(pinned)
        prefix = f"spherical_gap_{stokes.instance_number}_"; ksp.setOptionsPrefix(prefix)
        options = PETSc.Options(prefix)
        for key, value in linear_solver_options.items():
            options[key] = value
            cleanup.callback(options.delValue, key)
        ksp.setFromOptions()
        if use_norm_pc:
            context = norm_blocks.context(umap, coordinates, nodal_rotations, physical_space, prefix+"norm_velocity_")
            cleanup.callback(context.destroy, None)
            context.observer = observer if hasattr(observer, "pc_started") else None
            ksp.getPC().setPythonContext(context)
            ksp.setPCSide(PETSc.PC.Side.RIGHT)
            ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
        ksp.setUp()  # Include direct factorisation in setup, rather than the solve timer.
        if use_norm_pc:
            ksp.view()  # PETSc exposes the effective restart in its view, not a Python getter.
        setup_seconds = mpi.allreduce(time.perf_counter()-setup_started, op=MPI.MAX)
        resources["rss_after_setup_mib"] = _rss_mib(observer, "rss_after_setup_mib")
        solve_started = time.perf_counter()
        solution = rhs.duplicate(); cleanup.callback(solution.destroy)
        state = SphericalSolveState(dict(
            matrix=matrix, rhs=rhs, mpi=mpi, pressure_jump_integral=pressure_jump_integral,
            lift_gap=lift_gap, pmean=pmean, pressure_mode=pressure_mode, modes=modes,
            rotation_weights=rotation_weights, components=components, nu=nu, npres=npres,
            nboundary=nboundary, umap=umap, pmap=pmap, compressed=compressed,
            coordinates=coordinates, velocity=velocity, pressure=pressure, stokes=stokes,
            mesh=mesh, radius_inner=radius_inner, radius_outer=radius_outer,
        ))
        if observer is not None and hasattr(observer, "solve_started"):
            solve_with_observer(ksp, pinned_rhs, solution, state, observer)
        else:
            ksp.solve(pinned_rhs, solution)
        if ksp.getConvergedReason() <= 0:
            pc = ksp.getPC()
            factor_details = None
            if pc.getType() == "lu" and pc.getFactorSolverType() == "mumps":
                factor = pc.getFactorMatrix()
                factor_details = (factor.getMumpsInfog(1), factor.getMumpsInfog(2), factor.getMumpsIcntl(14))
            raise ArithmeticError(
                f"Spherical gap KSP failed: reason={ksp.getConvergedReason()}, "
                f"PC reason={pc.getFailedReason()}, factor diagnostics={factor_details}, pins={pins}"
            )
        solve_seconds = mpi.allreduce(time.perf_counter()-solve_started, op=MPI.MAX)
        initial_solve_seconds = solve_seconds
        resources["rss_after_solve_mib"] = _rss_mib(observer, "rss_after_solve_mib")
        lift_integral = state.lift_integral
        direct_lu = ksp.getType() == "preonly" and ksp.getPC().getType() == "lu"
        ungauged = solution.copy() if direct_lu else None
        if ungauged is not None:
            cleanup.callback(ungauged.destroy)

        restore_gauge = state.restore_gauge

        restore_gauge(solution)
        direct_history = None
        refinement_seconds = 0.0
        if direct_lu:
            refinement_started = time.perf_counter()
            direct_history = _refine_direct_solution(
                ksp, pinned, pinned_rhs, matrix, rhs, ungauged, solution, restore_gauge,
            )
            refinement_seconds = mpi.allreduce(time.perf_counter()-refinement_started, op=MPI.MAX)
            solve_seconds += refinement_seconds
        matrix.mult(solution, work); work.axpy(-1, rhs)
        relative = work.norm()/max(rhs.norm(), np.finfo(float).tiny)
        if not np.isfinite(relative) or relative > 1e-8:
            raise ArithmeticError(f"Spherical unpinned physical residual failed: {relative}")
        normal_field, tangent_field = state.assign_fields(solution)
        checks = dict(relative_residual=relative, null_relative_residuals=null_errors,
                      physical_volume=pmean.sum(), physical_pressure_integral=pmean.dot(solution)+lift_integral,
                      rotation_integrals=[weight.dot(solution) for weight in rotation_weights],
                      unknowns=size, multiplier_nodes=nboundary, ksp_reason=int(ksp.getConvergedReason()),
                      constraint_components=components, ranks=mpi.size,
                      surface_quadrature_order=quadrature,
                      radial_quadrature_order=int(radial_quadrature),
                      curtain_radial_quadrature_order=4,
                      inner_integration="physical tetrahedron outside ball",
                      inner_cut_quadrature_order=cut_order,
                      inner_cut_cells=mpi.allreduce(len(inner_cells), op=MPI.SUM),
                      wholly_void_cells=mpi.allreduce(len(void_cells), op=MPI.SUM),
                      unsupported_unknowns=mpi.allreduce(len(unsupported), op=MPI.SUM),
                      multiplier_mapping="scalar P2 normal" if components == 1 else "Cartesian P2 times R/|y|")
        checks.update(assembly_seconds=assembly_seconds, setup_seconds=setup_seconds,
                      solve_seconds=solve_seconds, ksp_iterations=int(ksp.getIterationNumber()))
        if direct_history is not None:
            checks.update(direct_refinement_steps=len(direct_history)-1,
                          direct_ksp_solve_calls=len(direct_history),
                          direct_refinement_seconds=refinement_seconds,
                          initial_solve_seconds=initial_solve_seconds,
                          direct_initial_pinned_relative_residual=direct_history[0]["pinned_relative"],
                          direct_final_pinned_relative_residual=direct_history[-1]["pinned_relative"],
                          direct_initial_relative_residual=direct_history[0]["full_relative"],
                          direct_final_relative_residual=direct_history[-1]["full_relative"],
                          direct_residual_history=direct_history)
        checks.update(graph_checks)
        # Durations and RSS snapshots use the largest local value, while matrix
        # counts above sum rank-local MatInfo. RSS snapshots are not peak RSS.
        checks.update({key: mpi.allreduce(value, op=MPI.MAX) for key, value in resources.items()})
        if use_norm_pc:
            checks.update(norm_preconditioner_length=float(norm_preconditioner_length),
                          norm_assembly_seconds=mpi.allreduce(norm_blocks.assembly_seconds, op=MPI.MAX),
                          norm_context=type(context).__name__, norm_velocity_pc=context.velocity_ksp.getPC().getType(),
                          norm_velocity_block_size=norm_blocks.h.getBlockSize(),
                          norm_pc_applications=context.applications,
                          rhs_removed_null_relative=removed_relative,
                          requested_gmres_restart=linear_solver_options.get("ksp_gmres_restart"),
                          restart_evidence="effective PETSc KSP view in run log",
                          rss_before_norm_setup_mib=checks["rss_before_setup_mib"])
        if ksp.getPC().getType() == "lu" and ksp.getPC().getFactorSolverType() == "mumps":
            checks["effective_mumps_icntl_14"] = ksp.getPC().getFactorMatrix().getMumpsIcntl(14)
        result = _SphericalGapResult(normal_field, tangent_field, checks)
        if observer is not None and hasattr(observer, "finalized"):
            observer.finalized(result)
        return result


def _solve_spherical_gap(stokes, inner_boundary, outer_boundary, radius_inner, radius_outer,
                         viscosity, pressure_jump, pressure_jump_integral, gap_bodyforce,
                         quadrature, linear_solver_options, batch_size, volume_force=None,
                         norm_preconditioner_length=None, radial_quadrature=4):
    return _solve_constraints(stokes, inner_boundary, outer_boundary, radius_inner, radius_outer,
                              viscosity, pressure_jump, pressure_jump_integral, gap_bodyforce,
                              quadrature, linear_solver_options, batch_size, 1, volume_force,
                              norm_preconditioner_length, radial_quadrature)


def _solve_spherical_gap_zero_slip(stokes, inner_boundary, outer_boundary, radius_inner, radius_outer,
                                   viscosity, pressure_jump, pressure_jump_integral, gap_bodyforce,
                                   quadrature, linear_solver_options, batch_size, volume_force=None,
                                   norm_preconditioner_length=None, radial_quadrature=4):
    return _solve_constraints(stokes, inner_boundary, outer_boundary, radius_inner, radius_outer,
                              viscosity, pressure_jump, pressure_jump_integral, gap_bodyforce,
                              quadrature, linear_solver_options, batch_size, 3, volume_force,
                              norm_preconditioner_length, radial_quadrature)
