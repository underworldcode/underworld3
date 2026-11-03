r"""Opt-in affine P2/P1 spherical-gap mixed Stokes assembly.

The physical-domain bilinear form adds outer gaps and subtracts inner gaps in
both momentum and continuity. Free slip uses scalar P2 normal multipliers.
Zero slip uses three Cartesian multiplier components with mapped basis
(R/|y|) N_i(y) on each planar facet parameter y. This custom mapping contains
exact sphere normals, preserving the paired pressure/reaction gauge; its
uniform stability is unproven. No polar tangent frame or pressure filtering is
used. Returned reactions are physical values at projected boundary nodes.

The gap extension and weak normal constraint are inspired by Collins et al.,
Gap-SBM (2025), https://arxiv.org/abs/2508.09613, and Urquiza, Garon and Farinas,
JCP 256 (2014), https://doi.org/10.1016/j.jcp.2013.08.045. They do not establish
stability of this three-dimensional mixed space or its signed inner gaps.
The cited Gap-SBM analysis is two-dimensional and does not supply the curtain
terms or mapped Cartesian multipliers used here.

For adjacent cap extensions A/B, let [v]=v_A-v_B and {q}=(q_A+q_B)/2.
With n pointing A to B and cap sign s, the additional continuity form is
+s integral({q} [v].n). The symmetric viscous correction is
-s integral([v].{2 mu epsilon(u) n} + [u].{2 mu epsilon(v) n}).
These cancel the artificial-curtain fluxes for smooth exact fields and retain
all three rigid rotations. Inner-cap subtraction is a signed continuation
method, so these identities do not prove coercivity on ker([B; C]) or an
inf-sup bound. Both require independent measured checks before acceptance.
"""

from contextlib import ExitStack
from itertools import combinations
from typing import NamedTuple

import numpy as np
import sympy as sp
from mpi4py import MPI
from petsc4py import PETSc
import underworld3 as uw

from underworld3.utilities._cylindrical_gap import _layout, _scatter_values
from underworld3.utilities._spherical_gap import _spherical_facet_quadrature
from underworld3.utilities._spherical_curtain import (
    _spherical_curtain_quadrature, _curtain_stokes_blocks,
)
from underworld3.utilities._tetrahedral_gap import (
    _tetrahedral_p2_basis, _tetrahedron_quadrature, _tetrahedral_stokes_blocks,
)


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


def _source_values(expression, vertices, points, cells, mesh):
    variables = {type(atom).meshvar() for atom in expression.atoms(sp.Function)
                 if hasattr(type(atom), "meshvar")}
    if not variables:
        return _coordinate_values(expression, points, 3, mesh)
    if len(variables) != 1:
        raise ValueError("Use one unmodified DG0/P1/P2 vector source field or a coordinate force.")
    field = variables.pop()
    if (field.mesh is not mesh or field.degree not in (0, 1, 2)
            or field.num_components != 3 or tuple(expression) != tuple(field.sym)):
        raise ValueError("Use one unmodified DG0/P1/P2 vector source field on the Stokes mesh.")
    rows = mesh._cell_node_indices(field.degree, field.continuous)[cells]
    if field.degree == 0:
        return np.broadcast_to(np.asarray(field.array)[rows[:, 0], 0, :][:, None, :], points.shape).copy()
    shape, _, bary = _tetrahedral_p2_basis(vertices[cells], np.asarray(field.coords)[rows])
    design = shape if field.degree == 2 else bary
    coefficients = np.linalg.solve(design, np.asarray(field.array)[rows, 0, :])
    shape, _, bary = _tetrahedral_p2_basis(vertices[cells], points)
    return np.einsum("eqi,eid->eqd", shape if field.degree == 2 else bary, coefficients)


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
    return matrix


def _insert(matrix, uid, pid, blocks, viscosity, sign):
    for u, p, a, b in zip(uid, pid, blocks["A"], blocks["B"], strict=True):
        matrix.setValues(u, u, np.ascontiguousarray(sign*viscosity*a), addv=PETSc.InsertMode.ADD_VALUES)
        matrix.setValues(p, u, np.ascontiguousarray(sign*b), addv=PETSc.InsertMode.ADD_VALUES)
        matrix.setValues(u, p, np.ascontiguousarray(sign*b.T), addv=PETSc.InsertMode.ADD_VALUES)


def _add(vector, ids, values):
    for index, value in zip(ids, values, strict=True):
        vector.setValues(index, np.asarray(value).ravel(), addv=PETSc.InsertMode.ADD_VALUES)


def _solve_constraints(stokes, inner_boundary, outer_boundary, radius_inner, radius_outer,
                       viscosity, pressure_jump, pressure_jump_integral, gap_bodyforce,
                       quadrature, linear_solver_options, batch_size, components, volume_force=None):
    mesh, velocity, pressure = stokes.mesh, stokes.u, stokes.p
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
    observer = getattr(stokes, "_spherical_gap_observer", None)
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
    curtains = _curtain_pairs(walls, vertices, uid, pid, mesh)
    with ExitStack() as cleanup:
        compressed, nboundary = _compressed_boundary_ids(umap.reshape(-1, 3)[:, 0]//3, nu//3, walls, comm, cleanup)
        size = nu+npres+components*nboundary
        for wall in walls:
            wall["lid"] = (nu+npres+components*compressed[wall["rows"], None]
                           + np.arange(components)).reshape(-1, 6*components).astype(PETSc.IntType)
        matrix = _create_matrix(size, uid[owned], pid[owned],
                                [(uid[w["cells"]], w["lid"]) for w in walls], curtains, comm)
        cleanup.callback(matrix.destroy)
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
            _add(pmean, pid[cells], blocks["pressure_mean"])
            for axis in range(3):
                _add(rotation_weights[axis], uid[cells], blocks["rotation"][..., axis])
            points = np.einsum("qi,eid->eqd", native_bary, vertices[cells])
            determinant = np.abs(np.linalg.det(np.swapaxes(vertices[cells, 1:]-vertices[cells, None, 0], 1, 2)))
            weights = determinant[:, None]*np.asarray(native_weights).reshape(1, -1)/8
            shape, _, _ = _tetrahedral_p2_basis(vertices[cells], points)
            force = _source_values(force_expr, vertices, points, cells, mesh)
            _add(rhs, uid[cells], np.einsum("eq,eqi,eqd->eid", weights, shape, force))
        if observer is not None:
            matrix.assemble()
            observer.background(matrix, nu, npres)
        lift_gap = 0.0
        for wall in walls:
            for start in range(0, len(wall["cells"]), batch_size):
                sl = slice(start, start+batch_size)
                cells, sign = wall["cells"][sl], wall["sign"]
                rule = _spherical_facet_quadrature(wall["facets"][sl], wall["radius"], quadrature, 4)
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
        ksp = PETSc.KSP().create(comm=comm); cleanup.callback(ksp.destroy); ksp.setOperators(pinned)
        prefix = f"spherical_gap_{stokes.instance_number}_"; ksp.setOptionsPrefix(prefix)
        options = PETSc.Options(prefix)
        for key, value in linear_solver_options.items():
            options[key] = value
            cleanup.callback(options.delValue, key)
        ksp.setFromOptions()
        solution = rhs.duplicate(); cleanup.callback(solution.destroy); ksp.solve(pinned_rhs, solution)
        if ksp.getConvergedReason() <= 0:
            pc = ksp.getPC()
            factor_details = None
            if pc.getFactorSolverType() == "mumps":
                factor = pc.getFactorMatrix()
                factor_details = (factor.getMumpsInfog(1), factor.getMumpsInfog(2), factor.getMumpsIcntl(14))
            raise ArithmeticError(
                f"Spherical gap KSP failed: reason={ksp.getConvergedReason()}, "
                f"PC reason={pc.getFailedReason()}, factor diagnostics={factor_details}, pins={pins}"
            )
        lift_integral = float(pressure_jump_integral)+mpi.allreduce(float(lift_gap), op=MPI.SUM)
        solution.axpy(-(pmean.dot(solution)+lift_integral)/pmean.sum(), pressure_mode)
        if components == 1:
            gram = np.array([[weight.dot(mode) for mode in modes[1:]] for weight in rotation_weights])
            coefficients = np.linalg.solve(gram, [weight.dot(solution) for weight in rotation_weights])
            for coefficient, mode in zip(coefficients, modes[1:], strict=True):
                solution.axpy(-coefficient, mode)
        matrix.mult(solution, work); work.axpy(-1, rhs)
        relative = work.norm()/max(rhs.norm(), np.finfo(float).tiny)
        if not np.isfinite(relative) or relative > 1e-8:
            raise ArithmeticError(f"Spherical unpinned physical residual failed: {relative}")
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
        checks = dict(relative_residual=relative, null_relative_residuals=null_errors,
                      physical_volume=pmean.sum(), physical_pressure_integral=pmean.dot(solution)+lift_integral,
                      rotation_integrals=[weight.dot(solution) for weight in rotation_weights],
                      unknowns=size, multiplier_nodes=nboundary, ksp_reason=int(ksp.getConvergedReason()),
                      constraint_components=components, ranks=mpi.size,
                      multiplier_mapping="scalar P2 normal" if components == 1 else "Cartesian P2 times R/|y|")
        if ksp.getPC().getType() == "lu" and ksp.getPC().getFactorSolverType() == "mumps":
            checks["effective_mumps_icntl_14"] = ksp.getPC().getFactorMatrix().getMumpsIcntl(14)
        return _SphericalGapResult(normal_field, tangent_field, checks)


def _solve_spherical_gap(stokes, inner_boundary, outer_boundary, radius_inner, radius_outer,
                         viscosity, pressure_jump, pressure_jump_integral, gap_bodyforce,
                         quadrature, linear_solver_options, batch_size, volume_force=None):
    return _solve_constraints(stokes, inner_boundary, outer_boundary, radius_inner, radius_outer,
                              viscosity, pressure_jump, pressure_jump_integral, gap_bodyforce,
                              quadrature, linear_solver_options, batch_size, 1, volume_force)


def _solve_spherical_gap_zero_slip(stokes, inner_boundary, outer_boundary, radius_inner, radius_outer,
                                   viscosity, pressure_jump, pressure_jump_integral, gap_bodyforce,
                                   quadrature, linear_solver_options, batch_size, volume_force=None):
    return _solve_constraints(stokes, inner_boundary, outer_boundary, radius_inner, radius_outer,
                              viscosity, pressure_jump, pressure_jump_integral, gap_bodyforce,
                              quadrature, linear_solver_options, batch_size, 3, volume_force)
