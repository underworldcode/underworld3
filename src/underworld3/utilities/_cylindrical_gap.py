r"""Coupled signed-domain Stokes integration for affine P2/P1 annuli.

Solution and test polynomials extend into outer circular caps; inner caps
are removed. Zero slip adds a second multiplier for the tangential velocity;
its rotation is constrained rather than removed as a gauge. Both viscous and incompressibility blocks use this same domain.
A continuous P2-in-angle multiplier enforces the physical-circle normal trace
weakly and solves lambda=-sigma_nn in the coupled system. This differs from
nodal rotated free slip and from lagged stress-derivative wall transfer.

References
----------
Collins, Li, Lozinski & Scovazzi (2025), "Gap-SBM: A New Conceptualization of
the Shifted Boundary Method with Optimal Convergence for the Neumann and
Dirichlet Problems", https://arxiv.org/abs/2508.09613: adopt the principle of
extending both trial and test functions and including gap virtual work.
Urquiza, Garon & Farinas (2014), "Weak imposition of the slip boundary condition
on curved boundaries for Stokes flow", JCP 256, 748-767,
https://doi.org/10.1016/j.jcp.2013.08.045: normal-velocity multiplier formulation.
Burman, Hansbo & Larson (2020), "Dirichlet boundary value correction using
Lagrange multipliers", BIT 60, 235-260,
https://doi.org/10.1007/s10543-019-00773-4: related multiplier boundary correction.

The signed circular-cap geometry, affine P2/P1 volume spaces, P2-in-angle
normal multiplier and paired pressure/reaction gauge below are this UW3
adaptation. It is not a verbatim implementation of any one paper, and their
convergence/stability theorems do not establish those properties for this space.
No analytical Stokes solution or harmonic coefficients enter the solve.
"""

from underworld3.utilities._cylindrical_stokes_state import CylindricalSolveState
from underworld3.utilities._stokes_observer import solve_with_observer
from contextlib import ExitStack
from typing import NamedTuple
from underworld3.utilities._field_layout import _layout, _scatter_values
from underworld3.utilities._cylindrical_gap_geometry import _p2_basis, _cap_quadrature, _triangle_rule, _volume_blocks
from underworld3.utilities._cylindrical_gap_assembly import _insert_blocks, _create_gap_matrix

import numpy as np
import sympy as sp
import underworld3 as uw
from mpi4py import MPI
from petsc4py import PETSc

from underworld3.utilities._cylindrical_dirichlet import _labelled_edge_keys, _edge_key


class _GapResult(NamedTuple):
    normal_reaction: object
    checks: dict












def _ordered_rows(mesh, field, vertices):
    rows = mesh._cell_node_indices(field.degree, field.continuous)
    standard = vertices
    if field.degree == 2:
        standard = np.concatenate(
            (vertices, (vertices[:, [0, 1, 2]] + vertices[:, [1, 2, 0]]) / 2), axis=1
        )
    distance = np.linalg.norm(
        np.asarray(field.coords)[rows][:, :, None] - standard[:, None], axis=-1
    )
    error = float(np.max(distance.min(axis=1), initial=0.0))
    if uw.mpi.comm.allreduce(error, op=MPI.MAX) > 1e-10:
        raise ValueError(
            "Cylindrical gap integration requires affine P2/P1 coordinates"
        )
    return np.take_along_axis(rows, distance.argmin(axis=1), axis=1)


def _native_force_rule(mesh, vertices):
    """Use the ordinary UW3 source quadrature for a matched bulk load."""
    fe = PETSc.FE().createDefault(2, 1, True, mesh.qdegree, "gap_source_")
    points, weights = fe.getQuadrature().getData()
    xy = (np.asarray(points).reshape(-1, 2) + 1) / 2
    bary = np.column_stack((1 - xy.sum(axis=1), xy))
    area = (
        abs(
            np.linalg.det(
                np.stack(
                    (vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]),
                    axis=-1,
                )
            )
        )
        / 2
    )
    physical = np.einsum("qi,eid->eqd", bary, vertices)
    measure = area[:, None] * np.asarray(weights).reshape(1, -1) / 2
    fe.destroy()
    return physical, measure


def _field_values(field, vertices, points, cells):
    """Continue a P1/P2 source from its actual PETSc interpolation nodes.

    DG nodes need not be triangle vertices or edge midpoints. Reconstruct the
    same cell polynomial in a canonical barycentric basis before quadrature.
    """
    rows = field.mesh._cell_node_indices(field.degree, field.continuous)[cells]
    nodes = np.asarray(field.coords)[rows]
    shape, _, bary = _p2_basis(vertices[cells], nodes)
    design = shape if field.degree == 2 else bary
    coefficients = np.linalg.solve(design, np.asarray(field.array)[rows, 0, :])
    shape, _, bary = _p2_basis(vertices[cells], points)
    basis = shape if field.degree == 2 else bary
    return np.einsum("eqi,eid->eqd", basis, coefficients)


def _coordinate_values(expression, points, components, mesh):
    expression = uw.function.expressions._unwrap_for_compilation(
        expression, keep_constants=False, return_self=False
    )
    if any(hasattr(type(atom), "meshvar") for atom in expression.atoms(sp.Function)):
        raise ValueError(
            "Cap sources and pressure lifts must depend on coordinates only"
        )
    if not expression.free_symbols and not expression.atoms(sp.Function):
        return np.broadcast_to(
            np.asarray(expression, dtype=float).reshape(components),
            (*points.shape[:-1], components),
        ).copy()
    values = uw.function.evaluate(
        expression, np.ascontiguousarray(points.reshape(-1, 2)), coord_sys=mesh.N
    )
    return np.asarray(values).reshape(*points.shape[:-1], components)








def _solve_circle_constraints(
    stokes,
    inner_boundary,
    outer_boundary,
    radius_inner,
    radius_outer,
    viscosity,
    pressure_jump,
    pressure_jump_integral,
    cap_bodyforce,
    quadrature,
    linear_solver_options,
    constraint_components,
):
    ncomponents = len(constraint_components)
    mesh, velocity, pressure = stokes.mesh, stokes.u, stokes.p
    if (
        mesh.dim != 2
        or mesh.cdim != 2
        or velocity.degree != 2
        or not velocity.continuous
        or pressure.degree != 1
        or not pressure.continuous
    ):
        raise ValueError(
            "Cylindrical gap Stokes requires affine 2D continuous P2/P1 fields"
        )
    if (
        uw.is_nondimensional_scaling_active()
        or uw.get_default_model().has_units_active()
    ):
        raise ValueError("Cylindrical gap Stokes supports nondimensional data only")
    if (
        not 0 < radius_inner < radius_outer
        or not np.isfinite(viscosity)
        or viscosity <= 0
        or int(quadrature) != quadrature
        or quadrature < 3
    ):
        raise ValueError(
            "Require positive ordered radii, positive constant viscosity and integer quadrature >= 3"
        )
    if (
        stokes.essential_bcs
        or stokes.natural_bcs
        or stokes._rotated_freeslip_bcs
        or getattr(stokes, "_block_constraint_bcs", [])
    ):
        raise ValueError(
            "Cylindrical gap Stokes replaces ordinary wall registration on both complete circles"
        )
    mu = uw.function.expressions._unwrap_for_compilation(
        stokes.constitutive_model.viscosity, keep_constants=False, return_self=False
    )
    if mu.free_symbols or float(mu) != float(viscosity):
        raise ValueError(
            "The constitutive viscosity must equal the supplied positive constant viscosity"
        )
    if not hasattr(stokes, "_cylindrical_gap_reaction"):
        stokes._cylindrical_gap_reaction = uw.discretisation.MeshVariable(
            f"CylindricalNormalReaction_{stokes.instance_number}",
            mesh,
            1,
            degree=2,
            continuous=True,
        )
    reactions = [stokes._cylindrical_gap_reaction]
    if "tangent" in constraint_components:
        if not hasattr(stokes, "_cylindrical_gap_tangent_reaction"):
            stokes._cylindrical_gap_tangent_reaction = uw.discretisation.MeshVariable(
                f"CylindricalTangentReaction_{stokes.instance_number}",
                mesh,
                1,
                degree=2,
                continuous=True,
            )
        reactions.append(stokes._cylindrical_gap_tangent_reaction)
    if stokes.constraints != sp.Matrix((stokes.div_u,)):
        raise ValueError(
            "Cylindrical gap Stokes requires the standard zero-divergence constraint"
        )
    penalty = uw.function.expressions._unwrap_for_compilation(
        stokes.penalty, keep_constants=False, return_self=False
    )
    if penalty != 0:
        raise ValueError("Cylindrical gap Stokes does not include a grad-div penalty")
    vertex_rows = mesh._cell_node_indices(1, True)
    vertices = np.asarray(mesh._get_coords_for_basis(1, True))[vertex_rows]
    urows, prows = _ordered_rows(mesh, velocity, vertices), _ordered_rows(
        mesh, pressure, vertices
    )
    umap, nu = _layout(velocity)
    pmap, npres = _layout(pressure)
    uid = umap.reshape(-1, 2)[urows].reshape(len(vertices), 12)
    pid = pmap[prows] + nu
    owned = np.flatnonzero(mesh._get_owned_cells_mask())
    walls = []
    boundary_uids = []
    for boundary, radius, sign in (
        (inner_boundary, radius_inner, -1),
        (outer_boundary, radius_outer, 1),
    ):
        keys = _labelled_edge_keys(mesh, boundary, radius, radius * 1e-10)
        owners, edge_rows, edges = [], [], []
        for i, j, mid in ((0, 1, 3), (1, 2, 4), (2, 0, 5)):
            selected = owned[
                np.all(
                    abs(np.linalg.norm(vertices[owned][:, [i, j]], axis=-1) - radius)
                    < 1e-10,
                    axis=1,
                )
            ]
            for cell in selected:
                edge = vertices[cell, [i, j]]
                if _edge_key(edge, radius * 1e-10) in keys:
                    owners.append(cell)
                    edge_rows.append(urows[cell, [i, j, mid]])
                    edges.append(edge)
        owners = np.asarray(owners, dtype=int)
        edges = np.asarray(edges, dtype=float).reshape(-1, 2, 2)
        edge_rows = np.asarray(edge_rows, dtype=int).reshape(-1, 3)
        found = set().union(
            *uw.mpi.comm.allgather({_edge_key(edge, radius * 1e-10) for edge in edges})
        )
        if not keys or found != keys:
            raise ValueError(
                "Owned cells do not cover the requested physical circular wall"
            )
        boundary_uids.extend(umap.reshape(-1, 2)[edge_rows, 0].ravel().tolist())
        walls.append(
            dict(
                owners=owners,
                edge_rows=edge_rows,
                radius=radius,
                sign=sign,
                rule=_cap_quadrature(edges, radius, quadrature, 4),
            )
        )
    boundary_uids = np.unique(
        np.concatenate(
            uw.mpi.comm.allgather(np.asarray(boundary_uids, dtype=PETSc.IntType))
        )
    )
    size = nu + npres + ncomponents * len(boundary_uids)
    for wall in walls:
        edge_uids = umap.reshape(-1, 2)[wall["edge_rows"], 0]
        wall["multiplier_ids"] = np.asarray(
            nu
            + npres
            + ncomponents * np.searchsorted(boundary_uids, edge_uids)[..., None]
            + np.arange(ncomponents),
            dtype=PETSc.IntType,
        ).reshape(-1, 3 * ncomponents)
    matrix = _create_gap_matrix(
        size,
        uid[owned],
        pid[owned],
        [(uid[wall["owners"]], wall["multiplier_ids"]) for wall in walls],
        mesh.dm.comm,
    )
    rhs = matrix.createVecRight()
    pmean = rhs.duplicate()
    rotation_weights = rhs.duplicate()
    for vector in (rhs, pmean, rotation_weights):
        vector.set(0.0)
    points, weights = _triangle_rule(vertices[owned], 4)
    blocks = _volume_blocks(vertices[owned], points, weights)
    _insert_blocks(matrix, uid[owned], pid[owned], blocks, viscosity, 1)
    load_points, load_weights = _native_force_rule(mesh, vertices[owned])
    bodyforce = uw.function.expressions._unwrap_for_compilation(
        stokes.bodyforce, keep_constants=False, return_self=False
    )
    variables = {
        type(atom).meshvar()
        for atom in bodyforce.atoms(sp.Function)
        if hasattr(type(atom), "meshvar")
    }
    if variables:
        if len(variables) != 1:
            raise ValueError(
                "Use one P1/P2 vector MeshVariable or a coordinate-only volume body force"
            )
        field = variables.pop()
        if (
            field.mesh is not mesh
            or field.degree not in (1, 2)
            or field.num_components != 2
            or tuple(bodyforce) != tuple(field.sym)
        ):
            raise ValueError(
                "Use the unmodified P1/P2 vector source field on the Stokes mesh"
            )
        force = _field_values(field, vertices, load_points, owned)
    else:
        force = _coordinate_values(bodyforce, load_points, 2, mesh)
    shape, _, _ = _p2_basis(vertices[owned], load_points)
    load = np.einsum("eq,eqi,eqd->eid", load_weights, shape, force)
    for ids, value in zip(uid[owned], load, strict=True):
        rhs.setValues(ids, value.ravel(), addv=PETSc.InsertMode.ADD_VALUES)
    for ids, mass in zip(pid[owned], blocks["pressure_mean"], strict=True):
        pmean.setValues(ids, mass, addv=PETSc.InsertMode.ADD_VALUES)
    for ids, rot in zip(uid[owned], blocks["rotation"], strict=True):
        rotation_weights.setValues(ids, rot.ravel(), addv=PETSc.InsertMode.ADD_VALUES)
    lift_cap_integral = 0.0
    for wall in walls:
        owners, sign, rule = wall["owners"], wall["sign"], wall["rule"]
        cap = _volume_blocks(vertices[owners], rule["cap_points"], rule["cap_weight"])
        _insert_blocks(matrix, uid[owners], pid[owners], cap, viscosity, sign)
        shape, _, _ = _p2_basis(vertices[owners], rule["cap_points"])
        force = _coordinate_values(cap_bodyforce, rule["cap_points"], 2, mesh)
        load = sign * np.einsum("eq,eqi,eqd->eid", rule["cap_weight"], shape, force)
        for ids, value, rot in zip(uid[owners], load, cap["rotation"], strict=True):
            rhs.setValues(ids, value.ravel(), addv=PETSc.InsertMode.ADD_VALUES)
            rotation_weights.setValues(
                ids, sign * rot.ravel(), addv=PETSc.InsertMode.ADD_VALUES
            )
        for ids, value in zip(pid[owners], cap["pressure_mean"], strict=True):
            pmean.setValues(ids, sign * value, addv=PETSc.InsertMode.ADD_VALUES)
        lift_cap_integral += sign * np.sum(
            rule["cap_weight"]
            * _coordinate_values(pressure_jump, rule["cap_points"], 1, mesh)[..., 0]
        )
        shape, _, _ = _p2_basis(vertices[owners], rule["circle_points"])
        t = (np.polynomial.legendre.leggauss(quadrature)[0] + 1) / 2
        multiplier = np.array(
            [(1 - t) * (1 - 2 * t), t * (2 * t - 1), 4 * t * (1 - t)]
        ).T
        radial = rule["circle_points"] / wall["radius"]
        tangent = np.stack((-radial[..., 1], radial[..., 0]), axis=-1)
        directions = np.stack((sign * radial, tangent), axis=-2)[..., :ncomponents, :]
        c = np.einsum(
            "eq,ql,eqi,eqcd->elcid",
            rule["circle_weight"],
            multiplier,
            shape,
            directions,
        )
        # p=q+p_J contributes +integral(p_J n.v) to the regular-pressure
        # momentum load, so the solved multiplier remains physical -sigma_nn.
        lift_trace = _coordinate_values(pressure_jump, rule["circle_points"], 1, mesh)[
            ..., 0
        ]
        lift_load = sign * np.einsum(
            "eq,eqi,eq,eqd->eid",
            rule["circle_weight"],
            shape,
            lift_trace,
            rule["circle_points"] / wall["radius"],
        )
        for u, value in zip(uid[owners], lift_load, strict=True):
            rhs.setValues(u, value.ravel(), addv=PETSc.InsertMode.ADD_VALUES)
        lid = wall["multiplier_ids"]
        for u, l, value in zip(
            uid[owners],
            lid.reshape(-1, 3 * ncomponents),
            c.reshape(-1, 3 * ncomponents, 12),
            strict=True,
        ):
            matrix.setValues(
                l, u, np.ascontiguousarray(value), addv=PETSc.InsertMode.ADD_VALUES
            )
            matrix.setValues(
                u, l, np.ascontiguousarray(value.T), addv=PETSc.InsertMode.ADD_VALUES
            )
    matrix.assemble()
    if not matrix.isSymmetric(tol=1e-12):
        raise ArithmeticError("Cylindrical gap mixed assembly lost transpose symmetry")
    for vector in (rhs, pmean, rotation_weights):
        vector.assemble()
    pressure_mode = rhs.duplicate()
    rotation_mode = rhs.duplicate()
    pressure_mode.set(0.0)
    rotation_mode.set(0.0)
    lo, hi = pressure_mode.getOwnershipRange()
    pressure_indices = np.arange(max(lo, nu), min(hi, nu + npres), dtype=PETSc.IntType)
    normal_indices = (
        nu + npres + ncomponents * np.arange(len(boundary_uids), dtype=PETSc.IntType)
    )
    normal_indices = normal_indices[(normal_indices >= lo) & (normal_indices < hi)]
    indices = np.concatenate((pressure_indices, normal_indices))
    pressure_mode.setValues(indices, np.ones(len(indices)))
    pressure_mode.assemble()
    rotation = np.column_stack(
        (-np.asarray(velocity.coords)[:, 1], np.asarray(velocity.coords)[:, 0])
    ).ravel()
    rotation_mode.setValues(umap, rotation)
    rotation_mode.assemble()
    work = rhs.duplicate()
    matrix.mult(pressure_mode, work)
    pn = work.norm() / (matrix.norm() * pressure_mode.norm())
    matrix.mult(rotation_mode, work)
    rn = work.norm() / (matrix.norm() * rotation_mode.norm())
    if pn > 1e-12 or (ncomponents == 1 and rn > 1e-12):
        raise ArithmeticError(
            f"Cylindrical gap null-mode assembly failed: pressure={pn}, rotation={rn}"
        )
    compatibility = max(
        abs(rhs.dot(pressure_mode)) / pressure_mode.norm(),
        abs(rhs.dot(rotation_mode)) / rotation_mode.norm(),
    )
    if ncomponents == 1 and compatibility > 1e-10 * max(rhs.norm(), 1.0):
        raise ValueError(
            "The prescribed volume/cap forces have incompatible net rotation torque"
        )
    # Two algebraic null-mode pins; physical volume gauges are restored below.
    best = float(np.max(abs(rotation), initial=0.0))
    maximum = uw.mpi.comm.allreduce(best, op=MPI.MAX)
    candidate = int(
        np.min(umap[np.isclose(abs(rotation), maximum, rtol=0, atol=1e-13)], initial=nu)
    )
    rotation_pin = uw.mpi.comm.allreduce(candidate, op=MPI.MIN)
    pins = np.asarray(
        ([rotation_pin] if ncomponents == 1 else []) + [nu], dtype=PETSc.IntType
    )
    pinned = matrix.copy()
    pinned.zeroRowsColumns(pins, diag=1.0)
    pinned_rhs = rhs.copy()
    lo, hi = pinned_rhs.getOwnershipRange()
    local_pins = pins[(pins >= lo) & (pins < hi)]
    pinned_rhs.setValues(local_pins, np.zeros(len(local_pins)))
    pinned_rhs.assemble()
    ksp = PETSc.KSP().create(comm=mesh.dm.comm)
    ksp.setOperators(pinned)
    prefix = f"cylindrical_gap_{stokes.instance_number}_"
    ksp.setOptionsPrefix(prefix)
    options = PETSc.Options(prefix)
    for key, value in linear_solver_options.items():
        options[key] = value
    ksp.setFromOptions()
    solution = rhs.duplicate()
    with ExitStack() as cleanup:
        for obj in (ksp, solution, pinned_rhs, pinned, work, pressure_mode, rotation_mode,
                    rhs, pmean, rotation_weights, matrix):
            cleanup.callback(obj.destroy)
        try:
            state = CylindricalSolveState(dict(
                matrix=matrix, rhs=rhs, mpi=uw.mpi.comm, pressure_jump_integral=pressure_jump_integral,
                lift_gap=lift_cap_integral, pmean=pmean, pressure_mode=pressure_mode,
                rotation_weight=rotation_weights, rotation_weights=[rotation_weights], rotation_mode=rotation_mode,
                components=ncomponents, nu=nu, npres=npres, velocity=velocity, pressure=pressure,
                mesh=mesh, umap=umap, pmap=pmap, boundary_uids=boundary_uids, reactions=reactions,
            ))
            observer = getattr(stokes, '_stokes_observer', None)
            if observer is None:
                ksp.solve(pinned_rhs, solution)
            else:
                solve_with_observer(ksp, pinned_rhs, solution, state, observer)
            if ksp.getConvergedReason() <= 0:
                raise ArithmeticError(
                    f"Cylindrical gap KSP failed: reason={ksp.getConvergedReason()}"
                )
            lift_integral = state.lift_integral
            state.restore_gauge(solution)
            matrix.mult(solution, work)
            work.axpy(-1.0, rhs)
            relative = work.norm() / max(rhs.norm(), np.finfo(float).tiny)
            checks = dict(
                relative_residual=relative,
                pressure_null_relative_residual=pn,
                rotation_null_relative_residual=rn,
                physical_area=pmean.sum(),
                physical_pressure_integral=pmean.dot(solution) + lift_integral,
                rotation_integral=rotation_weights.dot(solution),
                unknowns=size,
                ksp_reason=int(ksp.getConvergedReason()),
                ranks=uw.mpi.size,
                constraint_components=constraint_components,
                rotation_is_null=ncomponents == 1,
            )
            if relative > 1e-9:
                raise ArithmeticError(f"Cylindrical gap unpinned residual failed: {checks}")
            state.assign_fields(solution)
            result = (_GapResult(reactions[0], checks) if ncomponents == 1
                      else _ZeroSlipGapResult(reactions[0], reactions[1], checks))
            if observer is not None:
                observer.finalized(result)
            return result
        finally:
            for key in linear_solver_options:
                del options[key]



class _ZeroSlipGapResult(NamedTuple):
    normal_reaction: object
    tangential_reaction: object
    checks: dict


def _solve_cylindrical_gap(
    stokes,
    inner_boundary,
    outer_boundary,
    radius_inner,
    radius_outer,
    viscosity,
    pressure_jump,
    pressure_jump_integral,
    cap_bodyforce,
    quadrature,
    linear_solver_options,
):
    return _solve_circle_constraints(
        stokes,
        inner_boundary,
        outer_boundary,
        radius_inner,
        radius_outer,
        viscosity,
        pressure_jump,
        pressure_jump_integral,
        cap_bodyforce,
        quadrature,
        linear_solver_options,
        ("normal",),
    )



def _solve_cylindrical_gap_zero_slip(
    stokes,
    inner_boundary,
    outer_boundary,
    radius_inner,
    radius_outer,
    viscosity,
    pressure_jump,
    pressure_jump_integral,
    cap_bodyforce,
    quadrature,
    linear_solver_options,
):
    return _solve_circle_constraints(
        stokes,
        inner_boundary,
        outer_boundary,
        radius_inner,
        radius_outer,
        viscosity,
        pressure_jump,
        pressure_jump_integral,
        cap_bodyforce,
        quadrature,
        linear_solver_options,
        ("normal", "tangent"),
    )

