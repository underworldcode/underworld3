r"""Interface conditions on a split-node fault: the frictionless contact.

A split fault (:mod:`underworld3.utilities.fault_split`) carries two
geometrically coincident boundaries, ``<name>Plus`` / ``<name>Minus``, whose
degrees of freedom pair exactly through the split's clone map. This module
imposes the **frictionless (perfectly slippery) contact** on those pairs:

.. math::

    [\mathbf v]\cdot\hat n = 0 \quad\text{(no opening, strong)}, \qquad
    \hat t\cdot\sigma\cdot\hat n = 0 \quad\text{(zero shear traction)},

with the tangential slip :math:`V = [\mathbf v]\cdot\hat t` left to EMERGE —
the stress-driven crack, as opposed to the kinematic (prescribed-slip)
fault, which dictates both sides.

The mechanism is the rotated strong-constraint machinery of
:mod:`underworld3.utilities.rotated_bc`, extended from per-node blocks to
per-PAIR blocks: for each coincident pair the composite rotation ``Q`` gets an
orthogonal :math:`2\,\mathrm{dim}\times 2\,\mathrm{dim}` block that transforms
``(v^+, v^-)`` to mean/jump components in the fault frame
:math:`(\hat n,\hat t)`. The jump-normal row is strongly constrained to zero;
every other row — the two mean rows and the slip row — is left free, and a
free slip row IS the zero-shear-traction condition (its conjugate reaction is
the shear traction, and an unconstrained row carries none). Friction, later,
is a nonlinear relation placed on that same slip row; prescribed jump-only
slip is a datum on it.

Both points of a pair are rank-local (the split refuses a fault whose fans
touch the partition seam), so every pair block sits inside one rank's
diagonal portion of ``Q``, exactly like the single-node wall blocks.

Until the interface lands in the solver API proper, the entry points are
module functions::

    child = split_fault(cut_mesh, "Fault")
    stokes = uw.systems.Stokes(child, ...)
    ... wall BCs ...
    add_frictionless_fault_bc(stokes, "Fault")
    info = solve_with_fault(stokes)
    s, V, leak = fault_slip(stokes, "Fault", info)
"""

import numpy as np

from underworld3 import mpi

# PETSc section field id of the velocity unknown (solver field registration
# order: velocity first) — the same convention as rotated_bc.
_VELOCITY_FIELD = 0


def add_frictionless_fault_bc(solver, boundary):
    """Register the frictionless contact on the split fault ``boundary``.

    ``boundary`` is the fault's original surface name; the mesh must be one
    returned by :func:`fault_split.split_fault`, which records the
    Plus/Minus DOF pairing. The condition itself is imposed inside the
    rotated solve — run it with :func:`solve_with_fault`.
    """
    pairs = getattr(solver.mesh, "_fault_point_pairs", {})
    if boundary not in pairs:
        raise ValueError(
            f"mesh carries no split-fault pairing for {boundary!r} — the "
            "solver's mesh must come from fault_split.split_fault, which is "
            "what records the coincident DOF pairs. "
            f"Available: {sorted(pairs) or 'none'}")
    if solver.mesh.dim != 2:
        raise NotImplementedError(
            "fault contact is 2-D; the 3-D fault is not yet a split surface.")
    registered = getattr(solver, "_fault_contact_faults", [])
    if boundary not in registered:
        solver._fault_contact_faults = registered + [boundary]


def add_viscous_fault_bc(solver, conds, boundary):
    r"""Register a viscous (linear dashpot) law on the split fault ``boundary``:

    .. math::  \hat t\cdot\sigma\cdot\hat n = \eta_f\, V,
               \qquad V = [\mathbf v]\cdot\hat t,

    with ``conds`` = :math:`\eta_f`, the interface viscosity (bulk viscosity
    per unit length: the zero-thickness limit of a shear band of viscosity
    :math:`\eta_b` and width :math:`w` is :math:`\eta_f = \eta_b/w`). The
    no-opening constraint rides along as always. Limits: ``conds = 0`` is the
    frictionless contact; ``conds`` large welds the fault. The natural scale
    is :math:`\eta/a` (bulk viscosity over fault half-length), at which the
    fault slips at roughly half its free rate.

    This is the linear member of the interface-law family: the term enters
    the rotated system as :math:`2\eta_f M` on the slip rows (:math:`M` the
    1-D trace mass), which is exactly the tangent structure a friction law
    occupies with :math:`2\,(\partial\tau/\partial V)\,M` — so everything
    downstream of this hook is what friction will reuse.
    """
    add_frictionless_fault_bc(solver, boundary)
    eta_f = float(conds)
    if eta_f < 0.0:
        raise ValueError(f"interface viscosity must be >= 0 (got {eta_f}).")
    registered = dict(getattr(solver, "_fault_interface_viscosity", {}))
    registered[boundary] = eta_f
    solver._fault_interface_viscosity = registered


def solve_with_fault(solver, verbose=False, zero_init_guess=True, picard=0):
    """Solve with the registered fault contact(s) imposed; returns the
    rotated-solve result dict (see ``rotated_bc.solve_rotated_freeslip``).

    Drives the SAME manual Newton loop as ``add_rotated_freeslip_bc`` — the
    fault pair blocks ride in the rotation the loop already builds — so
    rotated wall free-slip and the fault contact compose. The result is also
    stashed on ``solver._rotated_freeslip_info`` so the reaction-based
    recoveries can find it.
    """
    from underworld3.utilities.rotated_bc import solve_rotated_freeslip

    if not getattr(solver, "_fault_contact_faults", []):
        raise RuntimeError(
            "no fault contact registered on this solver; call "
            "add_frictionless_fault_bc first.")
    info = solve_rotated_freeslip(
        solver, list(getattr(solver, "_rotated_freeslip_bcs", [])),
        verbose=verbose, zero_init_guess=zero_init_guess, picard=picard)
    solver._rotated_freeslip_info = info
    return info


def _fault_pair_nodes(solver, boundary):
    """The velocity-carrying coincident pairs of a split fault, with normals.

    Returns ``[(q_plus, q_minus, n_hat), ...]`` over this rank's pairs, where
    the points are DMPlex points of the solver's DM (vertices and fault
    facets — P2 midpoint DOFs live on the facets) and ``n_hat`` is the fault
    unit normal at the pair, oriented from the Plus side toward the Minus
    side. Orientation is derived per facet from its single support cell —
    after the split a fault facet is one-sided, which is what makes the
    orientation well-defined — and accumulated to the facet closure points.
    The sign only flips the constrained row's sign, so either orientation
    imposes the same no-opening condition.
    """
    mesh = solver.mesh
    dm = solver.dm
    dim = mesh.dim
    lsec = dm.getLocalSection()
    fS, fE = dm.getHeightStratum(1)

    pairs = mesh._fault_point_pairs[boundary]

    plus_name = f"{boundary}Plus"
    value = mesh.boundaries[plus_name].value
    # Empty-stratum guard: a rank owning no part of the fault is the normal
    # case in parallel, and a null IS segfaults in getIndices().
    if not (dm.hasLabel(plus_name)
            and dm.getLabel(plus_name).getStratumSize(value) > 0):
        return []
    facets = [int(p) for p in
              dm.getLabel(plus_name).getStratumIS(value).getIndices()
              if fS <= int(p) < fE]

    nacc = {}
    for f in facets:
        _, cent, nrm = dm.computeCellGeometryFVM(f)
        ne = np.asarray(nrm, dtype=float)
        ne = ne / (np.linalg.norm(ne) + 1e-30)
        support = dm.getSupport(f)
        _, ccent, _ = dm.computeCellGeometryFVM(int(support[0]))
        if np.dot(ne, np.asarray(cent) - np.asarray(ccent)) < 0:
            ne = -ne
        for q in (int(c) for c in dm.getTransitiveClosure(f)[0]):
            if lsec.getFieldDof(q, _VELOCITY_FIELD) > 0:
                nacc[q] = nacc.get(q, np.zeros(dim)) + ne

    out = []
    for q_minus, q_plus in pairs.items():
        if lsec.getFieldDof(int(q_plus), _VELOCITY_FIELD) <= 0:
            continue
        acc = nacc.get(int(q_plus))
        if acc is None:
            # A paired point off the Plus closure would mean the pairing and
            # the labels disagree — a split bug, not a configuration error.
            raise RuntimeError(
                f"fault_contact: paired point {q_plus} carries velocity DOFs "
                f"but is not in the {plus_name!r} facet closure.")
        out.append((int(q_plus), int(q_minus),
                    acc / (np.linalg.norm(acc) + 1e-30)))
    return out


def _interface_slip_operator(solver, Q):
    r"""The interface constitutive operator on the rotated slip rows, or None.

    For every registered fault with :math:`\eta_f > 0`, assembles
    :math:`\hat K = 2\eta_f M` where :math:`M` is the consistent 1-D P2 trace
    mass over the fault facets and the factor 2 converts between the slip
    :math:`V` and the rotated slip row's value :math:`V/\sqrt 2` (the term is
    :math:`\eta_f\int V\,\delta V\,d\Gamma = 2\eta_f\,\hat u^T M\,\delta\hat
    u`). Rows/columns live only on slip rows, so the operator adds to the
    rotated Jacobian and contributes :math:`\hat K\hat u` to the rotated
    residual — the solve driver owns both hooks.

    A tip vertex has no slip DOF — both sides share the point, the jump
    space vanishes there — so its trace-mass entries are simply dropped:
    the interface law inherits V = 0 at the tips topologically, with no
    boundary condition on the interface system.

    ``Q`` is accepted for interface laws that need the rotated frame; the
    linear law only needs the row map, which is rebuilt here from the same
    deterministic pairing ``build_rotation`` used.
    """
    from petsc4py import PETSc

    registered = {name: eta for name, eta in
                  getattr(solver, "_fault_interface_viscosity", {}).items()
                  if eta > 0.0}
    if not registered:
        return None

    dm = solver.dm
    dim = solver.mesh.dim
    lsec = dm.getLocalSection()
    l2g = dm.getLGMap()
    csec = dm.getCoordinateSection()
    cvec = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dim)
    v0, v1 = dm.getDepthStratum(0)
    fS, fE = dm.getHeightStratum(1)

    A = solver.snes.getJacobian()[0]
    rstart, rend = A.getOwnershipRange()
    nloc = rend - rstart
    N = A.getSize()[0]
    K = PETSc.Mat().create(comm=dm.comm)
    K.setSizes(((nloc, N), (nloc, N)))
    K.setType("aij")
    K.setPreallocationNNZ((3, 0))
    K.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

    # Consistent mass of a P2 line element of unit length, nodes ordered
    # (end, midpoint, end).
    M_e = np.array([[4.0, 2.0, -1.0],
                    [2.0, 16.0, 2.0],
                    [-1.0, 2.0, 4.0]]) / 30.0

    for name, eta_f in registered.items():
        pairs = solver.mesh._fault_point_pairs[name]
        minus_of = {q_plus: q_minus for q_minus, q_plus in pairs.items()}

        def slip_row(q_plus):
            """Global row of the pair's rotated slip component, or None at a
            tip (no replica, no slip DOF)."""
            q_minus = minus_of.get(int(q_plus))
            if q_minus is None:
                return None
            lo = lsec.getFieldOffset(q_minus, _VELOCITY_FIELD)
            g = int(l2g.apply([lo + 1])[0])
            return g if g >= 0 else None

        plus_name = f"{name}Plus"
        value = solver.mesh.boundaries[plus_name].value
        if not (dm.hasLabel(plus_name)
                and dm.getLabel(plus_name).getStratumSize(value) > 0):
            continue
        facets = [int(p) for p in
                  dm.getLabel(plus_name).getStratumIS(value).getIndices()
                  if fS <= int(p) < fE]
        for f in facets:
            va, vb = (int(q) for q in dm.getCone(f))
            L = float(np.linalg.norm(cvec[csec.getOffset(va) // dim]
                                     - cvec[csec.getOffset(vb) // dim]))
            rows = [slip_row(va), slip_row(f), slip_row(vb)]
            for i, gi in enumerate(rows):
                # off-rank rows are legal: at a seam crossing the pair is
                # owned across the seam, and PETSc communicates stashed
                # entries at assembly.
                if gi is None:
                    continue
                for j, gj in enumerate(rows):
                    if gj is None:
                        continue
                    K.setValue(gi, gj, 2.0 * eta_f * L * M_e[i, j],
                               addv=True)
    K.assemble()
    return K


def fault_slip(solver, boundary, solve_result):
    """Slip and leak across the fault, per coincident pair, from the solve.

    Returns ``(s, V, leak)`` on this rank: the along-fault coordinate of each
    pair (arc-length-like, measured along the fault's mean tangent from the
    first tip), the tangential velocity jump :math:`V = \\hat t\\cdot(v^+ -
    v^-)`, and the normal jump :math:`\\hat n\\cdot(v^+ - v^-)` — the leak,
    which the strong constraint holds at machine zero. Reads the composite
    solution ``solve_result["U"]`` through the pairing, which is the only
    correct route: the pair coordinates are identical, so field queries by
    position see one side only.
    """
    dm = solver.dm
    dim = solver.mesh.dim
    lsec = dm.getLocalSection()
    l2g = dm.getLGMap()
    U = solve_result["U"]
    rs, re = U.getOwnershipRange()
    ua = np.asarray(U.getArray())

    csec = dm.getCoordinateSection()
    cvec = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dim)
    v0, v1 = dm.getDepthStratum(0)

    from underworld3.utilities.rotated_bc import _point_coord

    nodes = _fault_pair_nodes(solver, boundary)
    coords, jumps, normals = [], [], []
    for q_plus, q_minus, nrm in nodes:
        vals = {}
        for tag, q in (("plus", q_plus), ("minus", q_minus)):
            lo = lsec.getFieldOffset(q, _VELOCITY_FIELD)
            grows = [int(l2g.apply([lo + c])[0]) for c in range(dim)]
            if not all(rs <= g < re for g in grows):
                break
            vals[tag] = ua[[g - rs for g in grows]]
        if len(vals) < 2:
            continue
        coords.append(_point_coord(dm, dim, cvec, csec, v0, v1, q_plus))
        jumps.append(vals["plus"] - vals["minus"])
        normals.append(nrm)
    if not coords:
        return (np.zeros(0),) * 3
    coords = np.array(coords)
    jumps = np.array(jumps)
    normals = np.array(normals)
    tangents = np.column_stack([-normals[:, 1], normals[:, 0]])
    # Along-fault coordinate from the mean tangent, origin at the trailing end.
    tbar = tangents.mean(axis=0)
    tbar /= np.linalg.norm(tbar) + 1e-30
    s = coords @ tbar
    s -= s.min()
    order = np.argsort(s)
    V = np.einsum("ij,ij->i", jumps, tangents)
    leak = np.einsum("ij,ij->i", jumps, normals)
    return s[order], V[order], leak[order]
