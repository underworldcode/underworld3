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
import sympy
from petsc4py import PETSc

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
    if eta_f > 0.0:
        _register_law(solver, boundary, ViscousFaultLaw(eta_f))


#: The canonical slip-rate symbol an interface law is written in: a fault
#: law is ``tau(slip_rate)`` as a SYMPY expression, and its consistent
#: Newton tangent is derived by ``sympy.diff`` — never hand-coded. This is
#: the same division of labour as the bulk constitutive models: the law is
#: symbolic; only the innermost trace quadrature evaluates compiled
#: (lambdified) callables.
slip_rate = sympy.Symbol(r"V_{slip}", real=True)


class SymbolicFaultLaw:
    r"""An interface law :math:`\tau(V)` given as a sympy expression in
    :data:`slip_rate`.

    The consistent tangent :math:`d\tau/dV` is derived symbolically and
    both are lambdified ONCE at registration into fast numpy callables —
    the interface analogue of the JIT step for bulk kernels. A physical law
    should be monotone (:math:`d\tau/dV > 0`), which keeps the Newton
    block positive; nothing enforces it, exactly as nothing stops a
    negative bulk viscosity.
    """

    def __init__(self, tau_expr):
        expr = sympy.sympify(tau_expr)
        stray = expr.free_symbols - {slip_rate}
        if stray:
            raise ValueError(
                f"a fault law must be an expression in fault_contact."
                f"slip_rate alone (found {sorted(map(str, stray))}); "
                "substitute parameters before registering.")
        self.expr = expr
        self._tau = sympy.lambdify(slip_rate, expr, "numpy")
        self._dtau = sympy.lambdify(slip_rate, sympy.diff(expr, slip_rate),
                                    "numpy")

    def tau(self, V):
        V = np.asarray(V, dtype=float)
        return np.broadcast_to(np.asarray(self._tau(V), dtype=float),
                               V.shape).copy()

    def dtau_dV(self, V):
        V = np.asarray(V, dtype=float)
        return np.broadcast_to(np.asarray(self._dtau(V), dtype=float),
                               V.shape).copy()


def ViscousFaultLaw(eta_f):
    r"""The linear member, :math:`\tau = \eta_f V`, as a symbolic law."""
    return SymbolicFaultLaw(float(eta_f) * slip_rate)


def CoulombFaultLaw(mu, sigma_n, V0):
    r"""Regularised Coulomb friction,
    :math:`\tau = \mu\sigma_n\,\tfrac{2}{\pi}\arctan(V/V_0)`.

    Smooth, bounded by the strength :math:`\mu\sigma_n`, monotone. Below
    the regularisation velocity :math:`V_0` the fault sticks (residual
    creep :math:`\sim V_0`); above it the shear traction saturates and the
    fault slides at constant stress drop. ``sigma_n`` is PRESCRIBED here;
    feeding it from the no-opening constraint's reaction (Picard-lagged) is
    the follow-up increment. The tangent comes from ``sympy.diff``, not a
    hand-coded derivative.
    """
    strength = float(mu) * float(sigma_n)
    return SymbolicFaultLaw(
        strength * (2 / sympy.pi) * sympy.atan(slip_rate / float(V0)))


def add_coulomb_fault_bc(solver, conds, boundary, sigma_n=None, V0=1.0e-5):
    """Regularised Coulomb friction on the split fault ``boundary``
    (value-first: ``conds`` is the friction coefficient mu).

    ``sigma_n`` is the effective normal stress, prescribed and required in
    this version (reaction-fed sigma_n is the next increment). ``V0`` is the
    regularisation velocity — choose it well below the slip rates the flow
    produces; below it the fault sticks (creep ~ V0), above it the shear
    traction saturates at mu*sigma_n and the fault slides at constant
    stress drop.
    """
    if sigma_n is None:
        raise ValueError(
            "add_coulomb_fault_bc requires sigma_n (the prescribed effective "
            "normal stress) in this version.")
    mu = float(conds)
    if mu <= 0.0 or float(sigma_n) <= 0.0 or float(V0) <= 0.0:
        raise ValueError("mu, sigma_n and V0 must all be positive.")
    add_frictionless_fault_bc(solver, boundary)
    _register_law(solver, boundary, CoulombFaultLaw(mu, sigma_n, V0))


def _register_law(solver, boundary, law):
    registered = dict(getattr(solver, "_fault_interface_laws", {}))
    registered[boundary] = law
    solver._fault_interface_laws = registered


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


# 3-point Gauss-Legendre on [0,1] (exact through quartics — the tangent's
# N_i N_j weighting of a P2 trace is quartic) and the quadratic line shape
# functions at those points, node order (end, midpoint, end).
_XI = 0.5 + (np.sqrt(15.0) / 10.0) * np.array([-1.0, 0.0, 1.0])
_WQ = np.array([5.0, 8.0, 5.0]) / 18.0
_NQ = np.column_stack([2.0 * (_XI - 0.5) * (_XI - 1.0),
                       4.0 * _XI * (1.0 - _XI),
                       2.0 * _XI * (_XI - 0.5)])


class _InterfaceAssembler:
    r"""Residual and consistent tangent of the interface laws, per iterate.

    The interface term is :math:`\int_\Gamma \tau(V)\,\delta V\,d\Gamma`
    on the fault trace. In the rotated frame the slip row carries
    :math:`V/\sqrt2`, so the residual lands on the slip rows as
    :math:`\sqrt2\,L\sum_q w_q\,\tau(V_q)N_i` per facet and the tangent
    as :math:`2\,L\sum_q w_q\,(d\tau/dV)(V_q)\,N_iN_j` — for a linear
    law exactly :math:`2\eta_f M`, the measured dashpot. Geometry (facet
    tables, pair offsets, tangent vectors) is cached at construction; each
    call evaluates the lambdified law at the CURRENT iterate's slip rates,
    which is what makes the tangent consistent Newton rather than Picard.

    Tip nodes have no slip DOF (the jump space vanishes there): they enter
    the quadrature with V = 0 and receive no row or column. A crossing
    pair's row may be owned across the seam; both Vec and Mat additions go
    through PETSc's off-process stash. All entry points are COLLECTIVE —
    ranks holding no fault still participate in the assemblies.
    """

    def __init__(self, solver):
        dm = solver.dm
        dim = solver.mesh.dim
        lsec = dm.getLocalSection()
        l2g = dm.getLGMap()
        csec = dm.getCoordinateSection()
        cvec = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dim)
        fS, fE = dm.getHeightStratum(1)

        rows, lo_p, lo_m, tans, laws_of, facets = [], [], [], [], [], []
        index_of = {}

        for name, law in getattr(solver, "_fault_interface_laws",
                                 {}).items():
            pairs = solver.mesh._fault_point_pairs[name]
            minus_of = {qp: qm for qm, qp in pairs.items()}
            normals = {q: n for q, _qm, n in
                       ((q_plus, q_minus, nrm) for q_plus, q_minus, nrm in
                        _fault_pair_nodes(solver, name))}

            def node_index(q):
                q = int(q)
                qm = minus_of.get(q)
                if qm is None:
                    return -1                     # a tip: V = 0, no DOF
                if q in index_of:
                    return index_of[q]
                lo = lsec.getFieldOffset(qm, _VELOCITY_FIELD)
                g = int(l2g.apply([lo + 1])[0])
                nrm = normals[q]
                index_of[q] = len(rows)
                rows.append(g)
                lo_p.append(lsec.getFieldOffset(q, _VELOCITY_FIELD))
                lo_m.append(lsec.getFieldOffset(qm, _VELOCITY_FIELD))
                tans.append((-nrm[1], nrm[0]))
                return index_of[q]

            plus_name = f"{name}Plus"
            value = solver.mesh.boundaries[plus_name].value
            if not (dm.hasLabel(plus_name) and
                    dm.getLabel(plus_name).getStratumSize(value) > 0):
                continue
            law_id = len(laws_of)
            laws_of.append(law)
            for f in (int(q) for q in dm.getLabel(plus_name)
                      .getStratumIS(value).getIndices()):
                if not (fS <= f < fE):
                    continue
                va, vb = (int(q) for q in dm.getCone(f))
                L = float(np.linalg.norm(
                    cvec[csec.getOffset(va) // dim]
                    - cvec[csec.getOffset(vb) // dim]))
                facets.append((node_index(va), node_index(f),
                               node_index(vb), L, law_id))

        # petsc4py refuses to narrow index arrays — Vec/Mat setValues need
        # PETSc's own integer type.
        self._rows = np.asarray(rows, dtype=PETSc.IntType)
        self._lo_p = np.asarray(lo_p, dtype=np.int64)
        self._lo_m = np.asarray(lo_m, dtype=np.int64)
        self._tan = np.asarray(tans, dtype=float).reshape(-1, 2)
        self._laws = laws_of
        self._facets = facets
        self._dim = dim

    def _nodal_slip(self, solver, uvec):
        """V at every law-carrying pair node, from the CURRENT iterate —
        read through the local vector so ghosted (cross-seam) values
        resolve, and padded with the tips' identically-zero slip."""
        dm = solver.dm
        lvec = dm.getLocalVec()
        dm.globalToLocal(uvec, lvec)
        a = np.asarray(lvec.getArray())
        V = np.zeros(len(self._rows) + 1)
        for k in range(len(self._rows)):
            du = (a[self._lo_p[k]:self._lo_p[k] + 2]
                  - a[self._lo_m[k]:self._lo_m[k] + 2])
            V[k] = float(self._tan[k] @ du)
        dm.restoreLocalVec(lvec)
        return V                                   # V[-1] == 0: the tip pad

    def residual_add(self, solver, uvec, Fh):
        """Add the interface force to the ROTATED residual, in place.
        COLLECTIVE (the Vec assembly): every rank calls, fault or not."""
        V = self._nodal_slip(solver, uvec)
        vals = np.zeros(len(self._rows))
        s2 = np.sqrt(2.0)
        for ia, im, ib, L, law_id in self._facets:
            idx = (ia, im, ib)
            Vq = _NQ @ V[list(idx)]
            tq = self._laws[law_id].tau(Vq)
            contrib = s2 * L * (_NQ.T @ (_WQ * tq))
            for j, k in enumerate(idx):
                if k >= 0:
                    vals[k] += contrib[j]
        if len(self._rows):
            Fh.setValues(self._rows, vals, addv=True)
        Fh.assemblyBegin()
        Fh.assemblyEnd()

    def tangent(self, solver, uvec):
        """The consistent interface tangent at the current iterate, as a
        fresh Mat on the composite layout (caller destroys). COLLECTIVE."""
        dm = solver.dm
        A = solver.snes.getJacobian()[0]
        rstart, rend = A.getOwnershipRange()
        K = PETSc.Mat().create(comm=dm.comm)
        K.setSizes(((rend - rstart, A.getSize()[0]),
                    (rend - rstart, A.getSize()[0])))
        K.setType("aij")
        K.setPreallocationNNZ((3, 0))
        K.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

        V = self._nodal_slip(solver, uvec)
        for ia, im, ib, L, law_id in self._facets:
            idx = (ia, im, ib)
            Vq = _NQ @ V[list(idx)]
            dq = self._laws[law_id].dtau_dV(Vq)
            Ke = 2.0 * L * (_NQ.T @ (_NQ * (_WQ * dq)[:, None]))
            for i, ki in enumerate(idx):
                if ki < 0:
                    continue
                for j, kj in enumerate(idx):
                    if kj >= 0:
                        K.setValue(int(self._rows[ki]),
                                   int(self._rows[kj]),
                                   float(Ke[i, j]), addv=True)
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
