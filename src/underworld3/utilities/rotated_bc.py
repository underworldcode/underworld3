"""Rotated strong free-slip for the Stokes saddle (the implementation behind
``solver.add_rotated_freeslip_bc``): build a per-node rotation Q from boundary
normals, then drive ONE manual Newton/Picard loop that rotates the residual and
tangent every iteration (Ĵ=Q J Qᵀ, F̂=Q F), imposes v_n = ũ_n strongly on the
rotated normal rows (ũ_n = 0 is pure free-slip; a datum enters cold starts via the
first increment's affine lift, feasible iterates thereafter), rotates back u=Qᵀû,
removes the rigid-rotation gauge, and exposes σ_nn as the constraint reaction.
A linear model is simply the loop converging after its first increment — there is
no separate linear path and no up-front nonlinearity probe.

Each increment is solved by a self-contained fieldsplit-Schur KSP by default: the
velocity block is geometric FMG on the custom prolongation whenever this solver has a
hierarchy — registered by ``set_custom_fmg`` or owned by an ``adapt()`` child mesh —
which is the PREFERRED route, else GAMG; the option bundle for both is the shared one
in ``underworld3.utilities.multigrid_options``, so the rotated velocity block is
configured identically to the native and standard custom-P routes.
The Schur complement is preconditioned by the native 1/mu pressure mass
(the Pmat p-p block, exactly the standard path's ``schur_precondition=a11``), with a
constant-pressure nullspace on the inner Schur solve for enclosed domains; direct
MUMPS LU is opt-in via ``solver._rotated_use_lu``.
σ_nn / dynamic topography reuse the shared Consistent-Boundary-Flux de-smear in
``underworld3.utilities.boundary_flux``.
"""
import numpy as np
import sympy
from petsc4py import PETSc

# Rank-safe printing only (mpi.pprint). Safe at module top: underworld3.mpi is a
# leaf module loaded before underworld3.utilities in the package __init__.
from underworld3 import mpi

# Shared Consistent-Boundary-Flux machinery (the σ_nn recovery is a rotated-frame reading
# of the same primitive): the consolidated-label boundary stratum, the lumped/consistent
# boundary-mass de-smear, and the scalar-field hand-off all live in `boundary_flux`.
from underworld3.utilities.boundary_flux import (
    _boundary_stratum_is, _desmear, write_boundary_scalar_field)

# The outward-orientation rule for a boundary facet is shared with the two sibling
# accumulators (Mesh._assemble_boundary_normal, boundary_flux._node_normals) so the
# three cannot drift apart again — they did, and #564 was the bill.
from underworld3.utilities.facet_normals import facet_measure_and_normal

# The multigrid option bundles are owned in ONE place and shared with the native
# and standard custom-P routes, so the rotated velocity block cannot be
# configured differently from the others (#468).
from underworld3.utilities import multigrid_options

# Monotonic counter so each rotated solve gets a UNIQUE PETSc options prefix. With a
# fixed prefix, sequential rotated solves (e.g. two solvers in one script, or one
# solver in a time-stepping loop) share and re-set the same global-options keys; the
# per-solve prefix plus the delValue cleanup below keeps each solve's options local
# and stops the global database growing / emitting "unused option" warnings.
_ROT_SOLVE_COUNT = 0


def _warn_if_ksp_diverged(ksp, kind):
    """Emit a rank-0 warning if a KSP finished on a KSP_DIVERGED_* reason
    (negative converged-reason). ``ksp.solve`` never raises on divergence, so
    without this the linear rotated-freeslip solvers here would return a
    partial answer to the caller silently — which is exactly what let the 3D
    rotation-nullspace bug (#306) go unnoticed until it produced clearly-wrong
    physics in a downstream test."""
    reason = int(ksp.getConvergedReason())
    # Convention: > 0 == converged, < 0 == diverged, 0 == KSP_CONVERGED_ITERATING
    # (should not happen after ksp.solve() returns; warn if it does). Matches
    # petsc_generic_snes_solvers.pyx:2979.
    if reason > 0:
        return
    its = int(ksp.getIterationNumber())
    try:
        rnorm = float(ksp.getResidualNorm())
    except Exception:
        rnorm = float("nan")
    mpi.pprint(f"[rotated_bc] WARNING: {kind} KSP did NOT converge "
               f"(reason={reason}, iterations={its}, |r|={rnorm:.3e}); "
               f"proceeding with the last iterate.")


# --------------------------------------------------------------------------- #
#  Rotation construction
# --------------------------------------------------------------------------- #

# PETSc section field ids of the Stokes saddle unknowns (fixed by the solver's
# DM field registration order: velocity first, pressure second).
_VELOCITY_FIELD = 0
_PRESSURE_FIELD = 1


def _boundary_spec(spec):
    """Normalize one entry of a rotated-free-slip boundary list to ``(name, normal)``:
    a bare boundary name means the geometric facet normal (``normal=None``); a
    ``(name, normal)`` pair carries an analytic/constant normal override (see
    ``_boundary_velocity_nodes`` for the normal conventions)."""
    return spec if isinstance(spec, tuple) else (spec, None)


def _point_coord(dm, dim, cvec, csec, v0, v1, q):
    """Coordinate of a DMPlex point (vertex → its coord; higher point → mean of
    its closure vertices)."""
    if v0 <= q < v1:
        return cvec[csec.getOffset(q) // dim]
    clo = dm.getTransitiveClosure(q)[0]
    verts = [int(c) for c in clo if v0 <= c < v1]
    return np.mean([cvec[csec.getOffset(v) // dim] for v in verts], axis=0)


def _boundary_velocity_nodes(solver, boundary, normal=None):
    """DMPlex points carrying velocity DOFs on `boundary`, each with an outward
    unit normal. Dimension-general (2D edges, 3D faces).

    `normal` selects the normal source (per boundary):
      * None      — geometric facet normal from PETSc ``computeCellGeometryFVM``,
                    accumulated to the facet closure points weighted by the facet
                    MEASURE (edge length in 2D, face area in 3D). The weight is
                    what makes the constraint consistent with the assembly — see
                    the note below.
      * sympy 1×dim Matrix — an analytic normal (function of mesh.X); evaluated
                    at each node's coordinate (radial ``X/|X|`` on a spherical
                    cap; a constant on a planar side of a regional spherical
                    box). Exact for the TRUE surface, which is a different thing
                    from being consistent with the straight-facet assembly — see
                    "Which normal to use" in
                    ``docs/developer/subsystems/rotated-freeslip.md``.
      * (dim,) array — a constant normal vector.
    Returns ``[(point, n̂), ...]``.

    Why the geometric normals are MEASURE-weighted (issue #560)
    -----------------------------------------------------------
    A boundary node sitting on more than one facet — a vertex in 2D, a vertex or an
    edge-midpoint in 3D — gets ONE nodal normal, while the assembler integrates the
    boundary term facet by facet. The rotated frame's free tangential row therefore
    collects, for a constant pressure :math:`p`,

    .. math::
        r_i = p \\sum_f \\left(\\int_f \\phi_i\\, ds\\right)\\, (\\hat t_i \\cdot n_f)

    which vanishes for every tangent :math:`\\hat t_i` only when the nodal normal is
    parallel to :math:`\\sum_f (\\int_f \\phi_i\\, ds)\\, n_f`. For simplicial P1 and P2
    velocity the basis integral over a facet is the facet measure times a constant that
    does not depend on which facet it is, so the measure IS the right weight. That is
    EXACT for every simplicial facet (2D P2 vertex :math:`|f|/6`; 3D P1 vertex
    :math:`|f|/3`; 3D P2 edge-midpoint :math:`|f|/3`; the 3D P2 vertex integral is
    identically zero, so that row is consistent whatever normal it is given) and for
    every 2D facet at any degree, a facet being a 1D element there. On a **non-affine
    3D quad facet** (a deformed hex) the Jacobian varies across the facet, so the
    measure is the leading-order weight rather than the exact one — measured, a
    deformed hex box retains a residual ~7-16x its flat control while the bisector
    leaves 3e8-2e9 times more, so the weighting is a large improvement there but not
    exact.

    Normalising each facet normal to unit length before accumulating — what this
    function used to do — gives the BISECTOR :math:`n_1 + n_2` instead, which is
    correct only where the facets are equal. On a kinked boundary with unequal facets
    the constant-pressure vector then stops being a null vector of the constrained
    operator, the pressure gauge goes unpinned, and the answer picks up a round-off
    seeded offset (#560).

    An axis-aligned wall is unchanged to the last bit: its facet normals have exactly
    0/±1 components, so :math:`\\sum_f |f| n_f` normalises to the same floats as
    :math:`\\sum_f n_f` whatever the weights. A flat but TILTED wall can move by one
    ulp, which is why this reads "axis-aligned" and not "flat".

    Parallel
    --------
    The sum is over ALL facets meeting the node, so it has to be completed ACROSS
    RANKS: each boundary facet is labelled on exactly one rank, and a node on a
    partition seam has its adjacent facets split between ranks (both edges are in the
    local mesh, only one carries the label). Accumulating rank-locally would give a
    seam node the wrong normal and make the answer rank-count dependent. The weighted
    contributions are therefore summed over the DMPlex point SF before normalising, so
    every rank ends with a bit-identical normal at a shared node. The reduction is
    COLLECTIVE — a rank owning no facet of this boundary still takes part.

    The analytic-normal path is NOT reduced: it evaluates a function of the node
    coordinate, so every rank already computes the same value and summing copies would
    only rescale it (and by a rank-count-dependent factor).
    """
    dm = solver.dm
    dim = solver.mesh.dim
    csec = dm.getCoordinateSection()
    cvec = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dim)
    v0, v1 = dm.getDepthStratum(0)
    lsec = dm.getLocalSection()
    # Boundary facets via the consolidated "UW_Boundaries" label (per-boundary labels do
    # not survive mesh adaptation); raises a clear error for an unknown boundary name.
    # In parallel a rank may own NO part of this boundary → a null IS; guard and return
    # no local nodes (calling getIndices() on a null IS would segfault).
    # A rank owning no part of this boundary gets a null/empty IS and contributes
    # nothing — but it must NOT return early: the cross-rank sum below is collective.
    sis = _boundary_stratum_is(dm, solver.mesh, boundary)
    facets = ([int(z) for z in sis.getIndices()]
              if (sis and sis.getSize() > 0) else [])
    fS, fE = dm.getHeightStratum(1)              # facets (edges in 2D, faces in 3D)

    def coord(q):
        return _point_coord(dm, dim, cvec, csec, v0, v1, q)

    # analytic / constant normal specs. An analytic (sympy) normal is LAMBDIFIED
    # once into a fast numpy callable — per-node sympy .subs() is orders of magnitude
    # slower and, in parallel, serialises on the rank that owns the boundary (the
    # others idle), which looked like a hang. Components are UNWRAPPED first so a
    # normal written in curvilinear terms (e.g. ``mesh.CoordinateSystem.unit_e_0``,
    # whose UWexpressions hide the Cartesian coordinates behind ``r`` etc.) reduces
    # to pure mesh.X before lambdify — otherwise the generated function references
    # the curvilinear symbol by bare name and dies with a NameError at call time.
    sym_fn = None
    const_normal = None
    if normal is not None:
        if isinstance(normal, sympy.Matrix):
            from underworld3.function.expressions import unwrap
            comps = [sympy.sympify(unwrap(normal[0, k], keep_constants=False,
                                          return_self=False))
                     for k in range(dim)]
            stray = set().union(*[c.free_symbols for c in comps]) \
                - set(solver.mesh.X)
            if stray:
                raise ValueError(
                    f"analytic normal for boundary '{boundary}' contains "
                    f"symbols {sorted(map(str, stray))} that are not mesh "
                    "coordinates — express it in mesh.X (e.g. radial X/|X|).")
            sym_fn = sympy.lambdify(list(solver.mesh.X), comps, "numpy")
        else:
            const_normal = np.asarray(normal, dtype=float).ravel()

    nacc = {}                                    # velocity node → Σ_f measure · n̂_f
    for f in facets:
        if not (fS <= f < fE):
            continue
        # facet outward normal, and the measure the boundary term is integrated over
        # (1.0 on the analytic path, whose normal does not come from the facet).
        #
        # The orientation rule lives in ONE place, `facet_measure_and_normal`:
        # outward = away from the one cell this boundary facet belongs to, which is
        # the domain's outward normal on ANY boundary, convex or not, and which needs
        # no global reference point. Read its docstring for why the obvious
        # alternative (away from the mean of the mesh coordinates) is both rank-local
        # and inward on a concave boundary, and for the internal-boundary guard.
        #
        # CONVENTION, and it is user-visible: on a concave boundary — an annulus or
        # shell INNER arc, the CMB — this is the opposite of what UW3 produced before
        # #560, so σ_nn and dynamic topography read there through the GEOMETRIC normal
        # reverse sign against earlier releases. The new sign is the one the
        # docstrings have always claimed ("outward"). An analytic ``normal=`` is used
        # exactly as given and is NOT reoriented, so ``X/|X|`` on an inner arc points
        # into the domain and disagrees in sign with the default — see "Which normal
        # to use" in ``docs/developer/subsystems/rotated-freeslip.md``.
        wgt = 1.0
        if normal is None:
            vol, ne, _exterior = facet_measure_and_normal(dm, f)
            wgt = vol
        # all velocity points on this facet (closure): verts + edges(3D) + the facet
        clo = dm.getTransitiveClosure(f)[0]
        for q in (int(c) for c in clo):
            if lsec.getFieldDof(q, _VELOCITY_FIELD) <= 0:
                continue
            if normal is not None:
                if sym_fn is not None:
                    cq = coord(q)
                    ne = np.asarray(sym_fn(*cq), dtype=float).ravel()
                else:
                    ne = const_normal.copy()
                ne = ne / (np.linalg.norm(ne) + 1e-30)
            nacc[q] = nacc.get(q, np.zeros(dim)) + wgt * ne

    if normal is None:
        nacc = _sum_facet_normals_across_ranks(solver, nacc)

    return [(q, v / (np.linalg.norm(v) + 1e-30))
            for q, v in sorted(nacc.items())
            if np.linalg.norm(v) > 0.0]


def _sum_facet_normals_across_ranks(solver, contribs):
    """Complete the per-node facet sum across ranks: ``{point: Σ_f |f| n̂_f}`` restricted
    to this rank's facets in, the sum over EVERY rank's facets out — the same value on
    every rank that holds the node.

    Each boundary facet is labelled on exactly one rank, and on the rank that owns it
    (measured: no facet is labelled twice and none is labelled away from its owner), so
    a plain ADD is exact and needs no de-duplication. The velocity field already has
    ``dim`` DOFs at exactly these points, so the sum rides the DM's own local↔global
    scatter: ADD into the global vector accumulates the ghost copies onto the owner,
    and scattering back gives every rank the identical total.

    COLLECTIVE — a rank owning no facet of this boundary still takes part.
    """
    dm = solver.dm
    dim = solver.mesh.dim
    if dm.comm.getSize() == 1:
        return contribs

    lsec = dm.getLocalSection()
    lvec = dm.getLocalVec()
    gvec = dm.getGlobalVec()
    try:
        lvec.set(0.0)
        larr = lvec.getArray()
        for q, v in contribs.items():
            lo = lsec.getFieldOffset(q, _VELOCITY_FIELD)
            larr[lo:lo + dim] = v
        gvec.set(0.0)
        dm.localToGlobal(lvec, gvec, addv=PETSc.InsertMode.ADD_VALUES)
        dm.globalToLocal(gvec, lvec)
        summed = lvec.getArray()
        out = {}
        for q in _local_boundary_candidates(dm, lsec) | set(contribs):
            lo = lsec.getFieldOffset(q, _VELOCITY_FIELD)
            w = np.array(summed[lo:lo + dim], dtype=float)
            # LOAD-BEARING: the candidate set over-collects (see
            # _local_boundary_candidates); a point off this boundary contributes
            # nothing anywhere, so its summed normal is exactly zero and it is
            # dropped here. This is what makes the over-collection safe.
            if w.any():
                out[q] = w
            elif q in contribs:
                # Velocity DOFs constrained out of the global vector (an essential
                # BC meeting this boundary) come back zero. Keep this rank's own
                # contribution so the node set is exactly what it always was.
                # build_rotation drops these nodes (l2g < 0); boundary_normal_traction
                # does NOT — it reads n̂·r_c at every returned node — so on such a node
                # it reports the reaction of the ESSENTIAL constraint along a
                # rank-locally accumulated normal. That was true before this change
                # too, and the nodes are the corners where a rotated wall meets an
                # essential one (measured: 2 of 75 on the test_1070 configuration, at
                # every rank count, and never shared between ranks there).
                out[q] = contribs[q]
        return out
    finally:
        dm.restoreLocalVec(lvec)
        dm.restoreGlobalVec(gvec)


def _local_boundary_candidates(dm, lsec):
    """Velocity points on a support-1 facet of this rank's LOCAL mesh — a SUPERSET of
    its boundary nodes, deliberately.

    The labelled facet list is not enough to enumerate a rank's boundary nodes: a rank
    can OWN a node every one of whose labelled facets lives on a neighbour (the label
    goes to one rank per facet, so a seam node's facets are split between them), and
    such a node would otherwise never get a constraint row.

    Support 1 does NOT mean "on the domain boundary": the facets on the outer fringe of
    the overlap layer also have support 1 locally while being interior globally, and
    measured they are 13-27% of what this returns. The caller is what makes the result
    correct — those points receive no contribution from any labelled facet, so their
    summed normal is exactly zero and the ``w.any()`` test in
    :func:`_sum_facet_normals_across_ranks` drops them. That test is load-bearing, not
    an aside; do not remove it while over-collecting here.
    """
    fS, fE = dm.getHeightStratum(1)
    out = set()
    for f in range(fS, fE):
        if dm.getSupportSize(f) != 1:
            continue
        for q in (int(c) for c in dm.getTransitiveClosure(f)[0]):
            if lsec.getFieldDof(q, _VELOCITY_FIELD) > 0:
                out.add(q)
    return out


def _eval_datum(dspec, coords, solver):
    """Prescribed wall-normal velocity datum at an array of node coordinates. A number
    is uniform; a sympy expression / MeshVariable ``.sym`` is evaluated (interpolated)
    at the coordinates."""
    if isinstance(dspec, (int, float, np.floating, np.integer)):
        return np.full(len(coords), float(dspec))
    import underworld3 as uw
    return np.asarray(uw.function.evaluate(dspec, np.ascontiguousarray(coords))).flatten()


def build_rotation(solver, boundaries, datum_specs=None):
    """Global sparse rotation Q on the composite saddle vector: identity except a
    per-node (normal,tangential) block at each velocity node of `boundaries`.
    Returns ``(Q, Qt, normal_rows, datum_map)`` where normal_rows are the global rows
    carrying the rotated NORMAL velocity component (v_n = n̂·v) to be strongly
    constrained, and ``datum_map`` is ``{global_row: value}`` giving the prescribed
    ``v_n`` at each simple (single-normal) constrained node — empty for pure free-slip
    (``u.n = 0``). ``datum_specs`` maps a boundary name to its datum (a constant, sympy
    expression, or field ``.sym``); absent/None ⇒ free-slip on that boundary.
    """
    dm = solver.dm
    lsec = dm.getLocalSection()
    l2g = dm.getLGMap()
    dim = solver.mesh.dim

    # gather all normals per velocity node across the boundaries (and, per node, the
    # datum of the boundary it came from — the surface's ũ_n for a prescribed-normal
    # solve; None for free-slip).
    node_normals = {}
    node_dspec = {}
    for spec in boundaries:
        name, normal = _boundary_spec(spec)
        dspec = None if datum_specs is None else datum_specs.get(name)
        for q, nrm in _boundary_velocity_nodes(solver, name, normal=normal):
            node_normals.setdefault(q, []).append(nrm)
            if dspec is not None:
                node_dspec.setdefault(q, dspec)

    # Split-fault contact pairs (fault_contact.add_frictionless_fault_bc):
    # their 2·dim mean/jump blocks widen the widest row of Q.
    fault_names = getattr(solver, "_fault_contact_faults", [])

    # Distributed Q with the assembled operator's ROW layout. Q is identity except a
    # per-node dim×dim orthonormal block; because a node's dim velocity components
    # live on a SINGLE DMPlex point owned by ONE rank, each block is entirely within
    # that rank's diagonal portion — no off-rank columns. Each rank sets ONLY its
    # owned rows (ghost copies of a shared boundary node are skipped: their global
    # rows fall outside [rstart, rend)). A fault PAIR block spans two points, but
    # both are rank-local (a seam-touching fault is redistributed onto one rank
    # before the split), so the diagonal-portion argument carries over unchanged.
    A = solver.snes.getJacobian()[0]
    rstart, rend = A.getOwnershipRange()
    nloc = rend - rstart
    N = A.getSize()[0]
    Q = PETSc.Mat().create(comm=dm.comm)
    Q.setSizes(((nloc, N), (nloc, N)))
    Q.setType("aij")
    Q.setPreallocationNNZ((2 * dim if fault_names else dim, 0))
    Q.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    for i in range(rstart, rend):
        Q.setValue(i, i, 1.0)                    # identity default (owned rows)

    normal_rows = []
    datum_nodes = []                             # (grows[0], q, dspec) for simple nodes
    for q, nrms in node_normals.items():
        lo = lsec.getFieldOffset(q, _VELOCITY_FIELD)
        grows = [int(l2g.apply([lo + c])[0]) for c in range(dim)]
        if any(g < 0 for g in grows):
            continue
        if not (rstart <= grows[0] < rend):      # not owned by this rank → skip
            continue
        # DIMENSION-GENERAL constraint frame: the accumulated normals span a rank-r
        # subspace the velocity must be orthogonal to (r=1 face → constrain v_n;
        # r=2 edge → constrain 2, free the edge tangent; r=dim corner → v=0). The SVD
        # right vectors are a complete orthonormal frame E; Q's node block rows = E
        # (rotated component i = E[i]·v); constrain the first r rotated rows.
        M = np.array(nrms, dtype=float)
        _, sv, Vt = np.linalg.svd(M)
        r = int((sv > 1e-8 * (sv[0] if sv.size else 1.0)).sum())
        for i in range(dim):
            for j in range(dim):
                Q.setValue(grows[i], grows[j], float(Vt[i, j]))
        normal_rows.extend(grows[:r])            # constrain the r normal-space rows
        # A prescribed v_n datum is only well-defined at a SIMPLE node (one normal,
        # r=1): the rotated row grows[0] carries E[0]·v with E[0]=Vt[0]=±n̂ (SVD sign
        # ambiguity), so the constrained component is sgn·v_n. Store sgn = sign(Vt[0]·n̂)
        # to impose v_n=datum via row value sgn·datum. A corner/edge (r>1) has no single
        # normal → free-slip there (datum ignored), consistent with u=0 pinning.
        if r == 1 and q in node_dspec:
            nn = M.sum(axis=0); nn = nn / (np.linalg.norm(nn) + 1e-30)
            sgn = 1.0 if float(np.dot(Vt[0], nn)) >= 0.0 else -1.0
            datum_nodes.append((grows[0], q, node_dspec[q], sgn))

    # Split-fault contact: one orthogonal 2·dim block per coincident DOF pair,
    # transforming (v⁺, v⁻) to mean/jump components in the fault (n̂,t̂) frame —
    # mean rows written on the Plus point's rows, jump rows on the Minus
    # point's. The jump-NORMAL row joins the constrained set (no opening,
    # datum 0); the jump-tangent (slip) row stays free, which IS the
    # frictionless zero-shear-traction condition — its conjugate reaction is
    # the shear traction and an unconstrained row carries none. (A future
    # datum on the slip row is jump-only prescribed slip; a nonlinear relation
    # on it is friction.)
    if fault_names:
        from underworld3.utilities import fault_contact
        s2 = 1.0 / np.sqrt(2.0)
        # Rank-local failures in this block (a pairing/label mismatch or a
        # vanishing analytic normal inside _fault_pair_nodes, a fault node
        # shared with a wall boundary, a pair straddling ranks) must not
        # leave the peers blocked in the collective Q assembly below —
        # collect the verdict and raise it identically on every rank.
        # ``fault_names`` is registration state, identical across ranks, so
        # every rank reaches the verdict exchange.
        problem = None
        try:
            for fname in fault_names:
                for q_plus, q_minus, nrm in fault_contact._fault_pair_nodes(
                        solver, fname):
                    if q_plus in node_normals or q_minus in node_normals:
                        raise ValueError(
                            f"fault {fname!r} shares a node with a rotated "
                            "free-slip boundary; a point cannot carry both a "
                            "wall block and a pair block.")
                    # The tangent frame: one in-fault direction in 2-D, two
                    # in 3-D. Shared with the interface assembler's slip
                    # rows — _tangent_frame is the single authority. The
                    # specific in-plane directions carry no physics: both
                    # slip rows are free (frictionless) or enter through
                    # |V| (laws).
                    frame = [nrm] + fault_contact._tangent_frame(nrm)
                    gp, gm = [], []
                    for q, rows in ((q_plus, gp), (q_minus, gm)):
                        lo = lsec.getFieldOffset(q, _VELOCITY_FIELD)
                        rows.extend(int(l2g.apply([lo + c])[0])
                                    for c in range(dim))
                    if any(g < 0 for g in gp + gm):
                        continue
                    owned_p = rstart <= gp[0] < rend
                    owned_m = rstart <= gm[0] < rend
                    if owned_p != owned_m:
                        # Structurally impossible after the fault-aware
                        # redistribution (the split runs rank-interior),
                        # but "impossible by the seam rules" is exactly
                        # where surprises have lived — keep the check,
                        # and make its verdict collective like the rest.
                        raise RuntimeError(
                            "fault_contact: a coincident pair straddles "
                            "ranks — the split's redistribution should "
                            "make this impossible.")
                    if not owned_p:
                        continue
                    for i, e in enumerate(frame):
                        for row, sgn in ((gp[i], +1.0), (gm[i], -1.0)):
                            for j in range(dim):
                                Q.setValue(row, gp[j], s2 * float(e[j]))
                                Q.setValue(row, gm[j],
                                           sgn * s2 * float(e[j]))
                    normal_rows.append(gm[0])      # [v]·n̂ = 0, datum 0
        except (RuntimeError, ValueError) as err:
            problem = str(err)
        fault_contact._collective_raise(problem)

    Q.assemble()
    Qt = Q.transpose(PETSc.Mat())

    # Evaluate the prescribed normal datum at each simple constrained node, batched per
    # distinct datum spec (usually one — the surface). Rotated row grows[0] == v_n.
    datum_map = {}
    if datum_nodes:
        csec = dm.getCoordinateSection()
        cvec = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dim)
        v0, v1 = dm.getDepthStratum(0)
        by_spec = {}
        for grow0, q, dspec, sgn in datum_nodes:
            by_spec.setdefault(id(dspec), (dspec, []))[1].append((grow0, q, sgn))
        for dspec, items in by_spec.values():
            coords = np.array([_point_coord(dm, dim, cvec, csec, v0, v1, q)
                               for _, q, _ in items])
            vals = _eval_datum(dspec, coords, solver)
            for (grow0, _, sgn), val in zip(items, vals):
                # A zero datum IS free-slip; leave those rows out so an all-zero datum
                # takes the untouched free-slip RHS path and is bit-identical to it.
                if val != 0.0:
                    datum_map[grow0] = float(sgn * val)
    return Q, Qt, sorted(set(normal_rows)), datum_map


def _zero_rows_local(vec, normal_rows):
    """Zero ``vec`` at the global rows ``normal_rows`` using ownership-relative
    local indices (indexing the local slice with global rows overflows on any rank
    whose ownership does not start at 0 — the np>1 crash class)."""
    rs, re = vec.getOwnershipRange()
    loc = np.asarray([g - rs for g in normal_rows if rs <= g < re], dtype=np.int64)
    a = vec.getArray()
    a[loc] = 0.0
    vec.setArray(a)


def _set_rows_local(vec, row_val_map):
    """Set ``vec[g] = val`` for ``{global_row: val}`` using ownership-relative local
    indices (the np>1-safe counterpart of :func:`_zero_rows_local`). Used to restore a
    prescribed rotated normal datum (v_n = ũ_n) after the solve's v_n=0 cleanup."""
    rs, re = vec.getOwnershipRange()
    a = vec.getArray()
    for g, val in row_val_map.items():
        if rs <= g < re:
            a[g - rs] = val
    vec.setArray(a)


# --------------------------------------------------------------------------- #
#  Cross-solve workspace reuse (issue #417)
#
#  Repeated rotated free-slip solves used to rebuild the whole linear
#  workspace every time: the rotation Q, the PtAP'd operator, the fieldsplit
#  Schur KSP/PC and its GAMG (or FMG) hierarchy. In a time-stepping loop that
#  is both slow (distributed assembly + PCSetUp every step) and an allocator
#  high-water problem (a fresh KSP/PC object graph per step). The cache below
#  keeps the workspace alive on the solver between solves, split by what each
#  piece actually depends on:
#
#  * GEOMETRY tier — Q/Qt, the constrained rows, the fault pair blocks and
#    the custom-FMG prolongation depend only on the mesh, the boundary specs
#    and the fault registration. They are reused whenever the geometry
#    signature matches, and torn down by the solver's own rebuild paths
#    (_reset / the _build full-rebuild teardown), which fire on mesh.deform,
#    BC changes and every is_setup=False invalidation.
#  * STRUCTURE tier — Ahat, the Schur pmat Mp and the KSP/PC context are
#    reused as OBJECTS with their values refreshed in place (ptap-with-result
#    / createSubMatrix-with-submat / setOperators), exactly the pattern the
#    Newton loop already uses between its own iterations. This is always
#    correct regardless of any change-detection: values are reassembled.
#  * FAST PATH — iteration 0 of the Newton loop may additionally skip the
#    Jacobian assembly / ptap / PCSetUp entirely when the operator key proves
#    nothing operator-relevant changed. A wrong skip cannot produce a wrong
#    answer: the loop measures the TRUE residual at every iterate and every
#    iteration after the first always reassembles, so a stale operator costs
#    one extra increment, never a stale solution (the structural safety net
#    that the state-counter-keyed original lacked).
#
#  The cache is forfeited entirely (correctness first) for the paths where
#  reuse is not provably sound: direct-LU solves, prescribed-datum solves
#  (the datum is re-evaluated from possibly-changed fields inside
#  build_rotation), and fault interface laws (the interface tangent and the
#  Picard-lagged normal stress are solution-dependent state).
# --------------------------------------------------------------------------- #

def _operator_coefficient_variables(solver):
    """Mesh variables that can change the assembled Stokes OPERATOR (not just
    the RHS): everything reachable from the constitutive parameters, the
    constraint term, the penalty and the saddle preconditioner. The solver's
    own unknowns are excluded (the solve writes them every time). Expressions
    are unwrapped first so variables hidden inside nested UWexpressions are
    seen (the "unwrap before extracting atoms" rule).

    Returns a tuple of variables, or ``None`` when the enumeration fails —
    the caller must then treat the operator as always-changed (no fast path;
    the structural refresh path is used every solve)."""
    from underworld3.function.expressions import mesh_vars_in_expression, unwrap

    expressions = []
    try:
        expressions.append(solver.constraints)
        expressions.append(solver.penalty)
        saddle_preconditioner = getattr(solver, "saddle_preconditioner", None)
        if saddle_preconditioner is not None:
            expressions.append(saddle_preconditioner)
        parameters = getattr(solver.constitutive_model, "Parameters", None)
        if parameters is not None:
            for name in parameters._list_valid_parameters(type(parameters)):
                expressions.append(getattr(parameters, name))
    except Exception:
        return None

    variables = set()
    for expression in expressions:
        expression = getattr(expression, "sym", expression)
        if expression is None or not hasattr(expression, "args"):
            continue
        try:
            expression = unwrap(expression, keep_constants=False,
                                return_self=False)
            _, regular, derivatives = mesh_vars_in_expression(
                sympy.sympify(expression))
        except Exception:
            return None
        variables.update(fn.meshvar() for fn in regular)
        variables.update(derivatives)

    unknowns = {getattr(variable, "_base_var", variable)
                for variable in solver.fields.values()}
    variables = {getattr(variable, "_base_var", variable)
                 for variable in variables}
    variables.difference_update(unknowns)
    return tuple(sorted(variables, key=lambda variable: variable._uw_id))


def _coefficient_states(variables):
    """UW data-version counters of the operator coefficient variables."""
    return tuple(variable._state for variable in variables)


def _operator_constants_signature(solver):
    """Signature of everything that reaches the compiled kernels OUTSIDE
    mesh-variable data: the packed ``constants[]`` values and the JIT bundle
    key.

    This closes the blind spot the original PR #418 cache had: a rampable
    UWexpression constant changes VALUE with no ``_state`` bump anywhere (the
    #416 live-constants contract — every δ-continuation and viscosity-ramp
    driver relies on it), so a key built from MeshVariable state counters
    alone reported "unchanged" for a 2x viscosity ramp. The packed constants
    ARE the values the kernels will assemble with, so comparing them closes
    the gap exactly — at the cost of also invalidating on RHS-only constant
    changes, which errs toward reassembly (robust generality). The JIT key
    covers a function rewire in place (new kernels on the same DM), where the
    manifest itself is replaced. If the manifest cannot be read, return None:
    the caller must forfeit the fast path."""
    from underworld3.utilities._jitextension import _pack_constants
    try:
        manifest = getattr(solver, "constants_manifest", None)
        values = tuple(float(v) for v in _pack_constants(manifest)) \
            if manifest else ()
    except Exception:
        return None
    jit_key = getattr(solver, "_current_jit_cache_key", None)
    if jit_key is None:
        jit_key = getattr(solver, "_last_jit_cache_key", None)
    return (str(jit_key), values)


def _rotated_geometry_signature(solver, boundaries):
    """Signature of everything the GEOMETRY tier of the workspace was built
    from: the boundary specs (name + normal), the fault-contact registration
    (names + analytic-normal overrides) and the DM identity. All entries are
    registration state, identical across ranks."""
    dm = solver.dm
    sig_boundaries = tuple((name, repr(normal))
                           for name, normal in map(_boundary_spec, boundaries))
    sig_faults = tuple(getattr(solver, "_fault_contact_faults", []) or [])
    sig_fault_normals = repr(sorted(
        (k, repr(v))
        for k, v in getattr(solver, "_fault_normal_overrides", {}).items()))
    return (sig_boundaries, sig_faults, sig_fault_normals,
            dm.handle if dm is not None else 0)


def _destroy_rotated_linear_cache(cache):
    """Release the persistent rotated-free-slip workspace. The custom-FMG
    prolongation list is only DEREFERENCED (its coarse Mats are shared with
    the solver's registered multigrid hierarchy)."""
    if not cache:
        return
    _destroy_rotated_ksp_ctx(cache.get("ctx"))
    cache["ctx"] = None
    for key in ("Ahat", "Q", "Qt"):
        obj = cache.get(key)
        if obj is not None:
            obj.destroy()
            cache[key] = None
    cache["custom_Pl"] = None


# --------------------------------------------------------------------------- #
#  The rotated solve
# --------------------------------------------------------------------------- #
def _naive_pressure_pin(dm):
    """One owned pressure DOF (datum) for the direct-LU gauge pin — row only, so
    the B^T coupling is kept. Only the direct solve needs it; the iterative path
    fixes the pressure gauge with the constant-pressure null space instead.

    TODO(BUG): a naive per-rank scan of the rank's OWN global-section chart, so at
    np>1 each rank pins a different pressure DOF (or none → the caller must skip).
    Parallel-unsafe; tolerated only because LU is opt-in via
    ``solver._rotated_use_lu`` (a serial PC-free diagnostic)."""
    gsec = dm.getGlobalSection()
    pS, pE = gsec.getChart()
    for q in range(pS, pE):
        if gsec.getFieldDof(q, _PRESSURE_FIELD) > 0 \
                and gsec.getFieldOffset(q, _PRESSURE_FIELD) >= 0:
            return gsec.getFieldOffset(q, _PRESSURE_FIELD)
    return None


def _finalize_rotated_solution(solver, U, Q, normal_rows, remove_rotation_gauge):
    """Remove the rigid-rotation gauge (if it is a genuine null space of the
    constrained problem), scatter the composite global vector ``U`` into the
    velocity/pressure fields, and refresh the enhanced-variable caches. Shared by
    every exit of the rotated solve. Returns whether the gauge was
    removed."""
    dm = solver.dm
    # Remove the rigid-rotation gauge — every mode that is a genuine null space of
    # the constrained problem (closed circular free-slip: one; full spherical
    # shell: all three); on straight walls the constraint pins the rotations, and
    # projecting would corrupt the solution.
    # Done on the GLOBAL vector U with PETSc dots (parallel-correct ownership — a
    # local nodal sum would double-count shared nodes at rank boundaries). The
    # surviving modes are orthonormalised (Gram-Schmidt) before projection — the
    # three 3D modes are not mutually orthogonal on a general mesh.
    # COLLECTIVE: all ranks walk the same mode list, same order.
    removed = False
    # What the gauge decision actually was, recorded for the caller. Two solves
    # of the same system that both converge can still differ by an admitted
    # gauge mode, and until this was recorded there was no way to see which of
    # them admitted what.
    gauge = {"offered": 0, "modes": [], "orthonormal": 0, "removed": False,
             "requested": bool(remove_rotation_gauge)}
    if remove_rotation_gauge:
        live = []
        for tg in _rigid_rotation_modes(solver):
            gauge["offered"] += 1
            if _mode_satisfies_constraints(solver, Q, normal_rows, tg,
                                           report=gauge["modes"]):
                live.append(tg.copy())          # owned copy — pool vec goes back below
            dm.restoreGlobalVec(tg)
        ortho = []
        for w in live:
            for q in ortho:
                w.axpy(-w.dot(q), q)
            nrm = w.norm()
            if nrm > 1e-14:
                w.scale(1.0 / nrm)
                ortho.append(w)
            else:
                w.destroy()
        gauge["orthonormal"] = len(ortho)
        for q in ortho:
            U.axpy(-U.dot(q), q)
            q.destroy()
            removed = True
    gauge["removed"] = removed
    solver._rotated_gauge_report = gauge

    # scatter U → velocity/pressure fields, completing each field's essential
    # (Dirichlet) DOFs. Those are absent from the global vector, so a plain
    # per-field scatter leaves them at ZERO — silently wrong wherever the datum
    # g != 0 (an inhomogeneous Dirichlet wall next to a rotated boundary, #497).
    # The insertion must run on the FULL dm, where the auxiliary vector lives:
    # a sub-DM insertion segfaults when a BC datum references another
    # MeshVariable. _scatter_global_to_fields does exactly that (boundary FEM on
    # the parent local vector, then the per-field split) and carries the same
    # cache-invalidation tail as the native post-solve copy-back.
    # TODO(BUG): DMPlexSNESComputeBoundaryFEM inserts at time=PETSC_MIN_REAL, so
    # a mesh.t-dependent essential datum is written as garbage — same defect on
    # the native copy-back path. See issue #410.
    solver._scatter_global_to_fields(U)
    return removed


def _gather_fields_to_global(solver):
    """Composite global vector built from the solver's current velocity/pressure
    field values (the warm-start initial guess for the rotated Newton loop)."""
    dm = solver.dm
    U = dm.createGlobalVec()
    U.set(0.0)
    for name, var in solver.fields.items():
        sg = U.getSubVector(solver._subdict[name][0])
        solver._subdict[name][1].localToGlobal(var.vec, sg)
        U.restoreSubVector(solver._subdict[name][0], sg)
    return U


def _impose_normal_constraint(u, Q, Qt, normal_rows, datum_map=None):
    """Impose the strong affine constraint ``v_n = ũ_n`` exactly on the composite
    vector ``u``, in place: rotate to the boundary frame (``û = Q u``), zero the
    constrained normal rows, set the prescribed-datum rows to ``sgn·ũ_n``
    (``datum_map``; empty/None ⇒ pure free-slip ``v_n = 0``), rotate back
    (``u = Qᵀ û``). Q is orthogonal, so this is the exact affine projection onto
    the constraint-satisfying set. Keeping every Newton iterate feasible this way
    is what lets the increment problem stay HOMOGENEOUS (``n̂·δ = 0``) — the
    datum never touches the tangent or the increment RHS."""
    uh = u.duplicate()                       # transient projection buffer
    Q.mult(u, uh)
    _zero_rows_local(uh, normal_rows)
    if datum_map:
        _set_rows_local(uh, datum_map)
    Qt.mult(uh, u)
    uh.destroy()


def _backtracking_line_search(u, d, rnorm, rotated_residual, Q, Qt, normal_rows,
                              datum_map=None, max_halvings=8):
    """Backtracking line search on ‖F̂‖ for the rotated Newton/Picard update
    ``u + α d`` (full step α=1 first, halved on failure). Cheap insurance far from
    the solution; α=1 is accepted immediately near it. Every trial iterate is
    snapped back onto the strong ``v_n = ũ_n`` constraint (``datum_map``; free-slip
    when empty) before its residual is measured, so ‖F̂‖ is always evaluated at a
    feasible point.

    Owns all its temporaries' destroys: on acceptance the input ``u`` is destroyed
    and replaced by the accepted iterate. Returns ``(u, improved)``; ``improved``
    is False when no step reduced the residual below ``rnorm`` — the iteration has
    stalled (typically because it is already at the solution) and the caller should
    stop rather than accept a non-decreasing step."""
    alpha = 1.0
    for _ls in range(max_halvings):
        utry = u.copy()
        utry.axpy(alpha, d)
        _impose_normal_constraint(utry, Q, Qt, normal_rows, datum_map)
        Ftry = rotated_residual(utry)
        fnorm = Ftry.norm()
        Ftry.destroy()
        if fnorm < rnorm:
            u.destroy()
            return utry, True
        utry.destroy()
        alpha *= 0.5
    return u, False


def solve_rotated_freeslip(solver, boundaries, remove_rotation_gauge=True,
                           verbose=False, zero_init_guess=True, picard=0,
                           rtol=None, atol=1.0e-11, stol=1.0e-8, max_it=50,
                           force_operator_refresh=False):
    """THE rotated strong-free-slip solve (linear and nonlinear models alike): a
    manual outer Newton/Picard loop that rotates the residual F(u), the Jacobian
    J(u) and the strong ``v_n = ũ_n`` constraint (``ũ_n = 0`` for pure free-slip;
    a prescribed wall-normal datum via ``solver._rotated_freeslip_datum``) EVERY
    iteration, reusing the validated self-contained rotated fieldsplit-Schur solve
    (``_solve_rotated_iterative``, incl. custom geometric FMG / GAMG velocity block
    and the rotated coupled null space) for each Newton increment.

    A LINEAR model needs no separate path: the first increment from a cold start is
    assembled at rest and already solves the problem (with a datum it is exactly the
    affine-lift solve), so the loop converges at the next residual check — one
    linear solve, plus two cheap residual evaluations. There is therefore no
    up-front nonlinearity probe; the loop self-terminates. Direct LU per increment
    (a serial PC-free diagnostic arbiter, e.g. for TI anisotropy experiments) is
    opt-in via ``solver._rotated_use_lu``.

    Why a manual loop rather than ``snes.solve()``: the rotated operator ``Q A Qᵀ``
    (a ``ptap`` result) carries no DM field information, so PETSc's DM-coupled
    fieldsplit + geometric-MG cannot precondition it (SUBPC_ERROR); the increment
    must be solved by the IS-based self-contained fieldsplit. Driving that from a
    manual Newton loop keeps the whole validated linear machinery and imposes the
    strong constraint exactly at every iterate.

    Each iteration (unknown carried in the CARTESIAN frame ``u``; the increment is
    solved in the rotated frame ``û = Q u``). Every ACCEPTED iterate is FEASIBLE
    (``v_n = ũ_n`` imposed exactly), so increments satisfy the homogeneous
    constraint ``n̂·δ = 0`` and the datum never enters the tangent. The one
    exception is a cold start with a non-zero datum, where the first increment
    carries the datum jump through the affine lift (see the initial-guess comment
    in the body — snapping the zero state onto the datum creates a boundary-strip
    strain state whose nonlinear tangent is unusable):
      * ``F = computeFunction(u)``  → Cartesian residual (native essential BCs on
        other boundaries already applied by the DM);
      * ``F̂ = Q F``, zero ``F̂`` at the constrained normal rows (the constraint
        residual is 0 there — the iterate is feasible); converge on ‖F̂‖;
      * ``Ĵ = Q J(u) Qᵀ`` with ``zeroRowsColumns(normal_rows)``;
      * solve ``Ĵ δ̂ = −F̂``, ``δ = Qᵀ δ̂``, with a ‖F̂‖ backtracking line search;
      * ``u += α δ`` and re-impose ``v_n = ũ_n`` exactly.

    The tangent used by ``computeJacobian`` is the solver's own (``consistent_jacobian``
    → Picard / Newton / continuation), so the rotated loop inherits the same tangent
    the standard path would use. The converged Cartesian residual is stashed as the
    constraint reaction for σ_nn recovery (``boundary_normal_traction``).

    Tangent / warmup handling (mirrors the standard ``solve`` path so nothing is
    silently ignored):

    * ``consistent_jacobian == "continuation"`` — a two-phase Picard→Newton solve via
      the ``constants[]``-routed blend α: phase 1 holds α=0 (frozen/Picard tangent) to
      the loose ``solver.newton_switch_rtol`` (and at least ``picard`` iterations if
      given) to enter Newton's basin; phase 2 sets α=1 (consistent Newton tangent) and
      drives to the requested tolerance. α is restored to 0 afterwards.
    * ``consistent_jacobian is True`` (pure Newton) with ``picard > 0`` — a Picard
      warmup needs the frozen tangent, which the pure-Newton compile does not carry, so
      this **raises** ``NotImplementedError`` pointing to ``"continuation"`` rather than
      silently ignoring the request.
    * ``consistent_jacobian is False`` (default, frozen/Picard) — the whole solve is
      already the frozen tangent, so ``picard`` is inherently satisfied (matches the
      standard path, whose post-warmup Newton phase also uses the frozen tangent here).

    Returns
    -------
    dict
        The solve result consumed by ``boundary_normal_traction`` /
        ``dynamic_topography_field`` (as their ``solve_result`` argument), with keys:

        * ``"Q"``, ``"Qt"`` — the rotation and its transpose (PETSc Mats);
        * ``"reaction"`` — the converged Cartesian residual F(u), stashed as the
          constraint reaction for σ_nn recovery (for a linear residual this equals
          the assembled ``A·u − b`` exactly);
        * ``"U"`` — the Cartesian solution;
        * ``"normal_rows"`` — global rows of the constrained normal components;
        * ``"boundaries"`` — the boundary specs the rotation was built from;
        * ``"rotation_gauge_removed"`` — whether a rigid-rotation gauge was projected out;
        * ``"rotation_gauge"`` — how that was decided: how many rigid-body modes
          were offered, and per mode the constraint violation, the operator
          violation and the verdict. Two solves of the same system that both
          converge can still differ by an admitted gauge mode, and this is the
          only place that decision is visible;
        * ``"ksp_reason"`` — converged-reason of the LAST Newton increment's KSP;
        * ``"ksp_its"`` — list of linear iteration counts, one per Newton iteration;
        * ``"nonlinear_iterations"``, ``"converged"`` — outer-loop count (the number
          of Newton increments solved, ``== len(ksp_its)``) and status;
        * ``"rnorm"``, ``"rnorm0"`` — final and initial rotated residual norms ‖F̂‖
          (feed the solve report);
        * ``"continuation_switched"`` — whether the Picard→Newton tangent switch fired;
        * ``"rotation_reused"`` — the GEOMETRY tier (Q/Qt/constrained rows/
          prolongation) came from the cross-solve cache;
        * ``"workspace_reused"`` — the iteration-0 operator fast path fired
          (Jacobian assembly, ptap and PCSetUp all skipped — see the workspace
          reuse block comment above ``_operator_coefficient_variables``).

    ``force_operator_refresh=True`` disables the iteration-0 fast path for this
    solve (used by ``time=`` solves: the DM time reaches the kernels through
    ``petsc_t``, which no state counter or constants value can see).
    """
    if getattr(solver, "snes", None) is None:
        solver._setup_pointwise_functions()
        solver._setup_discretisation()
        solver._setup_solver()
    dm = solver.dm
    snes = solver.snes
    snes.setUp()
    # Attach the auxiliary vector before ANY assembly. Callers that enter this
    # loop directly (fault_contact.solve_with_fault) bypass the native solve()
    # preamble that normally does this, so a form referencing an auxiliary
    # MeshVariable would read a NULL aux array in the first kernel (segfault),
    # and a repeat solve would read STALE auxiliary values after the fields
    # change between calls. Redundant (and cheap) when the native dispatch
    # already attached it.
    solver.mesh.update_lvec()
    solver.dm.setAuxiliaryVec(solver.mesh.lvec, None)
    solver._update_constants()
    if rtol is None:
        rtol = float(solver.tolerance)

    # Resolve the tangent / warmup policy (see the docstring). ``continuation`` runs a
    # staged α=0 → α=1 solve; pure Newton + picard>0 is unsupported (no frozen tangent
    # to warm up with) and errors loudly; frozen (default) already satisfies a warmup.
    mode = getattr(solver, "consistent_jacobian", False)
    continuation = (mode == "continuation")
    if picard and not continuation and mode is True:
        # The ONLY place linearity must be known in advance (hence the lazy probe —
        # two trial assemblies, paid only by this contradictory configuration): a
        # Picard warmup is meaningless for a linear residual, so ignore it there;
        # for a genuinely nonlinear model pure Newton has no frozen tangent
        # compiled in to warm up with, so fail loudly.
        if solver._residual_is_nonlinear():
            raise NotImplementedError(
                f"rotated free-slip: a Picard->Newton warmup (picard={picard}) with the "
                "consistent Newton tangent (consistent_jacobian=True) requires "
                "consistent_jacobian='continuation' — with pure Newton the frozen (Picard) "
                "tangent needed for the warmup is not compiled in. Use "
                "consistent_jacobian='continuation' (staged Picard then Newton), or drop "
                "picard to run pure Newton.")
        picard = 0
    switch_rtol = max(float(getattr(solver, "newton_switch_rtol", 1.0e-2)), rtol)
    if continuation:
        solver._set_newton_alpha(0.0)            # start in the Picard phase

    # Q, the custom-FMG prolongation, the coupled null space and the datum values
    # depend only on the geometry / normals / prescribed surface field (NOT the
    # solution), so build them ONCE and reuse each step. The datum ũ_n enters the
    # iteration ONLY through the feasible-iterate projection below: every iterate
    # carries v_n = ũ_n exactly, so the Newton increment satisfies the HOMOGENEOUS
    # constraint n̂·δ = 0 and the per-iteration operator treatment (zeroRowsColumns
    # with no lift, tangent transparency, custom_Pl, nullspace) is unchanged from
    # pure free-slip.
    datum_specs = getattr(solver, "_rotated_freeslip_datum", None)
    # Direct LU per increment: no FMG prolongation / null space to build — the
    # gauge is fixed by the naive pressure pin instead (see _naive_pressure_pin).
    use_lu = bool(getattr(solver, "_rotated_use_lu", False))
    interface_laws = bool(getattr(solver, "_fault_interface_laws", {}))

    # ---- cross-solve workspace cache (issue #417; see the block comment
    # above _operator_coefficient_variables for the tier design) ----
    # Forfeited for LU (per-solve pin/factorisation), prescribed datum (the
    # datum values are re-evaluated from possibly-changed fields inside
    # build_rotation) and fault interface laws (solution-dependent interface
    # tangent + Picard-lagged normal stress) — correctness first.
    cache_allowed = not (use_lu or datum_specs or interface_laws)
    cache = getattr(solver, "_rotated_linear_cache", None)
    if cache is not None and not cache_allowed:
        _destroy_rotated_linear_cache(cache)
        solver._rotated_linear_cache = None
        cache = None

    geometry_sig = _rotated_geometry_signature(solver, boundaries) \
        if cache_allowed else None
    coefficient_variables = None
    coefficient_states = None
    operator_sig = None
    if cache_allowed:
        if cache is not None and cache.get("geometry_sig") == geometry_sig \
                and cache.get("coefficient_variables") is not None:
            coefficient_variables = cache["coefficient_variables"]
        else:
            coefficient_variables = _operator_coefficient_variables(solver)
        if coefficient_variables is not None:
            coefficient_states = _coefficient_states(coefficient_variables)
            # The state counters alone are BLIND to rampable UWexpression
            # constants (#416: value changes bump nothing) — the constants
            # signature is the other half of the key.
            operator_sig = _operator_constants_signature(solver)

    op_ok = False
    if cache is not None:
        geom_ok = cache.get("geometry_sig") == geometry_sig
        op_ok = (geom_ok
                 and not force_operator_refresh
                 and coefficient_states is not None
                 and operator_sig is not None
                 and cache.get("coefficient_states") == coefficient_states
                 and cache.get("operator_sig") == operator_sig)
        # COLLECTIVE verdict: state counters are bumped by rank-local data
        # writes, so a rank-divergent match here would desync the collective
        # assembly/PCSetUp calls the verdict gates (the np>1 deadlock class).
        if mpi.size > 1:
            verdicts = mpi.comm.allgather((bool(geom_ok), bool(op_ok)))
            geom_ok = all(v[0] for v in verdicts)
            op_ok = all(v[1] for v in verdicts)
        if not geom_ok:
            # Null before destroying — same double-destroy discipline as
            # _reset_rotated_solver_cache (#543 review, m4).
            solver._rotated_linear_cache = None
            stale, cache = cache, None
            _destroy_rotated_linear_cache(stale)
        elif not op_ok:
            # The operator values are about to be refreshed in place; poison
            # the stored key NOW so an exception mid-refresh cannot leave a
            # stale signature that later matches half-updated values.
            cache["operator_sig"] = None

    if cache is not None:
        Q, Qt = cache["Q"], cache["Qt"]
        normal_rows = cache["normal_rows"]
        datum_map = {}                       # cache_allowed excludes datum solves
        custom_Pl = cache.get("custom_Pl")
        rotation_reused = True
    else:
        Q, Qt, normal_rows, datum_map = build_rotation(solver, boundaries, datum_specs)
        custom_Pl = None if use_lu else _build_rotated_custom_Pl(solver, Q, normal_rows)
        rotation_reused = False

    # The iteration-0 fast path: skip Jacobian assembly / ptap / PCSetUp when
    # the operator key proves nothing operator-relevant changed AND the
    # previous solve behaved linearly (converged in <= 1 increment). For a
    # genuinely nonlinear model the cached operator is the LAST solve's
    # converged tangent, not this iterate's, so the skip would only trade one
    # assembly for one wasted increment — the linear hint is self-measured,
    # no up-front nonlinearity probe.
    reuse_operator = bool(cache is not None and op_ok
                          and cache.get("linear_hint", False))
    workspace_reused = False

    # Interface constitutive laws (fault_contact.add_viscous_fault_bc /
    # add_coulomb_fault_bc): the assembler caches the fault-trace geometry
    # once; each iterate it adds the interface force ∫τ(V)δV to the rotated
    # residual and the CONSISTENT tangent 2·(dτ/dV)·M to the rotated
    # operator — full Newton in V, the stiff direction. (Only a future
    # reaction-fed σ_n argument would be Picard-lagged, by choice.)
    interface = None
    if interface_laws:
        from underworld3.utilities import fault_contact
        interface = fault_contact._InterfaceAssembler(solver)
    # The null space is built INSIDE the loop, after the first Jacobian assembly:
    # _mode_satisfies_constraints verifies each candidate mode against the
    # ASSEMBLED operator (‖J·m‖ ≈ 0), which an unassembled J cannot support.
    # A cached ctx carries the (geometry-tier) null space it was built with.
    nsp = cache["ctx"].get("nsp") if (cache is not None
                                      and cache.get("ctx")) else None

    # initial guess (cartesian, composite): warm-start from the fields or zero.
    # A prescribed datum on a COLD start is imposed through the FIRST increment's
    # affine lift (zeroRowsColumns x/b at the rest-state tangent),
    # NOT by snapping the zero state: v_n = ũ_n over a zero interior manufactures an
    # extreme boundary-strip strain rate whose shear-thinning tangent linearises so
    # badly that no line-searched step descends (measured: first step 4 orders too
    # large, f(α) > f(0) down to α=2⁻¹¹). Assembling the first tangent at rest and
    # letting the increment carry the datum jump gives the smooth regularised-
    # viscosity response instead; every later iterate is feasible and the
    # increments are homogeneous. A warm start is assumed smooth (the previous
    # converged state) and takes the exact affine snap directly.
    # x̂ (the prescribed rotated-frame datum as a composite vector) does double
    # duty, as in the linear path: its collective PETSc norm is the rank-consistent
    # datum-activity flag (datum_map is rank-local — per-rank branching on it
    # desyncs the zeroRowsColumns variant and the line-search collectives, the
    # np>1 deadlock class), and it is the ready-made lift vector for the
    # cold-start first increment.
    xhat = dm.createGlobalVec()
    xhat.set(0.0)
    _set_rows_local(xhat, datum_map)
    datum_active = xhat.norm() > 0.0
    lift_datum = datum_active and zero_init_guess
    if not lift_datum:
        xhat.destroy()
        xhat = None
    if zero_init_guess:
        u = dm.createGlobalVec()
        u.set(0.0)
    else:
        u = _gather_fields_to_global(solver)
    if not lift_datum:
        _impose_normal_constraint(u, Q, Qt, normal_rows, datum_map)

    J, Jp = snes.getJacobian()[:2]
    pres_is = solver._subdict["pressure"][0]
    Fc = dm.createGlobalVec()
    reaction = dm.createGlobalVec()          # the final (un-zeroed) Cartesian residual

    # Reused across Newton iterations: the rotated operator (ptap-with-result), the
    # 1/mu pressure-mass Schur pmat (values refreshed in place), the constraint
    # diagonal scale (frozen at the first tangent — only the magnitude matters),
    # and the KSP/PC context (fieldsplit ISs, FMG hierarchy, GAMG setup survive).
    # A cross-solve cache seeds all of them — the STRUCTURE tier: same objects,
    # values refreshed in place unless the iteration-0 fast path fires.
    Ahat = cache["Ahat"] if cache is not None else None
    Atot = None
    ctx = cache["ctx"] if cache is not None else None
    Mp = ctx["Mp"] if ctx is not None else None
    diag_scale = cache["diag_scale"] if cache is not None else None
    lin_its = []
    # velocity / pressure sub-KSP LAST-APPLICATION counts, one per Newton increment
    # (iterative path only — the direct-LU path has no sub-KSPs and records None)
    vel_its_last = []
    pres_its_last = []

    # With the constant-pressure nullspace active, the outer residual is measured
    # in the pressure-gauge QUOTIENT space: the inner KSP projects the constant
    # mode out of every increment (it is the gauge of an enclosed incompressible
    # domain), so the loop cannot reduce that component and must not measure it.
    # On a DEFORMED faceted boundary the component is not exactly zero — free
    # tangential DOFs carry a small net flux through the node-vs-facet normal
    # mismatch, an irreducible discrete incompatibility (measured: an unprojected
    # outer norm floors at rel ~2e-3, 100% pressure rows, and the line search
    # stalls against the constant offset). This mirrors PETSc's own projected
    # residual for singular systems. The Cartesian reaction stash is UNPROJECTED
    # (σ_nn reads velocity rows only).
    use_pnull = bool(getattr(solver, "_petsc_use_pressure_nullspace", False))
    # L2 norm of the LAST projected-out gauge component (|mean|·√N over the
    # pressure rows) — kept observable so an INCOMPATIBLE datum cannot hide
    # behind the projection (see the guard after the Newton loop).
    pnull_gauge = [0.0]

    def rotated_residual(uvec, keep_cartesian=False):
        snes.computeFunction(uvec, Fc)
        if keep_cartesian:
            Fc.copy(reaction)                # stash the Cartesian reaction for σ_nn
            if interface is not None:
                # Picard-lag the reaction-fed normal stress into the laws:
                # the stash IS the pure transmitted traction at this iterate
                # (the interface force is added below, after it). Updated
                # only at ITERATION STARTS — line-search trials
                # (keep_cartesian=False) evaluate against the frozen sigma,
                # which is what makes the lag a consistent comparison.
                # Iteration 0 sees a zero state: a reaction-fed law starts
                # frictionless and tightens as the stress field forms.
                interface.update_normal_stress(solver, reaction)
        Fh = Fc.duplicate()
        Q.mult(Fc, Fh)
        if interface is not None:
            # interface constitutive force on the slip rows, evaluated at
            # THIS iterate (the pair's equal-and-opposite tractions, already
            # in the rotated frame)
            interface.residual_add(solver, uvec, Fh)
        _zero_rows_local(Fh, normal_rows)
        if use_pnull:
            sp = Fh.getSubVector(pres_is)
            n_p = sp.getSize()
            mean = sp.sum() / max(n_p, 1)
            sp.shift(-mean)                  # project out the pressure-gauge mode
            Fh.restoreSubVector(pres_is, sp)
            pnull_gauge[0] = abs(mean) * (max(n_p, 1) ** 0.5)
        return Fh

    # Convergence reference: max(initial residual, REST-STATE residual ‖F̂(0)‖).
    # A warm start's own initial residual is small, and rtol relative to it
    # demands ever-more absolute accuracy the better the guess (the warm-start
    # rtol trap: the FS time loop stalled at rel~2e-3 of its warm start while
    # being far below tolerance on the physical scale). The rest-state residual
    # is the problem's intrinsic forcing scale; a cold start's r0 IS that scale,
    # so nothing changes there.
    fnorm_rest = None
    if not zero_init_guess:
        z = dm.getGlobalVec()
        z.set(0.0)
        Fz = rotated_residual(z)
        fnorm_rest = Fz.norm()
        Fz.destroy()
        dm.restoreGlobalVec(z)

    r0 = None
    ref = None
    last_reason = 0
    iters = 0
    did_assemble = False
    converged = False
    phase = "picard" if continuation else "newton"
    for iters in range(max_it):
        Fhat = rotated_residual(u, keep_cartesian=True)
        rnorm = Fhat.norm()
        if r0 is None:
            r0 = rnorm
            ref = max(r0, fnorm_rest) if fnorm_rest is not None else r0
        if verbose:
            mpi.pprint(f"[rotated_bc] nonlinear iter {iters:2d}  |F̂|={rnorm:.6e}  "
                       f"rel={rnorm/(ref+1e-300):.3e}  [{phase}]")
        # residual convergence (relative to the reference scale, plus an absolute
        # floor so an already-converged warm start does not chase machine noise).
        if rnorm <= rtol * ref + atol:
            converged = True
            Fhat.destroy()
            break
        # Continuation: switch the frozen (Picard, α=0) tangent to the consistent
        # (Newton, α=1) tangent once the residual has dropped into Newton's basin (the
        # loose newton_switch_rtol) and at least `picard` Picard iterations have run.
        if continuation and phase == "picard" and rnorm <= switch_rtol * r0 and iters >= picard:
            solver._set_newton_alpha(1.0)
            phase = "newton"
            if verbose:
                mpi.pprint(f"[rotated_bc] continuation: Picard→Newton at iter {iters} "
                           f"(rel |F̂| {rnorm/(r0+1e-300):.2e})")
        # Iteration 0 may ride the cross-solve fast path: the cached Ahat still
        # holds the rotated, constraint-eliminated operator and the cached
        # KSP/PC is set up on it. Every LATER iteration always reassembles —
        # that, plus the exact residual measured at every iterate, is the
        # structural safety net: a wrong skip costs one extra increment, never
        # a stale solution.
        assemble = not (reuse_operator and iters == 0)
        if assemble:
            did_assemble = True
            snes.computeJacobian(u, J, Jp)   # Jp carries the 1/mu mass (Schur pmat)
            if Ahat is None:
                Ahat = J.ptap(Qt)
            else:
                J.ptap(Qt, result=Ahat)      # same nonzero pattern → in-place refresh
        else:
            workspace_reused = True
        # The interface tangent is REASSEMBLED at every iterate (that is
        # what makes it consistent Newton, dtau/dV at the current slip
        # rates) and added to a COPY: the ptap-with-result refresh above
        # owns Ahat's structure, and injected entries would be
        # stale-or-doubled on the next refresh.
        if interface is not None:
            Khat = interface.tangent(solver, u)
            if Atot is not None:
                Atot.destroy()
            Atot = Ahat.copy()
            Atot.axpy(1.0, Khat,
                      structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
            Khat.destroy()
            # A recreated operator loses the null space the first solve
            # attached; re-attach for iterations after the first (a nonlinear
            # interface law is what reaches them).
            if ctx is not None and ctx.get("nsp") is not None:
                Atot.setNullSpace(ctx["nsp"])
                Atot.setTransposeNullSpace(ctx["nsp"])
            Aop = Atot
        else:
            Aop = Ahat
        if not use_lu:
            if ctx is None:
                Mp = _pressure_mass_schur_pmat(solver)
                nsp = _rotated_nullspace(solver, Q, normal_rows)   # J assembled above
            elif assemble and Mp is not None:
                Jp.createSubMatrix(pres_is, pres_is, submat=Mp)   # viscosity may be u-dependent
        if diag_scale is None:
            diag_scale = _velocity_diag_scale(Aop, solver)
        bhat = Fhat.copy()
        bhat.scale(-1.0)
        if lift_datum and iters == 0:
            # cold-start lift: the increment FROM ZERO carries the datum jump —
            # b̂[rows] = diag·x̂ and the eliminated columns' A[:,cols]·x̂ are
            # subtracted from the other rows. For a LINEAR model this increment IS
            # the whole solve; the loop converges at the next residual check.
            Aop.zeroRowsColumns(normal_rows, diag=diag_scale, x=xhat, b=bhat)
            xhat.destroy()
            xhat = None
        elif assemble:
            Aop.zeroRowsColumns(normal_rows, diag=diag_scale)
        # else: the cached Aop already carries the eliminated constraint
        # rows/cols from the solve that built it — re-zeroing would bump the
        # Mat state and force PCSetUp, defeating the fast path.
        if use_lu:
            if ctx is None:
                ksp_lu = PETSc.KSP().create(comm=dm.comm)
                ksp_lu.setType("preonly")
                pc_lu = ksp_lu.getPC()
                pc_lu.setType("lu")
                pc_lu.setFactorSolverType("mumps")
                ctx = {"ksp": ksp_lu, "Mp": None,
                       "pin": _naive_pressure_pin(dm)}
            pin = ctx["pin"]
            if pin is not None:
                Aop.zeroRows([pin], diag=1.0)
                _zero_rows_local(bhat, [pin])
            ctx["ksp"].setOperators(Aop)      # values changed in place → refactor
            dhat = Aop.createVecRight()
            dhat.set(0.0)
            ctx["ksp"].solve(bhat, dhat)
            last_reason = ctx["ksp"].getConvergedReason()
            _warn_if_ksp_diverged(ctx["ksp"], kind="rotated direct-LU")
            # parity with the iterative path's exact v_n cleanup (LU is exact only
            # to factorisation round-off at the decoupled constraint rows)
            _zero_rows_local(dhat, normal_rows)
        else:
            dhat, last_reason, ctx = _solve_rotated_iterative(
                solver, Aop, bhat, Q, Qt, normal_rows,
                custom_Pl=custom_Pl, nsp=nsp, Mp=Mp, verbose=False, ctx=ctx,
                refresh_operator=assemble)
        lin_its.append(ctx["ksp"].getIterationNumber())
        vel_its_last.append(ctx.get("vel_its_last"))
        pres_its_last.append(ctx.get("pres_its_last"))
        d = dm.createGlobalVec()
        Qt.mult(dhat, d)
        # step-norm convergence (SNES_CONVERGED_SNORM): a tiny Newton step means we
        # are at the solution — the exit for a warm start that is already converged
        # (otherwise the relative test above, with a tiny r0, chatters near machine
        # level). ‖u‖=0 on a cold start ⇒ this never fires prematurely (d is large).
        # A tiny step alone does NOT prove convergence — a stiff tangent (power-law
        # at the regularisation floor) also produces tiny increments at a large
        # residual — so the exit is VERIFIED against the reference scale (which
        # includes the rest-state residual ‖F̂(0)‖ for warm starts): a warm start
        # at the solution passes; a stagnating crawl does not. On verification
        # failure the tiny step goes through the NORMAL line search instead: if it
        # still improves, the crawl proceeds (bounded by max_it); if not, the
        # stall exit ends the loop honestly.
        step_converged = d.norm() <= stol * (u.norm() + 1e-30)
        if step_converged:
            step_converged = rnorm <= rtol * ref + atol
            if not step_converged:
                mpi.pprint(f"[rotated_bc] step-norm at rel |F̂| = "
                           f"{rnorm / (ref + 1e-300):.2e} of the reference "
                           f"residual — stagnation-or-crawl, continuing to iterate.")
        improved = False
        if lift_datum and iters == 0:
            # accept the lift step unconditionally (it is the smooth rest-state
            # response — the standard first-Picard warmup; a line search against
            # ‖F̂(0)‖, which ignores the datum mismatch, would be meaningless) and
            # snap the datum rows exactly (the KSP cleanup zeroed them).
            u.axpy(1.0, d)
            _impose_normal_constraint(u, Q, Qt, normal_rows, datum_map)
            step_converged = False
            improved = True
        elif not step_converged:
            u, improved = _backtracking_line_search(
                u, d, rnorm, rotated_residual, Q, Qt, normal_rows, datum_map)
        # every per-iteration temporary dies HERE, on all exit paths
        dhat.destroy()
        d.destroy()
        bhat.destroy()
        Fhat.destroy()
        if step_converged:
            converged = True
            break
        if not improved:
            break

    # Restore a clean frozen (Picard) tangent for any subsequent solve (next time
    # step), matching the standard _continuation_solve which leaves alpha at 0.
    if continuation:
        solver._set_newton_alpha(0.0)

    # The true outer-iteration COUNT is the number of Newton increments solved — one
    # linear solve per increment, so len(lin_its). The loop index `iters` undercounts
    # by one on the step-norm / line-search-stall / max_it exits (the increment of the
    # final pass is solved before the break) and matches only on the residual exit.
    newton_its = len(lin_its)

    # The loop can exhaust max_it or stall in the line search (`not improved`) without
    # meeting the residual / step-norm criteria. Warn — as the standard SNES path does
    # on divergence — so an unconverged iterate left in the fields is not silent.
    if not converged:
        rel = (rnorm / (ref + 1e-300)) if ref is not None else float("nan")
        mpi.pprint(f"[rotated_bc] WARNING: nonlinear rotated free-slip did NOT converge "
                   f"in {newton_its} iterations (rel |F̂| = {rel:.2e} of the reference "
                   f"residual); the fields hold the last (unconverged) iterate.")

    # The quotient projection makes the outer test blind to the pressure-gauge
    # component. A COMPATIBLE datum leaves that component at discretisation
    # level; an INCOMPATIBLE one (net wall-normal flux into an enclosed
    # incompressible domain) parks an O(forcing) constant there — and the loop
    # above would report convergence while mass conservation is violated.
    # Surface the component: readable state always, loud when it dominates.
    if use_pnull:
        solver._rotated_pressure_gauge_residual = pnull_gauge[0]
        if converged and pnull_gauge[0] > 10.0 * max(rnorm, atol):
            mpi.pprint(f"[rotated_bc] WARNING: converged in the pressure-gauge "
                       f"QUOTIENT space, but the projected-out gauge component "
                       f"(|F̂_p·1|/√N = {pnull_gauge[0]:.2e}) dominates the "
                       f"converged residual ({rnorm:.2e}). The wall-normal datum "
                       f"carries a net boundary flux this enclosed incompressible "
                       f"domain cannot absorb — the velocity field violates mass "
                       f"conservation at that level. Check the datum's surface "
                       f"integral (solver._rotated_pressure_gauge_residual holds "
                       f"this number).")

    Fc.destroy()                     # residual output buffer (reaction persists in the result dict)
    if cache_allowed and ctx is not None and Ahat is not None and not use_lu:
        # Persist the workspace for the next solve. The stored key describes
        # the operator values Ahat now holds; linear_hint records whether
        # this solve behaved linearly (<= 1 increment), which is what
        # licenses the iteration-0 fast path next time. Q/Qt are SHARED with
        # the result dict below — one Python wrapper each, so teardown and
        # GC compose.
        solver._rotated_linear_cache = {
            "geometry_sig": geometry_sig,
            "Q": Q, "Qt": Qt, "normal_rows": normal_rows,
            "custom_Pl": custom_Pl,
            "Ahat": Ahat, "diag_scale": diag_scale, "ctx": ctx,
            "coefficient_variables": coefficient_variables,
            "coefficient_states": coefficient_states,
            # The stored key must describe what Ahat HOLDS: valid if this
            # solve reassembled (values current), or if the fast path rode a
            # key-matched operator untouched. A poisoned solve that exited at
            # iteration 0 without assembling must not persist a fresh key
            # against unrefreshed values (#543 review, m2).
            "operator_sig": (operator_sig if (did_assemble or reuse_operator)
                             else None),
            "linear_hint": bool(converged and newton_its <= 1),
        }
    else:
        if cache_allowed:
            # nothing solvable was built (e.g. an already-converged warm start
            # on a fresh solver) — leave no partial cache behind.
            solver._rotated_linear_cache = None
        _destroy_rotated_ksp_ctx(ctx)        # KSP/PC + the owned Schur pmat
        if Ahat is not None:
            Ahat.destroy()                   # the reused rotated operator
    if Atot is not None:
        Atot.destroy()                       # rotated operator + interface term
    if xhat is not None:
        xhat.destroy()                       # unused lift vector (loop exited before it)
    removed = _finalize_rotated_solution(solver, u, Q, normal_rows, remove_rotation_gauge)

    return {"Q": Q, "Qt": Qt, "reaction": reaction, "U": u,
            "normal_rows": normal_rows, "boundaries": list(boundaries),
            "rotation_gauge_removed": removed,
            # per-mode record of the gauge decision (offered / accepted / the
            # violations that decided it) — see _finalize_rotated_solution
            "rotation_gauge": getattr(solver, "_rotated_gauge_report", None),
            "ksp_reason": last_reason,
            "nonlinear_iterations": newton_its, "converged": converged,
            "ksp_its": lin_its, "rnorm": rnorm, "rnorm0": r0,
            "vel_its_last": vel_its_last, "pres_its_last": pres_its_last,
            # keyed on use_lu, NOT on `ctx is None`: the iterative path also exits
            # with ctx unset when a warm start is already converged at increment 0.
            "velocity_pc": "direct-LU" if use_lu else (ctx or {}).get("velocity_pc"),
            "schur_pre": "none" if use_lu else (ctx or {}).get("schur_pre"),
            "velocity_pc_type": None if use_lu else (ctx or {}).get("velocity_pc_type"),
            "continuation_switched": continuation and phase == "newton",
            "rotation_reused": rotation_reused,
            "workspace_reused": workspace_reused}


def _build_rotated_custom_Pl(solver, Q, normal_rows):
    """The rotated custom-FMG prolongation list [*coarse, Q_v·P_fine] for the
    velocity block, or None if this solver has no hierarchy. Depends only on Q and
    the mesh (NOT the solution), so the rotated Newton loop builds it ONCE and reuses
    it across Newton iterations (the prolongation build is the expensive part).

    The hierarchy is resolved by ``custom_mg.build_transfers``, which is the same
    rule the standard path uses: an explicit ``set_custom_fmg`` registration wins,
    otherwise a MESH-OWNED coarse tail (what ``mesh.adapt()`` leaves on a
    refinement child) is picked up opportunistically. That mesh-owned pickup used
    to be unreachable from here — the standard path's injection hook runs after
    the rotated dispatch has already returned — so an adapt child under rotated
    free-slip silently lost its multigrid and solved on GAMG (#467). This is the
    ``adapt-on-top-faults`` workflow's own configuration."""
    from underworld3.utilities import custom_mg
    # Same None discipline as auto_inject_custom_mg: the contract is a 2-tuple,
    # but "no hierarchy" must degrade to the default preconditioner, never raise.
    resolved = custom_mg.build_transfers(solver, field_id=0)
    h, Ps = resolved if resolved is not None else (None, None)
    if h is None or Ps is None:
        return None
    vel_is = solver._subdict["velocity"][0]
    vis = np.asarray(vel_is.getIndices())
    g2blk = {int(g): k for k, g in enumerate(vis)}
    Qv = Q.createSubMatrix(vel_is, vel_is)
    nrows_blk = sorted({g2blk[g] for g in normal_rows if g in g2blk})
    Pfine = Qv.matMult(Ps[-1])
    Pfine.zeroRows(nrows_blk, diag=0.0)
    # Remember an opportunistic mesh-owned pickup, so a later solve on this solver
    # (rotated or not) reuses the hierarchy instead of re-resolving it.
    if getattr(solver, "_custom_mg", None) is None:
        solver._custom_mg = {"mode": "hierarchy", "hierarchy": h, "verbose": False}
    return list(Ps[:-1]) + [Pfine]


def _pressure_mass_schur_pmat(solver):
    """The native 1/mu pressure-mass Schur preconditioner block, extracted from the
    solver's assembled Pmat — the SAME p-p block the standard path uses via
    ``pc_fieldsplit_schur_precondition=a11`` (the DS JacobianPreconditioner term
    ``_pp_G0``). Q is identity on pressure, so the block needs no rotation.
    Returns None (→ selfp fallback) when the Pmat is not distinct from the operator
    or the block was not assembled. The CALLER must have assembled the Pmat
    (``snes.computeJacobian(x, J, Jp)``) and owns the returned Mat."""
    A, P = solver.snes.getJacobian()[:2]
    if P is None or P.handle == A.handle:
        return None
    pres_is = solver._subdict["pressure"][0]
    Mp = P.createSubMatrix(pres_is, pres_is)
    if Mp.norm() == 0.0:                               # not assembled → useless
        Mp.destroy()
        return None
    return Mp


def _velocity_diag_scale(Ahat, solver):
    """Mean |diag| of the (rotated) velocity block — the representative diagonal
    for constraint rows (see the zeroRowsColumns call sites). Collective."""
    d = Ahat.getDiagonal()
    vis = solver._subdict["velocity"][0]
    sub = d.getSubVector(vis)
    n = sub.getSize()
    s = sub.norm(PETSc.NormType.NORM_1) / max(n, 1)
    d.restoreSubVector(vis, sub)
    d.destroy()
    return float(s) if s > 0.0 else 1.0


def _destroy_rotated_ksp_ctx(ctx):
    """Release a reusable rotated-KSP context and its owned PETSc objects."""
    if ctx is None:
        return
    if ctx.get("ksp") is not None:
        ctx["ksp"].destroy()
        ctx["ksp"] = None
    if ctx.get("Mp") is not None:
        ctx["Mp"].destroy()
        ctx["Mp"] = None
    if ctx.get("nsp") is not None:
        ctx["nsp"].destroy()
        ctx["nsp"] = None
    if ctx.get("cns") is not None:
        ctx["cns"].destroy()
        ctx["cns"] = None


def _solve_rotated_iterative(solver, Ahat, bhat, Q, Qt, normal_rows, verbose=False,
                             custom_Pl=None, nsp=None, Mp=None, ctx=None,
                             refresh_operator=True):
    """Solve the rotated saddle with a SELF-CONTAINED fieldsplit-Schur KSP on the
    rotated operator. The velocity block is geometric FMG on the CUSTOM prolongation
    (PR#290, rotated) whenever the solver has a hierarchy — ``set_custom_fmg`` or a
    mesh-owned ``adapt()`` tail — else GAMG. Both bundles come from
    ``utilities.multigrid_options``, shared with the native and standard routes.

    A plain rotated Mat has no DM field info, so UW3's DM-coupled fieldsplit cannot
    split it — we build the split from EXPLICIT velocity/pressure index sets. For the
    custom-FMG case the velocity sub-PC gets our prolongation via ``setMGInterpolation``
    (needs no DM); the rotated block A_vv = Q_v A_vv Q_vᵀ is formed from Âhat
    automatically and only the FINE prolongation is rotated (Galerkin coarse ops
    auto-correct). NO direct solve of the fine system.

    Preconditioning parity with the native solve (the Schur-iteration fix):
      * ``Mp`` — the 1/mu pressure-mass block from the native Pmat
        (``_pressure_mass_schur_pmat``) is installed as a USER Schur preconditioner,
        exactly what the standard path uses via ``schur_precondition=a11``. Without
        it (Mp=None) the Schur pre falls back to selfp, which degrades badly on
        curved/deformed boundaries and variable viscosity.
      * the pressure sub-solve mirrors the native FGMRES+GASM at the solver
        tolerance; the constant-pressure nullspace is attached to the Schur
        complement for enclosed domains (the hand-built IS fieldsplit does not
        inherit it from the operator).

    ``custom_Pl`` / ``nsp`` are built by the CALLER (once, reused each Newton step;
    the null-space mode verification needs the assembled Jacobian, which only the
    caller can guarantee); None simply means "no hierarchy" / "no null modes".

    Returns ``(Uhat, reason, ctx)``. Passing ``ctx`` back in reuses the KSP/PC
    across Newton iterations — the fieldsplit ISs, Schur USER pmat and FMG
    prolongations survive; only the operator-values refresh is paid. The caller
    must keep ``Ahat``/``Mp`` the SAME Mat objects (values updated in place) and
    release the context with ``_destroy_rotated_ksp_ctx`` when done.

    ``refresh_operator=False`` is the cross-solve fast path for an exactly
    unchanged matrix: the KSP/PC/GAMG hierarchy is left untouched and only the
    right-hand side changes."""
    from underworld3.utilities import custom_mg
    dm = solver.dm
    vel_is = solver._subdict["velocity"][0]
    pres_is = solver._subdict["pressure"][0]

    if ctx is None:
        # rotated coupled null space (pressure-const ⊕ Q·rotation) on the operator —
        # built by the CALLER (the Newton loop, after the first Jacobian assembly:
        # the mode verification needs the assembled operator).
        if nsp is not None:
            Ahat.setNullSpace(nsp)
            Ahat.setTransposeNullSpace(nsp)

        # UNIQUE prefix per KSP (see _ROT_SOLVE_COUNT) so concurrent rotated solves
        # do not share global-options state; the keys are removed after setup.
        global _ROT_SOLVE_COUNT
        _ROT_SOLVE_COUNT += 1
        pfx = f"rotfs{_ROT_SOLVE_COUNT}_"
        opts = PETSc.Options()
        tol = float(solver.tolerance)
        cfg = {
            "ksp_type": "fgmres", "ksp_rtol": str(tol), "ksp_max_it": "300",
            "pc_type": "fieldsplit", "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_fact_type": "full",
            # Pressure sub-solve at native-path parity (pyx Stokes defaults):
            # FGMRES at 0.1 x tol, GASM on the 1/mu mass (jacobi if only selfp
            # exists).
            #
            # The 0.1 is the MARGIN, and it is the point. The Citcom design this
            # configuration descends from (Moresi & Solomatov 1995) makes the inner
            # solves deliberately inexact, which is why the outer/Schur Krylov must
            # be flexible (fgmres, above) — inexact inner solves perturb the search
            # directions. But inexact is bounded: an inner solve must still converge
            # WELL BELOW the tolerance demanded of the outer solve. This path used to
            # ask for `tol` while the outer KSP also asks for `tol` — no margin at
            # all, the inner solve permitted to be no better than the answer it
            # feeds. Restoring the margin costs ~17% wall clock with identical outer
            # iteration counts (measured, both velocity-block routes, isotropic and
            # TI); a too-loose inner solve fails by silent stagnation, not loudly.
            "fieldsplit_pres_ksp_type": "fgmres",
            "fieldsplit_pres_ksp_rtol": str(tol * 0.1),
            "fieldsplit_pres_ksp_max_it": "200",
        }
        if Mp is not None:
            cfg["fieldsplit_pres_pc_type"] = "gasm"
        else:
            cfg["pc_fieldsplit_schur_precondition"] = "selfp"
            cfg["fieldsplit_pres_pc_type"] = "jacobi"
        if custom_Pl is None:
            # GAMG fallback velocity block. The bundle is the SHARED one (see
            # utilities.multigrid_options), so it cannot drift from what the
            # standard path applies. NOTE: the custom-FMG route is the preferred
            # velocity block — this applies only when no hierarchy is available.
            # No stale keys to clear: `pfx` is unique per rotated solve.
            cfg.update({
                "fieldsplit_vel_ksp_type": "fgmres",
                "fieldsplit_vel_ksp_rtol": str(tol * 0.033),
                "fieldsplit_vel_ksp_max_it": "200",
            })
            cfg.update({f"fieldsplit_vel_{key}": value for key, value
                        in multigrid_options.gamg_bundle().settings.items()})
        else:
            # Custom-FMG velocity block, wrapped in a short FGMRES — NOT `preonly`.
            # PCFieldSplit applies the Schur complement S = A11 - A10 A00^-1 A01
            # through THIS velocity KSP, so `preonly` replaces A00^-1 with a single
            # multigrid cycle and the pressure Krylov is handed a different system
            # S~ != S, preconditioned by a 1/mu mass built for S.
            #
            # Measured (annulus, weak plane reaching the constrained boundary,
            # eta_1/eta_0 = 1e-3, transversely isotropic): under `preonly` the
            # pressure residual falls 4.4e4 in ~16 iterations and then STAGNATES at
            # a floor ~3.1e-7, burning the remaining 184 iterations of its cap for
            # nothing, every outer iteration — 9 outer iterations total. Under
            # FGMRES it converges 1.1e8 monotonically in 17 pressure iterations — 1
            # outer. The isotropic control moves the same way (5 -> 1), so this is
            # the Schur application and not the anisotropy.
            #
            # NB the Schur application is bitwise reproducible under both settings
            # (measured), so the floor is NOT "the operator changes between
            # applications" — it is S~ being the wrong system. The precise origin of
            # the floor (most likely a range/nullspace inconsistency between S~ and
            # the constant-pressure null space attached to S below) is not isolated.
            #
            # rtol and max_it are the native path's (0.033 x tol, 200), as is the
            # GAMG fallback above. The FMG cycle reaches 0.1 x tol in ~11 iterations
            # and would be cheaper there, but a sub-solve tolerance is a guardrail:
            # the conservative value is the default and loosening it is the caller's
            # risk to take.
            cfg.update({
                "fieldsplit_vel_ksp_type": "fgmres",
                "fieldsplit_vel_ksp_rtol": str(tol * 0.033),
                "fieldsplit_vel_ksp_max_it": "200",
            })
        for k, v in cfg.items():
            opts[pfx + k] = v
        try:
            ksp = PETSc.KSP().create(comm=dm.comm)
            ksp.setOptionsPrefix(pfx)
            ksp.setOperators(Ahat)
            pc = ksp.getPC()
            pc.setType("fieldsplit")
            pc.setFieldSplitIS(("vel", vel_is), ("pres", pres_is))
            ksp.setFromOptions()
            if Mp is not None:
                pc.setFieldSplitSchurPreType(
                    PETSc.PC.FieldSplitSchurPreType.USER, Mp)
            pc.setUp()
            if custom_Pl is not None:                 # geometric FMG via custom P
                vel_pc = pc.getFieldSplitSubKSP()[0].getPC()
                A_vv, P_vv = vel_pc.getOperators()
                vel_pc.reset()
                vel_pc.setOperators(A_vv, P_vv)
                # coarse="svd" ONLY when the rotated problem carries VERIFIED
                # null modes (nsp): the Galerkin-coarsened ROTATED velocity
                # block inherits every rigid-rotation mode of the constrained
                # problem (a closed circle: one; a spherical shell: three),
                # and the redundant/LU coarse solve hits a zero pivot there
                # (SUBPC_ERROR, outer reason -11). When there are NO null
                # modes — Dirichlet walls, a split-fault contact box — the
                # blanket SVD is pure cost: a 3-D P2 coarse level is a DENSE
                # factorisation (measured: most of ~0.8 s per V-cycle
                # application at healthy iteration counts, #622). Everything
                # else in the bundle is identical to the native and standard
                # custom-P routes.
                # nsp alone cannot discriminate: it is non-None whenever the
                # constant-PRESSURE mode is attached (enclosed domains), which
                # says nothing about the velocity block. The recorded count of
                # verified ROTATION modes does.
                _coarse = ("svd"
                           if getattr(solver, "_rotated_velocity_null_modes",
                                      1 if nsp is not None else 0)
                           else "redundant")
                custom_mg._configure_pcmg(
                    vel_pc, custom_Pl, coarse=_coarse,
                    smoother=solver._mg_smoother_variant)
                vel_pc.setUp()
                # The bundle went into the GLOBAL options DB under the velocity
                # sub-PC's prefix, which is derived from `pfx` and so is unique to
                # this solve — drop it again now it is consumed, as the `finally`
                # below does for the KSP's own keys.
                vopts = PETSc.Options()
                vpfx = vel_pc.getOptionsPrefix() or ""
                for key in multigrid_options.geometric_mg_bundle(
                        coarse=_coarse,
                        smoother=solver._mg_smoother_variant).settings:
                    vopts.delValue(vpfx + key)
            # Constant-pressure nullspace on the Schur COMPLEMENT (enclosed
            # domains): the IS-built fieldsplit does not propagate the coupled
            # nullspace to the inner Schur solve, which is otherwise singular
            # and grinds against its iteration cap every outer iteration.
            cns = None
            if getattr(solver, "_petsc_use_pressure_nullspace", False):
                S = pc.getFieldSplitSubKSP()[1].getOperators()[0]
                cns = PETSc.NullSpace().create(constant=True, comm=dm.comm)
                S.setNullSpace(cns)
        finally:
            # all options consumed by setFromOptions/setUp — drop them so the
            # global database stays clean (and bounded under time-stepping).
            for k in cfg:
                try:
                    opts.delValue(pfx + k)
                except Exception:
                    pass
        ctx = {"ksp": ksp, "pc": pc, "Mp": Mp, "nsp": nsp, "cns": cns,
               "custom_Pl": custom_Pl, "pfx": pfx,
               # which preconditioner this solve actually got — reported so a model
               # (or a test) can assert on it instead of inferring it from timings
               "velocity_pc": "custom-FMG" if custom_Pl is not None else "GAMG",
               "schur_pre": "1/mu-mass" if Mp is not None else "selfp"}
        # Mirror the degraded arms into the solver-wide fallback record (#484):
        # the ctx keys above stay authoritative for this path's own tests, but
        # "was anything substituted?" must be answerable in ONE place for any
        # solver. Recorded only when a substitution actually happened.
        if custom_Pl is None:
            solver._record_pc_fallback(
                "rotated.velocity_pc",
                requested="custom-P geometric MG on the rotated velocity block",
                installed="GAMG",
                reason="unavailable",
                detail="no multigrid hierarchy (set_custom_fmg or a mesh-owned "
                       "adapt tail) is available to the rotated path")
        if Mp is None:
            solver._record_pc_fallback(
                "rotated.schur_pre",
                requested="1/mu pressure-mass Schur preconditioner",
                installed="selfp + jacobi",
                reason="unavailable",
                detail="the native pressure-mass Pmat block could not be built; "
                       "selfp degrades on curved/deformed boundaries and "
                       "variable viscosity")
    else:
        ksp = ctx["ksp"]
        nsp = ctx["nsp"]
        # Same Mat objects, new values (ptap-with-result / createSubMatrix-with-
        # submat) — poke the KSP so PCSetUp refreshes on the changed operator.
        # refresh_operator=False is the cross-solve fast path for a provably
        # unchanged matrix: leave the KSP/PC/GAMG setup untouched entirely.
        if refresh_operator:
            ksp.setOperators(Ahat)

    if nsp is not None:
        nsp.remove(bhat)                              # project EVERY rhs

    Uhat = Ahat.createVecRight()
    Uhat.set(0.0)
    ksp.solve(bhat, Uhat)
    _warn_if_ksp_diverged(ksp, kind="rotated fieldsplit-Schur")
    # An identity constraint row in an ITERATIVE solve only drives its residual
    # (= û_i) below tolerance, so û_i ~ tol, not exactly 0. Because zeroRowsColumns
    # made these DOFs fully decoupled (row AND column zeroed), û_i affects no other
    # equation → setting them to exactly 0 here makes the strong v_n=0 BC exact
    # independent of the iterative tolerance, without perturbing the rest.
    _zero_rows_local(Uhat, normal_rows)
    # Sub-KSP diagnostics: the OUTER count alone hides a degraded inner solve — a
    # full Schur factorisation with a good pressure mass converges in ~1 outer
    # iteration while the pressure sub-solve grinds against its cap underneath.
    #
    # These are LAST-APPLICATION samples, not work: KSPGetIterationNumber reports
    # the most recent solve, and the velocity KSP is applied once per Schur MatMult
    # (i.e. once per pressure Krylov iteration). They are named `_last` so they are
    # not mistaken for the summed counts the solver_health report uses as its work
    # axis — wiring the rotated path into SolverInstrumentation.sub_reports() is the
    # proper fix and is not done here.
    vel_ksp, pres_ksp = ctx["pc"].getFieldSplitSubKSP()
    ctx["vel_its_last"] = vel_ksp.getIterationNumber()
    ctx["pres_its_last"] = pres_ksp.getIterationNumber()
    ctx["velocity_pc_type"] = vel_ksp.getPC().getType()
    # A velocity FGMRES that exhausts its cap returns KSP_DIVERGED_ITS, which
    # KSPCheckSolve deliberately does NOT escalate — so without this it degrades
    # silently. (`preonly` could never fail, which is why the check is new.)
    _warn_if_ksp_diverged(vel_ksp, kind="rotated fieldsplit velocity sub-solve")
    if verbose:
        mpi.pprint(f"[rotated_bc] velocity block = {ctx['velocity_pc']} "
                   f"(PC {ctx['velocity_pc_type']}); Schur pre = {ctx['schur_pre']}; "
                   f"outer KSP {ksp.getConvergedReason()} in "
                   f"{ksp.getIterationNumber()} its (last apply: vel "
                   f"{ctx['vel_its_last']}, pres {ctx['pres_its_last']})")
    return Uhat, ksp.getConvergedReason(), ctx


def _rotated_nullspace(solver, Q, normal_rows):
    """Coupled Stokes null space in the rotated frame: constant pressure, plus every
    rigid rotation Q·mode that is a genuine null space of the constraints — one
    mode (-y,x) in 2D, up to THREE (e_k×r) in 3D (all three on a spherical shell
    with free-slip inner and outer boundaries; leaving them off makes the outer
    Krylov grind against a near-singular operator and pollutes the solution with
    arbitrary rotation content). Returns a PETSc.NullSpace on the composite
    vector, or None.

    Each vector is ZEROED at the constrained normal rows so it is exactly compatible
    with the strong v_n=0 constraint. Without this, a rotation-mode vector that is
    only ~O(h²)-small at the constrained rows (curved boundary: the normal in Q is
    sampled at the chord midpoint, the rotation at the true P2 node) would, when
    PETSc projects the solution onto the null space, inject a spurious wall-normal
    component — a strong BC must be exact, independent of the iterative solve.
    """
    dm = solver.dm
    vecs = []
    # constant pressure (Q = identity on pressure → unchanged)
    if getattr(solver, "_petsc_use_pressure_nullspace", False):
        pv = dm.createGlobalVec()                  # persists inside the returned NullSpace
        pv.set(0.0)
        pis = solver._subdict["pressure"][0]
        sp = pv.getSubVector(pis)
        sp.set(1.0)
        pv.restoreSubVector(pis, sp)
        pv.normalize()
        vecs.append(pv)
    # rigid rotations (rotated), each only if it satisfies the constraints.
    # COLLECTIVE: all ranks walk the same mode list, same order.
    n_rot = 0
    for tg in _rigid_rotation_modes(solver):
        if _mode_satisfies_constraints(solver, Q, normal_rows, tg):
            tr = tg.duplicate()                    # tr persists in the NullSpace
            Q.mult(tg, tr)
            vecs.append(tr)
            n_rot += 1
        dm.restoreGlobalVec(tg)                    # tg transient → return to pool
    # The VELOCITY-block null-mode count, recorded for the coarse-solve
    # choice in _solve_rotated_iterative: only a rotation mode makes the
    # Galerkin-coarsened velocity block singular (the constant-pressure
    # mode lives in the pressure block), so only then is the SVD coarse
    # solve required.
    solver._rotated_velocity_null_modes = n_rot
    if not vecs:
        return None
    # Make every null-space vector EXACTLY compatible with the strong v_n=0
    # constraint (zero at the constrained rows), then orthonormalise. This is what
    # keeps the wall-normal velocity exact under an iterative solve.
    for w in vecs:
        _zero_rows_local(w, normal_rows)
    ortho = []
    for w in vecs:
        for u in ortho:
            w.axpy(-w.dot(u), u)
        nrm = w.norm()
        if nrm > 1e-14:
            w.scale(1.0 / nrm)
            ortho.append(w)
    return PETSc.NullSpace().create(constant=False, vectors=ortho, comm=dm.comm)


def _mode_satisfies_constraints(solver, Q, normal_rows, tg, tol=1e-8,
                                report=None):
    """True iff the rigid-body mode ``tg`` is a genuine null mode of the
    constrained problem: it satisfies all rotated v_n=0 constraints (Q·tg ~0 on
    every constrained normal row) AND it is a null vector of the assembled
    operator (‖J·tg‖ ~0 against the velocity diagonal scale). The operator test
    is what catches pinning the rotated rows cannot see — an ESSENTIAL no-slip
    on another boundary leaves the mode tangential to the rotated wall (first
    test passes) while the eliminated operator is NOT null on it; admitting it
    then projects an irreducible component out of every increment RHS, and the
    Newton residual floors at that component's magnitude instead of converging
    (measured: rel ~2e-5 plateau on the essential-inner + rotated-outer annulus).
    (A closed circular free-slip boundary admits its one rotation; a full
    spherical shell admits all three; straight/partial walls and any essential
    BC pin them.)

    The caller must have ASSEMBLED the solver's Jacobian at some iterate; an
    unassembled J (norm 0) skips the operator test rather than passing every
    mode through a vacuous 0 ≈ 0.

    COLLECTIVE: every rank runs the same global-vector ops. Do NOT early-return on
    a per-rank ``not normal_rows`` — in parallel a rank may own no boundary node
    (empty normal_rows) while others do, and an early return there would desync the
    collective norms below and deadlock.

    ``report``, if given, is a list this appends one dict to per call: the
    constraint violation, the operator violation (``None`` when the constraint
    test already rejected the mode) and the verdict. Whether a mode is admitted
    decides whether a gauge component is projected out of the answer, so when
    two solves of the same system disagree this is the first thing to look at
    — but nothing in the returned dict was observable before."""
    tr = tg.duplicate()
    Q.mult(tg, tr)
    full = tr.norm()                              # parallel norm
    # norm of tr restricted to the constrained rows: zero everything else, then .norm()
    rs, re = tr.getOwnershipRange()
    loc = np.asarray([g - rs for g in normal_rows if rs <= g < re], dtype=np.int64)
    trc = tr.duplicate()
    trc.set(0.0)
    tra = trc.getArray()
    tga = tr.getArray()
    tra[loc] = tga[loc]
    trc.setArray(tra)
    viol = trc.norm() / (full + 1e-30)            # parallel (collective on all ranks)
    # transient duplicates
    tr.destroy()
    trc.destroy()
    if viol >= tol:
        if report is not None:
            report.append({"viol": float(viol), "op_viol": None,
                           "accepted": False, "rejected_by": "constraint"})
        return False
    # operator-nullity: rigid rotation has exactly zero strain in the discrete
    # space (P2 contains linear fields, affine cells integrate the form exactly),
    # so a genuine null mode gives assembly round-off; a pinned mode leaves O(1)
    # boundary-strip rows.
    J = solver.snes.getJacobian()[0]
    Jm = tg.duplicate()
    J.mult(tg, Jm)
    jn = Jm.norm()
    Jm.destroy()
    if jn == 0.0 and J.norm() == 0.0:             # J never assembled → cannot verify
        if report is not None:
            report.append({"viol": float(viol), "op_viol": None,
                           "accepted": True, "rejected_by": "unassembled-J"})
        return True
    op_viol = jn / (_velocity_diag_scale(J, solver) * (tg.norm() + 1e-30))
    if report is not None:
        report.append({"viol": float(viol), "op_viol": float(op_viol),
                       "accepted": bool(op_viol < tol),
                       "rejected_by": None if op_viol < tol else "operator"})
    return op_viol < tol


def boundary_normal_traction(solver, boundary, solve_result, mass="auto"):
    """Boundary normal traction σ_nn on `boundary` from the constraint reaction of the
    last rotated-free-slip solve (``solve_result`` is the dict returned by
    ``solve_rotated_freeslip`` — see its
    Returns section for the keys). Returned mean-removed (the ρg·h gauge), as
    ``(xs, sigma)`` with one entry per boundary velocity node on this rank.

    σ_nn is recovered from the CARTESIAN nodal reaction r_c = A·u − b: the nodal load
    is R_i = n̂_i · r_c(node_i), where n̂_i is THIS boundary's outward normal at node i.

    SIGN. ``n̂_i`` is whatever :func:`_boundary_velocity_nodes` produced for this
    boundary, so σ_nn is measured along that direction. The geometric default is the
    DOMAIN's outward normal everywhere, which on a concave boundary (an annulus or
    shell inner arc) points toward the centre of curvature — the opposite of what UW3
    produced before #560, so σ_nn and the dynamic topography built on it reverse sign
    there against earlier releases. An analytic ``normal=`` is used exactly as given
    and is not reoriented, so ``X/|X|`` on an inner arc yields the other sign. See
    "Which way is outward" in ``docs/developer/subsystems/rotated-freeslip.md``.
    Projecting the Cartesian reaction onto the boundary normal (rather than reading the
    rotated frame's normal row) is corner-correct — at a node shared with another
    rotated-free-slip boundary the rotated frame's first row is a mix of both walls'
    normals, but n̂·r_c is the true normal traction for this boundary. The pointwise
    σ_nn is the boundary-mass de-smear of R.

    ``mass`` selects the de-smear:
      * ``"auto"`` (default) — lumped for 2D traces and 3D P1 triangles, consistent for
        3D P2 triangles.
      * ``"lumped"`` — the diagonal row-sum mass. It is monotone for supported traces,
        but invalid for 3D P2 triangles because their vertex row sums are exactly zero.
      * ``"consistent"`` — the full trace mass. Pointwise-exact 3D P2 recovery, but
        carries the vertex-integral checkerboard on P2 triangles (#404 hold).
      * ``"p1"`` — P1-PROJECTED recovery on a 3D P2 trace (edge-midpoint loads folded
        onto vertices, lumped P1 triangle mass). Sound where the consistent P2 path
        checkerboards; the FreeSurface default in 3D. On a P1 trace, identical to
        ``"lumped"``.

    Parallel-safe: r_c is scattered to a local vector (ghosts included) and read by LOCAL
    section offset; the boundary mass is assembled globally by a coordinate-keyed
    allgather of the boundary elements, so every rank produces the same de-smear and the
    mean-removal gauge is global. In 3D, only triangular P1/P2 traces are supported.
    """
    dm = solver.dm
    dim = solver.mesh.dim
    # Cartesian nodal reaction r_c = F(u) at the converged state, stashed by the
    # solve driver (the final ``computeFunction`` residual; for a linear residual
    # this equals the assembled A·u−b exactly).
    rc = solve_result["reaction"]                # owned by the result dict — do NOT destroy
    rcl = dm.getLocalVec()
    dm.globalToLocal(rc, rcl)
    rca = np.asarray(rcl.getArray())

    lsec = dm.getLocalSection()
    csec = dm.getCoordinateSection()
    cvec = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dim)
    v0, v1 = dm.getDepthStratum(0)
    normal = dict(_boundary_spec(s) for s in solve_result["boundaries"]).get(boundary)
    nodes = _boundary_velocity_nodes(solver, boundary, normal=normal)
    xs = []
    Rn = []
    for q, nrm in nodes:
        lo = lsec.getFieldOffset(q, _VELOCITY_FIELD)
        rcv = rca[lo:lo + dim]                    # Cartesian reaction at this node (local)
        xs.append(_point_coord(dm, dim, cvec, csec, v0, v1, q))
        Rn.append(float(np.dot(nrm, rcv)))        # R_i = n̂·r_c  (corner-correct)
    dm.restoreLocalVec(rcl)
    xs = np.array(xs)
    Rn = np.array(Rn)
    # σ_nn = −(nodal reaction), de-smeared by the SHARED boundary-mass primitive and
    # mean-removed (the ρg·h gauge). partial_reaction=False: the reaction here comes from
    # the ASSEMBLED global operator Q(A·u−b), already complete at every node (ranks agree
    # at a partition-cut node — must OVERWRITE, not sum, else shared nodes double-count).
    return xs, _desmear(solver, boundary, xs, -Rn, mass,
                        remove_mean=True, partial_reaction=False)


def dynamic_topography_field(solver, boundary, solve_result, field,
                             buoyancy_scale=1.0, mass="auto"):
    """Populate a scalar MeshVariable ``field`` with the dynamic topography
    :math:`h = -(\\sigma_{nn}-\\overline{\\sigma_{nn}})/(\\Delta\\rho\\,g)` on ``boundary``,
    recovered from the rotated-free-slip constraint reaction. ``mass="auto"`` uses
    lumped recovery where valid and the consistent surface mass for 3D P2 triangles.
    Interior nodes are left untouched. Returns ``field``.

    This is the hand-off to the free-surface machinery: the 3-number topography
    integrator drives node motion from a surface field, so σ_nn is written onto the
    boundary nodes of a P1 (or higher) scalar field it can read / BdIntegral. Parallel-
    safe: the recovery is partition-independent and the write is local (each rank fills
    its own boundary nodes, matched by coordinate to the recovery output).
    """
    dim = solver.mesh.dim
    xs, sig = boundary_normal_traction(solver, boundary, solve_result, mass=mass)
    # σ_nn is already mean-removed (the ρg·h gauge); topography h = -σ_nn / (Δρ g).
    def key(c):
        return tuple(round(float(t), 9) for t in np.asarray(c).ravel()[:dim])
    hmap = {key(x): -float(s) / buoyancy_scale for x, s in zip(np.asarray(xs), np.asarray(sig))}
    # shared bulk field write (parallel-safe: write ONCE, not per node)
    return write_boundary_scalar_field(solver, field, hmap, dim)


def _rigid_rotation_modes(solver):
    """The Cartesian rigid-body rotation mode(s) as composite GLOBAL vectors
    (velocity DOFs only, zero pressure): ``[(-y,x)]`` in 2D, ``[e_k×r]`` for
    k=x,y,z in 3D — a spherical shell with free-slip on both boundaries has all
    THREE. Parallel-safe: built via localToGlobal on the velocity sub-DM, so
    shared nodes are handled by PETSc, not double-counted. The vectors are
    borrowed from the DM pool — every one must go back via
    ``dm.restoreGlobalVec``. COLLECTIVE: all ranks build the same mode list."""
    dm = solver.dm
    v = solver.Unknowns.u
    c = v.coords
    saved = v.data.copy()
    if solver.mesh.dim == 2:
        fields = [np.column_stack([-c[:, 1], c[:, 0]])]
    else:
        x, y, z = c[:, 0], c[:, 1], c[:, 2]
        zero = np.zeros_like(x)
        fields = [np.column_stack([zero, -z, y]),      # e_x × r
                  np.column_stack([z, zero, -x]),      # e_y × r
                  np.column_stack([-y, x, zero])]      # e_z × r
    vis = solver._subdict["velocity"][0]
    modes = []
    for f in fields:
        v.data[...] = f
        tg = dm.getGlobalVec()
        tg.set(0.0)
        sg = tg.getSubVector(vis)
        solver._subdict["velocity"][1].localToGlobal(v.vec, sg)
        tg.restoreSubVector(vis, sg)
        modes.append(tg)
    v.data[...] = saved
    return modes
