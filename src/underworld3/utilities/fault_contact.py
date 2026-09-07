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

Both points of a pair are rank-local (a seam-touching fault is
redistributed onto one rank before the split; a direct low-level split of
one is refused), so every pair block sits inside one rank's diagonal
portion of ``Q``, exactly like the single-node wall blocks.

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

# One rule for a facet's measure and normal, shared with the three wall accumulators
# (rotated_bc._boundary_velocity_nodes, Mesh._assemble_boundary_normal,
# boundary_flux._node_normals). This was a fourth verbatim copy.
from underworld3.utilities.facet_normals import facet_measure_and_normal

# PETSc section field id of the velocity unknown (solver field registration
# order: velocity first) — the same convention as rotated_bc.
_VELOCITY_FIELD = 0


def _collective_raise(problem):
    """One synchronisation point for rank-local failures in collective paths.

    ``problem`` is ``None`` or an error message string. The verdicts are
    allgathered and, if any rank failed, the SAME error is raised on EVERY
    rank — the same contract as fault_split's refusals. Without this, a
    rank-local raise inside the rotation build or the interface assembler
    leaves the other ranks blocked in the next collective (Q assembly, the
    trace-mass exchange, the Newton loop) instead of aborting: a deadlock
    where an error was meant. COLLECTIVE — every rank must reach it.
    """
    if mpi.size > 1:
        gathered = mpi.comm.allgather(problem)
        problems = [p for p in gathered if p is not None]
        problem = problems[0] if problems else None
    if problem is not None:
        raise RuntimeError(problem)


def add_frictionless_fault_bc(solver, boundary, normal=None):
    """Register the frictionless contact on the split fault ``boundary``.

    ``boundary`` is the fault's original surface name; the mesh must be one
    returned by :func:`fault_split.split_fault`, which records the
    Plus/Minus DOF pairing. The condition itself is imposed inside the
    rotated solve — run it with :func:`solve_with_fault`.

    ``normal`` optionally supplies the fault's ANALYTIC unit normal — a
    sympy ``1×dim`` Matrix in ``mesh.X`` or a constant ``(dim,)`` array,
    the same conventions as ``add_rotated_freeslip_bc``. Without it the
    per-node normal is the average of the adjacent facet normals, which is
    exact on a straight fault but on a SAMPLED CURVE zig-zags at node
    scale: slip past each sampling kink is then geometrically incompatible
    with the no-opening constraint, producing slip notches and normal-
    traction spikes that INTENSIFY under mesh refinement. The analytic
    normal restores the smooth curve's mechanics on the same kinked mesh
    (measured: an order of magnitude off the sawtooth, h-convergent).
    """
    pairs = getattr(solver.mesh, "_fault_point_pairs", {})
    if boundary not in pairs:
        raise ValueError(
            f"mesh carries no split-fault pairing for {boundary!r} — the "
            "solver's mesh must come from fault_split.split_fault, which is "
            "what records the coincident DOF pairs. "
            f"Available: {sorted(pairs) or 'none'}")
    registered = getattr(solver, "_fault_contact_faults", [])
    if boundary not in registered:
        solver._fault_contact_faults = registered + [boundary]
    if normal is not None:
        _compile_normal_spec(normal, solver.mesh, boundary)   # fail fast
        overrides = dict(getattr(solver, "_fault_normal_overrides", {}))
        overrides[boundary] = normal
        solver._fault_normal_overrides = overrides


def _trace_normal_fn(poly):
    """The smoothed normal field of a sampled 2-D trace, from the trace.

    Tangent ANGLE at each control point by central difference (one-sided
    at the tips), interpolated linearly along each segment — the smooth
    reconstruction of the curve the polyline sampled, with error
    O(kink angle squared) and no dependence on an analytic formula. Fault
    nodes lie ON the polyline (every control point is pulled onto a mesh
    vertex before the cut), so the nearest-segment projection is exact.
    """
    pts = np.asarray(poly, dtype=float)
    d = np.diff(pts, axis=0)
    seg2 = np.einsum("ij,ij->i", d, d)
    seg_ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    ang = np.empty(len(pts))
    ang[0], ang[-1] = seg_ang[0], seg_ang[-1]
    ang[1:-1] = 0.5 * (seg_ang[:-1] + seg_ang[1:])

    def fn(X):
        t = np.clip(np.einsum("ij,ij->i", X[None, :2] - pts[:-1], d)
                    / seg2, 0.0, 1.0)
        proj = pts[:-1] + t[:, None] * d
        i = int(np.argmin(np.einsum("ij,ij->i", X[None, :2] - proj,
                                    X[None, :2] - proj)))
        th = (1.0 - t[i]) * ang[i] + t[i] * ang[i + 1]
        return np.array([-np.sin(th), np.cos(th)])

    return fn


def _compile_normal_spec(spec, mesh, boundary):
    """Compile an analytic-normal spec into ``callable(X) -> n``.

    Mirrors the rotated-free-slip normal conventions
    (:func:`rotated_bc._boundary_velocity_nodes`): a sympy ``1×dim`` Matrix
    in ``mesh.X`` is unwrapped and lambdified ONCE into a numpy callable
    (per-node ``.subs()`` is orders of magnitude slower); the string
    ``"trace"`` builds the smoothed normal from the fault's own stored
    polyline (:func:`_trace_normal_fn` — for digitized traces with no
    analytic form); anything else is a constant ``(dim,)`` vector. The
    result need not be unit length — the caller normalises — but must be
    nonzero at every fault node.
    """
    dim = mesh.dim
    if isinstance(spec, str):
        if spec == "trace":
            if dim != 2:
                raise NotImplementedError(
                    "normal=\"trace\" is 2-D; in 3-D pass the "
                    "FaultSurface (or normal=\"surface\").")
            poly = getattr(mesh, "_fault_traces", {}).get(boundary)
            if poly is None:
                raise ValueError(
                    f"mesh carries no stored trace for {boundary!r} — "
                    "normal=\"trace\" needs a mesh built by "
                    "Mesh.add_fault.")
            return _trace_normal_fn(poly)
        if spec == "surface":
            fs = getattr(mesh, "_fault_surfaces", {}).get(boundary)
            if fs is None:
                raise ValueError(
                    f"mesh carries no stored surface object for "
                    f"{boundary!r} — normal=\"surface\" needs a mesh "
                    "built from one (Mesh.add_fault(Surface) in 2-D, "
                    "BoxInternalPatch(patch_points=fault_surface) in "
                    "3-D).")
            if dim == 2:
                # a 2-D Surface's on-fault normal IS the smoothed
                # normal of its control polyline (its signed-distance
                # gradient is discontinuous exactly ON the trace, so
                # the director is the wrong object here)
                return _trace_normal_fn(
                    np.asarray(fs.control_points, dtype=float)[:, :2])
            return _compile_normal_spec(fs, mesh, boundary)
        raise ValueError(
            f"unknown normal spec {spec!r} for fault {boundary!r}: pass "
            "\"trace\" (2-D), \"surface\"/a FaultSurface (3-D), a sympy "
            "1×dim Matrix in mesh.X, or a constant vector.")
    if hasattr(spec, "triangles") and hasattr(spec, "normals"):
        # a FaultSurface: the frame comes from the surface's OWN
        # geometry — nearest-face normal (exact for a planar patch,
        # unchanged for future curved sheets)
        if dim != 3:
            raise NotImplementedError(
                "a FaultSurface normal source is 3-D; in 2-D use "
                "normal=\"trace\".")
        if spec.normals is None:
            spec.compute_normals()
        centres = np.asarray(spec.face_centers, dtype=float)
        face_normals = np.asarray(spec.normals, dtype=float)
        from scipy.spatial import cKDTree
        tree = cKDTree(centres)
        return lambda X: face_normals[int(tree.query(X)[1])]
    if isinstance(spec, sympy.MatrixBase):
        if len(spec) != dim:
            raise ValueError(
                f"analytic normal for fault {boundary!r} has {len(spec)} "
                f"components; the mesh is {dim}-D.")
        from underworld3.function.expressions import unwrap
        comps = [sympy.sympify(unwrap(spec[k], keep_constants=False,
                                      return_self=False))
                 for k in range(dim)]
        # unwrap canonicalises coordinate BaseScalars onto the FIRST
        # mesh's frame (multi-mesh symbol identity, see issue #501), so
        # on any later mesh the unwrapped x is not ``mesh.X[0]`` by
        # identity even though it prints the same. Re-tag by NAME onto
        # this mesh's coordinates before the stray check and the
        # lambdify.
        by_name = {str(xk): xk for xk in mesh.X}
        retag = {s: by_name[str(s)] for c in comps for s in c.free_symbols
                 if str(s) in by_name and s is not by_name[str(s)]}
        if retag:
            comps = [c.subs(retag) for c in comps]
        stray = set().union(*[c.free_symbols for c in comps]) \
            - set(mesh.X)
        if stray:
            raise ValueError(
                f"analytic normal for fault {boundary!r} contains symbols "
                f"{sorted(map(str, stray))} that are not mesh coordinates "
                "— express it in mesh.X.")
        fn = sympy.lambdify(list(mesh.X), comps, "numpy")
        return lambda X: np.asarray(fn(*X), dtype=float).ravel()
    const = np.asarray(spec, dtype=float).ravel()
    if const.shape != (dim,):
        raise ValueError(
            f"constant normal for fault {boundary!r} must have shape "
            f"({dim},); got {const.shape}.")
    return lambda X: const


def _compiled_normal_override(solver, boundary):
    """The fault's registered analytic-normal override, compiled — or None."""
    spec = getattr(solver, "_fault_normal_overrides", {}).get(boundary)
    if spec is None:
        return None
    return _compile_normal_spec(spec, solver.mesh, boundary)


def add_viscous_fault_bc(solver, conds, boundary, normal=None):
    r"""Register a viscous (linear dashpot) law on the split fault ``boundary``:

    .. math::  \hat t\cdot\sigma\cdot\hat n = \eta_f\, V,
               \qquad V = [\mathbf v]\cdot\hat t,

    with ``conds`` = :math:`\eta_f`, the interface viscosity (viscosity per
    unit length: the zero-thickness limit of a shear band is
    :math:`\eta_f = \eta_{band}/w` with :math:`\eta_{band}` the band's OWN
    — weak-zone — viscosity and :math:`w` its width. Not the background
    viscosity: matching a layer means :math:`V = \tau w/\eta_{band}`, and
    with the background value the layer would not be weak). The
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
    add_frictionless_fault_bc(solver, boundary, normal=normal)
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

#: The effective normal stress symbol, SIGNED: positive in compression,
#: negative in tension. The assembler feeds it per node from the no-opening
#: constraint's REACTION, Picard-lagged once per Newton iteration — the one
#: deliberately lagged quantity (its derivative is nonlocal and non-stiff;
#: the stiff direction dtau/dV stays full Newton). A law must clamp its OWN
#: strength at zero — e.g. ``Max(0, C + mu*normal_stress)`` — because the
#: physics of losing strength lives in the law: a cohesive fault keeps
#: shear strength into MILD tension (down to sigma = -C/mu), a cohesionless
#: one loses it the moment the normal stress turns tensile. Where the
#: strength is zero AND the normal stress is tensile the fault would OPEN:
#: the bilateral no-opening constraint then holds it shut with a tensile
#: reaction (glue), and the computed solution is no longer physical —
#: detect this by the sign of ``fault_normal_traction``.
normal_stress = sympy.Symbol(r"\sigma_{n,eff}", real=True)

#: The interface state variable (rate-state theta), per node like the
#: normal stress: a law may use it; the values are held per fault on the
#: solver (``solver._fault_state``) and advanced between solves by
#: :func:`update_fault_state` — the ageing-law ODE integrated exactly for
#: piecewise-constant V. (When manifold MeshVariables land, theta moves
#: there; the law layer is unchanged either way.)
state_variable = sympy.Symbol(r"\theta_{state}", real=True, positive=True)


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
        stray = expr.free_symbols - {slip_rate, normal_stress,
                                     state_variable}
        if stray:
            raise ValueError(
                f"a fault law must be an expression in fault_contact."
                f"slip_rate (and optionally normal_stress and "
                f"state_variable) alone (found {sorted(map(str, stray))}); "
                "substitute other parameters before registering.")
        self.expr = expr
        args = (slip_rate, normal_stress, state_variable)
        self._tau = sympy.lambdify(args, expr, "numpy")
        self._dtau = sympy.lambdify(args, sympy.diff(expr, slip_rate),
                                    "numpy")

    def tau(self, V, S, T):
        V = np.asarray(V, dtype=float)
        return np.broadcast_to(np.asarray(self._tau(V, S, T), dtype=float),
                               V.shape).copy()

    def dtau_dV(self, V, S, T):
        V = np.asarray(V, dtype=float)
        return np.broadcast_to(np.asarray(self._dtau(V, S, T), dtype=float),
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
    if sigma_n == "reaction":
        # the strength clamp is the law's own: bare friction loses all
        # strength the moment the normal stress turns tensile
        strength = float(mu) * sympy.Max(normal_stress, 0)
    else:
        strength = float(mu) * float(sigma_n)
    return SymbolicFaultLaw(
        strength * (2 / sympy.pi) * sympy.atan(slip_rate / float(V0)))


def add_coulomb_fault_bc(solver, conds, boundary, sigma_n=None, V0=1.0e-5,
                         normal=None):
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
            "add_coulomb_fault_bc requires sigma_n: a prescribed effective "
            "normal stress, or \"reaction\" to feed it from the no-opening "
            "constraint's reaction (Picard-lagged).")
    mu = float(conds)
    if mu <= 0.0 or float(V0) <= 0.0 or (
            sigma_n != "reaction" and float(sigma_n) <= 0.0):
        raise ValueError("mu, sigma_n and V0 must all be positive.")
    add_frictionless_fault_bc(solver, boundary, normal=normal)
    _register_law(solver, boundary, CoulombFaultLaw(mu, sigma_n, V0))


def RateStateFaultLaw(a, b, V0, f0, Dc, sigma_n="reaction"):
    r"""Regularised rate-and-state friction (arcsinh form — never the raw
    logarithmic law, which is singular at V = 0):

    .. math:: \tau = a\,\sigma_n\,\mathrm{asinh}\!\left[
        \frac{V}{2V_0}\exp\!\left(\frac{f_0 + b\ln(V_0\theta/D_c)}
        {a}\right)\right],

    smooth and odd in V (sign-correct for either slip direction), with the
    state :math:`\theta` entering per node through
    :data:`state_variable` and evolving by the ageing law between solves
    (:func:`update_fault_state`). At steady state
    (:math:`\theta = D_c/V`) it reduces to
    :math:`\tau = \sigma_n(f_0 + (a-b)\ln V/V_0)`: velocity
    strengthening for a > b, weakening for a < b. The consistent tangent
    in V comes from ``sympy.diff``, exactly as for every other law.
    """
    a, b, V0, f0, Dc = (float(q) for q in (a, b, V0, f0, Dc))
    sn = (sympy.Max(normal_stress, 0) if sigma_n == "reaction"
          else float(sigma_n))
    return SymbolicFaultLaw(
        a * sn * sympy.asinh(
            (slip_rate / (2 * V0)) * sympy.exp(
                (f0 + b * sympy.log(V0 * state_variable / Dc)) / a)))


def add_rate_state_fault_bc(solver, conds, boundary, *, a, b, V0, Dc,
                            sigma_n="reaction", theta0=None, normal=None):
    """Rate-and-state friction on the split fault ``boundary``
    (value-first: ``conds`` is the reference friction coefficient f0).

    ``a``/``b`` are the direct-effect and evolution coefficients, ``V0``
    the reference slip rate, ``Dc`` the state-evolution distance;
    ``sigma_n`` as for Coulomb. The state starts at ``theta0`` (default
    ``Dc/V0``, steady at the reference rate) and is advanced after each
    solve with :func:`update_fault_state`.
    """
    add_frictionless_fault_bc(solver, boundary, normal=normal)
    _register_law(solver, boundary,
                  RateStateFaultLaw(a, b, V0, conds, Dc, sigma_n=sigma_n))
    states = dict(getattr(solver, "_fault_state", {}))
    states.setdefault(boundary, {"theta0": float(
        theta0 if theta0 is not None else float(Dc) / float(V0)),
        "Dc": float(Dc), "by_point": {}})
    solver._fault_state = states


def update_fault_state(solver, boundary, dt, solve_result):
    """Advance the fault's state variable by the ageing law over ``dt``:

    d(theta)/dt = 1 - V*theta/Dc, integrated EXACTLY for the solve's
    (piecewise-constant) slip rates:
    theta' = theta*exp(-x) + (Dc/|V|)*(1 - exp(-x)), x = |V| dt / Dc —
    written with expm1 so the V -> 0 limit degrades smoothly to pure
    ageing, theta' = theta + dt. Returns the median of theta*|V|/Dc over
    the slipping nodes (1.0 at steady state — the convergence monitor).
    """
    state = solver._fault_state[boundary]
    Dc = state["Dc"]
    assembler = _InterfaceAssembler(solver, include=(boundary,))
    V = np.linalg.norm(assembler._nodal_slip(solver, solve_result["U"])[:-1],
                       axis=1)
    pts = sorted(assembler._points, key=assembler._points.get)
    theta = np.array([state["by_point"].get(q, state["theta0"])
                      for q in pts])
    x = V * float(dt) / Dc
    decay = np.exp(-x)
    growth = np.where(x > 1e-8,
                      (Dc / np.maximum(V, 1e-300)) * (-np.expm1(-x)),
                      float(dt) * (1.0 - 0.5 * x))
    theta = theta * decay + growth
    state["by_point"] = {q: float(t) for q, t in zip(pts, theta)}
    slipping = V > 0.2 * V.max() if V.size and V.max() > 0 else \
        np.zeros(0, dtype=bool)
    if not slipping.any():
        return float("nan")
    return float(np.median(theta[slipping] * V[slipping] / Dc))


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

    An analytic-normal override registered for this fault (the ``normal``
    argument of the ``add_*_fault_bc`` family) replaces the accumulated
    facet-average DIRECTION per node, sign-aligned to the Plus→Minus
    orientation. This is the single authority for the pair normal: the
    rotated pair blocks, the interface-law rows, and every diagnostic
    (slip, leak, normal traction) all read the frame from here, so the
    override is consistent everywhere by construction.
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

    # Facet normals accumulated to the pair nodes, weighted by the facet MEASURE for
    # the same reason as the wall normals in rotated_bc (#560): the assembler
    # integrates facet by facet, so a node on two facets is only consistent when its
    # normal is parallel to Σ_f |f| n̂_f. The fault is not protected by being an
    # interior surface — a constant pressure cancels exactly in the MEAN rows (the two
    # sides carry opposite outward normals) but DOUBLES in the jump rows, and the
    # jump-tangential (slip) row is free, so the bisector leaves √2·p·sin(δ/2)·(|f₁|−|f₂|)/6
    # there: a pressure-driven spurious slip at every kink node, plus the same lost
    # pressure gauge. Rank-local by construction — a seam-touching fault is
    # redistributed onto one rank before the split, so no cross-rank sum is needed.
    # Same rule, same one place, as the three wall accumulators — see
    # `facet_normals.facet_measure_and_normal`. `orient="support0"` is the ONE
    # difference and it is deliberate: a fault facet is interior by construction (two
    # support cells), so the "exterior" rule would decline to orient it at all. Every
    # facet of a split fault lists the same side first, so "away from support[0]" IS
    # the ±side split and is coherent along the surface. Keeping the normalise, the
    # +1e-30 epsilon and the measure weight shared is the point: this was a fourth
    # verbatim copy of those six lines, and copies of them drifting apart is what
    # produced #560 and then #564.
    nacc = {}
    for f in facets:
        vol, ne, _exterior = facet_measure_and_normal(dm, f, orient="support0")
        for q in (int(c) for c in dm.getTransitiveClosure(f)[0]):
            if lsec.getFieldDof(q, _VELOCITY_FIELD) > 0:
                nacc[q] = nacc.get(q, np.zeros(dim)) + float(vol) * ne
    # A pair node ON the partition seam (the split through the seam,
    # fault_split.split_fault(across_seams=True)) has one of its two
    # facets on the other rank; the split records the whole sum from the
    # global chain, and it replaces the half this rank accumulated. No
    # exchange here — the value was formed where the chain was known.
    for q, whole in getattr(mesh, "_fault_seam_normals", {}).get(
            boundary, {}).items():
        if int(q) in nacc:
            nacc[int(q)] = np.asarray(whole, dtype=float)

    override = _compiled_normal_override(solver, boundary)
    if override is not None:
        from underworld3.utilities.rotated_bc import _point_coord
        csec = dm.getCoordinateSection()
        cvec = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dim)
        v0, v1 = dm.getDepthStratum(0)

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
        nrm = acc / (np.linalg.norm(acc) + 1e-30)
        if override is not None:
            X = np.asarray(_point_coord(dm, dim, cvec, csec, v0, v1,
                                        int(q_plus)), dtype=float)
            ne = np.asarray(override(X), dtype=float).ravel()
            norm = np.linalg.norm(ne)
            if not norm > 0:
                raise ValueError(
                    f"analytic normal for fault {boundary!r} vanishes at "
                    f"node {int(q_plus)} (coordinate {X}).")
            ne = ne / norm
            # keep the Plus→Minus orientation of the facet-derived normal
            if np.dot(ne, nrm) < 0:
                ne = -ne
            nrm = ne
        out.append((int(q_plus), int(q_minus), nrm))
    return out


def _tangent_frame(nrm):
    """The in-fault tangent(s) completing a pair block's rotation frame.

    One tangent in 2-D (the quarter-turn of the normal), two in 3-D (t̂1
    from the coordinate axis least aligned with the normal — deterministic
    per node — and t̂2 completing the right-handed triad). The SINGLE
    authority for the frame: the pair blocks in ``rotated_bc`` and the
    interface assembler's slip components must agree row for row.
    """
    nrm = np.asarray(nrm, dtype=float)
    if len(nrm) == 2:
        return [np.array([-nrm[1], nrm[0]])]
    axis = np.zeros(3)
    axis[int(np.argmin(np.abs(nrm)))] = 1.0
    t1 = np.cross(nrm, axis)
    t1 = t1 / np.linalg.norm(t1)
    return [t1, np.cross(nrm, t1)]


# 3-point Gauss-Legendre on [0,1] (exact through quartics — the tangent's
# N_i N_j weighting of a P2 trace is quartic) and the quadratic line shape
# functions at those points, node order (end, midpoint, end).
_XI = 0.5 + (np.sqrt(15.0) / 10.0) * np.array([-1.0, 0.0, 1.0])
_WQ = np.array([5.0, 8.0, 5.0]) / 18.0
_NQ = np.column_stack([2.0 * (_XI - 0.5) * (_XI - 1.0),
                       4.0 * _XI * (1.0 - _XI),
                       2.0 * _XI * (_XI - 0.5)])

# The 3-D trace element is the P2 TRIANGLE: 6-point degree-4 Dunavant rule
# (N_i N_j of P2 is quartic, as on the line) with weights summing to 1 on
# the unit-area reference, and the quadratic shapes in barycentric
# coordinates, node order (v0, v1, v2, e01, e12, e20).
_L1, _W1 = 0.445948490915965, 0.223381589678011
_L2, _W2 = 0.091576213509771, 0.109951743655322
_LAMBDA3 = np.array([[1 - 2 * _L1, _L1, _L1], [_L1, 1 - 2 * _L1, _L1],
                     [_L1, _L1, 1 - 2 * _L1],
                     [1 - 2 * _L2, _L2, _L2], [_L2, 1 - 2 * _L2, _L2],
                     [_L2, _L2, 1 - 2 * _L2]])
_WQ3 = np.array([_W1, _W1, _W1, _W2, _W2, _W2])
_NQ3 = np.column_stack(
    [_LAMBDA3[:, i] * (2.0 * _LAMBDA3[:, i] - 1.0) for i in range(3)]
    + [4.0 * _LAMBDA3[:, i] * _LAMBDA3[:, j]
       for i, j in ((0, 1), (1, 2), (2, 0))])


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
    the quadrature with V = 0 and receive no row or column. All entry
    points are COLLECTIVE — ranks holding no fault still participate in
    the assemblies.
    """

    def __init__(self, solver, include=()):
        dm = solver.dm
        dim = solver.mesh.dim
        ncomp = dim - 1                       # slip components per node
        lsec = dm.getLocalSection()
        l2g = dm.getLGMap()
        csec = dm.getCoordinateSection()
        cvec = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dim)
        fS, fE = dm.getHeightStratum(1)

        rows, lo_p, lo_m, tans, nrms, laws_of, facets = \
            [], [], [], [], [], [], []
        index_of = {}
        fault_of = []
        current = [None]                      # the fault being walked

        registered = dict(getattr(solver, "_fault_interface_laws", {}))
        for name in include:
            registered.setdefault(name, None)   # geometry only (diagnostics)
        if include:
            # A PER-FAULT view: the diagnostics (fault_normal_traction,
            # update_fault_state) ask for one fault, and their rows, slip
            # vectors and theta stores must not be interleaved with the
            # other registered faults' nodes (measured: a two-fault network
            # returned A-union-B rows for fault A). The SOLVE path passes no
            # ``include`` and keeps every law-carrying fault, as it must.
            registered = {name: registered[name] for name in include}
        problem = None
        for name, law in registered.items():
            current[0] = name
            pairs = solver.mesh._fault_point_pairs[name]
            minus_of = {qp: qm for qm, qp in pairs.items()}
            # A rank-local geometry failure (pairing/closure mismatch, a
            # vanishing analytic normal) must not leave the peers blocked
            # in the collective trace-mass exchange below — collect the
            # verdict, raise collectively after the walk.
            try:
                pair_nodes = _fault_pair_nodes(solver, name)
            except (RuntimeError, ValueError) as err:
                problem = str(err)
                break
            normals = {q_plus: nrm for q_plus, _q_minus, nrm in pair_nodes}

            def node_index(q):
                q = int(q)
                qm = minus_of.get(q)
                if qm is None:
                    return -1              # a tip/rim node: V = 0, no DOF
                if q in index_of:
                    return index_of[q]
                lo = lsec.getFieldOffset(qm, _VELOCITY_FIELD)
                nrm = normals[q]
                index_of[q] = len(rows)
                fault_of.append(current[0])
                # one global row per slip component, on the Minus point's
                # rotated rows — the jump-tangent rows of the pair block
                rows.append([int(l2g.apply([lo + 1 + a])[0])
                             for a in range(ncomp)])
                lo_p.append(lsec.getFieldOffset(q, _VELOCITY_FIELD))
                lo_m.append(lo)
                tans.append(_tangent_frame(nrm))
                nrms.append(nrm)
                return index_of[q]

            plus_name = f"{name}Plus"
            value = solver.mesh.boundaries[plus_name].value
            if not (dm.hasLabel(plus_name) and
                    dm.getLabel(plus_name).getStratumSize(value) > 0):
                continue
            law_id = len(laws_of) if law is not None else -1
            if law is not None:
                laws_of.append(law)
            for f in (int(q) for q in dm.getLabel(plus_name)
                      .getStratumIS(value).getIndices()):
                if not (fS <= f < fE):
                    continue
                if dim == 2:
                    va, vb = (int(q) for q in dm.getCone(f))
                    L = float(np.linalg.norm(
                        cvec[csec.getOffset(va) // dim]
                        - cvec[csec.getOffset(vb) // dim]))
                    facets.append(((node_index(va), node_index(f),
                                    node_index(vb)), L, law_id))
                else:
                    # P2 triangle: vertex nodes carry barycentric shapes
                    # in list order, edge nodes pair them (v0v1, v1v2,
                    # v2v0) — the _NQ3 node convention.
                    edges = [int(q) for q in dm.getCone(f)]
                    epair = {e: tuple(int(q) for q in dm.getCone(e))
                             for e in edges}
                    verts = sorted({v for pr in epair.values() for v in pr})
                    Xv = [cvec[csec.getOffset(v) // dim] for v in verts]
                    area = 0.5 * float(np.linalg.norm(
                        np.cross(Xv[1] - Xv[0], Xv[2] - Xv[0])))
                    edge_of_pair = {frozenset(pr): e
                                    for e, pr in epair.items()}
                    idx = tuple(node_index(v) for v in verts) + tuple(
                        node_index(edge_of_pair[frozenset(
                            (verts[i], verts[j]))])
                        for i, j in ((0, 1), (1, 2), (2, 0)))
                    facets.append((idx, area, law_id))
        _collective_raise(problem)

        # petsc4py refuses to narrow index arrays — Vec/Mat setValues need
        # PETSc's own integer type.
        self._ncomp = ncomp
        self._rows = np.asarray(rows, dtype=PETSc.IntType).reshape(-1, ncomp)
        self._lo_p = np.asarray(lo_p, dtype=np.int64)
        self._lo_m = np.asarray(lo_m, dtype=np.int64)
        self._tan = np.asarray(tans, dtype=float).reshape(-1, ncomp, dim)
        self._nrm = np.asarray(nrms, dtype=float).reshape(-1, dim)
        self._NQ = _NQ if dim == 2 else _NQ3
        self._WQ = _WQ if dim == 2 else _WQ3
        self._laws = laws_of
        self._facets = facets
        self._dim = dim
        self._points = {q: k for q, k in index_of.items()}

        # Positive trace mass per pair node, for de-smearing the constraint
        # reaction into a pointwise normal traction. On the P2 LINE the
        # lumped (row-sum) mass is positive (L/6, 2L/3, L/6) and is used
        # directly. On the P2 TRIANGLE the lumped VERTEX rows vanish (the
        # known trap: row sums are 0, 0, 0, A/3, A/3, A/3), so the triangle
        # uses the P1 SUB-LUMPING instead — each straight P2 triangle is
        # four P1 sub-triangles of area A/4, whose lumped masses give every
        # node a positive weight (A/12 per vertex, A/4 per midpoint).
        # The star-forest completion below is defensive: after the
        # redistribution that precedes every parallel split the fault is
        # rank-interior and the exchange is a no-op on the fault rows.
        mass = np.zeros(len(rows))
        for idx, measure, _law_id in facets:
            if dim == 2:
                weights = (measure / 6.0, 2.0 * measure / 3.0,
                           measure / 6.0)
            else:
                weights = (measure / 12.0,) * 3 + (measure / 4.0,) * 3
            for k, w in zip(idx, weights):
                if k >= 0:
                    mass[k] += w
        if mpi.size > 1:
            pStart, pEnd = dm.getChart()
            chart_mass = np.zeros(pEnd - pStart, dtype=np.float64)
            for q, k in index_of.items():
                chart_mass[q - pStart] = mass[k]
            sf = dm.getPointSF()
            try:
                nroots, _il, _ir = sf.getGraph()
            except (ValueError, TypeError):
                nroots = -1
            if nroots >= 0:
                from mpi4py import MPI
                sf.reduceBegin(MPI.DOUBLE, chart_mass, chart_mass, MPI.SUM)
                sf.reduceEnd(MPI.DOUBLE, chart_mass, chart_mass, MPI.SUM)
                sf.bcastBegin(MPI.DOUBLE, chart_mass, chart_mass,
                              MPI.REPLACE)
                sf.bcastEnd(MPI.DOUBLE, chart_mass, chart_mass, MPI.REPLACE)
            for q, k in index_of.items():
                mass[k] = float(chart_mass[q - pStart])
        self._mass = mass
        # Effective normal stress per pair node (positive in compression),
        # fed by update_normal_stress; zero until the first reaction exists,
        # which makes a reaction-fed law START frictionless — a natural
        # homotopy from the crack toward the frictional state.
        self._sigma = np.zeros(len(rows) + 1)
        # Interface state (rate-state theta) per pair node, loaded from the
        # solver's per-fault store; 1.0 for faults without state (the value
        # is inert unless the law references state_variable).
        self._theta = np.ones(len(rows) + 1)
        fault_states = getattr(solver, "_fault_state", {})
        points_in_order = sorted(index_of, key=index_of.get)
        for k, fname in enumerate(fault_of):
            st = fault_states.get(fname)
            if st is not None:
                self._theta[k] = st["by_point"].get(points_in_order[k],
                                                    st["theta0"])

    def _nodal_slip(self, solver, uvec):
        """The slip components at every law-carrying pair node, from the
        CURRENT iterate — shape ``(n + 1, dim - 1)``, one column per
        in-fault tangent — read through the local vector so ghosted
        (cross-seam) values resolve, and padded with the tips'/rim's
        identically-zero slip in the final row."""
        dm = solver.dm
        dim = self._nrm.shape[1]
        lvec = dm.getLocalVec()
        dm.globalToLocal(uvec, lvec)
        a = np.asarray(lvec.getArray())
        V = np.zeros((len(self._rows) + 1, self._ncomp))
        for k in range(len(self._rows)):
            du = (a[self._lo_p[k]:self._lo_p[k] + dim]
                  - a[self._lo_m[k]:self._lo_m[k] + dim])
            V[k] = self._tan[k] @ du
        dm.restoreLocalVec(lvec)
        return V                          # V[-1] == 0: the tip/rim pad

    def residual_add(self, solver, uvec, Fh):
        """Add the interface force to the ROTATED residual, in place.
        COLLECTIVE (the Vec assembly): every rank calls, fault or not.

        The slip is a scalar in 2-D (the law reads it SIGNED — every law
        is odd in V) and an in-plane vector in 3-D, where the traction is
        collinear with the slip: tau_vec = tau(|V|) V-hat.
        """
        NQ, WQ = self._NQ, self._WQ
        V = self._nodal_slip(solver, uvec)
        S = self._sigma
        vals = np.zeros((len(self._rows), self._ncomp))
        s2 = np.sqrt(2.0)
        for idx, measure, law_id in self._facets:
            if law_id < 0:
                continue                       # geometry-only (diagnostics)
            Vq = NQ @ V[list(idx)]
            Sq = NQ @ S[list(idx)]
            # theta varies by orders of magnitude along a fault and only ever
            # enters a law through ln(theta): interpolate the LOG, which
            # is also what keeps a positive-only state positive under
            # quadratic shape functions (their undershoot sent theta
            # negative at a quadrature point the plain way).
            Tq = np.exp(NQ @ np.log(self._theta[list(idx)]))
            if self._ncomp == 1:
                tvec = self._laws[law_id].tau(Vq[:, 0], Sq, Tq)[:, None]
            else:
                mag = np.linalg.norm(Vq, axis=1)
                tq = self._laws[law_id].tau(mag, Sq, Tq)
                # collinear traction; where |V| vanishes so does tau(|V|)
                scale = np.where(mag > 1e-300, tq / np.maximum(mag, 1e-300),
                                 0.0)
                tvec = scale[:, None] * Vq
            contrib = s2 * measure * (NQ.T @ (WQ[:, None] * tvec))
            for j, k in enumerate(idx):
                if k >= 0:
                    vals[k] += contrib[j]
        if len(self._rows):
            Fh.setValues(self._rows.ravel(), vals.ravel(), addv=True)
        Fh.assemblyBegin()
        Fh.assemblyEnd()

    def nodal_normal_traction(self, solver, reaction):
        """Signed normal traction sigma_nn per pair node from the Cartesian
        reaction (negative in compression): the jump-normal constraint's
        nodal load l_i = (n.r+ - n.r-)/2, de-smeared by the lumped trace
        mass. The reaction must be the PURE bulk residual — the loop
        stashes it before the interface force is added."""
        dm = solver.dm
        dim = self._nrm.shape[1]
        lvec = dm.getLocalVec()
        dm.globalToLocal(reaction, lvec)
        a = np.asarray(lvec.getArray())
        sig = np.zeros(len(self._rows))
        for k in range(len(self._rows)):
            rp = a[self._lo_p[k]:self._lo_p[k] + dim]
            rm = a[self._lo_m[k]:self._lo_m[k] + dim]
            load = 0.5 * float(self._nrm[k] @ (rp - rm))
            sig[k] = load / max(self._mass[k], 1e-300)
        dm.restoreLocalVec(lvec)
        return sig

    def update_normal_stress(self, solver, reaction):
        """Picard-lag the SIGNED effective normal stress into the laws:
        positive in compression, negative in tension. Each law clamps its
        own strength at zero (Max in the sympy expression) — the clamp
        cannot live here, because where strength vanishes is constitutive:
        at sigma = 0 for bare friction, at sigma = -C/mu with cohesion."""
        sig = self.nodal_normal_traction(solver, reaction)
        self._sigma[:len(sig)] = -sig

    def tangent(self, solver, uvec):
        """The consistent interface tangent at the current iterate, as a
        fresh Mat on the composite layout (caller destroys). COLLECTIVE.

        In 3-D the per-quadrature block on the two slip components is the
        collinear-traction tangent

        .. math:: (d\\tau/dV)\\,\\hat V\\hat V^T
                  + (\\tau/|V|)\\,(I - \\hat V\\hat V^T),

        regularised at :math:`|V| \\to 0` by the laws' own smoothness —
        arctan/asinh forms give :math:`\\tau/|V| \\to d\\tau/dV(0)`, so the
        zero-slip block is the isotropic :math:`d\\tau/dV(0)\\,I`.
        """
        dm = solver.dm
        NQ, WQ = self._NQ, self._WQ
        A = solver.snes.getJacobian()[0]
        rstart, rend = A.getOwnershipRange()
        K = PETSc.Mat().create(comm=dm.comm)
        K.setSizes(((rend - rstart, A.getSize()[0]),
                    (rend - rstart, A.getSize()[0])))
        K.setType("aij")
        K.setPreallocationNNZ((3 if self._ncomp == 1 else 40, 0))
        K.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

        V = self._nodal_slip(solver, uvec)
        S = self._sigma
        for idx, measure, law_id in self._facets:
            if law_id < 0:
                continue                       # geometry-only (diagnostics)
            Vq = NQ @ V[list(idx)]
            Sq = NQ @ S[list(idx)]
            # theta varies by orders of magnitude along a fault and only ever
            # enters a law through ln(theta): interpolate the LOG, which
            # is also what keeps a positive-only state positive under
            # quadratic shape functions (their undershoot sent theta
            # negative at a quadrature point the plain way).
            Tq = np.exp(NQ @ np.log(self._theta[list(idx)]))
            if self._ncomp == 1:
                dq = self._laws[law_id].dtau_dV(Vq[:, 0], Sq, Tq)
                Ke = 2.0 * measure * (NQ.T @ (NQ * (WQ * dq)[:, None]))
                for i, ki in enumerate(idx):
                    if ki < 0:
                        continue
                    for j, kj in enumerate(idx):
                        if kj >= 0:
                            K.setValue(int(self._rows[ki, 0]),
                                       int(self._rows[kj, 0]),
                                       float(Ke[i, j]), addv=True)
                continue
            mag = np.linalg.norm(Vq, axis=1)
            tq = self._laws[law_id].tau(mag, Sq, Tq)
            dq = self._laws[law_id].dtau_dV(mag, Sq, Tq)
            for q in range(len(WQ)):
                if mag[q] > 1e-12:
                    vhat = Vq[q] / mag[q]
                    P = np.outer(vhat, vhat)
                    B = dq[q] * P + (tq[q] / mag[q]) * (np.eye(2) - P)
                else:
                    B = dq[q] * np.eye(2)
                w = 2.0 * measure * WQ[q]
                for i, ki in enumerate(idx):
                    if ki < 0:
                        continue
                    for j, kj in enumerate(idx):
                        if kj < 0:
                            continue
                        NiNj = w * NQ[q, i] * NQ[q, j]
                        for aa in range(2):
                            for bb in range(2):
                                K.setValue(int(self._rows[ki, aa]),
                                           int(self._rows[kj, bb]),
                                           float(NiNj * B[aa, bb]),
                                           addv=True)
        K.assemble()
        return K


def fault_normal_traction(solver, boundary, solve_result):
    """Signed normal traction sigma_nn along the split fault ``boundary``
    (negative in compression), per coincident pair on this rank —
    recovered from the solve's stashed reaction, exactly as the interface
    laws read their effective normal stress. In 2-D returns
    ``(s, sigma_nn)`` ordered by the along-fault coordinate; a 3-D patch
    has no such ordering, so there the first element is the pair
    COORDINATES instead: ``(coords, sigma_nn)``.
    """
    from underworld3.utilities.rotated_bc import _point_coord

    assembler = _InterfaceAssembler(solver, include=(boundary,))
    sig = assembler.nodal_normal_traction(solver, solve_result["reaction"])

    dm = solver.dm
    dim = solver.mesh.dim
    csec = dm.getCoordinateSection()
    cvec = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dim)
    v0, v1 = dm.getDepthStratum(0)
    pts = sorted(assembler._points, key=assembler._points.get)
    coords = np.array([_point_coord(dm, dim, cvec, csec, v0, v1, q)
                       for q in pts]) if pts else np.zeros((0, dim))
    if not len(coords):
        return np.zeros(0), np.zeros(0)
    if dim == 3:
        return coords, sig
    tbar = assembler._tan[:, 0, :].mean(axis=0)
    tbar /= np.linalg.norm(tbar) + 1e-30
    s_coord = coords @ tbar
    s_coord -= s_coord.min()
    order = np.argsort(s_coord)
    return s_coord[order], sig[order]


def fault_pair_jumps(solver, boundary, solve_result, gather=False):
    """The velocity jump at every coincident pair, from the solve.

    Returns ``(coords, jumps, normals)`` — the pair position, the full
    jump vector :math:`v^+ - v^-`, and the fault unit normal — in any
    dimension. Reads the composite solution ``solve_result["U"]``
    through the pairing, which is the only correct route: the pair
    coordinates are identical, so field queries by position see one side
    only. The tangential part of the jump is the slip (a scalar against
    the in-fault tangent in 2-D, an in-plane vector in 3-D); the normal
    part is the leak, held at machine zero by the strong constraint.

    The pairs are rank-local (the split keeps a fault rank-interior), so
    a rank without the fault returns empty arrays. ``gather=True``
    all-gathers the three arrays so every rank holds the whole fault —
    the form a diagnostic that goes on to make collective calls
    (``evaluate``, a write) must use, or the ranks diverge and hang.
    """
    coords, jumps, normals = _fault_pair_jumps_local(solver, boundary,
                                                     solve_result)
    if not gather:
        return coords, jumps, normals
    comm = solver.mesh.dm.comm.tompi4py()
    if comm.size == 1:
        return coords, jumps, normals
    dim = solver.mesh.dim
    parts = comm.allgather((np.asarray(coords, dtype=float).reshape(-1, dim),
                            np.asarray(jumps, dtype=float).reshape(-1, dim),
                            np.asarray(normals, dtype=float).reshape(-1, dim)))
    return tuple(np.vstack([p[k] for p in parts]) for k in range(3))


def _fault_pair_jumps_local(solver, boundary, solve_result):
    """The rank-local half of :func:`fault_pair_jumps`."""
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
        return (np.zeros((0, dim)),) * 3
    return np.array(coords), np.array(jumps), np.array(normals)


def fault_slip(solver, boundary, solve_result):
    """Slip and leak along a 2-D fault, per coincident pair, from the solve.

    Returns ``(s, V, leak)`` on this rank: the along-fault coordinate of each
    pair (arc-length-like, measured along the fault's mean tangent from the
    first tip), the tangential velocity jump :math:`V = \\hat t\\cdot(v^+ -
    v^-)`, and the normal jump :math:`\\hat n\\cdot(v^+ - v^-)` — the leak,
    which the strong constraint holds at machine zero. The 2-D profile view
    of :func:`fault_pair_jumps`; a 3-D patch has no along-fault ordering, so
    read the jumps directly there.
    """
    coords, jumps, normals = fault_pair_jumps(solver, boundary, solve_result)
    if not len(coords):
        return (np.zeros(0),) * 3
    if solver.mesh.dim != 2:
        raise NotImplementedError(
            "fault_slip's arc-length profile is 2-D; use fault_pair_jumps "
            "for a 3-D patch.")
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
