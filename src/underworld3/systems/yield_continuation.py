r"""Multi-solve δ-continuation for hard viscoplastic (Drucker–Prager) yield.

A single solve straight onto a sharp yield surface (soft-min δ → 0) stalls or
diverges from a cold start. The robust, trusted route is a *continuation*: hold δ
**constant** for a full nonlinear solve to tolerance, warm-start the next (smaller)
δ from that converged state, and march δ down toward the sharp surface.

This module ships that march. It is what ``solver.solve(homotopy=True)`` runs; the
constitutive model says what δ *is* for it (:class:`YieldHomotopyControl`, built by
``constitutive_model._yield_homotopy_control()``) and this driver marches the number.

Do **not** try to ramp δ inside a single SNES solve. That is proven not to work: the
continuation only sharpens δ from a *converged*, well-conditioned iterate, whereas an
in-solve ramp sharpens it mid-solve where the consistent-Newton Jacobian on a
sharpening yield surface is ill-conditioned and the linear solve fails.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import underworld3 as uw


@dataclass(frozen=True)
class YieldHomotopyControl:
    """How to march one constitutive model's yield homotopy.

    Built by ``constitutive_model._yield_homotopy_control()``, which also switches the
    model into its smooth (δ-parameterised) yield mode. The model owns the meaning of
    the parameter; the solver only moves it.

    Attributes
    ----------
    set_delta
        Sets the yield softness δ. Model-owned, because *how* δ is applied differs:
        the isotropic models update a ``constants[]`` atom (no recompile), while the
        transverse-isotropic VEP model rebuilds.
    tangent
        The ``consistent_jacobian`` value to solve with — Newton for a viscous-plastic
        yield, the frozen/Picard tangent for the elastic (VEP) models.
    delta
        The δ ``constants[]`` atom itself, for diagnostics.
    delta0
        The entry δ this family should START from. Model-owned because δ does NOT mean
        the same thing in the two soft-min families: for the power mean the sharpness is
        ``s = 1/(δ + 0.001)``, so δ ≤ 1 and δ = 1 IS the harmonic mean, while for the
        sqrt family δ is a percentage stress deviation and a generous entry is O(10).
        Carrying one number across both families is a category error.
    """

    set_delta: Callable[[float], None]
    tangent: Any
    delta: Optional[Any] = None
    delta0: float = 1.0


@dataclass(frozen=True)
class RateStrengtheningControl:
    """How to ladder one constitutive model's declared plastic rate strengthening.

    Built by ``constitutive_model._rate_strengthening_control()``; ``None`` when the model
    states no ``plastic_rate_strengthening``. The model owns the meaning (``eta_pl ->
    tau_y/(2 edot_II) + m eta_reg``); the solver only moves ``m``, from the viscous limit
    down to 1, and never past 1 — the end of the ladder is the problem as stated.

    Attributes
    ----------
    set_scale
        Sets the multiplier ``m`` (a ``constants[]`` atom: no recompile).
    tangent
        The ``consistent_jacobian`` value the model pairs with its yield law.
    scale, parameter
        The ``m`` atom and the ``eta_reg`` parameter, for diagnostics.
    probe_terms
        Cell-wise expressions the ladder reads to place its first rungs (see
        ``ViscousFlowModel._rate_strengthening_probe_terms``), or ``None`` when the
        model does not provide them; the ladder then needs an explicit ``scale0``.
    """
    set_scale: Callable[[float], None]
    tangent: Any
    scale: Optional[Any] = None
    parameter: Optional[Any] = None
    probe_terms: Optional[dict] = None


def rate_strengthening_continuation(
    solver,
    control=None,
    scale0=None,
    scale_big=None,
    entry_factor=1.0e3,
    down=0.5,
    rung_tangent="continuation",
    entry_maxit=60,
    step_maxit=80,
    retries=2,
    max_steps=40,
    solve_kwargs=None,
    verbose=True,
):
    r"""Ladder the declared plastic rate strengthening from the viscous limit to the
    stated problem (``m`` from ``scale0`` down to 1), each rung a full solve warm-started
    from the previous one.

    Measured on the Spiegelman notch (2026-10-04): the equivalent ``xi`` ladder keeps the
    velocity block healthy on every rung (~14 multigrid cycles per Krylov iteration) and
    reaches a converged state on a case no single solve could; the blend
    (``consistent_jacobian="continuation"``) on every rung is 3-4x cheaper than Newton
    with a blend fallback, which is why it is the rung tangent here.

    ViscoPlasticFlowModel only: the elastic models advance a stress history on every
    solve, so a ladder of solves has to be driven around one history update by hand.

    Parameters
    ----------
    solver
        A configured Stokes solver whose model states ``plastic_rate_strengthening``.
    control : RateStrengtheningControl, optional
        Defaults to ``solver.constitutive_model._rate_strengthening_control()``.
    scale0 : float, optional
        First rung. Default: the viscous limit found from the fields — one solve at the
        entry multiplier (``scale_big``), at which nothing yields, then ``0.95 *`` the
        largest ``m`` at which a cell still yields there. If that is ``<= 1`` the stated
        problem does not yield beyond its onset and the ladder is one solve.
    scale_big : float, optional
        The entry multiplier. Default ``entry_factor * max(eta_ve / eta_reg)`` over the
        domain, so that ``m * eta_reg`` exceeds the viscous branch everywhere whatever the
        model's own ratio of the two scales.
    down : float
        Multiplicative step in ``m`` per rung (adapted: a cheap rung widens it, a
        laboured one narrows it); the last rung is clamped to exactly 1.
    rung_tangent
        ``consistent_jacobian`` for the rungs where the model's tangent is Newton;
        ``None`` keeps the model's own.
    entry_maxit, step_maxit, retries, max_steps, solve_kwargs, verbose
        As for :func:`yield_continuation`.

    Returns
    -------
    dict
        ``scale0``, ``rungs`` (list of ``(m, reason, its)``), ``reached_one``,
        ``converged``, ``reason``, ``message``. A ladder that stops above ``m = 1`` leaves
        the model at ``m = 1`` (as stated), the fields at the last converged rung, and
        warns.
    """
    import math
    import warnings

    if not (0.0 < down < 1.0):
        raise ValueError(f"down must satisfy 0 < down < 1, got {down}")
    if getattr(getattr(solver, "Unknowns", None), "DFDt", None) is not None:
        raise NotImplementedError(
            "the rate-strengthening ladder runs several solves and would advance a stress "
            "history each time; drive it yourself around one history update")
    cm = getattr(solver, "constitutive_model", None)
    if control is None:
        control = cm._rate_strengthening_control() if cm is not None and hasattr(cm, "_rate_strengthening_control") else None
    if control is None:
        raise TypeError(
            "rate_strengthening_continuation needs a constitutive model that STATES a "
            "plastic_rate_strengthening (Parameters.plastic_rate_strengthening > 0); the "
            "scale is the model's, not a solver default.")
    if scale0 is None and control.probe_terms is None:
        raise NotImplementedError(
            f"{type(cm).__name__} does not provide the terms the ladder reads to find the "
            f"viscous limit; pass scale0 explicitly")

    saved_probe = solver._difficulty_probe
    saved_probe_maxit = solver._difficulty_max_it
    saved_resume = solver._resume_abs_target
    saved_tangent = solver.consistent_jacobian
    saved_scale = cm.rate_strengthening_scale
    if rung_tangent is not None:
        solver.consistent_jacobian = rung_tangent if control.tangent is True else control.tangent
    u, p = solver.Unknowns.u, solver.Unknowns.p
    rungs = []
    reason = 0
    reached_one = False
    message = None
    try:
        if scale0 is None:
            terms = control.probe_terms
            if scale_big is None:
                visc, reg = _cell_values(solver, (terms["viscous"], terms["eta_reg"]))
                scale_big = entry_factor * _masked_max_ratio(visc, reg)
                if not (math.isfinite(scale_big) and scale_big > 0.0):
                    message = (f"the entry multiplier could not be found from the fields "
                               f"(max(eta_ve/eta_reg) = {scale_big / entry_factor:g})")
                    return dict(scale0=None, rungs=rungs, reached_one=False, converged=False,
                                reason=0, message=message)
            # the viscous limit, found from the fields rather than from the problem
            control.set_scale(float(scale_big))
            solver.is_setup = False
            solver._difficulty_probe = True
            solver._difficulty_max_it = entry_maxit
            solver._resume_abs_target = None
            solver.solve(zero_init_guess=not solver.has_solution, **dict(solve_kwargs or {}))
            reason = int(solver.snes.getConvergedReason())
            rungs.append((float(scale_big), reason, int(solver.snes.getIterationNumber())))
            if reason <= 0:
                message = f"the viscous-limit solve (m={scale_big:g}) failed (reason={reason})"
                return dict(scale0=None, rungs=rungs, reached_one=False, converged=False,
                            reason=reason, message=message)
            num, den = _cell_values(solver, (terms["onset_num"], terms["onset_den"]))
            m_star = _masked_max_ratio(num, den)
            if math.isnan(m_star) or m_star == math.inf:
                message = f"the viscous limit could not be read from the fields (m* = {m_star})"
                return dict(scale0=None, rungs=rungs, reached_one=False, converged=False,
                            reason=0, message=message)
            scale0 = 0.95 * m_star if m_star > 1.0 else 1.0
            if verbose:
                uw.pprint(f"  [rate-strengthening] viscous limit m*={m_star:.4g}; ladder starts at m={scale0:.4g}")
        m = max(float(scale0), 1.0)
        step = float(down)
        import numpy as np
        u_good, p_good = np.array(u.array), np.array(p.array)
        failures = 0
        first = not rungs
        while len(rungs) < max_steps:
            control.set_scale(m)
            if first:
                solver.is_setup = False
            else:
                solver._update_constants()
            budget = entry_maxit if first else step_maxit
            solver._difficulty_probe = True
            solver._difficulty_max_it = budget
            solver._resume_abs_target = None
            solver.solve(zero_init_guess=not solver.has_solution, **dict(solve_kwargs or {}))
            reason = int(solver.snes.getConvergedReason())
            nit = int(solver.snes.getIterationNumber())
            rungs.append((m, reason, nit))
            first = False
            if reason > 0:
                u_good[...] = u.array; p_good[...] = p.array
                if verbose:
                    uw.pprint(f"  [rate-strengthening] m={m:<10.4g} its={nit:3d} -> converged")
                if m <= 1.0:
                    reached_one = True
                    break
                if nit <= max(2, budget // 5):
                    step = max(step * step, 0.05)
                elif nit >= 0.8 * budget:
                    step = min(step ** 0.5, 0.95)
                m = max(m * step, 1.0)
            else:
                with uw.synchronised_array_update("rate_strengthening revert"):
                    u.array[...] = u_good
                    p.array[...] = p_good
                solver._record_convergence_status(converged=True)
                failures += 1
                if failures > retries:
                    message = f"m={m:.4g} failed (reason={reason}) after {retries} retries"
                    break
                step = min(step ** 0.5, 0.95)
                # retry from the LAST converged rung — the state just restored
                last_ok = next((r[0] for r in reversed(rungs) if r[1] > 0), m / step)
                m = max(last_ok * step, 1.0)
                if verbose:
                    uw.pprint(f"  [rate-strengthening] failed (reason={reason}); retrying at m={m:.4g}")
        else:
            message = f"max_steps={max_steps} rungs used, last converged m={rungs[-1][0]:.4g}" if rungs else None
    finally:
        solver._difficulty_probe = saved_probe
        solver._difficulty_max_it = saved_probe_maxit
        solver._resume_abs_target = saved_resume
        solver.consistent_jacobian = saved_tangent
        if not reached_one:
            # leave the model as it was stated, never at an intermediate m; the next
            # solve repacks the constant, so no repack is attempted here (it would
            # mask the exception that brought us to this block)
            control.set_scale(saved_scale)
    if not reached_one:
        message = message or "the ladder stopped above m = 1"
        if verbose:
            uw.pprint(f"  [rate-strengthening] {message}; the model is back at m={saved_scale:g}")
        warnings.warn(
            f"[{solver.name}] rate-strengthening ladder did not reach the stated problem: "
            f"{message}. The fields are the last converged rung's; the model is back at "
            f"m={saved_scale:g}.", RuntimeWarning, stacklevel=2)
    return dict(scale0=scale0, rungs=rungs, reached_one=reached_one,
                converged=bool(reached_one and reason > 0), reason=reason, message=message)


def _cell_values(solver, exprs):
    """Cell averages (P0 projections) of ``exprs`` on this rank's cells.

    One P0 variable and one projector per solver, created on first use and reused, so
    every rank creates the same objects in the same order. Raises if a projection
    fails rather than returning the previous contents of the variable.
    """
    import numpy as np
    probe = getattr(solver, "_rate_strengthening_probe", None)
    if probe is None:
        mesh = solver.mesh
        var = uw.discretisation.MeshVariable(
            f"rs_probe_{solver.instance_number}", mesh, 1, degree=0, continuous=False)
        proj = uw.systems.Projection(mesh, var)
        proj.petsc_options.delValue("ksp_monitor")
        probe = solver._rate_strengthening_probe = (var, proj)
    var, proj = probe
    values = []
    for expr in exprs:
        proj.uw_function = expr
        proj.solve()
        if proj.snes.getConvergedReason() <= 0:
            raise RuntimeError(
                f"rate-strengthening probe projection failed "
                f"(reason={proj.snes.getConvergedReason()}) for {expr}")
        values.append(np.asarray(var.array).reshape(-1).copy())
    return values


def _masked_max_ratio(num, den):
    """Global ``max(num / den)`` over cells with ``den > 0`` (collective).

    ``-inf`` when no cell on any rank qualifies; NaN if any qualifying value is NaN.
    """
    import numpy as np
    from mpi4py import MPI
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    ok = den > 0.0
    ratio = num[ok] / den[ok]
    # NaN is reduced as its own flag: MPI.MAX compares with >, which drops a NaN
    if int(uw.mpi.comm.allreduce(int(np.isnan(ratio).any()), op=MPI.MAX)):
        return float("nan")
    local = float(np.max(ratio)) if ratio.size else -np.inf
    return float(uw.mpi.comm.allreduce(local, op=MPI.MAX))


def yield_continuation(
    solver,
    control=None,
    smoother=None,
    anchor=None,
    delta0=None,
    down=0.5,
    dmin=1.0e-3,
    entry_maxit=30,
    step_maxit=10,
    retries=2,
    max_steps=60,
    solve_kwargs=None,
    verbose=True,
):
    r"""March the yield regularisation δ down to the sharp surface as a sequence of
    constant-δ solves, each warm-started from the previous.

    Each δ is held **constant** for one full nonlinear solve to tolerance; the
    converged solution warm-starts the next (smaller) δ. The march **settles** at the
    smallest feasible δ — the closest automatic approach to the hard ``Min``. A δ that
    fails to converge is reverted and retried with a gentler step; when the retries are
    used up, the last converged δ is the answer.

    The step size is **residual-guided**: a δ-step that converges in very few Newton
    iterations means there is slack, so the next step is bigger; a step that uses most
    of its iteration budget means the march is close to the feasible edge, so the next
    step is gentler.

    Starting δ is deliberately **large**, where the power-mean soft-min stays bounded by
    the background viscosity even as :math:`\dot\varepsilon \to 0`. That is what makes
    the first (cold) solve well posed, and why no separate viscous pre-solve is needed.

    Parameters
    ----------
    solver :
        A configured Stokes/SNES solver whose constitutive model supports the homotopy.
    control : YieldHomotopyControl, optional
        How to set δ and which tangent to use. Defaults to
        ``solver.constitutive_model._yield_homotopy_control()``.
    smoother : {"powermean", "sqrt"}, optional
        Which soft-min FAMILY to march. Default (``None``) leaves the model's own
        choice, which is the power mean. The families differ in how much room δ gives
        them: the sqrt law's smoothing saturates at a factor :math:`2f/(f+2)` in the
        overstress ratio :math:`f = \eta_{ve}/\eta_{pl}`, so where f is O(1) it can
        barely build an easier problem however large δ is, while the power mean does not
        saturate. Where f is large the position reverses. f is cheap to measure on the
        viscous seed, so choose the family from it rather than by reputation.
    anchor : {"onset", "yield"}, optional
        Which point of the smoothed law is pinned to the exact one — see
        ``ViscousFlowModel.yield_anchor``. Applies to BOTH families. Default (``None``)
        leaves the model's own choice. ``"yield"`` keeps the stress exact at the yield
        point for every δ and puts the law on or above the exact one throughout;
        ``"onset"`` (the historical default) is exact on the unyielded branch but sits
        BELOW the exact law at and above the yield point, which makes the entry problem
        WEAKER than the sharp problem it is supposed to lead to.
    delta0 : float, optional
        Starting (smooth) δ. Defaults to the family's own entry (``control.delta0``),
        because δ is NOT the same parameter in the two families — see
        :class:`YieldHomotopyControl`. Do not carry one number across both.
    down : float
        Nominal multiplicative step, :math:`0 < down < 1` (δ ← δ·down per success).
        Adapted during the march as described above.
    dmin : float
        Stop once δ reaches this floor (the sharp surface is effectively reached).
    entry_maxit : int
        Nonlinear iteration budget for the first solve.
    step_maxit : int
        Budget for each warm step. A feasible warm step converges in a few iterations;
        a tight budget lets a too-hard step abort cheaply.
    retries : int
        How many times to retry a failed δ with a gentler step before settling.
    max_steps : int
        Hard cap on the number of δ-solves, so a march that is easing off in small
        steps cannot run unbounded.
    solve_kwargs : dict, optional
        Extra arguments forwarded to every inner ``solver.solve()`` — notably
        ``timestep`` for a visco-elastic-plastic model, whose solves need one.
    verbose : bool
        Report each δ (rank-safe).

    Returns
    -------
    dict
        ``settled_delta`` (smallest δ that converged, or ``None`` if even the first
        solve failed), ``reason`` (final SNES converged reason), ``steps`` (number of
        δ-solves attempted), ``reached_dmin``, and ``converged``.
    """
    if not (0.0 < down < 1.0):
        raise ValueError(f"down must satisfy 0 < down < 1, got {down}")

    if control is None:
        cm = getattr(solver, "constitutive_model", None)
        if cm is None or not getattr(cm, "supports_yield_homotopy", False):
            raise TypeError(
                "yield_continuation needs a constitutive model that supports the yield "
                "homotopy (supports_yield_homotopy). Got "
                f"{type(cm).__name__ if cm is not None else None}."
            )
        control = cm._yield_homotopy_control(smoother=smoother, anchor=anchor)
    elif smoother is not None or anchor is not None:
        raise ValueError(
            "smoother=/anchor= configure the soft-min when this driver builds the "
            "control; they cannot override a control that was passed in ready-made. "
            "Set them on the model before building the control instead."
        )

    # Each family has its own natural entry, and they are not interchangeable numbers.
    if delta0 is None:
        delta0 = control.delta0
    if not delta0 > 0.0:
        raise ValueError(f"delta0 must be positive, got {delta0}")

    # A march is a SEQUENCE of solves, so a solver that integrates history in time
    # would advance that history once per delta-step rather than once per timestep.
    # Refuse rather than silently corrupt it -- see the design note.
    if getattr(getattr(solver, "Unknowns", None), "DFDt", None) is not None:
        raise NotImplementedError(
            "solve(homotopy=True) is not available on a solver carrying stress "
            "history (visco-elastic-plastic): the march runs several solves, each of "
            "which would advance the elastic stress history by a full timestep. Solve "
            "the VEP model without the homotopy, or drive the march yourself around a "
            "single history update."
        )

    # Iteration budgets have to be applied where they survive the solve path's own
    # setFromOptions (which pushes a hardcoded snes_max_it); petsc_options alone is
    # silently overwritten. This is the same hook estimate_difficulty() uses.
    saved_probe = solver._difficulty_probe
    saved_probe_maxit = solver._difficulty_max_it
    saved_resume = solver._resume_abs_target
    saved_tangent = solver.consistent_jacobian

    solver.consistent_jacobian = control.tangent

    u, p = solver.Unknowns.u, solver.Unknowns.p
    # Warm-state snapshot: a raw copy of the (already consistent) solution values,
    # restored if a too-hard delta corrupts the iterate.
    u_good, p_good = u.data.copy(), p.data.copy()

    # Whether there is a solution to warm-start the FIRST delta from. Read before the
    # rebuild below, which invalidates the solver and clears the flag.
    warm_entry = solver.has_solution

    d = float(delta0)
    step = float(down)
    settled = None
    reason = 0
    steps = 0
    failures = 0
    reached_dmin = False
    first = True

    try:
        while steps < max_steps:
            control.set_delta(d)
            if first:
                solver.is_setup = False       # build the operators once, at delta0
            else:
                solver._update_constants()    # recompile-free delta update

            budget = entry_maxit if first else step_maxit
            solver._difficulty_probe = True
            solver._difficulty_max_it = budget
            solver._resume_abs_target = None

            warm = warm_entry if first else True
            solver.solve(zero_init_guess=not warm, **dict(solve_kwargs or {}))
            reason = int(solver.snes.getConvergedReason())
            nit = int(solver.snes.getIterationNumber())
            steps += 1
            first = False

            if reason > 0:
                settled = d
                u_good[...] = u.data
                p_good[...] = p.data
                if verbose:
                    uw.pprint(f"  [yield-continuation] d={d:<11g} its={nit:3d} -> converged")
                if d <= dmin:
                    reached_dmin = True
                    break
                # Residual-guided: plenty of slack => take a bigger bite next time;
                # near the iteration budget => ease off. Clamped so the march can
                # neither stall (step -> 1) nor leap past the feasible edge in one go.
                if nit <= max(2, budget // 5):
                    step = max(step * step, 0.05)
                elif nit >= 0.8 * budget:
                    step = min(step ** 0.5, 0.95)
                d *= step
            else:
                failed_delta = d
                with uw.synchronised_array_update("yield_continuation revert"):
                    u.data[...] = u_good
                    p.data[...] = p_good
                # The reverted fields ARE a converged state, so say so: otherwise the
                # retry (and the caller's next solve) would auto-cold-start off a
                # solution that is perfectly good.
                if settled is not None:
                    solver._record_convergence_status(converged=True)
                failures += 1
                if settled is None or failures > retries:
                    if verbose:
                        uw.pprint(f"  [yield-continuation] d={failed_delta:<11g} its={nit:3d} "
                                  f"-> failed (reason={reason}); settling at d={settled}")
                    break
                # Retry from the last good delta with a gentler step.
                step = min(step ** 0.5, 0.95)
                d = settled * step
                if verbose:
                    uw.pprint(f"  [yield-continuation] d={failed_delta:<11g} its={nit:3d} "
                              f"-> failed (reason={reason}); retrying at d={d:g}")
        else:
            if verbose:
                uw.pprint(f"  [yield-continuation] stopped at the {max_steps}-step cap; "
                          f"settled at d={settled}")
    finally:
        # Leave the model holding the delta that actually converged, and restore every
        # setting the march borrowed -- including on the exception path.
        if settled is not None and settled != d:
            control.set_delta(settled)
            solver._update_constants()
        solver._difficulty_probe = saved_probe
        solver._difficulty_max_it = saved_probe_maxit
        solver._resume_abs_target = saved_resume
        solver.consistent_jacobian = saved_tangent

    return {
        "settled_delta": settled,
        "reason": reason,
        "steps": steps,
        "reached_dmin": reached_dmin,
        "converged": settled is not None,
    }
