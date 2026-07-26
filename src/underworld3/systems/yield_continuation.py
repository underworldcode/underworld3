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
    """

    set_delta: Callable[[float], None]
    tangent: Any
    delta: Optional[Any] = None


def yield_continuation(
    solver,
    control=None,
    delta0=1.0,
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

    Starting δ is deliberately **large**, where the power-mean soft-min is the harmonic
    mean and is bounded by the background viscosity even as :math:`\dot\varepsilon \to
    0`. That is what makes the first (cold) solve well posed, and why no separate
    viscous pre-solve is needed.

    Parameters
    ----------
    solver :
        A configured Stokes/SNES solver whose constitutive model supports the homotopy.
    control : YieldHomotopyControl, optional
        How to set δ and which tangent to use. Defaults to
        ``solver.constitutive_model._yield_homotopy_control()``.
    delta0 : float
        Starting (smooth) δ. Large δ is about as cheap to solve as small, so be generous.
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
    if not delta0 > 0.0:
        raise ValueError(f"delta0 must be positive, got {delta0}")

    if control is None:
        cm = getattr(solver, "constitutive_model", None)
        if cm is None or not getattr(cm, "supports_yield_homotopy", False):
            raise TypeError(
                "yield_continuation needs a constitutive model that supports the yield "
                "homotopy (supports_yield_homotopy). Got "
                f"{type(cm).__name__ if cm is not None else None}."
            )
        control = cm._yield_homotopy_control()

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
