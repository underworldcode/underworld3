r"""Multi-solve δ-continuation for hard viscoplastic (Drucker–Prager) yield.

A single solve straight onto a sharp yield surface (soft-min δ → 0) stalls or
diverges from a viscous guess. The robust, trusted route is a *continuation*: hold
δ **constant** for a full nonlinear solve to tolerance, warm-start the next
(smaller) δ from that converged state, and march δ down to the sharp surface.

This module ships that march as a reusable driver — the production form of the
hand-tuned harness used in the Spiegelman notch studies. It orchestrates solves; it
does not change the solver's tangent, preconditioner, or line-search options (the
caller configures those — see :func:`yield_continuation` notes).
"""

import sympy

import underworld3 as uw


def yield_continuation(
    solver,
    delta=None,
    delta0=1.0,
    down=0.5,
    dmin=1.0e-3,
    entry_maxit=30,
    step_maxit=10,
    verbose=True,
):
    r"""March the yield soft-min regularisation δ down to the sharp surface as a
    sequence of constant-δ solves, each warm-started from the previous.

    Each δ is held **constant** for one full nonlinear solve to tolerance; the
    converged :math:`(v, p)` warm-starts the next (smaller) δ. δ is a
    ``constants[]`` atom, so the operators are built once at ``delta0`` and every
    later step is a recompile-free ``PetscDSSetConstants`` update. The march halves
    δ (``× down``) while solves keep converging and **settles** at the smallest
    feasible δ — the closest automatic approach to hard-Min. A δ that fails to
    converge is reverted, and the previous δ is the reported answer.

    Parameters
    ----------
    solver :
        A configured Stokes/SNES solver whose viscosity uses a δ-parameterised
        soft-min yield law, already warm-started with a viscous (no-yield) presolve.
    delta : UWexpression, optional
        The δ ``constants[]`` atom to march. Defaults to the constitutive model's
        yield-softness atom, ``solver.constitutive_model._get_yield_softness()``.
    delta0 : float
        Starting (smooth) δ. Large δ is as cheap to solve as small, so be generous.
    down : float
        Multiplicative step, :math:`0 < \mathrm{down} < 1` (δ ← δ·down per success).
    dmin : float
        Stop once δ reaches this floor (hard-Min effectively reached).
    entry_maxit : int
        Nonlinear iteration budget for the first solve (from the viscous guess).
    step_maxit : int
        Budget for each warm step. A feasible warm step converges in a few
        iterations; a tight budget lets a too-hard step abort cheaply.
    verbose : bool
        Print per-δ convergence (rank-safe).

    Returns
    -------
    dict
        ``settled_delta`` (smallest δ that converged, or ``None`` if the first solve
        failed), ``reason`` (final SNES converged reason), ``steps`` (number of
        δ-solves), ``reached_dmin`` (whether the march made it to the floor).

    Notes
    -----
    Configure the solver **before** calling — this driver changes none of it:

    - the constitutive model in a δ-parameterised soft-min mode
      (``yield_mode="softmin"``);
    - the consistent-Newton tangent (``solver.consistent_jacobian = True``);
    - for the stiff, non-symmetric consistent-Newton velocity operator, a
      non-symmetry-safe FMG smoother —
      ``solver.petsc_options["fieldsplit_velocity_mg_levels_ksp_type"] = "gmres"``
      (with ``mg_levels_pc_type = "sor"``). The default Chebyshev/Richardson
      smoothers stall on it, turning a solve into an effectively unbounded grind;
      bound the outer Krylov (``ksp_max_it`` ~ 80) so a hostile step fails fast.
    """
    if delta is None:
        cm = getattr(solver, "constitutive_model", None)
        if cm is None or not hasattr(cm, "_get_yield_softness"):
            raise TypeError(
                "yield_continuation: no δ soft-min atom found. Put the constitutive "
                "model in a δ-parameterised mode (yield_mode='softmin') or pass an "
                "explicit `delta` atom."
            )
        delta = cm._get_yield_softness()

    if not (0.0 < down < 1.0):
        raise ValueError(f"down must satisfy 0 < down < 1, got {down}")

    u, p = solver.Unknowns.u, solver.Unknowns.p
    # Warm-state snapshot: a raw copy of the (already consistent) solution values,
    # restored if a too-hard δ corrupts the iterate.
    u_good, p_good = u.data.copy(), p.data.copy()
    saved_maxit = solver.petsc_options.getInt("snes_max_it", 50)

    d = float(delta0)
    settled = None
    reason = 0
    steps = 0
    reached_dmin = False
    first = True
    while True:
        delta.sym = sympy.Float(d)
        if first:
            solver.is_setup = False  # build the operators once, at delta0
        else:
            solver._update_constants()  # recompile-free δ update
        solver.petsc_options["snes_max_it"] = entry_maxit if first else step_maxit
        solver.solve(zero_init_guess=False)
        reason = int(solver.snes.getConvergedReason())
        nit = int(solver.snes.getIterationNumber())
        steps += 1

        if reason > 0:
            settled = d
            u_good[...] = u.data
            p_good[...] = p.data
            if verbose:
                uw.pprint(0, f"  [yield-continuation] δ={d:<11g} its={nit:3d} → converged")
            if d <= dmin:
                reached_dmin = True
                break
            d *= down
            first = False
        else:
            with uw.synchronised_array_update("yield_continuation revert"):
                u.data[...] = u_good
                p.data[...] = p_good
            if verbose:
                uw.pprint(0, f"  [yield-continuation] δ={d:<11g} its={nit:3d} → failed "
                             f"(reason={reason}); settling at δ={settled}")
            break

    solver.petsc_options["snes_max_it"] = saved_maxit
    return {
        "settled_delta": settled,
        "reason": reason,
        "steps": steps,
        "reached_dmin": reached_dmin,
    }
