r"""
The multigrid option bundles for a solver's managed preconditioner block.

Three routes reach a multigrid velocity block in UW3, and all three must be
configured the same way:

============================  ==============================================
route                         prolongation
============================  ==============================================
native                        PETSc ``DMCreateInterpolation`` between refined
                              DMPlex levels (mesh built with ``refinement>=1``)
custom-P, standard path       barycentric / RBF, Galerkin coarse operators
                              (:mod:`underworld3.utilities.custom_mg`)
custom-P, rotated path        as above, with the FINE prolongation rotated
                              (:mod:`underworld3.utilities.rotated_bc`)
============================  ==============================================

custom-P is mandatory wherever native cannot go — rotated boundary conditions
(the DM-coupled hierarchy cannot express the per-node rotation) and non-nested
grids (``adapt()`` children have no DMPlex refinement relation) — so the routes
are not alternatives, they are the same preconditioner reached three ways.

This module is where the bundles live so that they cannot drift apart. Each
writer reads a bundle from here and applies it to its own options object under
its own prefix; nobody writes a multigrid option value anywhere else.

A bundle carries two things: the settings it *sets*, and the **stale** keys it
must *remove*. The stale list is derived, not hand-written: it is every key any
other bundle sets that this one does not. That matters because these bundles are
written into a shared options database under a shared prefix — toggling a block
from GAMG to geometric MG leaves the GAMG-only keys behind, and ``setFromOptions``
will happily re-read them. Hand-maintained delete lists are exactly what let the
smoother iteration count go unset and be inherited from whatever ran before
(issue #468).

Examples
--------
Apply the geometric bundle to a solver's managed block::

    from underworld3.utilities import multigrid_options
    multigrid_options.geometric_mg_bundle().apply(
        PETSc.Options(), self.petsc_options_prefix + prefix,
        owned=self._managed_pc_options)

Read the settings without writing them (the rotated path stages its options in a
dict so it can drop them again after ``setUp``)::

    cfg.update({f"fieldsplit_vel_{k}": v
                for k, v in multigrid_options.gamg_bundle().settings.items()})
"""

from typing import NamedTuple

__all__ = ["MGSettings", "geometric_mg_bundle", "gamg_bundle", "option_string",
           "GEOMETRIC_MG_COARSE_SOLVERS", "GEOMETRIC_MG_SMOOTHERS"]


def option_string(value):
    """How PETSc stores ``value`` in its options database, so a value UW3 wrote can
    be compared against what is in there now. ``None`` is a bare flag (reads back as
    an empty string); booleans lower-case."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _user_owns(opts, name, owned):
    """Was ``name`` set by someone other than UW3's own option writers?

    True when the key is present and its current value is not the one UW3 last
    recorded writing. ``owned is None`` disables the check (the caller owns the
    prefix outright)."""
    if owned is None or not opts.hasName(name):
        return False
    return opts.getString(name) != owned.get(name)


#: Coarse-solve variants of the geometric bundle. ``"redundant"`` (redundant+LU)
#: is the default and is np-safe; ``"svd"`` is required whenever the coarse
#: operator inherits a null space — see :func:`geometric_mg_bundle`.
GEOMETRIC_MG_COARSE_SOLVERS = ("redundant", "svd")

#: Smoother variants. These are the two measured regimes, not a taste setting:
#: ``"robust"`` survives a badly-conditioned operator where a stationary smoother
#: stalls, ``"fast"`` is cheaper per cycle where the operator is benign. Selected by
#: ``solver.strategy``; see :func:`geometric_mg_bundle` for the numbers.
GEOMETRIC_MG_SMOOTHERS = ("robust", "fast")


class MGSettings(NamedTuple):
    """A preconditioner option bundle: the keys to set, and the keys to clear.

    ``settings`` maps an option suffix (no prefix, no leading ``-``) to its
    value; ``None`` means a bare PETSc flag. ``stale`` lists suffixes this
    bundle does not own but a sibling bundle does, which must be removed before
    ``setFromOptions`` so a previously-applied bundle cannot leak through.
    """

    settings: dict
    stale: tuple

    def apply(self, opts, prefix="", owned=None):
        """Write this bundle into ``opts`` (a ``PETSc.Options``) under ``prefix``.

        ``prefix`` is the managed block's own prefix — ``""`` for a top-level PC,
        ``"fieldsplit_velocity_"`` for the Stokes velocity sub-block — and is
        applied on top of whatever prefix ``opts`` itself carries.

        Parameters
        ----------
        owned : dict or None
            Bookkeeping of the option values UW3 itself has written, keyed by full
            option name. A key that is already present but whose current value is
            **not** what UW3 last wrote was set by the user, and is left alone —
            both here and in the stale-key clear, because it is not ours to remove.
            Pass ``None`` to write unconditionally, which is right when the caller
            owns the whole prefix (the rotated path builds a fresh, per-solve
            prefix no user can address).

            Ownership is *recorded*, never inferred from the value. Guessing by
            value fails as soon as a second internal writer touches the same key —
            which is exactly how the ``tolerance`` and ``strategy`` setters defeated
            an earlier attempt (#477). Every internal writer of a bundle key goes
            through this bookkeeping (``SolverBaseClass._push_managed_option``), so
            anything unrecorded is the user's by construction.
        """
        for key, value in self.settings.items():
            name = prefix + key
            if _user_owns(opts, name, owned):
                continue
            opts.setValue(name, value)
            if owned is not None:
                owned[name] = option_string(value)
        for key in self.stale:
            name = prefix + key
            if _user_owns(opts, name, owned):
                continue
            opts.delValue(name)
            if owned is not None:
                owned.pop(name, None)


def _geometric_mg_settings(coarse, smoother="robust"):
    """The geometric-MG settings for one coarse-solve and smoother variant.

    Both variants set the SAME keys — only values differ — so the derived stale-key
    sets are variant-independent."""
    settings = {
        "pc_type": "mg",
        "pc_mg_type": "full",              # FMG (F-cycle)
        # Galerkin (RAP) coarse operators are REQUIRED: UW3 installs no
        # residual/Jacobian callbacks on the coarse DMs, so PETSc cannot
        # re-discretise the operator there.
        "pc_mg_galerkin": "both",
        # gmres+sor, sized for a DEEP hierarchy (the only kind worth having: a
        # two-level cycle is a coarse-grid correction, not a V-cycle, and is not
        # worth special-casing). Chebyshev needs eigenvalue estimates of the
        # smoothed operator, which are fragile on the indefinite /
        # variable-viscosity velocity block and diverge.
        #
        # The measurement, which is what this setting rests on: Spiegelman notch
        # (Drucker-Prager, eta contrast 1e26) over a nested 4-level hierarchy,
        # contraction per V-cycle rho = 0.75 (richardson) vs 0.56 (gmres) at the
        # SAME four smoother iterations, the margin growing with depth (5% at 3
        # levels, 25% at 4). Four iterations, not more: per unit work gmres/4
        # (rho^(1/4) = 0.87) beats gmres/8 (0.91).
        #
        # TODO(DESIGN): the MECHANISM on record for that result is wrong. Both
        # this comment and design/nonlinear-solver-homotopy-warmstart.md (Layer 3)
        # said richardson degrades because the consistent-Newton tangent makes the
        # velocity block NON-SYMMETRIC. Measured (velocity_block_symmetry.py in the
        # rotated_pmat_audit study), that block stays symmetric to machine precision
        # -- 5e-17, against a linear calibration of 5e-17 -- under shear-thinning,
        # power-law AND Drucker-Prager with pressure-dependent yield, with a control
        # confirming the Newton term contributed 5-40% of ||A||. For an isotropic
        # eta(eps_II) the Newton term is a rank-one outer product a (x) a with
        # a = eps_dot(u), which is symmetric. Non-symmetry does appear under
        # transverse isotropy, but it appears under the PICARD tangent there too
        # (7e-2), where a frozen-coefficient form is symmetric by construction --
        # i.e. a defect signature, see #457. The likely real mechanism is operator
        # CONDITIONING: at eta contrast 1e26 the SOR-preconditioned spectrum is
        # spread far enough that a stationary iteration stalls where a Krylov
        # smoother adapts its polynomial. That also predicts the cost measured on
        # well-conditioned problems (see the note on geometric_mg_bundle).
        "mg_levels_pc_type": "sor",
        # Run EXACTLY max_it smoother iterations: no residual-norm computation and
        # no convergence test, so every V-cycle costs the same. A Krylov smoother
        # makes the cycle non-stationary, which is why the velocity block is
        # fgmres (flexible) rather than gmres.
        "mg_levels_ksp_norm_type": "none",
        "mg_levels_ksp_converged_maxits": None,
    }
    if smoother == "robust":
        settings["mg_levels_ksp_type"] = "gmres"
        settings["mg_levels_ksp_max_it"] = 4
    elif smoother == "fast":
        # Cheaper per cycle and measurably faster where the operator is benign,
        # at the cost of the regime "robust" exists for. See the docstring.
        settings["mg_levels_ksp_type"] = "richardson"
        settings["mg_levels_ksp_max_it"] = 3
    else:
        raise ValueError(
            f"smoother must be one of {GEOMETRIC_MG_SMOOTHERS} (got {smoother!r})")
    # SET the iteration count, never inherit it. PCMG's own default is 2 and the
    # GAMG bundle leaves 3 under the same prefix, so a bundle that omits this key
    # smooths differently depending on what ran before it (#468).
    if coarse == "redundant":
        # redundant+lu, not bare lu: a bare serial LU cannot factor a distributed
        # coarse matrix and fails at np>1 (DIVERGED_LINEAR_SOLVE after 0
        # iterations). redundant gathers the (small) coarse system to one rank and
        # is identical to lu in serial — np-safe by default without surprising
        # small-np users.
        settings["mg_coarse_pc_type"] = "redundant"
        settings["mg_coarse_redundant_pc_type"] = "lu"
    elif coarse == "svd":
        # The Galerkin-coarsened ROTATED velocity block inherits every
        # rigid-rotation null-space mode of the constrained problem (a closed
        # circle: one; a spherical shell: three), and redundant/LU hits a zero
        # pivot there (SUBPC_ERROR, outer reason -11 — the #306 fix). SVD is
        # null-space robust and the coarse level is small. Same choice as the
        # native spherical FMG setups.
        settings["mg_coarse_pc_type"] = "svd"
    else:
        raise ValueError(
            f"coarse must be one of {GEOMETRIC_MG_COARSE_SOLVERS} (got {coarse!r})")
    return settings


def _gamg_settings():
    """The algebraic-multigrid (GAMG) settings — the fallback whenever no
    geometric hierarchy is available."""
    return {
        "pc_type": "gamg",
        "pc_gamg_type": "agg",
        "pc_gamg_repartition": True,
        "pc_mg_type": "additive",
        "pc_gamg_agg_nsmooths": 2,
        "mg_levels_ksp_max_it": 3,
        "mg_levels_ksp_converged_maxits": None,
    }


def _all_keys():
    """Every option key any bundle here owns — the basis for the stale lists."""
    keys = set(_gamg_settings())
    for coarse in GEOMETRIC_MG_COARSE_SOLVERS:
        keys |= set(_geometric_mg_settings(coarse))
    return keys


def _bundle(settings):
    return MGSettings(settings=settings,
                    stale=tuple(sorted(_all_keys() - set(settings))))


def geometric_mg_bundle(coarse="redundant", smoother="robust"):
    """Geometric multigrid (FMG F-cycle) on an explicit hierarchy.

    Parameters
    ----------
    smoother : {"robust", "fast"}
        Which of the two MEASURED regimes to configure. ``"robust"`` is
        ``gmres``/4 and is the default; ``"fast"`` is ``richardson``/3. Chosen by
        ``solver.strategy``, not usually here.
    coarse : {"redundant", "svd"}
        The coarse-level solve. ``"redundant"`` (redundant+LU) is the default.
        Use ``"svd"`` when the coarse operator inherits a null space — which the
        Galerkin-coarsened *rotated* velocity block always does, because the
        rigid rotations survive the constraint.

    Returns
    -------
    MGSettings
        Settings and stale keys; see :meth:`MGSettings.apply`.

    Notes
    -----
    The two smoother variants are two measured regimes, and neither dominates.

    ``"robust"`` (``gmres``/4) survives an operator a stationary smoother stalls on.
    Spiegelman notch, :math:`\eta` contrast 1e26, nested 4-level hierarchy:
    per-V-cycle contraction :math:`\rho` = 0.56 against richardson's 0.75 at the same
    four iterations, the margin growing with depth. Transversely isotropic rotated
    annulus: 11 velocity iterations down to 5, and 1.74x on the linear solve.

    ``"fast"`` (``richardson``/3) is cheaper per cycle and measurably quicker where
    the operator is benign. Nested annulus, LINEAR (symmetric) velocity block at
    :math:`\eta` contrast 1e6, outer KSP timed in isolation: robust wins on
    iterations at every depth (x1.45, x1.60, x1.20 at 2, 3, 4 levels) and **loses on
    wall clock at every depth** (x0.86, x0.77, x0.55). The extra sweep costs ~3% and
    ``gmres`` itself ~40% per cycle; custom-P builds its Galerkin coarse operators
    from barycentric transfers, denser than the native nested ones, so the smoother
    is a larger share of the cycle on that route.

    ``"robust"`` is the default because the failure it avoids is worse than the cost
    it carries, and it carries that cost exactly where the problem is easy. Choose
    between them with ``solver.strategy``, which is the named-intent layer over
    these values.
    """
    return _bundle(_geometric_mg_settings(coarse, smoother))


def gamg_bundle():
    """Algebraic multigrid — the fallback when no geometric hierarchy exists.

    Returns
    -------
    MGSettings
        Settings and stale keys; see :meth:`MGSettings.apply`.
    """
    return _bundle(_gamg_settings())
