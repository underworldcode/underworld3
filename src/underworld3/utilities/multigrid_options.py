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
    multigrid_options.geometric_mg_bundle().apply(self.petsc_options, prefix)

Read the settings without writing them (the rotated path stages its options in a
dict so it can drop them again after ``setUp``)::

    cfg.update({f"fieldsplit_vel_{k}": v
                for k, v in multigrid_options.gamg_bundle().settings.items()})
"""

from typing import NamedTuple

__all__ = ["MGBundle", "geometric_mg_bundle", "gamg_bundle",
           "GEOMETRIC_MG_COARSE_SOLVERS"]


#: Coarse-solve variants of the geometric bundle. ``"redundant"`` (redundant+LU)
#: is the default and is np-safe; ``"svd"`` is required whenever the coarse
#: operator inherits a null space — see :func:`geometric_mg_bundle`.
GEOMETRIC_MG_COARSE_SOLVERS = ("redundant", "svd")


class MGBundle(NamedTuple):
    """A preconditioner option bundle: the keys to set, and the keys to clear.

    ``settings`` maps an option suffix (no prefix, no leading ``-``) to its
    value; ``None`` means a bare PETSc flag. ``stale`` lists suffixes this
    bundle does not own but a sibling bundle does, which must be removed before
    ``setFromOptions`` so a previously-applied bundle cannot leak through.
    """

    settings: dict
    stale: tuple

    def apply(self, opts, prefix=""):
        """Write this bundle into ``opts`` (a ``PETSc.Options``) under ``prefix``.

        ``prefix`` is the managed block's own prefix — ``""`` for a top-level PC,
        ``"fieldsplit_velocity_"`` for the Stokes velocity sub-block — and is
        applied on top of whatever prefix ``opts`` itself carries.
        """
        for key, value in self.settings.items():
            opts.setValue(prefix + key, value)
        for key in self.stale:
            opts.delValue(prefix + key)


def _geometric_mg_settings(coarse):
    """The geometric-MG settings for one coarse-solve variant."""
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
        # variable-viscosity velocity block and diverge. Richardson is stationary
        # and degrades on the NON-SYMMETRIC operator produced by the
        # consistent-Newton tangent. Measured on the Spiegelman notch
        # (Drucker-Prager, eta contrast 1e26) over a nested 4-level hierarchy:
        # contraction per V-cycle rho = 0.75 (richardson) vs 0.56 (gmres) at the
        # SAME four smoother iterations -- and the gmres margin GROWS with depth
        # (5% at 3 levels, 25% at 4), because deeper cycles apply the smoother on
        # more coarse operators. Four iterations, not more: per unit work gmres/4
        # (rho^(1/4) = 0.87) beats gmres/8 (0.91).
        "mg_levels_ksp_type": "gmres",
        "mg_levels_pc_type": "sor",
        # SET the count, never inherit it. PCMG's own default is 2 and the GAMG
        # bundle below leaves 3 under the same prefix, so a bundle that omits
        # this key smooths differently depending on what ran before it (#468).
        "mg_levels_ksp_max_it": 4,
        # Run EXACTLY max_it smoother iterations: no residual-norm computation and
        # no convergence test, so every V-cycle costs the same. A Krylov smoother
        # makes the cycle non-stationary, which is why the velocity block is
        # fgmres (flexible) rather than gmres.
        "mg_levels_ksp_norm_type": "none",
        "mg_levels_ksp_converged_maxits": None,
    }
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
    return MGBundle(settings=settings,
                    stale=tuple(sorted(_all_keys() - set(settings))))


def geometric_mg_bundle(coarse="redundant"):
    """Geometric multigrid (FMG F-cycle) on an explicit hierarchy.

    Parameters
    ----------
    coarse : {"redundant", "svd"}
        The coarse-level solve. ``"redundant"`` (redundant+LU) is the default.
        Use ``"svd"`` when the coarse operator inherits a null space — which the
        Galerkin-coarsened *rotated* velocity block always does, because the
        rigid rotations survive the constraint.

    Returns
    -------
    MGBundle
        Settings and stale keys; see :meth:`MGBundle.apply`.
    """
    return _bundle(_geometric_mg_settings(coarse))


def gamg_bundle():
    """Algebraic multigrid — the fallback when no geometric hierarchy exists.

    Returns
    -------
    MGBundle
        Settings and stale keys; see :meth:`MGBundle.apply`.
    """
    return _bundle(_gamg_settings())
