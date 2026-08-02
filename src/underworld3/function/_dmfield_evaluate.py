"""
.. currentmodule:: underworld3.function._dmfield_evaluate

**Internal primitive — user code should call ``uw.function.evaluate``
for general evaluation.**

``dmfield_evaluate`` uses PETSc's ``DMFieldEvaluate`` directly, bypassing
several contracts that ``uw.function.evaluate`` provides:

- Units and non-dimensionalisation frame
- Point-location fallback ladder (RBF rescue for unlocatable points)
- Sympy expression composition (this takes a single MeshVariable)
- Collective ``global_evaluate`` semantics

**Sanctioned uses** (internal / expert only):

- Computing exact derivatives of solved FE fields (velocity strain rates,
  temperature gradients) without L2 projection
- Hessians for adaptivity metrics (no other UW3 source exists)
- One-sided element-boundary evaluation for diagnostic probes
- Internal machinery that needs to bypass the expression stack

For smooth fields NOT in the FE space, the Clement recovery path
(``uw.function.evaluate``) is superconvergent and more accurate.
"""
import numpy as np

from ._dmfield_wrapper import DMFieldEvaluator


def dmfield_evaluate(var, coords, gradient=True, hessian=False):
    """Internal primitive — FE-exact field and derivative evaluation.

    Evaluates a single ``MeshVariable`` at arbitrary coordinates using
    PETSc's ``DMFieldEvaluate`` (FE basis interpolation, no L2 projection).

    **User code should call ``uw.function.evaluate``**, which provides
    units handling, location fallback, expression composition, and
    collective parallel evaluation. This function bypasses those
    contracts and is intended for internal machinery and expert use.

    Parameters
    ----------
    var : MeshVariable
        The field to evaluate (e.g. ``velocity``, ``temperature``).
    coords : ndarray (n_points, dim)
        Coordinates at which to evaluate.  Accepted forms:
        - ``mesh.X.coords`` (non-dimensional [0-1])
        - plain numpy array (assumed non-dimensional)
        - ``UnitAwareArray`` (auto-converted to non-dimensional)
    gradient : bool, optional
        Return first spatial derivatives D.  Default ``True``.
    hessian : bool, optional
        Return second spatial derivatives H (Hessian).  Default ``False``.

    Returns
    -------
    B : ndarray (n_points, nc) or None
        Field values at each point.
    D : ndarray (n_points, nc, dim) or None
        First derivatives.  ``D[k, j, i]`` = partial(component *j*) /
        partial x_i at point *k*.  ``None`` when ``gradient=False``.
    H : ndarray (n_points, nc, dim, dim) or None
        Second derivatives.  ``H[k, j, i, l]`` = partial^2(component *j*) /
        partial x_i partial x_l at point *k*.  ``None`` when ``hessian=False``.

    Notes
    -----
    **Parallel** — ``DMLocatePoints`` (called internally by
    ``DMFieldEvaluate``) is COLLECTIVE on the mesh DM's communicator.
    **All ranks must call this function**, even with zero local points.
    Each point in the query set must lie in the **calling rank's local
    subdomain** — a point owned by another rank may give ``NaN`` or
    PETSc error 62.  Results are local to each rank and must be gathered
    explicitly if a global result is needed.

    **Unlocated points** — Points that ``DMLocatePoints`` cannot place
    in any cell may return ``NaN`` or raise PETSc error 62 (the outcome
    depends on batching and whether the point enters the location-hash
    candidate set but fails the interior test).  Pre-filter points with
    ``mesh.points_in_domain()`` or catch the error.  Use
    ``uw.function.evaluate`` for RBF-interpolated fallback values at
    domain boundaries.

    **DMField lifecycle** — A fresh DMField is created and destroyed on
    each call.  The overhead (~0.5 us) is negligible vs. evaluation cost
    (~50-200 us).  No cache, no manual cleanup needed.

    **Mesh variable data** — ``mesh.update_lvec()`` is called internally
    to sync the variable's data to the PETSc local vector before
    evaluation.  No manual sync is required.

    Examples
    --------
    >>> B, D, H = dmfield_evaluate(velocity, mesh.X.coords)
    >>>
    >>> # Gradient of x-velocity:  D[:, 0, :]  (component 0, all dims)
    >>> dvx_dx = D[:, 0, 0]
    >>> dvx_dy = D[:, 0, 1]
    >>>
    >>> # Divergence (trace of gradient tensor):
    >>> div_v = D[:, 0, 0] + D[:, 1, 1]
    """
    # --- Normalise coordinates ------------------------------------------
    if hasattr(coords, "magnitude"):
        from underworld3.scaling import non_dimensionalise
        coords_array = np.asarray(non_dimensionalise(coords), dtype=np.float64)
    else:
        coords_array = np.asarray(coords, dtype=np.float64)

    if coords_array.ndim == 1:
        coords_array = coords_array.reshape(-1, var.mesh.dim)

    mesh = var.mesh

    # --- Sync local vector (collective — all ranks must call) ----------
    mesh.update_lvec()

    # --- Create -> evaluate -> destroy (no cache) ------------------------
    evaluator = DMFieldEvaluator()
    try:
        evaluator.create(mesh, var)
        B, D, H = evaluator.evaluate(
            coords_array, mesh, var,
            gradient=gradient, hessian=hessian,
        )
    finally:
        evaluator.destroy()

    return B, D, H
