"""
Evaluate a MeshVariable (with spatial derivatives) at arbitrary points
using PETSc's ``DMFieldEvaluate`` — FE-exact values and gradients, no projection.

.. currentmodule:: underworld3.function

When to use ``dmfield_evaluate`` vs. ``uw.function.evaluate``
-------------------------------------------------------------
- **DMField** wins for: fields that ARE in the FE space (machine-precision
  values/gradients), one-sided evaluation at element boundaries, and
  Hessians (not available via other UW3 paths).
- **Clement path** (``uw.function.evaluate``) wins for: smooth fields
  NOT in the FE space — Clement recovery is superconvergent on regular
  meshes and can give better accuracy than direct FE interpolation.

Unlocated points
----------------
Points outside the domain, or that ``DMLocatePoints`` cannot place in
any element, are returned as ``NaN`` in all output arrays.  No automatic
fallback interpolation is performed — use ``uw.function.evaluate`` if
you need RBF-filled values at boundary points.

Public Function
---------------
dmfield_evaluate  -- Evaluate a MeshVariable at coordinates, returning
                    field values and spatial derivatives.

Examples
--------
>>> import underworld3 as uw
>>> import numpy as np
>>> from underworld3.function import dmfield_evaluate
>>>
>>> # Velocity and its gradient at mesh nodes
>>> B, D, H = dmfield_evaluate(velocity, mesh.X.coords)
>>> D.shape   # (n_nodes, nc, dim) — e.g. (1024, 2, 2) for 2D velocity
>>>
>>> # Strain rate components.  D[k, j, i] = partial(component j)/partial x_i:
>>> exx = D[:, 0, 0]  # partial u_x / partial x
>>> eyy = D[:, 1, 1]  # partial u_y / partial y
>>> exy = 0.5 * (D[:, 0, 1] + D[:, 1, 0])  # symmetric shear rate
>>>
>>> # Fill a mesh variable directly (no projection, no boundary artifact)
>>> eII = np.sqrt((exx**2 + eyy**2 + 2*exy**2) / 2)
>>> visc_var.data[:, 0] = yield_stress / (2 * (eII + eps_min))
"""
import numpy as np

from ._dmfield_wrapper import DMFieldEvaluator


def dmfield_evaluate(var, coords, gradient=True, hessian=False):
    """Evaluate a MeshVariable (and its spatial derivatives) at *coords*.

    Uses PETSc's ``DMFieldEvaluate`` to compute the FE basis functions at
    each query point, giving FE-exact values and gradients **without** an
    L2 projection or mass-matrix solve.  This avoids the projection
    artifacts that appear when evaluating derivative-dependent quantities
    (strain rate, viscosity, ...) via ``uw.systems.Projection``.

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
    Results are local to each rank and must be gathered explicitly if
    a global result is needed.

    **Unlocated points** — Points outside the domain, or that the mesh
    cannot locate, are returned as ``NaN`` in all output arrays.  Use
    ``uw.function.evaluate`` if you need RBF-interpolated fallback values
    at domain boundaries.

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
