"""
Evaluate a MeshVariable (with spatial derivatives) at arbitrary points
using PETSc's DMFieldEvaluate — FE-exact values and gradients, no projection.

.. currentmodule:: underworld3.function

Public Function
---------------
dmfield_evaluate  -- Evaluate a MeshVariable at coordinates, returning
                    field values and spatial derivatives (gradient, Hessian).

Examples
--------
>>> import underworld3 as uw
>>> import numpy as np
>>> from underworld3.function.dmfield_evaluate import dmfield_evaluate
>>>
>>> # Create mesh and solve Stokes (not shown)
...
>>> # Get velocity + gradient at mesh nodes
>>> B, D, H = dmfield_evaluate(velocity, mesh.X.coords)
>>> D.shape   # (n_nodes, dim, nc) — e.g. (1024, 2, 2) for 2D velocity
>>>
>>> # Strain rate components from gradient tensor
>>> exx = D[:, 0, 0]  # ∂u_x/∂x
>>> eyy = D[:, 1, 1]  # ∂u_y/∂y
>>> exy = 0.5 * (D[:, 1, 0] + D[:, 0, 1])  # 0.5*(∂u_x/∂y + ∂u_y/∂x)
>>>
>>> # Fill a mesh variable directly (no Projection ... no boundary artifact)
>>> eII = np.sqrt((exx**2 + eyy**2 + 2*exy**2) / 2)
>>> visc_var.data[:, 0] = yield_stress / (2 * (eII + eps_min))
"""

import numpy as np
import weakref

from ._dmfield_wrapper import CachedDMField


# ---------------------------------------------------------------------------
# Global cache: (id(mesh), id(var)) -> CachedDMField
# The weakref container ensures DMFields are GC'd when the mesh/var go away.
# ---------------------------------------------------------------------------
_field_cache = {}  # int -> CachedDMField
_cache_owners = {}  # int -> (weakref.ref, weakref.ref)  keep key alive


def _cache_key(mesh, var):
    return (id(mesh), id(var))


def _cache_cleanup(key):
    """Drop a cache entry — called via weakref finalizer."""
    _field_cache.pop(key, None)
    _cache_owners.pop(key, None)


def dmfield_evaluate(var, coords, gradient=True, hessian=False):
    """Evaluate a MeshVariable (and its spatial derivatives) at *coords*.

    Uses PETSc's ``DMFieldEvaluate`` to compute the FE basis functions at
    each query point, giving FE-exact values and gradients **without** an
    L2 projection or mass-matrix solve.   This avoids the boundary artifacts
    that appear when projecting derivative-dependent quantities (strain rate,
    viscosity, …) via ``uw.systems.Projection``.

    Parameters
    ----------
    var : MeshVariable
        The field to evaluate (e.g. ``velocity``, ``temperature``).
    coords : ndarray (n_points, dim)
        Coordinates at which to evaluate.  Accepted forms:
        - ``mesh.X.coords`` (non-dimensional [0-1])
        - plain numpy array (assumed non-dimensional)
        - UnitAwareArray (auto-converted to non-dimensional)
    gradient : bool, optional
        Return first spatial derivatives (gradient).  Default ``True``.
    hessian : bool, optional
        Return second spatial derivatives (Hessian).  Default ``False``.

    Returns
    -------
    B : ndarray (n_points, nc) or None
        Field values at each point.  ``None`` if the variable has zero
        components (should not happen in practice).
    D : ndarray (n_points, dim, nc) or None
        First spatial derivatives.  ``D[k, i, j]`` = ∂(component *j*) /
        ∂xᵢ at point *k*.  ``None`` when ``gradient=False``.
    H : ndarray (n_points, dim, dim, nc) or None
        Second spatial derivatives.  ``H[k, i, j, l]`` = ∂²(component *l*)
        / ∂xᵢ∂xⱼ at point *k*.  ``None`` when ``hessian=False``.

    Notes
    -----
    **Data freshness** — The variable's internal PETSc local vector
    (``_lvec``) is read at the time of the *first* call.  If the variable's
    data changes later (e.g. after a Stokes solve), call
    ``mesh.update_lvec()`` before ``dmfield_evaluate()`` to sync the local
    vector.

    **Parallel** — This function is not yet collective across MPI ranks.
    Each rank evaluates its own set of coordinates.  For parallel-safe
    evaluation across the whole domain, use ``uw.function.global_evaluate``
    with an expression that has been pre-projected.

    **Cache** — The underlying ``DMField`` C object is cached per
    ``(mesh, var)`` and reused on subsequent calls.  Use
    ``dmfield_evaluate_clear_cache()`` to release the cache.

    Examples
    --------
    >>> B, D, H = dmfield_evaluate(velocity, mesh.X.coords)
    >>>
    >>> # Gradient of x-velocity component:  D[:, :, 0]
    >>> dvx_dx = D[:, 0, 0]
    >>> dvx_dy = D[:, 1, 0]
    >>>
    >>> # Fill a strain-rate variable without projection
    >>> SR = D[:, 0, 0] + D[:, 1, 1]  # divergence / trace of strain rate
    >>> srii.data[:, 0] = SR
    """
    import underworld3 as uw

    # --- Normalise coordinates -------------------------------------------
    # Strip pint / UnitAwareArray wrappers if present; convert to ND [0-1].
    if hasattr(coords, "magnitude"):
        # UnitAwareArray or pint Quantity
        from underworld3.scaling import non_dimensionalise

        coords_nd = non_dimensionalise(coords)
        coords_array = np.asarray(coords_nd, dtype=np.float64)
    elif isinstance(coords, np.ndarray):
        coords_array = np.asarray(coords, dtype=np.float64)
    else:
        coords_array = np.asarray(coords, dtype=np.float64)

    if coords_array.ndim == 1:
        coords_array = coords_array.reshape(-1, var.mesh.dim)

    # --- Ensure the local vector is in sync --------------------------------
    # update_lvec() creates the mesh's full lvec (needed by DMFieldCreateDS)
    # and syncs the global vector data to it. Must be called BEFORE creating
    # or using the cached DMField.
    mesh = var.mesh
    mesh.update_lvec()

    # --- Get or create cached DMField ------------------------------------
    key = _cache_key(mesh, var)
    cdf = _field_cache.get(key)

    if cdf is None or not cdf.is_valid:
        cdf = CachedDMField()
        cdf.create(mesh, var)
        _field_cache[key] = cdf

        # Keep weakrefs so the cache is cleaned up when mesh/var die
        _cache_owners[key] = (weakref.ref(mesh), weakref.ref(var))

    # --- Evaluate --------------------------------------------------------
    B, D, H = cdf.evaluate(coords_array, gradient=gradient, hessian=hessian)

    return B, D, H


def dmfield_evaluate_clear_cache():
    """Release all cached DMField C objects.

    Call this if mesh variables have been destroyed or re-created, or
    to free the underlying PETSc resources explicitly.
    """
    global _field_cache, _cache_owners

    for cdf in _field_cache.values():
        try:
            cdf.destroy()
        except Exception:
            pass

    _field_cache.clear()
    _cache_owners.clear()
