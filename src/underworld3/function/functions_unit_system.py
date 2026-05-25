"""
Clean unit-aware wrapper functions for evaluate and global_evaluate.

This module handles all unit conversion logic at the Python layer, providing
a clean interface that automatically converts between dimensional and
non-dimensional coordinate systems.

Architecture:
- Cython layer (_function.pyx): Works ONLY with non-dimensional [0-1] coordinates
  - Functions: evaluate_nd(), global_evaluate_nd()
  - Contract: Expects plain numpy arrays in [0-1] space, returns [0-1] results

- Python wrapper layer (this module): Handles ALL unit conversions
  - Functions: evaluate(), global_evaluate()
  - Responsibilities:
    1) Convert dimensional coords → non-dimensional [0-1]
    2) Call Cython functions with plain arrays
    3) Dimensionalise results [0-1] → dimensional values
    4) Wrap with unit-aware types (UWQuantity or UnitAwareArray)

This clean separation ensures:
- No spaghetti code between Python and Cython layers
- Clear contract: Cython = computation, Python = user interface
- Easy to maintain and test each layer independently
- Single source of truth for unit handling logic
"""

import numpy as np
import underworld3 as uw


def _evaluate_impl(
    expr,
    coords,
    coord_sys=None,
    other_arguments=None,
    simplify=False,
    verbose=False,
    evalf=False,
    mode="default",
    data_layout=None,
    check_extrapolated=False,
    smoothing=1e-6,
    # Expert overrides (override mode settings)
    rbf=None,
    force_l2=None,
):
    """
    Evaluate expression at coordinates with automatic unit handling.

    Internal implementation of :func:`evaluate`. The public wrapper
    delegates here and then optionally applies the ``monotone``
    bounded-interpolation post-process. This body is unchanged from the
    historical ``evaluate`` so that ``monotone=False`` is bit-identical.

    This function wraps the Cython evaluate_nd implementation to automatically
    handle unit conversions and return unit-aware results.

    Parameters
    ----------
    expr : sympy expression or UWexpression
        Expression to evaluate
    coords : array-like
        Coordinates at which to evaluate. Can be:
        - numpy array of doubles (shape: n_points x n_dims) in non-dimensional form
        - UnitAwareArray with dimensional coordinates (e.g., from mesh.X.coords)
        - Both work transparently - dimensional coords are auto-converted
    coord_sys : mesh.N vector coordinate system, optional
        Coordinate system to use (default: None)
    other_arguments : dict, optional
        Additional arguments for evaluation (default: None)
    simplify : bool, optional
        Whether to simplify expression (default: True)
    verbose : bool, optional
        Verbose output (default: False)
    evalf : bool, optional
        Force numerical evaluation via sympy evalf (default: False)
    mode : str, optional
        Evaluation mode controlling accuracy vs speed tradeoff.
        Options: ``"default"`` (accurate, projection for derivatives),
        ``"fast"`` (Clement gradient, RBF everywhere),
        ``"projection"`` (always L2 projection).
        Default: ``"default"``
    data_layout : callable, optional
        Data layout specification (default: None)
    check_extrapolated : bool, optional
        Check for extrapolated values (default: False)
    smoothing : float, optional
        Smoothing parameter for L2 projection (dimensionless).
        Only used when projection is active. Default: 1e-6
    rbf : bool, optional
        Expert override: Force RBF interpolation everywhere. Overrides mode.
    force_l2 : bool, optional
        Expert override: Force L2 projection path. Overrides mode.

    Returns
    -------
    UWQuantity, UnitAwareArray, or ndarray
        If non-dimensional scaling is active, returns plain ndarray.
        If expression has units, returns UWQuantity (scalar) or UnitAwareArray.
        Otherwise returns plain ndarray.

    Notes
    -----
    **Evaluation Modes:**

    - ``"fast"``: Clement gradient (no solve), direct calculation, RBF everywhere
    - ``"default"``: Projection for derivatives (solve), direct otherwise, DMInterp + RBF
    - ``"projection"``: Always use L2 projection (solve), DMInterp + RBF

    The `rbf` and `force_l2` parameters are expert overrides that take
    precedence over the mode setting when explicitly provided.

    Examples
    --------
    >>> # Works with both dimensional and non-dimensional coords
    >>> result = uw.function.evaluate(T.sym, T.coords)  # dimensional coords
    >>> result = uw.function.evaluate(T.sym, mesh.data[:, :2])  # non-dimensional
    >>> if hasattr(result, 'to'):
    ...     result_K = result.to('K')  # Unit conversion
    """
    from ._function import evaluate_nd as _evaluate_nd
    from .unit_conversion import has_units
    from underworld3.units import get_units
    from .expressions import unwrap_for_evaluate  # NEW: Use evaluate-specific unwrapper
    from .quantities import quantity, UWQuantity
    from ..utilities.unit_aware_array import UnitAwareArray
    from .pure_sympy_evaluator import is_pure_sympy_expression, evaluate_pure_sympy
    from .expressions import UWexpression

    # Map mode to internal flags (rbf, force_l2)
    # Expert overrides take precedence when explicitly provided
    if rbf is None and force_l2 is None:
        # Use mode settings
        if mode == "fast":
            rbf_flag, force_l2_flag = True, False
        elif mode == "projection":
            rbf_flag, force_l2_flag = False, True
        elif mode == "default":
            rbf_flag, force_l2_flag = False, False
        else:
            raise ValueError(
                f"Unknown mode: '{mode}'. Use 'default', 'fast', or 'projection'."
            )
    else:
        # Expert overrides - use provided values or defaults
        rbf_flag = rbf if rbf is not None else False
        force_l2_flag = force_l2 if force_l2 is not None else False

    # Step 0: CHECK for mixed-mesh expressions
    # All MeshVariable symbols must belong to the same mesh.
    from .expressions import extract_meshes
    expr_meshes = extract_meshes(expr)
    if len(expr_meshes) > 1:
        raise ValueError(
            f"Expression contains MeshVariable symbols from {len(expr_meshes)} "
            f"different meshes. All variables in an expression must belong to "
            f"the same mesh. Use var.copy_into() to transfer data first."
        )

    # Step 1: UNWRAP to canonical form (preprocessing/compiler IR)
    # This converts ALL expressions to a standardized form:
    # - UWexpressions substituted with base SI numeric values
    # - Result dimensionality tracked for correct unit wrapping
    # This is the single source of truth for unit handling!
    from .expressions import unwrap_for_evaluate

    scaling_is_active = uw.is_nondimensional_scaling_active()
    expr_unwrapped, result_dimensionality = unwrap_for_evaluate(expr, scaling_active=scaling_is_active)

    # IMPORTANT: Check if the unwrapped value is already in dimensional form.
    # This only applies when scaling is NOT active - in that case, unwrap_for_evaluate()
    # returns expr.value (dimensional), not expr.data (non-dimensional).
    #
    # When scaling IS active, unwrap_for_evaluate() always returns ND values (expr.data),
    # so we MUST re-dimensionalize regardless of whether the original expression has units.
    #
    # The case where this matters:
    #   T_anal = T_left + (T_right - T_left) * sympy.erf(theta)
    # where T_left, T_right are UWQuantity with units 'K'.
    # Without scaling: unwrap returns dimensional values (300 K) - don't re-dim
    # With scaling: unwrap returns ND values (0.3 if T_scale=1000K) - MUST re-dim
    expr_already_dimensional = (
        not scaling_is_active
        and isinstance(expr, UWexpression)
        and expr.has_units
    )

    # Step 2: OPTIMIZE - Check if unwrapped form is pure sympy (no mesh data)
    # If so, use fast lambdified evaluation instead of full RBF machinery
    is_pure_sympy, free_symbols, symbol_type = is_pure_sympy_expression(expr_unwrapped)

    if is_pure_sympy:
        # Pure sympy expression - use optimized lambdify path
        # Note: expr_unwrapped is already in canonical form (base SI units)

        # Convert coordinates
        if isinstance(coords, UnitAwareArray):
            coords_nondim = uw.non_dimensionalise(coords)
            coords_for_eval = np.asarray(coords_nondim, dtype=np.double)
        elif isinstance(coords, UWQuantity):
            coords_nondim = uw.non_dimensionalise(coords)
            if hasattr(coords_nondim, 'value'):
                coords_for_eval = np.asarray(coords_nondim.value, dtype=np.double)
            else:
                coords_for_eval = np.asarray(coords_nondim, dtype=np.double)
        elif isinstance(coords, np.ndarray):
            coords_for_eval = np.asarray(coords, dtype=np.double)
        else:
            coords_nondim = uw.non_dimensionalise(coords)
            coords_for_eval = np.asarray(coords_nondim, dtype=np.double)

        # Evaluate using optimized lambdification
        raw_values = evaluate_pure_sympy(expr_unwrapped, coords_for_eval)

        # Step 3: Re-dimensionalize and wrap with units
        # GATEWAY PRINCIPLE: evaluate() ALWAYS returns dimensional values when units are known
        result_units = get_units(expr)
        if result_units is not None:
            if expr_already_dimensional:
                # Expression was built from UWQuantity arithmetic - values are ALREADY dimensional
                # Just wrap with units, do NOT re-dimensionalise
                if np.isscalar(raw_values):
                    result_with_units = quantity(float(raw_values), result_units)
                else:
                    result_with_units = UnitAwareArray(raw_values, units=result_units)

                if check_extrapolated:
                    extrapolated = np.full((coords_for_eval.shape[0],), False, dtype=bool)
                    return result_with_units, extrapolated
                else:
                    return result_with_units
            elif result_dimensionality is not None:
                # Values are non-dimensional - need to re-dimensionalise
                try:
                    # Re-dimensionalize: ND value → dimensional value
                    # This multiplies by the appropriate reference scale
                    dimensional_result = uw.dimensionalise(raw_values, target_dimensionality=result_dimensionality)

                    if check_extrapolated:
                        extrapolated = np.full((coords_for_eval.shape[0],), False, dtype=bool)
                        return dimensional_result, extrapolated
                    else:
                        return dimensional_result
                except Exception:
                    # If dimensionalise fails, wrap raw values with units
                    if np.isscalar(raw_values):
                        result_with_units = quantity(float(raw_values), result_units)
                    else:
                        result_with_units = UnitAwareArray(raw_values, units=result_units)

                    if check_extrapolated:
                        extrapolated = np.full((coords_for_eval.shape[0],), False, dtype=bool)
                        return result_with_units, extrapolated
                    else:
                        return result_with_units
            else:
                # Has units but no dimensionality - just wrap with units
                if np.isscalar(raw_values):
                    result_with_units = quantity(float(raw_values), result_units)
                else:
                    result_with_units = UnitAwareArray(raw_values, units=result_units)

                if check_extrapolated:
                    extrapolated = np.full((coords_for_eval.shape[0],), False, dtype=bool)
                    return result_with_units, extrapolated
                else:
                    return result_with_units

        # No dimensionality - return plain array
        if check_extrapolated:
            extrapolated = np.full((coords_for_eval.shape[0],), False, dtype=bool)
            return raw_values, extrapolated
        else:
            return raw_values

    # Step 1: Unwrap expression with proper unit handling for evaluate path
    # NEW: unwrap_for_evaluate() handles:
    # - UWexpression composition: Recursively unwraps (rho0 * alpha) etc.
    # - Constants: Non-dimensionalized (e.g., 1500 K → 1.0)
    # - Variables: Preserved as symbols (already ND in PETSc)
    # - Dimensionality: Tracked for re-dimensionalization
    scaling_is_active = uw.is_nondimensional_scaling_active()
    expr_unwrapped, result_dimensionality = unwrap_for_evaluate(expr, scaling_active=scaling_is_active)

    # Step 2: Convert dimensional coordinates to non-dimensional [0-1] form
    # Internal mesh KDTrees are built with non-dimensional [0-1] coordinates from PETSc
    # User-facing var.coords properties may have dimensional units (meters)

    # Handle different coordinate input types
    # IMPORTANT: Check for UnitAwareArray BEFORE np.ndarray since it inherits from ndarray
    if isinstance(coords, UnitAwareArray):
        # Unit-aware array - need to non-dimensionalise
        coords_nondim = uw.non_dimensionalise(coords)
        coords_for_eval = np.asarray(coords_nondim, dtype=np.double)
    elif isinstance(coords, UWQuantity):
        # UWQuantity from arithmetic operations (e.g., coords - dt * velocity)
        coords_nondim = uw.non_dimensionalise(coords)
        if hasattr(coords_nondim, 'value'):
            coords_for_eval = np.asarray(coords_nondim.value, dtype=np.double)
        else:
            coords_for_eval = np.asarray(coords_nondim, dtype=np.double)
    elif isinstance(coords, np.ndarray):
        # Plain numpy array - assume it's already [0-1] non-dimensional
        coords_for_eval = np.asarray(coords, dtype=np.double)
    else:
        # Other type - try to non-dimensionalise
        coords_nondim = uw.non_dimensionalise(coords)
        coords_for_eval = np.asarray(coords_nondim, dtype=np.double)

    # Ensure coordinates are 2D: shape (N, ndim) not (ndim,)
    # This handles single coordinate evaluation: coords[60] -> shape (2,) -> (1, 2)
    if coords_for_eval.ndim == 1:
        coords_for_eval = np.atleast_2d(coords_for_eval)

    # Step 3: Call Cython implementation with plain non-dimensional arrays
    # NOTE: Returns NON-DIMENSIONAL values [0-1] since we queried
    # the KDTree with non-dimensional coordinates
    raw_result_nondim = _evaluate_nd(
        expr_unwrapped,
        coords_for_eval,
        coord_sys=coord_sys,
        other_arguments=other_arguments,
        simplify=simplify,
        verbose=verbose,
        evalf=evalf,
        rbf=rbf_flag,
        data_layout=data_layout,
        check_extrapolated=check_extrapolated,
        force_l2=force_l2_flag,
        smoothing=smoothing,
    )

    # Step 4: Unpack extrapolation flag if needed
    if check_extrapolated:
        raw_values, extrapolated = raw_result_nondim
    else:
        raw_values = raw_result_nondim
        extrapolated = None

    # Step 5: Re-dimensionalize and wrap with units
    # GATEWAY PRINCIPLE: evaluate() ALWAYS returns dimensional values when units are known
    # The user sees dimensional results; non-dimensional is for internal solver use only

    if result_dimensionality is not None:
        # We have unit information - re-dimensionalize and wrap the result
        result_units = get_units(expr)
        if result_units is not None:
            try:
                # Re-dimensionalize: ND value → dimensional value
                dimensional_result = uw.dimensionalise(raw_values, target_dimensionality=result_dimensionality)

                if check_extrapolated:
                    return dimensional_result, extrapolated
                else:
                    return dimensional_result
            except Exception:
                # If dimensionalise fails, wrap raw values with units
                if np.isscalar(raw_values):
                    result_with_units = quantity(float(raw_values), result_units)
                else:
                    result_with_units = UnitAwareArray(raw_values, units=result_units)

                if check_extrapolated:
                    return result_with_units, extrapolated
                else:
                    return result_with_units

    # No dimensionality info - return plain array (no units)
    if check_extrapolated:
        return raw_values, extrapolated
    else:
        return raw_values


def _global_evaluate_impl(
    expr,
    coords=None,
    coord_sys=None,
    other_arguments=None,
    simplify=False,
    verbose=False,
    evalf=False,
    mode="default",
    data_layout=None,
    check_extrapolated=False,
    smoothing=1e-6,
    # Expert overrides (override mode settings)
    rbf=None,
    force_l2=None,
):
    """
    Global evaluate with automatic unit-aware results.

    Internal implementation of :func:`global_evaluate`. The public
    wrapper delegates here and then optionally applies the ``monotone``
    bounded-interpolation post-process. This body is unchanged from the
    historical ``global_evaluate`` so that ``monotone=False`` is
    bit-identical.

    Similar to evaluate() but performs global evaluation across all processes.
    Returns unit-aware objects when expression has units.

    Parameters
    ----------
    expr : sympy expression or UWexpression
        Expression to evaluate
    coords : array-like, optional
        Coordinates at which to evaluate (default: None)
    coord_sys : CoordinateSystem, optional
        Coordinate system to use (default: None)
    other_arguments : dict, optional
        Additional arguments for evaluation (default: None)
    simplify : bool, optional
        Whether to simplify expression (default: True)
    verbose : bool, optional
        Verbose output (default: False)
    evalf : bool, optional
        Force numerical evaluation via sympy evalf (default: False)
    mode : str, optional
        Evaluation mode controlling accuracy vs speed tradeoff.
        Options: ``"default"`` (accurate, projection for derivatives),
        ``"fast"`` (Clement gradient, RBF everywhere),
        ``"projection"`` (always L2 projection).
        Default: ``"default"``
    data_layout : callable, optional
        Data layout specification (default: None)
    check_extrapolated : bool, optional
        Check for extrapolated values (default: False)
    smoothing : float, optional
        Smoothing parameter for L2 projection (dimensionless).
        Only used when projection is active. Default: 1e-6
    rbf : bool, optional
        Expert override: Force RBF interpolation everywhere. Overrides mode.
    force_l2 : bool, optional
        Expert override: Force L2 projection path. Overrides mode.

    Returns
    -------
    UWQuantity, UnitAwareArray, or ndarray
        If non-dimensional scaling is active, returns plain ndarray.
        Otherwise returns result with appropriate unit tracking.

    Notes
    -----
    See :func:`evaluate` for details on evaluation modes.
    """
    from ._function import global_evaluate_nd as _global_evaluate_nd
    from ..units import get_units
    from .quantities import quantity, UWQuantity
    from ..utilities.unit_aware_array import UnitAwareArray
    from .pure_sympy_evaluator import is_pure_sympy_expression, evaluate_pure_sympy

    # Map mode to internal flags (rbf, force_l2)
    # Expert overrides take precedence when explicitly provided
    if rbf is None and force_l2 is None:
        # Use mode settings
        if mode == "fast":
            rbf_flag, force_l2_flag = True, False
        elif mode == "projection":
            rbf_flag, force_l2_flag = False, True
        elif mode == "default":
            rbf_flag, force_l2_flag = False, False
        else:
            raise ValueError(
                f"Unknown mode: '{mode}'. Use 'default', 'fast', or 'projection'."
            )
    else:
        # Expert overrides - use provided values or defaults
        rbf_flag = rbf if rbf is not None else False
        force_l2_flag = force_l2 if force_l2 is not None else False

    # OPTIMIZATION: Check if this is a pure sympy expression (no UW3 variable data)
    # If so, use fast lambdified evaluation instead of full RBF machinery
    # This includes expressions with mesh coordinates (BaseScalar) or pure symbols
    is_pure_sympy, free_symbols, symbol_type = is_pure_sympy_expression(expr)

    if is_pure_sympy and coords is not None and (rbf_flag or evalf):
        # Pure sympy expression with explicit coords - use optimized path
        # This is the same as evaluate() but for global context

        # Convert coordinates
        if isinstance(coords, UnitAwareArray):
            coords_nondim = uw.non_dimensionalise(coords)
            coords_for_eval = np.asarray(coords_nondim, dtype=np.double)
        elif isinstance(coords, UWQuantity):
            coords_nondim = uw.non_dimensionalise(coords)
            if hasattr(coords_nondim, 'value'):
                coords_for_eval = np.asarray(coords_nondim.value, dtype=np.double)
            else:
                coords_for_eval = np.asarray(coords_nondim, dtype=np.double)
        elif isinstance(coords, np.ndarray):
            coords_for_eval = np.asarray(coords, dtype=np.double)
        else:
            coords_nondim = uw.non_dimensionalise(coords)
            coords_for_eval = np.asarray(coords_nondim, dtype=np.double)

        # Evaluate using optimized lambdification
        raw_result = evaluate_pure_sympy(expr, coords_for_eval)

        # Handle units (same logic as evaluate())
        if uw.is_nondimensional_scaling_active():
            return raw_result

        result_units = get_units(expr)
        if result_units is None:
            return raw_result

        if np.isscalar(raw_result):
            return quantity(float(raw_result), result_units)
        else:
            return UnitAwareArray(raw_result, units=result_units)

    # CRITICAL: Extract base numpy array from UnitAwareArray before passing to Cython
    # Cython code uses typed memory views that don't work with numpy subclasses
    if coords is not None:
        if isinstance(coords, UnitAwareArray):
            # Extract base array and non-dimensionalize if needed
            coords_nondim = uw.non_dimensionalise(coords)
            coords_for_cython = np.asarray(coords_nondim, dtype=np.double)
        elif isinstance(coords, UWQuantity):
            # UWQuantity from arithmetic operations (e.g., coords - dt * velocity)
            # Extract the underlying value and non-dimensionalize
            coords_nondim = uw.non_dimensionalise(coords)
            # coords_nondim might be a scalar or array - ensure it's an array
            if hasattr(coords_nondim, 'value'):
                coords_for_cython = np.asarray(coords_nondim.value, dtype=np.double)
            else:
                coords_for_cython = np.asarray(coords_nondim, dtype=np.double)
        elif isinstance(coords, np.ndarray):
            coords_for_cython = np.asarray(coords, dtype=np.double)
        else:
            coords_for_cython = coords
    else:
        coords_for_cython = coords

    # CRITICAL FIX (2025-11-28): Match evaluate() gateway behavior.
    # global_evaluate must return dimensional results, just like evaluate().
    # Previously returned raw ND values when scaling was active - this was WRONG.

    # Step 1: Track dimensionality for re-dimensionalization (same as evaluate())
    # Note: We pass the ORIGINAL expression to _global_evaluate_nd, not an unwrapped version,
    # because _global_evaluate_nd expects the raw expression and handles evaluation internally.
    from .expressions import unwrap_for_evaluate
    scaling_is_active = uw.is_nondimensional_scaling_active()
    _, result_dimensionality = unwrap_for_evaluate(expr, scaling_active=scaling_is_active)

    # Call the original Cython implementation with all parameters
    # Note: Pass original expr, not unwrapped - _global_evaluate_nd handles the expression directly
    raw_result = _global_evaluate_nd(
        expr,  # Use original expression - Cython handles it
        coords=coords_for_cython,
        coord_sys=coord_sys,
        other_arguments=other_arguments,
        simplify=simplify,
        verbose=verbose,
        evalf=evalf,
        rbf=rbf_flag,
        data_layout=data_layout,
        check_extrapolated=check_extrapolated,
        force_l2=force_l2_flag,
        smoothing=smoothing,
    )

    # Step 2: Re-dimensionalize and wrap with units (GATEWAY PRINCIPLE)
    # global_evaluate() ALWAYS returns dimensional values when units are known,
    # exactly like evaluate(). The user sees dimensional results.

    if result_dimensionality is not None:
        # We have unit information - re-dimensionalize and wrap the result
        result_units = get_units(expr)
        if result_units is not None:
            try:
                # Re-dimensionalize: ND value → dimensional value
                dimensional_result = uw.dimensionalise(raw_result, target_dimensionality=result_dimensionality)
                return dimensional_result
            except Exception:
                # Fall through to simple wrapping if dimensionalise fails
                pass

    # Fallback: Try to get units from expression and wrap
    result_units = get_units(expr)

    if result_units is None:
        # No units - return as-is
        return raw_result

    # Expression has units but couldn't dimensionalize - wrap with units
    if np.isscalar(raw_result):
        # Scalar result - wrap as UWQuantity
        return quantity(float(raw_result), result_units)
    else:
        # Array result - wrap as UnitAwareArray
        return UnitAwareArray(raw_result, units=result_units)


# ---------------------------------------------------------------------------
# Monotone (bounded) interpolation
#
# An opt-in post-process that bounds an interpolated result to the local
# data range of its source field. Lifted from the semi-Lagrangian
# advection-diffusion trace-back (``SemiLagrangian.update_pre_solve``),
# where FE Lagrange-P3 overshoot at non-nodal upstream points ignited
# catastrophic ringing in high-Ra free-surface convection (PR #186/#188).
# Exposed here so any resampling path (remesh re-interpolation, projection,
# restart, the SL DDt itself) can request the same bounded result from one
# place. It operates on the already-computed numbers and never touches the
# symbolic evaluation path.
# ---------------------------------------------------------------------------


def _normalize_monotone(monotone):
    """Map the ``monotone`` kwarg to a canonical mode string or ``None``.

    ``False`` / ``None`` → ``None`` (no limiting); ``True`` → ``"clamp"``
    (the cheap, always-safe choice); ``"clamp"`` / ``"pick"`` pass
    through. Anything else raises ``ValueError``.
    """
    if monotone is False or monotone is None:
        return None
    if monotone is True:
        return "clamp"
    if monotone in ("clamp", "pick"):
        return monotone
    raise ValueError(
        f"Unknown monotone option: {monotone!r}. "
        f"Use False, True, 'clamp', or 'pick'."
    )


def _apply_monotone_limit(expr, coords, value, mode, evalf=False):
    """Bound an interpolated result to the local data range of its source
    field (monotone / bounded interpolation).

    For each evaluation coordinate this finds the ``mesh.dim + 1`` nearest
    source-field DOFs and bounds ``value`` to their nodal min/max:

    - ``mode == "clamp"`` clips the result into ``[nbr_min, nbr_max]``
      (cheap, always bounded).
    - ``mode == "pick"`` keeps the in-bounds result and re-evaluates only
      the out-of-bounds subset via RBF (``evalf=True``), which is
      intrinsically bounded (more accurate, more expensive).

    MVP scope: a single source MeshVariable. ``expr`` must resolve
    directly to one mesh variable (or one of its components, e.g.
    ``T.sym``); composite / multi-field expressions raise ``ValueError``
    because the neighbour bound across fields is ill-defined.

    The algorithm is lifted verbatim from the SL trace-back limiter so
    that routing the DDt through here is bit-identical.
    """
    from ..utilities.unit_aware_array import UnitAwareArray
    from .expressions import extract_meshes

    # --- Resolve the single source MeshVariable --------------------------
    meshes = extract_meshes(expr)
    if len(meshes) != 1:
        raise ValueError(
            "monotone interpolation requires an expression backed by a "
            f"single MeshVariable, but the expression references "
            f"{len(meshes)} mesh(es). Composite / multi-field expressions "
            "are not supported (the neighbour bound across fields is "
            "ill-defined)."
        )
    mesh = next(iter(meshes))
    hit = uw.discretisation.meshVariable_lookup_by_symbol(mesh, expr)
    if hit is None:
        raise ValueError(
            "monotone interpolation requires an expression that is a "
            "single MeshVariable (or one of its components), e.g. `T.sym`. "
            "Composite expressions such as `a*T.sym + b` cannot be bounded "
            "against a single source field."
        )
    var, _comp = hit  # use the full var.data (matches the SL limiter)

    # --- Convert coords to the ND space of the source DOF cloud ----------
    # In unit-aware runs `coords` is dimensional (e.g. metres) while
    # `var.coords_nd` is [0,1] non-dimensional, so the kdtree must compare
    # like with like. Same conversion as the SL trace-back.
    if hasattr(coords, "magnitude"):
        coords_raw = uw.non_dimensionalise(coords)
        if isinstance(coords_raw, UnitAwareArray):
            coords_nd = np.array(coords_raw)
        elif hasattr(coords_raw, "magnitude"):
            coords_nd = coords_raw.magnitude
        else:
            coords_nd = np.asarray(coords_raw)
    else:
        coords_nd = np.asarray(coords)

    psi_coords_nd = np.asarray(var.coords_nd)
    if hasattr(psi_coords_nd, "magnitude"):
        psi_coords_nd = np.asarray(psi_coords_nd.magnitude)

    # --- kNN neighbour stats from the source nodal data ------------------
    # TODO(parallel): the KDTree is built from rank-local `var.coords_nd`,
    # so near a partition seam the neighbour stats bound against a
    # truncated neighbourhood. This matches the validated SL behaviour. For
    # full parallel correctness the bound should include halo / global DOF
    # neighbours -- see the nav-only overlap-clone machinery
    # (project_parallel_point_eval_decision) as the hook if hardened.
    # TODO(units): nbr bounds come from `var.data` (always non-dimensional)
    # while `value` is dimensional in a units-active run -- a pre-existing
    # latent mismatch (scaling is inactive in the validated baseline so it
    # never bites). Do not "fix" without re-validating the trajectory.
    nnn = mesh.dim + 1
    kdt = uw.kdtree.KDTree(np.ascontiguousarray(psi_coords_nd))
    _, idxs = kdt.query(
        np.ascontiguousarray(coords_nd), k=nnn, sqr_dists=False)
    psi_data = np.asarray(var.data)
    # Flatten trailing dims to compute per-coord nbr stats, then restore
    # the original shape afterwards.
    psi_data_flat = psi_data.reshape(psi_data.shape[0], -1)
    nbr_vals = psi_data_flat[idxs]
    nbr_min = nbr_vals.min(axis=1)
    nbr_max = nbr_vals.max(axis=1)

    veep_np = np.asarray(value)
    orig_shape = veep_np.shape
    veep_flat = veep_np.reshape(nbr_min.shape)

    if mode == "clamp":
        veep_lim = np.clip(veep_flat, nbr_min, nbr_max)
    else:
        # "pick": re-evaluate via RBF only at the subset of coords whose
        # result is out-of-bounds. Keeps cost dominated by the cheap pass
        # when most points are in-bounds.
        out_of_bounds = ((veep_flat < nbr_min) | (veep_flat > nbr_max))
        oob_mask = out_of_bounds.any(
            axis=tuple(range(1, out_of_bounds.ndim)))
        veep_lim = veep_flat.copy()
        # TODO(parallel): this re-evaluation is gated on a rank-local
        # `oob_mask.any()`, so in parallel one rank may enter the
        # (collective) global_evaluate while another skips it -> deadlock.
        # Lifted verbatim from the validated (serial) SL limiter; pick is
        # serial-only until the rank-local bound is hardened (see the
        # TODO(parallel) above and project_parallel_point_eval_decision).
        if oob_mask.any():
            oob_coords = coords[oob_mask]
            value_rbf_oob = global_evaluate(expr, oob_coords, evalf=True)
            vrbf_flat = np.asarray(value_rbf_oob).reshape(
                (-1,) + veep_flat.shape[1:])
            # Only overwrite entries individually out of bounds
            # (multi-component case).
            sub_oob = out_of_bounds[oob_mask]
            veep_sub = veep_lim[oob_mask]
            veep_sub = np.where(sub_oob, vrbf_flat, veep_sub)
            veep_lim[oob_mask] = veep_sub

    limited = veep_lim.reshape(orig_shape)

    # Re-wrap units after the numpy ops, mirroring the source field.
    var_units = getattr(var, "units", None)
    if var_units is not None and not isinstance(limited, UnitAwareArray):
        limited = UnitAwareArray(limited, units=var_units)
    return limited


@uw.timing.routine_timer_decorator
def evaluate(
    expr,
    coords,
    coord_sys=None,
    other_arguments=None,
    simplify=False,
    verbose=False,
    evalf=False,
    mode="default",
    data_layout=None,
    check_extrapolated=False,
    smoothing=1e-6,
    # Expert overrides (override mode settings)
    rbf=None,
    force_l2=None,
    monotone=False,
):
    """Evaluate ``expr`` at ``coords`` with automatic unit handling.

    Thin wrapper over :func:`_evaluate_impl`. See that function for the
    full parameter documentation and evaluation-mode notes. With the
    default ``monotone=False`` this is bit-identical to the historical
    ``evaluate``.

    Parameters
    ----------
    monotone : bool or str, optional
        Opt-in bounded (monotone) interpolation, applied as a
        post-process to the computed result. ``False`` (default) → no
        limiting. ``True`` / ``"clamp"`` → clip the result into the
        ``[min, max]`` of the ``mesh.dim + 1`` nearest source-field DOFs.
        ``"pick"`` → keep in-bounds values and re-evaluate only the
        out-of-bounds subset via (bounded) RBF interpolation. Only
        single-MeshVariable expressions are supported; composites raise
        ``ValueError``. See :func:`_apply_monotone_limit`.
    """
    result = _evaluate_impl(
        expr,
        coords,
        coord_sys=coord_sys,
        other_arguments=other_arguments,
        simplify=simplify,
        verbose=verbose,
        evalf=evalf,
        mode=mode,
        data_layout=data_layout,
        check_extrapolated=check_extrapolated,
        smoothing=smoothing,
        rbf=rbf,
        force_l2=force_l2,
    )

    monotone_mode = _normalize_monotone(monotone)
    if monotone_mode is None:
        return result

    if check_extrapolated:
        value, extrapolated = result
        limited = _apply_monotone_limit(
            expr, coords, value, monotone_mode, evalf=evalf)
        return limited, extrapolated

    return _apply_monotone_limit(
        expr, coords, result, monotone_mode, evalf=evalf)


@uw.timing.routine_timer_decorator
def global_evaluate(
    expr,
    coords=None,
    coord_sys=None,
    other_arguments=None,
    simplify=False,
    verbose=False,
    evalf=False,
    mode="default",
    data_layout=None,
    check_extrapolated=False,
    smoothing=1e-6,
    # Expert overrides (override mode settings)
    rbf=None,
    force_l2=None,
    monotone=False,
):
    """Parallel-safe evaluate with automatic unit-aware results.

    Thin wrapper over :func:`_global_evaluate_impl`. See that function and
    :func:`evaluate` for the full parameter documentation. With the
    default ``monotone=False`` this is bit-identical to the historical
    ``global_evaluate``.

    Parameters
    ----------
    monotone : bool or str, optional
        Opt-in bounded (monotone) interpolation post-process. See
        :func:`evaluate` for semantics. Not supported together with
        ``check_extrapolated`` (raises ``NotImplementedError``).
    """
    result = _global_evaluate_impl(
        expr,
        coords=coords,
        coord_sys=coord_sys,
        other_arguments=other_arguments,
        simplify=simplify,
        verbose=verbose,
        evalf=evalf,
        mode=mode,
        data_layout=data_layout,
        check_extrapolated=check_extrapolated,
        smoothing=smoothing,
        rbf=rbf,
        force_l2=force_l2,
    )

    monotone_mode = _normalize_monotone(monotone)
    if monotone_mode is None:
        return result

    if check_extrapolated:
        raise NotImplementedError(
            "monotone interpolation is not supported together with "
            "check_extrapolated in global_evaluate."
        )

    return _apply_monotone_limit(
        expr, coords, result, monotone_mode, evalf=evalf)
