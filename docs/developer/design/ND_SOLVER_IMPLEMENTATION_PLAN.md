# Non-Dimensional Solver Implementation Plan

## Executive Summary

Plan to integrate non-dimensional scaling into UW3 solvers using the existing dimensionality tracking system. The approach minimizes risk by leveraging the existing `unwrap()` integration point and building incrementally with comprehensive validation at each step.

---

## Phase 1: Foundation - Length Scaling and Mesh Synchronization

### Problem: Length Scale Must Stay Synchronized with Mesh

**Critical Issue**: Coordinate scaling (length) affects:
- Mesh coordinates (x, y, z)
- Coordinate derivatives (dx, dy, dz)
- Spatial operators (grad, div, curl)
- Integration measures (dV, dS)

**Risk**: If length scale changes independently of mesh, all spatial derivatives become incorrect.

### Solution: Lock Length Scale to Mesh Coordinate System

**Implementation Strategy**:

1. **Mesh owns the length scale** (not variables)
   - Add `mesh.length_scale` property
   - Set once during mesh creation
   - Immutable after mesh is created

2. **Variables inherit coordinate dimensionality**
   - Temperature: `[temperature]`
   - Velocity: `[length]/[time]`
   - Pressure: `[mass]/[length]/[time]²`
   - Spatial derivatives automatically scaled by `1/length_scale`

3. **Coordinate unit integration** (ALREADY IMPLEMENTED ✅)
   - Mesh coordinates already carry unit information (2025-10-15)
   - `mesh.X[0]` has units (e.g., "kilometer")
   - `uw.get_units(mesh.X[0])` extracts coordinate units
   - See: `docs/developer/COORDINATE-UNITS-TECHNICAL-NOTE.md`

### Implementation Details

**File**: `src/underworld3/discretisation/discretisation_mesh.py`

```python
class Mesh:
    def __init__(self, ..., length_units=None):
        self._length_scale = 1.0  # Default: no scaling
        self._length_units = length_units  # e.g., "kilometer"

        # Lock length scale based on reference quantities
        model = uw.get_default_model()
        if model._reference_quantities:
            if 'length' in model._reference_quantities:
                self._length_scale = float(model._reference_quantities['length'].magnitude)
            elif 'domain_depth' in model._reference_quantities:
                self._length_scale = float(model._reference_quantities['domain_depth'].magnitude)

    @property
    def length_scale(self):
        """Length scale for non-dimensionalization (immutable after creation)."""
        return self._length_scale

    @property
    def length_units(self):
        """Unit string for mesh coordinates."""
        return self._length_units

    @property
    def nd_coords(self):
        """Non-dimensional coordinate access."""
        return self.coords / self._length_scale
```

**Validation**:
```python
# Test that mesh length scale is immutable
mesh = uw.meshing.UnstructuredSimplexBox(...)
L = mesh.length_scale
with pytest.raises(AttributeError):
    mesh.length_scale = 100.0  # Should fail - immutable
```

---

## Phase 2: Solver Integration - Make It Work

### 2.1 Add ND Mode Flag to Solvers

**File**: `src/underworld3/cython/petsc_generic_snes_solvers.pyx`

```python
class SolverBaseClass:
    def __init__(self, mesh):
        self.mesh = mesh
        self._use_nondimensional = False  # Default: dimensional solving
        self._nd_scaling_applied = False   # Track if scaling was applied

    @property
    def use_nondimensional(self):
        """Whether to solve in non-dimensional form."""
        return self._use_nondimensional

    @use_nondimensional.setter
    def use_nondimensional(self, value):
        if self.is_setup:
            raise RuntimeError(
                "Cannot change use_nondimensional after solver setup. "
                "Set this property before calling solve()."
            )
        self._use_nondimensional = bool(value)
```

**User-Facing API** (in `solvers.py`):
```python
class SNES_Poisson(SNES_Scalar):
    def __init__(self, ...):
        super().__init__(...)

    def solve_nondimensional(self, **kwargs):
        """Solve in non-dimensional form using reference scales."""
        self.use_nondimensional = True
        return self.solve(**kwargs)
```

### 2.2 Modify Unwrap to Apply Scaling When ND Mode Active

**File**: `src/underworld3/function/expressions.py`

```python
def unwrap(fn, keep_constants=True, return_self=True, apply_scaling=None):
    """
    Convert UWexpression to pure SymPy.

    Args:
        fn: Expression to unwrap
        keep_constants: Preserve constant values
        return_self: Return original if already SymPy
        apply_scaling: Override auto-detect for scaling application
            - None: Auto-detect from solver context (default)
            - True: Force scaling application
            - False: No scaling
    """
    # ... existing unwrap logic ...

    # Check if we should apply scaling
    if apply_scaling is None:
        # Auto-detect: check if we're in a solver context with ND mode
        apply_scaling = _should_apply_nd_scaling()

    if apply_scaling:
        result = _apply_scaling_to_unwrapped(result)

    return result


def _should_apply_nd_scaling():
    """Check if we're in a context that requires ND scaling."""
    # Strategy: Thread-local storage or global flag set by solver
    import underworld3 as uw

    # Check if we have an active solver in ND mode
    if hasattr(uw, '_active_solver'):
        solver = uw._active_solver
        if solver is not None and hasattr(solver, 'use_nondimensional'):
            return solver.use_nondimensional

    return False
```

### 2.3 Enhanced Scaling Application

**File**: `src/underworld3/function/expressions.py`

```python
def _apply_scaling_to_unwrapped(expr):
    """
    Apply non-dimensional scaling to unwrapped SymPy expression.

    Scaling Rules:
    1. Variables scaled by their reference scales: u → u/u_ref
    2. Spatial derivatives scaled by length: ∂u/∂x → (∂u/∂x) * L_ref
    3. Time derivatives scaled by time: ∂u/∂t → (∂u/∂t) * t_ref
    """
    model = uw.get_default_model()
    mesh = getattr(uw, '_active_solver', None).mesh if hasattr(uw, '_active_solver') else None

    # Build substitution map
    substitutions = {}

    # 1. Scale variable function symbols
    for var_name, variable in model._variables.items():
        if hasattr(variable, 'scaling_coefficient') and variable.scaling_coefficient != 1.0:
            # Get variable's SymPy function symbol
            if hasattr(variable, 'sym'):
                var_syms = variable.sym.atoms(sympy.Function)
                for var_sym in var_syms:
                    # Find matching symbols in expression
                    expr_syms = expr.atoms(sympy.Function)
                    for expr_sym in expr_syms:
                        if _symbols_match(expr_sym, var_sym):
                            # Scale: u(x,y) → u(x,y)/u_ref
                            scale = 1.0 / variable.scaling_coefficient
                            substitutions[expr_sym] = expr_sym * scale

    # 2. Scale spatial derivatives (chain rule)
    if mesh is not None and mesh.length_scale != 1.0:
        # Derivatives have attribute 'diffindex' (from coordinate system)
        for atom in expr.atoms():
            if hasattr(atom, 'diffindex'):
                # This is a derivative ∂f/∂x_i
                # Scale by 1/L: (∂f/∂x) → (∂f/∂x)/L_ref
                # But this is already handled by coordinate scaling
                pass  # Handled automatically via coordinate units

    # 3. Apply substitutions
    if substitutions:
        scaled_expr = expr.xreplace(substitutions)
        return scaled_expr

    return expr


def _symbols_match(sym1, sym2):
    """Check if two SymPy symbols represent the same variable."""
    # Compare function names and argument count
    if not (isinstance(sym1, sympy.Function) and isinstance(sym2, sympy.Function)):
        return False

    return (str(sym1.func) == str(sym2.func) and
            len(sym1.args) == len(sym2.args))
```

### 2.4 Solver Setup Integration

**File**: `src/underworld3/cython/petsc_generic_snes_solvers.pyx`

```python
def _setup_pointwise_functions(self, verbose=False, debug=False):
    """Setup symbolic equations and compile to C code."""

    # Set active solver for unwrap() context detection
    import underworld3 as uw
    uw._active_solver = self

    try:
        # Existing unwrap calls - now with automatic ND scaling if enabled
        f0 = sympy.Array(uw.function.expressions.unwrap(
            self.F0.sym, keep_constants=False, return_self=False
        )).reshape(1).as_immutable()

        F1 = sympy.Array(uw.function.expressions.unwrap(
            self.F1.sym, keep_constants=False, return_self=False
        )).reshape(dim).as_immutable()

        # ... rest of jacobian computation ...

    finally:
        # Clear active solver
        uw._active_solver = None
```

---

## Phase 3: Validation - Test Simple Cases First

### 3.1 Test 1: Poisson Equation (Simplest Case)

**Equation**: ∇²u = f

**Dimensional form**:
```python
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0, 0), maxCoords=(1.0, 1.0), cellSize=0.1
)

u = uw.discretisation.MeshVariable('u', mesh, 1, units='kelvin')
u.set_reference_scale(100.0)  # T_ref = 100 K

poisson = uw.systems.Poisson(mesh, u_Field=u)
poisson.constitutive_model = uw.constitutive_models.DiffusionModel
poisson.constitutive_model.Parameters.diffusivity = 1.0  # dimensionless
poisson.f = 10.0  # K (dimensional source)

# Essential BC
poisson.add_dirichlet_bc(u.sym[0], "Left", 0.0)
poisson.add_dirichlet_bc(u.sym[0], "Right", 100.0)

# Solve dimensional (current method)
poisson.solve()
u_dim = u.array.copy()
```

**Non-dimensional form**:
```python
# Same setup, but solve in ND mode
poisson2 = uw.systems.Poisson(mesh, u_Field=u)
poisson2.constitutive_model = uw.constitutive_models.DiffusionModel
poisson2.constitutive_model.Parameters.diffusivity = 1.0
poisson2.f = 10.0 / 100.0  # Scale source: f* = f/T_ref

poisson2.add_dirichlet_bc(u.sym[0] / 100.0, "Left", 0.0)  # BC: u*/T_ref
poisson2.add_dirichlet_bc(u.sym[0] / 100.0, "Right", 1.0)

poisson2.use_nondimensional = True
poisson2.solve()

# Convert back to dimensional
u_from_nd = u.from_nd(u.array)

# Validate: solutions should match
assert np.allclose(u_dim, u_from_nd, rtol=1e-10)
```

**Expected Result**: Identical solutions (within numerical precision)

### 3.2 Test 2: Stokes Flow (More Complex)

**Equations**:
- ∇·σ - ∇p = f
- ∇·u = 0

**Scaling**:
- Velocity: u_ref = 1 cm/year
- Pressure: p_ref = η * u_ref / L (derived)
- Length: L = 100 km
- Viscosity: η = 1e21 Pa·s

**Test Setup**:
```python
# Set reference quantities
model = uw.Model()
model.set_reference_quantities(
    domain_depth=uw.quantity(100, "km"),
    plate_velocity=uw.quantity(1, "cm/year"),
    viscosity=uw.quantity(1e21, "Pa*s")
)

# Create variables with units
v = uw.discretisation.MeshVariable('v', mesh, mesh.dim, units='m/s')
p = uw.discretisation.MeshVariable('p', mesh, 1, units='Pa')

# Auto-derive scales
# v.scaling_coefficient → 1 cm/year (from model)
# p.scaling_coefficient → derived from η*u/L

# Solve dimensional
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1e21  # Pa·s
stokes.bodyforce = sympy.Matrix([0, -3300*9.81])  # kg/m³ * m/s²
stokes.solve()
v_dim = v.array.copy()

# Solve non-dimensional
stokes2 = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes2.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes2.constitutive_model.Parameters.viscosity = 1.0  # Non-dimensional
stokes2.bodyforce = stokes2.bodyforce / (p.scaling_coefficient / v.length_scale)
stokes2.use_nondimensional = True
stokes2.solve()

v_from_nd = v.from_nd(v.array)
assert np.allclose(v_dim, v_from_nd, rtol=1e-8)
```

### 3.3 Test 3: Advection-Diffusion (Time-Dependent)

**Additional Complexity**: Time derivatives require time scaling

**Equation**: ∂T/∂t + v·∇T = κ∇²T

**Scaling**:
- Temperature: T_ref = 1000 K
- Time: t_ref = L/u_ref
- Diffusivity: κ_ref = L²/t_ref

**Validation**: Compare time evolution in dimensional vs ND

---

## Phase 4: User Experience - Make It Bulletproof

### 4.1 Automatic Reference Quantity Validation

```python
class Model:
    def validate_reference_quantities(self):
        """
        Ensure reference quantities form a complete dimensional system.

        Raises:
            ValueError: If dimensional system is incomplete or inconsistent
        """
        if not self._reference_quantities:
            return  # No scaling, nothing to validate

        # Check for minimum required quantities
        required_dims = set()
        for var in self._variables.values():
            if hasattr(var, 'dimensionality') and var.dimensionality:
                required_dims.add(var.dimensionality)

        # Build dimensional matrix
        from .utilities.nondimensional import _build_dimensional_matrix
        matrix, dims = _build_dimensional_matrix(self._reference_quantities)

        rank = matrix.rank()
        if rank < len(dims):
            raise ValueError(
                f"Incomplete dimensional system: rank={rank}, need={len(dims)}.\n"
                f"Reference quantities: {list(self._reference_quantities.keys())}\n"
                f"Missing dimensions: {set(dims) - set(self._reference_quantities.keys())}"
            )
```

### 4.2 Clear Error Messages

```python
class SNES_Scalar:
    def solve(self, **kwargs):
        # Before solving in ND mode, validate setup
        if self.use_nondimensional:
            self._validate_nd_setup()

        # ... existing solve logic ...

    def _validate_nd_setup(self):
        """Validate that ND mode is properly configured."""
        model = uw.get_default_model()

        # Check 1: Reference quantities set
        if not model._reference_quantities:
            raise ValueError(
                "Non-dimensional solving requires reference quantities.\n"
                "Call model.set_reference_quantities(...) before solving.\n"
                "Example:\n"
                "  model.set_reference_quantities(\n"
                "      length=100*u.km,\n"
                "      time=1*u.megayear,\n"
                "      temperature=1000*u.kelvin\n"
                "  )"
            )

        # Check 2: All variables have units
        for field in [self.u]:
            if not hasattr(field, 'has_units') or not field.has_units:
                raise ValueError(
                    f"Variable '{field.name}' has no units.\n"
                    "Non-dimensional solving requires all variables to have units.\n"
                    f"Add units='...' when creating {field.name}."
                )

        # Check 3: All variables have scaling coefficients
        for field in [self.u]:
            if field.scaling_coefficient == 1.0:
                warnings.warn(
                    f"Variable '{field.name}' has scaling_coefficient=1.0.\n"
                    "This may indicate reference quantities don't match variable units.\n"
                    f"Variable units: {field.units}\n"
                    f"Reference quantities: {list(model._reference_quantities.keys())}"
                )

        # Check 4: Mesh has length scale
        if self.mesh.length_scale == 1.0 and model._reference_quantities:
            warnings.warn(
                "Mesh length_scale is 1.0 despite reference quantities being set.\n"
                "Did you create the mesh before calling set_reference_quantities()?\n"
                "For automatic length scaling, set reference quantities first."
            )
```

### 4.3 Helpful Debug Output

```python
class SNES_Scalar:
    def _setup_pointwise_functions(self, verbose=False, debug=False):
        if verbose and self.use_nondimensional:
            print("\n" + "="*60)
            print("NON-DIMENSIONAL SOLVE CONFIGURATION")
            print("="*60)

            model = uw.get_default_model()
            print("\nReference Quantities:")
            for name, qty in model._reference_quantities.items():
                print(f"  {name}: {qty}")

            print("\nVariable Scaling:")
            for field in [self.u]:
                print(f"  {field.name}:")
                print(f"    Units: {field.units}")
                print(f"    Dimensionality: {field.dimensionality}")
                print(f"    Scale coefficient: {field.scaling_coefficient}")

            print("\nMesh Scaling:")
            print(f"  Length scale: {self.mesh.length_scale}")
            print(f"  Length units: {self.mesh.length_units}")

            print("="*60 + "\n")

        # ... continue with setup ...
```

### 4.4 Context Manager for ND Solving

```python
# In src/underworld3/__init__.py

@contextmanager
def nondimensional_solve():
    """
    Context manager for non-dimensional solving.

    Usage:
        with uw.nondimensional_solve():
            poisson.solve()
            stokes.solve()

    All solvers within the context will automatically use ND mode.
    """
    global _default_nd_mode
    old_mode = _default_nd_mode
    _default_nd_mode = True

    try:
        yield
    finally:
        _default_nd_mode = old_mode
```

---

## Phase 5: Code Cleanup and Deprecation

### 5.1 Deprecate `to_model_units()`

**File**: `src/underworld3/units.py`

```python
@deprecated(
    version="3.1.0",
    reason="Use dimensionality tracking system with .to_nd() and .from_nd() instead",
    alternative="See docs/examples/Dimensionality-Demo.py for new workflow"
)
def to_model_units(value, units):
    """
    DEPRECATED: Convert value to model's reference units.

    This function is deprecated in favor of the new dimensionality tracking
    system which provides:
    - .to_nd() for converting to non-dimensional form
    - .from_nd() for converting back to dimensional form
    - Automatic scaling during solver.solve()
    """
    # ... existing implementation ...
```

### 5.2 Remove Legacy Scaling Module

**Files to Deprecate**:
- `src/underworld3/scaling/_scaling.py` (keep `units` registry)
- `src/underworld3/scaling/_utils.py` (migrate useful utilities)

**Migration Strategy**:
1. Mark as deprecated in v3.1
2. Add warnings pointing to new system
3. Remove in v3.2 (after 1-2 release cycles)

### 5.3 Update Documentation

**Files to Update**:
- `README.md` - Add non-dimensional solving to feature list
- `docs/examples/` - Add comprehensive ND examples
- `docs/developer/` - Document ND architecture
- Docstrings in all solver classes

---

## Phase 6: Advanced Features (Post-MVP)

### 6.1 Automatic Characteristic Scale Detection

```python
class Model:
    def auto_detect_reference_scales(self):
        """
        Automatically detect reference scales from problem setup.

        Strategy:
        1. Scan all variables for typical values
        2. Use boundary conditions to infer scales
        3. Use constitutive model parameters
        """
        detected = {}

        # Length from mesh bounds
        for mesh in self._meshes.values():
            bounds = mesh.get_global_bounds()
            domain_size = max(bounds[:, 1] - bounds[:, 0])
            detected['length'] = uw.quantity(domain_size, mesh.length_units)

        # Temperature from BC values
        for var in self._variables.values():
            if 'temperature' in var.name.lower():
                # Scan Dirichlet BCs for typical values
                # ...

        return detected
```

### 6.2 Preconditioner Scaling Hints

```python
class SNES_Stokes:
    def _setup_solver(self, verbose=False):
        # ... existing setup ...

        if self.use_nondimensional:
            # Provide scaling hints to PETSc for better preconditioning
            # Schur complement scaling
            self.petsc_options["fieldsplit_pressure_scale"] = p.scaling_coefficient / (
                v.scaling_coefficient / self.mesh.length_scale
            )
```

### 6.3 Dimensional Analysis Validation

```python
def validate_equation_dimensionality(equation, variables):
    """
    Check that equation has consistent dimensions across all terms.

    Raises:
        DimensionalityError: If dimensions don't match
    """
    from .utilities.nondimensional import get_dimensionality

    # Split equation into terms
    terms = equation.as_ordered_terms()

    # Get dimensionality of each term
    dims = [get_dimensionality(term, variables) for term in terms]

    # Check all dimensions match
    if len(set(dims)) > 1:
        raise DimensionalityError(
            f"Equation has inconsistent dimensions:\n"
            f"  Terms: {terms}\n"
            f"  Dimensions: {dims}\n"
            "All terms in an equation must have the same dimensionality."
        )
```

---

## Critical Implementation Notes

### Length Scaling: The Central Constraint

**Key Insight**: Length scale is special because:
1. **Mesh coordinates are dimensional** (x, y, z have units)
2. **Spatial operators depend on coordinates** (grad, div, curl)
3. **Integration measures scale with length** (dV ~ L^dim, dS ~ L^(dim-1))

**Solution**:
- Mesh creation locks in length scale
- Coordinate system (mesh.N) carries units
- Derivatives automatically include coordinate scaling
- No separate length scaling needed for spatial operators

**Already Implemented** (2025-10-15):
- `uw.get_units(mesh.X[0])` returns coordinate units
- Patching system adds unit awareness to coordinates
- Chain rule for derivatives handled automatically

**What Remains**:
- Ensure mesh creation respects `model._reference_quantities['length']`
- Validate that all meshes in a model use same length scale
- Prevent mixed-scale meshes in same problem

### Time Scaling: Second-Order Issue

**Strategy**: Handle time separately from space:
- Time scale set via `model._reference_quantities['time']`
- Time derivatives (`∂u/∂t`, `DuDt`) scaled explicitly
- Advection terms: `v·∇T` naturally scaled (v has time dimension)

**Implementation**: After spatial scaling works, add time via:
```python
# In _apply_scaling_to_unwrapped()
for atom in expr.atoms():
    if hasattr(atom, 'time_derivative'):
        # Scale: ∂u/∂t → (∂u/∂t) * t_ref
        time_scale = model.get_reference_quantity('time')
        # Apply scaling...
```

---

## Testing Strategy

### Unit Tests (Fast, Isolated)
1. `test_mesh_length_scale.py` - Mesh creation and locking
2. `test_variable_scaling.py` - Scale coefficient computation
3. `test_unwrap_scaling.py` - Symbolic expression scaling
4. `test_nd_validation.py` - Error messages and validation

### Integration Tests (Slower, Full Pipeline)
1. `test_poisson_nd.py` - Dimensional vs ND comparison
2. `test_stokes_nd.py` - Stokes with derived pressure scale
3. `test_advdiff_nd.py` - Time-dependent ND solving

### Regression Tests (Ensure No Breaking Changes)
1. Run full test suite with `use_nondimensional=False` (default)
2. Verify all existing tests still pass
3. Confirm no performance degradation

---

## Risk Mitigation

### Risk 1: Breaking Existing Solvers
**Mitigation**:
- Default `use_nondimensional=False`
- All changes behind feature flag
- Comprehensive regression testing

### Risk 2: Length Scale Synchronization Issues
**Mitigation**:
- Lock mesh length scale on creation
- Validate all meshes use same scale
- Clear error messages if mismatch detected

### Risk 3: Constitutive Model Scaling Complexity
**Mitigation**:
- Start with simple models (Diffusion, ViscousFlow)
- Document scaling behavior for each model
- Provide helper functions for complex tensor scaling

### Risk 4: User Confusion
**Mitigation**:
- Clear documentation with examples
- Helpful error messages
- `verbose=True` shows scaling configuration
- Context manager for easy enablement

---

## Success Criteria

### Phase 1 (Foundation): Complete when...
- ✅ Mesh length scale implemented and locked
- ✅ Variables auto-derive scales from model
- ✅ Tests pass: length scale synchronization

### Phase 2 (Integration): Complete when...
- ✅ Solver flag `use_nondimensional` works
- ✅ Unwrap applies scaling in ND mode
- ✅ Tests pass: Poisson ND vs dimensional match

### Phase 3 (Validation): Complete when...
- ✅ All 3 test cases pass (Poisson, Stokes, AdvDiff)
- ✅ Solutions match to machine precision
- ✅ No regression in existing tests

### Phase 4 (UX): Complete when...
- ✅ Clear error messages for all failure modes
- ✅ Validation functions prevent invalid setups
- ✅ Documentation complete with examples

### Phase 5 (Cleanup): Complete when...
- ✅ `to_model_units` deprecated with warnings
- ✅ Legacy scaling module marked deprecated
- ✅ All docstrings updated

---

## Timeline Estimate

| Phase | Tasks | Estimated Time | Dependency |
|-------|-------|----------------|------------|
| 1. Foundation | Length scale + mesh sync | 1-2 days | None |
| 2. Integration | Solver flags + unwrap | 2-3 days | Phase 1 |
| 3. Validation | 3 test cases | 1-2 days | Phase 2 |
| 4. UX Polish | Errors + validation | 1 day | Phase 3 |
| 5. Cleanup | Deprecation | 0.5 days | Phase 4 |
| **Total** | | **5.5-8.5 days** | Sequential |

**Note**: Includes debugging time, iteration, and comprehensive testing.

---

## File Modification Checklist

### Must Modify
- [x] `src/underworld3/discretisation/discretisation_mesh.py` - Add length_scale
- [ ] `src/underworld3/cython/petsc_generic_snes_solvers.pyx` - Add use_nondimensional flag
- [ ] `src/underworld3/function/expressions.py` - Enhanced scaling in unwrap()
- [ ] `src/underworld3/systems/solvers.py` - User-facing ND methods

### Should Modify
- [ ] `src/underworld3/model.py` - Validation functions
- [ ] `src/underworld3/units.py` - Deprecation warnings
- [ ] `tests/` - New ND validation tests

### May Modify (Later)
- [ ] `src/underworld3/constitutive_models.py` - Model-specific scaling helpers
- [ ] `src/underworld3/scaling/` - Deprecation of legacy module

---

## Open Questions

1. **Should ND mode be per-solver or model-wide?**
   - Proposal: Per-solver (more flexible)
   - Alternative: Model-wide flag + context manager

2. **How to handle mixed-dimensional problems?**
   - E.g., 2D mesh embedded in 3D space
   - Proposal: Use mesh.dim for length scale dimensionality

3. **Should boundary conditions be auto-scaled?**
   - Dirichlet BCs: User provides dimensional, we scale?
   - Or user must provide pre-scaled in ND mode?
   - Proposal: Auto-scale if units match variable

4. **Time integration in ND mode?**
   - Timesteppers need time scale
   - CFL conditions change with scaling
   - Proposal: Defer to later (focus on steady-state first)

---

## References

- Existing dimensionality implementation: `src/underworld3/utilities/dimensionality_mixin.py`
- Solver architecture: `src/underworld3/cython/petsc_generic_snes_solvers.pyx`
- Unwrap mechanism: `src/underworld3/function/expressions.py:73-173`
- Coordinate units: `docs/developer/COORDINATE-UNITS-TECHNICAL-NOTE.md`
- Mathematical objects: `src/underworld3/utilities/mathematical_mixin.py`
