# Anisotropic Tensor Pattern Analysis

**Date**: 2025-11-12
**Context**: Understanding tensor-valued parameters for future parameter descriptor migration

---

## Overview

Constitutive models in UW3 support both **isotropic** (scalar) and **anisotropic** (tensor-valued) material parameters. This document analyzes the existing patterns and explains why anisotropic cases require special handling for Parameter descriptor migration.

---

## Isotropic vs Anisotropic Parameters

### Isotropic (Scalar) Parameters

**Definition**: Single value applies equally in all directions

**Examples**:
- Scalar viscosity: `η = 1e21 Pa·s`
- Scalar diffusivity: `κ = 1e-6 m²/s`
- Scalar permeability: `k = 1e-10 m²`

**Tensor Construction**: Multiply identity tensor by scalar
```python
# For viscosity (rank-4):
c_ijkl = 2 * I_ijkl * η

# For diffusivity (rank-2):
c_ij = δ_ij * κ
```

**Migration Status**: ✅ **Completed** for ViscousFlowModel, DiffusionModel, DarcyFlowModel

### Anisotropic (Tensor-Valued) Parameters

**Definition**: Material properties vary with direction

**Examples**:
- Anisotropic diffusivity: `κ = [κ_x, κ_y, κ_z]` (diagonal)
- Anisotropic viscosity: Full rank-4 tensor with directional dependence
- Layered materials: Different properties parallel vs perpendicular to layers

**Tensor Construction**: Parameter itself is a tensor
```python
# For anisotropic diffusivity (rank-2):
c = diag(κ_x, κ_y, κ_z)

# For anisotropic viscosity (rank-4):
c_ijkl = provided directly or via Mandel form
```

**Migration Status**: ⚠️ **Not yet migrated** - Requires special handling

---

## Current Implementations

### 1. ViscousFlowModel - Three Forms Supported

**File**: `src/underworld3/constitutive_models.py` lines 494-540

**Form 1: Scalar Viscosity (Isotropic)** ✅ Migrated
```python
viscosity = self.Parameters.shear_viscosity_0  # Scalar UWexpression

# Tensor construction (lines 512-535):
identity = uw.maths.tensor.rank4_identity(d)
result = sympy.MutableDenseNDimArray.zeros(d, d, d, d)

for i, j, k, l in itertools.product(range(d), repeat=4):
    val = 2 * identity[i, j, k, l] * viscosity
    # Wrap bare UWexpression to avoid Iterable check
    if hasattr(val, '__getitem__') and not isinstance(val, (sympy.MatrixBase, sympy.NDimArray)):
        val = sympy.Mul(sympy.S.One, val, evaluate=False)
    result[i, j, k, l] = val

self._c = result  # Shape: (d, d, d, d)
```

**Form 2: Mandel Matrix (Compressed Rank-4)** ⚠️ Not migrated
```python
dv = uw.maths.tensor.idxmap[d][0]  # Number of independent components
viscosity = sympy.Matrix(dv, dv)  # Mandel form

# Tensor construction (lines 505-507):
if viscosity.shape == (dv, dv):
    self._c = 2 * uw.maths.tensor.mandel_to_rank4(viscosity, d)
```

**Mandel Form Details**:
- 2D: `dv=3` (xx, yy, xy components)
- 3D: `dv=6` (xx, yy, zz, yz, xz, xy components)
- Exploits symmetry of stress/strain tensors
- More compact than full rank-4 representation

**Form 3: Full Rank-4 Tensor** ⚠️ Not migrated
```python
viscosity = sympy.Array(d, d, d, d)  # Full anisotropic form

# Tensor construction (lines 508-510):
if viscosity.shape == (d, d, d, d):
    self._c = 2 * viscosity
```

**Complexity**:
- 2D: 81 components (3⁴)
- 3D: 256 components (4⁴)
- With symmetry constraints: fewer independent components

---

### 2. DiffusionModel - Scalar Only

**File**: `src/underworld3/constitutive_models.py` lines 1465-1527

**Status**: ✅ **Migrated** with scalar diffusivity only

**Parameter**:
```python
diffusivity = api_tools.Parameter(
    r"\upkappa",
    lambda params_instance: params_instance._owning_model.create_unique_symbol(
        r"\upkappa", 1, "Diffusivity"
    ),
    "Diffusivity",
    units="m**2/s"
)
```

**Tensor Construction** (lines 1503-1527):
```python
# Element-wise loops for consistency
result = sympy.Matrix.zeros(d, d)

for i in range(d):
    for j in range(d):
        if i == j:
            val = kappa  # Diagonal only
            if hasattr(val, '__getitem__') and not isinstance(val, (sympy.MatrixBase, sympy.NDimArray)):
                val = sympy.Mul(sympy.S.One, val, evaluate=False)
            result[i, j] = val

self._c = result  # Shape: (d, d)
```

---

### 3. AnisotropicDiffusionModel - Diagonal Tensor

**File**: `src/underworld3/constitutive_models.py` lines 1543-1601

**Status**: ⚠️ **Not migrated** - Uses old instance-level pattern

**Current Implementation**:
```python
class _Parameters:
    def __init__(inner_self, _owning_model):
        dim = _owning_model.dim
        inner_self._owning_model = _owning_model

        # Create diagonal diffusivity with validate_parameters
        default_diffusivity = sympy.ones(dim, 1)
        elements = [default_diffusivity[i] for i in range(dim)]
        validated = []
        for i, v in enumerate(elements):
            comp = validate_parameters(
                rf"\upkappa_{{{i}}}", v, f"Diffusivity in x_{i}", allow_number=True
            )
            if comp is not None:
                validated.append(comp)

        # Store as diagonal matrix
        inner_self._diffusivity = sympy.diag(*validated)

    @property
    def diffusivity(inner_self):
        return inner_self._diffusivity

    @diffusivity.setter
    def diffusivity(inner_self, value: sympy.Matrix):
        # Validate each component
        # Store as diagonal matrix
        inner_self._diffusivity = sympy.diag(*validated)
        inner_self._reset()

def _build_c_tensor(self):
    """Constructs the anisotropic (diagonal) tensor from the diffusivity vector."""
    self._c = self.Parameters.diffusivity  # Direct assignment!
    self._is_setup = True
```

**Key Features**:
- Accepts vector of diffusivities: `[κ_x, κ_y, κ_z]`
- Converts to diagonal matrix: `diag(κ_x, κ_y, κ_z)`
- Simple tensor construction: direct assignment

---

## Why Anisotropic Cases Are More Complex

### Challenge 1: Multiple Parameter Components

**Scalar case**:
```python
# Single parameter
shear_viscosity_0 = Parameter(..., units="Pa*s")
```

**Anisotropic case**:
```python
# Multiple parameters - how to structure?
# Option A: Vector parameter
diffusivity = Parameter(..., units="m**2/s")  # But value is a vector?

# Option B: Separate parameters
diffusivity_x = Parameter(..., units="m**2/s")
diffusivity_y = Parameter(..., units="m**2/s")
diffusivity_z = Parameter(..., units="m**2/s")

# Option C: Single parameter accepting tensor
viscosity = Parameter(..., units="Pa*s")  # But value is rank-4 tensor?
```

### Challenge 2: Units for Tensor-Valued Parameters

**Question**: How to specify units for a matrix/tensor parameter?

**Scalar case** (straightforward):
```python
uw.quantity(1e21, "Pa*s")  # Clear semantics
```

**Tensor case** (ambiguous):
```python
# All components have same units?
uw.quantity(sympy.diag(1e-6, 1e-7, 1e-8), "m**2/s")  # How to attach units to Matrix?

# Or separate unit quantities per component?
[uw.quantity(1e-6, "m**2/s"),
 uw.quantity(1e-7, "m**2/s"),
 uw.quantity(1e-8, "m**2/s")]
```

### Challenge 3: JIT Unwrapping for Tensors

**Scalar case**: UWexpression wrapped in Mul, unwrapper finds it
```python
val = 2 * viscosity  # viscosity is UWexpression
# Unwrapper: finds viscosity atom, substitutes non-dimensional value
```

**Tensor case**: Matrix/Array of UWexpressions
```python
viscosity = sympy.Matrix([
    [η_xx, η_xy, η_xz],
    [η_yx, η_yy, η_yz],
    [η_zx, η_zy, η_zz]
])  # Each element might be a UWexpression

# Unwrapper must:
# 1. Traverse matrix structure
# 2. Find all UWexpression atoms in all elements
# 3. Substitute each with non-dimensional value
# 4. Preserve matrix structure
```

### Challenge 4: Consistent Pattern Across Models

**Current status**:
- Scalar parameters: Element-wise loops with Mul wrapping ✅
- Mandel matrix parameters: Direct multiplication ⚠️
- Rank-4 tensor parameters: Direct multiplication ⚠️
- Diagonal matrix parameters: Direct assignment ⚠️

**Goal**: Unified pattern that works for all forms

---

## Design Options for Anisotropic Migration

### Option A: Matrix/Tensor-Valued Parameters

**Concept**: Extend Parameter descriptor to accept matrix/tensor values

**Implementation**:
```python
class _Parameters:
    import underworld3.utilities._api_tools as api_tools

    # Single parameter, value can be scalar or matrix
    diffusivity = api_tools.Parameter(
        r"\upkappa",
        lambda params_instance: params_instance._owning_model.create_unique_symbol(
            r"\upkappa", params_instance._owning_model.dim, "Diffusivity"
        ),
        "Diffusivity",
        units="m**2/s",
        tensor_type="diagonal"  # New: specify tensor structure
    )
```

**Pros**:
- Single parameter for conceptually unified property
- Follows physics intuition (diffusivity is one property)
- Units apply to all components uniformly

**Cons**:
- Parameter descriptor must handle matrix assignment
- `uw.quantity()` doesn't currently support matrix values
- JIT unwrapping more complex
- Units attachment to matrices not well-defined

### Option B: Component Parameters

**Concept**: Separate Parameter descriptor for each component

**Implementation**:
```python
class _Parameters:
    import underworld3.utilities._api_tools as api_tools

    diffusivity_x = api_tools.Parameter(
        r"\upkappa_x",
        lambda params_instance: params_instance._owning_model.create_unique_symbol(
            r"\upkappa_x", 1, "Diffusivity in x"
        ),
        "Diffusivity in x",
        units="m**2/s"
    )

    diffusivity_y = api_tools.Parameter(
        r"\upkappa_y",
        lambda params_instance: params_instance._owning_model.create_unique_symbol(
            r"\upkappa_y", 1, "Diffusivity in y"
        ),
        "Diffusivity in y",
        units="m**2/s"
    )

    @property
    def diffusivity(inner_self):
        """Construct diagonal matrix from components"""
        return sympy.diag(
            inner_self.diffusivity_x,
            inner_self.diffusivity_y
        )
```

**Pros**:
- Each parameter is scalar - existing Pattern works
- Units assignment clear: `uw.quantity(1e-6, "m**2/s")`
- JIT unwrapping works with current pattern
- Consistent with migrated models

**Cons**:
- Multiple parameters for single physical property
- API verbosity: `model.Parameters.diffusivity_x = ...`
- Doesn't scale to full rank-4 tensors (81/256 components!)

### Option C: Hybrid Approach

**Concept**: Use component parameters with convenience property

**Implementation**:
```python
class _Parameters:
    import underworld3.utilities._api_tools as api_tools

    # Individual components as Parameters
    diffusivity_x = api_tools.Parameter(...)
    diffusivity_y = api_tools.Parameter(...)
    diffusivity_z = api_tools.Parameter(...)

    @property
    def diffusivity(inner_self):
        """Convenience property returning diagonal matrix"""
        return sympy.diag(
            inner_self.diffusivity_x,
            inner_self.diffusivity_y,
            inner_self.diffusivity_z
        )

    @diffusivity.setter
    def diffusivity(inner_self, value):
        """Accept vector/matrix and distribute to components"""
        if isinstance(value, (list, tuple)):
            inner_self.diffusivity_x = value[0]
            inner_self.diffusivity_y = value[1]
            if len(value) > 2:
                inner_self.diffusivity_z = value[2]
        elif isinstance(value, sympy.Matrix):
            # Extract diagonal
            diag = value.diagonal()
            inner_self.diffusivity_x = diag[0]
            inner_self.diffusivity_y = diag[1]
            if len(diag) > 2:
                inner_self.diffusivity_z = diag[2]
```

**Pros**:
- Flexibility: set components individually or as vector
- Maintains scalar Parameter pattern
- Convenience setter for user-friendly API
- Extends to moderately complex tensors

**Cons**:
- Still doesn't scale to full rank-4 (need 81 descriptors!)
- More complex setter logic
- Need to handle partial updates correctly

---

## Recommendations

### Short-Term: Complete Isotropic Models First

1. ✅ **ViscousFlowModel** - Scalar viscosity migrated
2. ✅ **DiffusionModel** - Scalar diffusivity migrated
3. ✅ **DarcyFlowModel** - Scalar permeability migrated
4. ⚠️ **GenericFluxModel** - Needs investigation (may not need migration)
5. ⚠️ **MultiMaterialConstitutiveModel** - Complex, defer until after anisotropic

**Status**: 3 of 5 models complete (60%)

### Medium-Term: Design Anisotropic Pattern

**Action Items**:
1. Prototype Option C (Hybrid Approach) with AnisotropicDiffusionModel
2. Test units assignment: `model.Parameters.diffusivity = [uw.quantity(...), uw.quantity(...)]`
3. Validate JIT unwrapping with component Parameters
4. Document pattern if successful

### Long-Term: Full Rank-4 Anisotropic Viscosity

**Challenge**: 81/256 components not feasible with individual Parameters

**Potential Solutions**:
1. **Matrix-valued Parameter descriptor**: Extend Parameter to accept/preserve Matrix values
2. **Structured tensors**: Define symmetry classes (transverse isotropic, orthotropic, etc.)
3. **Custom tensor Parameter**: Special Parameter subclass for rank-4 tensors
4. **Mandel form Parameters**: Component Parameters in Mandel (compressed) space

**Recommendation**: Defer until clear use case emerges

---

## Next Steps

1. **Document this analysis** ✅ (this document)
2. **Complete remaining isotropic models** (GenericFluxModel, MultiMaterialConstitutiveModel)
3. **Prototype AnisotropicDiffusionModel migration** using Option C
4. **Test and validate** anisotropic pattern
5. **Update Parameter descriptor** if matrix-valued Parameters needed

---

## References

- **ViscousFlowModel**: `src/underworld3/constitutive_models.py` lines 420-540
- **DiffusionModel**: `src/underworld3/constitutive_models.py` lines 1465-1527
- **AnisotropicDiffusionModel**: `src/underworld3/constitutive_models.py` lines 1543-1601
- **Mandel notation**: `src/underworld3/maths/tensor.py`
- **Parameter descriptor**: `src/underworld3/utilities/_api_tools.py` lines 252-330
