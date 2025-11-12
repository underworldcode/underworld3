# Parameter Descriptor Rollout Status

**Date**: 2025-11-12
**Goal**: Migrate all constitutive models to use Parameter descriptor pattern for unit-aware, lazy-evaluation parameters

---

## ✅ COMPLETED

### ViscousFlowModel
**File**: `/src/underworld3/constitutive_models.py` lines 420-432

**Status**: ✅ **FULLY IMPLEMENTED**

**Parameters**:
- `shear_viscosity_0` - Class-level Parameter descriptor with units="Pa*s"

**Implementation**:
```python
class _Parameters:
    import underworld3.utilities._api_tools as api_tools

    shear_viscosity_0 = api_tools.Parameter(
        r"\eta",
        lambda params_instance: params_instance._owning_model.create_unique_symbol(
            r"\eta", 1, "Shear viscosity"
        ),
        "Shear viscosity",
        units="Pa*s"
    )
```

**Tensor Construction**: Element-wise loops with Mul wrapping (lines 512-535)
- Handles scalar, Mandel matrix, and full rank-4 tensor viscosity
- Preserves UWexpression for JIT unwrapping
- All 20 regression tests passing ✅

**Features Working**:
- ✅ Units assignment: `model.Parameters.shear_viscosity_0 = uw.quantity(1e21, "Pa*s")`
- ✅ Unit metadata preserved
- ✅ Lazy evaluation maintained
- ✅ Parameter updates trigger tensor rebuild via `_reset()`
- ✅ JIT unwrapping substitutes numeric values correctly

---

## ⚠️ PARTIALLY COMPLETE

### DiffusionModel
**File**: `/src/underworld3/constitutive_models.py` lines 1443-1522

**Status**: ⚠️ **OLD PATTERN - NEEDS MIGRATION**

**Current Implementation**: Instance-level attribute with property setter (lines 1479-1492)
```python
class _Parameters:
    def __init__(inner_self, _owning_model):
        inner_self._diffusivity = expression(R"\upkappa", 1, "Diffusivity")
        inner_self._owning_model = _owning_model

    @property
    def diffusivity(inner_self):
        return inner_self._diffusivity

    @diffusivity.setter
    def diffusivity(inner_self, value):
        diff = validate_parameters(R"{\upkappa}", value, "Diffusivity", allow_number=True)
        if diff is not None:
            inner_self._diffusivity.copy(diff)
            inner_self._reset()
```

**Tensor Construction** (line 1502-1511):
```python
def _build_c_tensor(self):
    d = self.dim
    kappa = self.Parameters.diffusivity
    # Direct construction to avoid SymPy Matrix scalar multiplication issues
    eye_matrix = sympy.Matrix.eye(d)
    self._c = sympy.Matrix(d, d, lambda i, j: eye_matrix[i, j] * kappa)
```

**Issues**:
- ❌ No Parameter descriptor - can't assign with units
- ❌ Uses `.copy()` pattern instead of metadata preservation
- ⚠️ Tensor construction uses Matrix lambda - may have same UWexpression Iterable issues
- ❌ No automatic unit metadata transfer

**Migration Needed**:
1. Convert `diffusivity` to class-level Parameter descriptor
2. Update tensor construction to handle UWexpression (similar to ViscousFlowModel)
3. Add units specification (should be thermal diffusivity units: m²/s or conductivity/heat_capacity)
4. Test with units assignment

---

## 📋 NOT STARTED

### GenericFluxModel
**File**: `/src/underworld3/constitutive_models.py` lines 1587-1659

**Status**: ❓ **UNKNOWN - NEEDS INVESTIGATION**

**Description**: Generic model for custom flux definitions. May not need parameter migration if users provide flux directly.

**Action**: Review usage to determine if parameter descriptors applicable.

---

### DarcyFlowModel
**File**: `/src/underworld3/constitutive_models.py` lines 1661-1968

**Status**: ⚠️ **OLD PATTERN - NEEDS MIGRATION**

**Current Implementation**: Instance-level attributes (lines 1716-1829)

**Parameters Identified**:
- `permeability` (line 1746)
- `buoyancy_forcing_fn` (line 1770)
- `b_vector` (line 1792)

**Migration Needed**: Convert all parameters to descriptor pattern with appropriate units
- Permeability: k [m²]
- Buoyancy forcing: body force density [N/m³] or [kg/(m²·s²)]

---

### MultiMaterialConstitutiveModel
**File**: `/src/underworld3/constitutive_models.py` lines 1970-2095

**Status**: 🔄 **COMPLEX - SPECIAL HANDLING REQUIRED**

**Description**: Container model that selects between multiple material-specific constitutive models

**Current Implementation**: Uses `materialFn` to switch between models per-element

**Migration Considerations**:
- Individual material models should use Parameter pattern
- Multi-material container needs special handling for unit consistency across materials
- May need material-specific unit validation

**Action**: After migrating individual models, design multi-material parameter interface

---

## ⚠️ REMAINING ISSUES

### 1. Non-Dimensionalisation Problem (User Reported)

**Issue**: Unknown problem at non-dimensionalisation stage

**Status**: 🔍 **NEEDS INVESTIGATION**

**Questions**:
- Does it occur during `to_model_units()` conversion?
- Is it during scaling system setup with reference quantities?
- Does it affect JIT compilation or just symbolic manipulation?

**Action Required**: User to provide error details or test case demonstrating the issue

---

### 2. Tensor Construction Pattern Consistency

**Current State**:
- ViscousFlowModel: Element-wise loops (rank-4 tensor) ✅
- DiffusionModel: Matrix lambda (rank-2 matrix) ⚠️

**Question**: Should DiffusionModel adopt same element-wise pattern for consistency?

**Recommendation**: Test DiffusionModel with UWexpression parameters. If Matrix lambda works, keep it (simpler). If not, migrate to loop pattern.

---

### 3. Unit Specifications

**Needed for Migration**:
- DiffusionModel.diffusivity: `"m**2/s"` or `"W/(m*K)"` depending on formulation
- DarcyFlowModel.permeability: `"m**2"`
- DarcyFlowModel.buoyancy_forcing_fn: `"N/m**3"` or equivalent

**Action**: Consult physics definitions to determine correct unit dimensions

---

## 📊 ROLLOUT PRIORITY

### High Priority
1. **DiffusionModel** - Widely used, straightforward migration similar to ViscousFlowModel
2. **Investigation of non-dimensionalisation issue** - May be blocking further rollout

### Medium Priority
3. **DarcyFlowModel** - More complex parameter set, needs careful unit selection
4. **Tensor construction pattern validation** - Ensure DiffusionModel's Matrix lambda handles UWexpression

### Low Priority
5. **GenericFluxModel** - May not need migration (user-defined flux)
6. **MultiMaterialConstitutiveModel** - Depends on individual model migrations completing first

---

## 🎯 NEXT STEPS

1. **Investigate non-dimensionalisation issue** (user to provide details)
2. **Test DiffusionModel tensor construction** with UWexpression parameters
3. **Migrate DiffusionModel** to Parameter descriptor pattern
4. **Create test suite** for each migrated model validating:
   - Units assignment
   - Lazy evaluation
   - JIT unwrapping
   - Tensor construction correctness

---

## 📝 MIGRATION CHECKLIST (Per Model)

When migrating a constitutive model:

- [ ] Convert parameters to class-level Parameter descriptors
- [ ] Add unit specifications to Parameter constructors
- [ ] Update `_build_c_tensor()` to handle UWexpression (loops if needed)
- [ ] Test tensor construction with bare UWexpression edge cases
- [ ] Verify JIT unwrapping produces numeric values
- [ ] Add regression tests for units assignment
- [ ] Update model documentation with units examples

---

**Status**: 1 of 5 models complete (20%). Non-dimensionalisation issue needs investigation before continuing rollout.
