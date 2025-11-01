# Reduction Operations Analysis - Complete Codebase Search

## Summary
Found reduction operation methods in 2 primary locations:
1. **MeshVariable class** - `discretisation_mesh_variables.py` (lines 1757-2042)
2. **UnitAwareArray class** - `unit_aware_array.py` (lines 490-1221)

Array view classes (SimpleMeshArrayView, TensorMeshArrayView) do NOT have reduction operations defined.

---

## 1. MeshVariable Class Reduction Operations
**File**: `/Users/lmoresi/+Underworld/underworld-pixi-2/underworld3/src/underworld3/discretisation/discretisation_mesh_variables.py`

### 1.1 min() - Line 1757-1779
```python
def min(self) -> Union[float, tuple]:
    """
    The global variable minimum value.
    Returns the value only (not the rank). For multi-component variables,
    returns a tuple of minimum values for each component.
    """
```
**Implementation**:
- Single component: Uses `self._gvec.min()` → returns float
- Multi-component: Uses `self._gvec.strideMin(i)[1]` for each component → returns tuple

**Line 1772-1779 (key logic)**:
```python
if self.num_components == 1:
    rank, value = self._gvec.min()
    return value
else:
    return tuple(
        [self._gvec.strideMin(i)[1] for i in range(self.num_components)]
    )
```

**Handles num_components**: ✅ YES
- num_components == 1 → float scalar
- num_components > 1 → tuple of values

**Shape handling**: 
- Works with (N, 1, 1), (N, 1, d), (N, d, d) - calls on flat PETSc vector

---

### 1.2 max() - Line 1781-1803
```python
def max(self) -> Union[float, tuple]:
    """
    The global variable maximum value.
    Returns the value only (not the rank). For multi-component variables,
    returns a tuple of maximum values for each component.
    """
```
**Implementation**:
- Single component: Uses `self._gvec.max()` → returns float
- Multi-component: Uses `self._gvec.strideMax(i)[1]` for each component → returns tuple

**Line 1796-1803 (key logic)**:
```python
if self.num_components == 1:
    rank, value = self._gvec.max()
    return value
else:
    return tuple(
        [self._gvec.strideMax(i)[1] for i in range(self.num_components)]
    )
```

**Handles num_components**: ✅ YES
- num_components == 1 → float scalar
- num_components > 1 → tuple of values

**Shape handling**: Works with all shapes

---

### 1.3 sum() - Line 1805-1825
```python
def sum(self) -> Union[float, tuple]:
    """
    The global variable sum value.
    """
```
**Implementation**:
- Single component: Uses `self._gvec.sum()` → returns float
- Multi-component: Uses `self._gvec.strideSum(i)` for each component → returns tuple

**Line 1818-1825 (key logic)**:
```python
if self.num_components == 1:
    return self._gvec.sum()
else:
    cpts = []
    for i in range(0, self.num_components):
        cpts.append(self._gvec.strideSum(i))
    return tuple(cpts)
```

**Handles num_components**: ✅ YES
- num_components == 1 → float scalar
- num_components > 1 → tuple of values

**Shape handling**: Works with all shapes

---

### 1.4 norm() - Line 1827-1856
```python
def norm(self, norm_type) -> Union[float, tuple]:
    """
    The global variable norm value.

    norm_type: type of norm, one of
        - 0: NORM 1 ||v|| = sum_i | v_i |
        - 1: NORM 2 ||v|| = sqrt(sum_i |v_i|^2) (vectors only)
        - 3: NORM INFINITY ||v|| = max_i |v_i|
    """
```
**Implementation**:
- Single component: Uses `self._gvec.norm(norm_type)` → returns float
- Multi-component: Uses `self._gvec.strideNorm(i, norm_type)` for each component → returns tuple

**Line 1848-1856 (key logic)**:
```python
if self.num_components == 1:
    return self._gvec.norm(norm_type)
else:
    return tuple(
        [
            self._gvec.strideNorm(i, norm_type)
            for i in range(self.num_components)
        ]
    )
```

**Handles num_components**: ✅ YES
- num_components == 1 → float scalar
- num_components > 1 → tuple of values

**Special case**: Requires `norm_type` parameter (unlike numpy/standard reductions)

---

### 1.5 mean() - Line 1858-1878
```python
def mean(self) -> Union[float, tuple]:
    """
    The global variable mean value.
    """
```
**Implementation**:
- Single component: `self._gvec.sum() / vecsize` → returns float
- Multi-component: `self._gvec.strideSum(i) / vecsize` for each component → returns tuple

**Line 1871-1878 (key logic)**:
```python
if self.num_components == 1:
    vecsize = self._gvec.getSize()
    return self._gvec.sum() / vecsize
else:
    vecsize = self._gvec.getSize() / self.num_components
    return tuple(
        [self._gvec.strideSum(i) / vecsize for i in range(self.num_components)]
    )
```

**Handles num_components**: ✅ YES
- num_components == 1 → float scalar
- num_components > 1 → tuple of values

**Important**: Divides vecsize by num_components for multi-component calculations

---

### 1.6 stats() - Line 1880-1916
```python
@uw.collective_operation
def stats(self):
    """
    Universal statistics method for all variable types.
    Returns various statistical measures appropriate for the variable type.
    ...
    """
```
**Implementation**:
- Dispatches to `_scalar_stats()`, `_vector_stats()`, or `_tensor_stats()`
- Returns dict with keys: 'type', 'components', 'size', 'mean', 'min', 'max', 'sum', 'norm2', 'rms'

**NOT a direct reduction** - Uses `_scalar_stats()`, `_vector_stats()`, `_tensor_stats()`

---

### 1.7 _scalar_stats() - Line 1944-1966
**For num_components == 1**:
```python
def _scalar_stats(self):
    """Statistics for scalar variables (original implementation)."""
    vsize = self._gvec.getSize()
    vmean = self.mean()
    vmax = self.max()  # Now returns value directly, not tuple
    vmin = self.min()  # Now returns value directly, not tuple
    vsum = self.sum()
    vnorm2 = self.norm(NormType.NORM_2)
    vrms = vnorm2 / numpy.sqrt(vsize)
```

**Returns**: dict with scalar values

---

### 1.8 _vector_stats() - Line 1968-2004
**For num_components > 1 (vector variables)**:
```python
def _vector_stats(self):
    """Statistics for vector variables using magnitude."""
    # Creates temporary scalar variable for magnitude
    # Computes |v| = sqrt(v·v)
    # Returns dict with magnitude-based statistics
```

**Computes magnitude**: sqrt(sum of component squares)

**Returns**: dict with magnitude statistics

---

### 1.9 _tensor_stats() - Line 2006-2042
**For tensor variables**:
```python
def _tensor_stats(self):
    """Statistics for tensor variables using Frobenius norm."""
    # Creates temporary scalar variable for Frobenius norm
    # Computes ||A||_F = sqrt(sum(A_ij^2))
```

**Computes Frobenius norm**: sqrt(sum of all components squared)

**Returns**: dict with Frobenius norm statistics

---

## 2. UnitAwareArray Class Reduction Operations
**File**: `/Users/lmoresi/+Underworld/underworld-pixi-2/underworld3/src/underworld3/utilities/unit_aware_array.py`

### 2.1 max() - Line 500-509
```python
def max(self, axis=None, out=None, keepdims=False, initial=None, where=True):
    """Return maximum with units preserved."""
```
**Implementation**:
- Calls `super().max()` (numpy method)
- Wraps scalar results with units via `_wrap_scalar_result()`
- Wraps array results with `_wrap_result()`

**Handles num_components**: ✅ Indirectly (via axis parameter)

**Units handling**:
- Scalar results: `uw.function.quantity(float(value), self._units)`
- Array results: `UnitAwareArray(result, units=self._units)`

---

### 2.2 min() - Line 511-520
```python
def min(self, axis=None, out=None, keepdims=False, initial=None, where=True):
    """Return minimum with units preserved."""
```
**Same pattern as max()**

**Units handling**: Preserves units in results

---

### 2.3 mean() - Line 522-531
```python
def mean(self, axis=None, dtype=None, out=None, keepdims=False, where=True):
    """Return mean with units preserved."""
```
**Same pattern as max()**

**Units handling**: Preserves units in results

---

### 2.4 sum() - Line 533-542
```python
def sum(self, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
    """Return sum with units preserved."""
```
**Same pattern as max()**

**Units handling**: Preserves units in results

---

### 2.5 std() - Line 544-568
```python
def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True):
    """Return standard deviation with units preserved."""
```
**Implementation**:
- Computes `std = sqrt(variance)`
- Uses custom unit-aware variance calculation
- Wraps results with units

**Units handling**:
- Scalar: `uw.function.quantity(std_value, self._units)`
- Array: `UnitAwareArray(..., units=self._units)`

---

### 2.6 var() - Line 570-621
```python
def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True):
    """Return variance with units squared."""
```
**Implementation**:
- Custom unit-aware calculation: `var = mean((x - mean(x))**2)`
- Handles squared units: `var_units = f"({self._units})**2"`
- Applies ddof correction

**Units handling**: Squares units for variance

---

### 2.7 global_max() - Line 626-712
```python
def global_max(self, axis=None, out=None, keepdims=False):
    """Return maximum across all MPI ranks with units preserved."""
```
**MPI-aware version of max()**

**Implementation**:
- Local max: `self.max(axis=axis, ...)`
- Global reduction: `uw.mpi.comm.allreduce(local_val, op=MPI.MAX)`
- Wraps with units

---

### 2.8 global_min() - Line 714-800
```python
def global_min(self, axis=None, out=None, keepdims=False):
    """Return minimum across all MPI ranks with units preserved."""
```
**MPI-aware version of min()**

---

### 2.9 global_sum() - Line 802-869
```python
def global_sum(self, axis=None, dtype=None, out=None, keepdims=False):
    """Return sum across all MPI ranks with units preserved."""
```
**MPI-aware version of sum()**

---

### 2.10 global_mean() - Line 871-941
```python
def global_mean(self, axis=None, dtype=None, out=None, keepdims=False):
    """Return mean across all MPI ranks with units preserved."""
```
**MPI-aware version of mean()**
- Computes: `mean = global_sum / global_count`

---

### 2.11 global_var() - Line 943-1045
```python
def global_var(self, axis=None, dtype=None, ddof=0, keepdims=False):
    """Return variance across all MPI ranks with units squared preserved."""
```
**MPI-aware version of var()**
- Uses parallel variance algorithm (Welford/Chan)

---

### 2.12 global_std() - Line 1047-1094
```python
def global_std(self, axis=None, dtype=None, ddof=0, keepdims=False):
    """Return standard deviation across all MPI ranks with units preserved."""
```
**MPI-aware version of std()**

---

### 2.13 global_norm() - Line 1096-1144
```python
def global_norm(self, ord=None):
    """Return norm across all MPI ranks."""
```
**MPI-aware norm computation**
- Default: 2-norm (Euclidean)
- Only supports ord=None or ord=2

---

### 2.14 global_size() - Line 1146-1173
```python
def global_size(self):
    """Return total number of elements across all MPI ranks."""
```
**Not a reduction operation** - Returns total element count

---

### 2.15 global_rms() - Line 1175-1220
```python
def global_rms(self):
    """Return root mean square across all MPI ranks with units preserved."""
```
**Computed as**: `rms = global_norm() / sqrt(global_size())`

---

## 3. Array View Classes
**File**: `discretisation_mesh_variables.py`

### SimpleMeshArrayView (lines 1401-1527)
**NO reduction operations defined**
- Has `shape`, `dtype`, `units` properties
- Has `__getitem__`, `__setitem__`, `__array__`, `__array_ufunc__` methods
- Delegates to numpy via `__array_ufunc__`

### TensorMeshArrayView (lines 1534-1659)
**NO reduction operations defined**
- Same structure as SimpleMeshArrayView
- Uses pack/unpack for tensor layouts

---

## 4. Summary Table

| Operation | MeshVariable | UnitAwareArray | Notes |
|-----------|--------------|----------------|-------|
| `.max()` | Line 1781 ✅ | Line 500 ✅ | Returns float/tuple; UnitAwareArray has axis param |
| `.min()` | Line 1757 ✅ | Line 511 ✅ | Returns float/tuple; UnitAwareArray has axis param |
| `.sum()` | Line 1805 ✅ | Line 533 ✅ | Returns float/tuple; UnitAwareArray has axis param |
| `.mean()` | Line 1858 ✅ | Line 522 ✅ | Divides by vecsize properly; UnitAwareArray has axis param |
| `.std()` | ❌ None | Line 544 ✅ | Only in UnitAwareArray (unit-aware) |
| `.var()` | ❌ None | Line 570 ✅ | Only in UnitAwareArray (unit-aware) |
| `.norm()` | Line 1827 ✅ | ❌ None | PETSc-based; requires norm_type param |
| `.global_max()` | ❌ None | Line 626 ✅ | MPI-aware reduction |
| `.global_min()` | ❌ None | Line 714 ✅ | MPI-aware reduction |
| `.global_sum()` | ❌ None | Line 802 ✅ | MPI-aware reduction |
| `.global_mean()` | ❌ None | Line 871 ✅ | MPI-aware reduction |
| `.global_std()` | ❌ None | Line 1047 ✅ | MPI-aware reduction (unit-aware) |
| `.global_var()` | ❌ None | Line 943 ✅ | MPI-aware reduction (unit-aware) |
| `.global_norm()` | ❌ None | Line 1096 ✅ | MPI-aware 2-norm |
| `.global_rms()` | ❌ None | Line 1175 ✅ | MPI-aware RMS |
| `.stats()` | Line 1880 ✅ | ❌ None | Returns dict of statistics |

---

## 5. Pattern for Consistent Implementation

### Current Pattern in MeshVariable
```python
def min(self) -> Union[float, tuple]:
    if self.num_components == 1:
        return scalar_result
    else:
        return tuple([component_result_i for i in range(self.num_components)])
```

**Problem**: 
- `max()`, `min()`, `sum()`, `mean()` return tuples for multi-component
- `.stats()` returns dicts instead
- No `.std()` or `.var()` methods
- No axis parameter support
- No unit awareness

### Recommended Pattern for SimpleMeshArrayView/TensorMeshArrayView
```python
def max(self, axis=None, keepdims=False):
    """Return maximum with proper component handling."""
    array_data = self._get_array_data()
    
    if axis is None and not keepdims:
        # Scalar reduction (all dimensions)
        if self.parent.num_components == 1:
            return np.max(array_data)  # Scalar
        else:
            # Multi-component: return component-wise maxima
            return tuple(np.max(array_data[:, i, :], axis=None) 
                        for i in range(self.parent.shape[0]))
    else:
        # Axis-specific reduction
        result = np.max(array_data, axis=axis, keepdims=keepdims)
        return self._wrap_result(result) if self.parent.units else result
```

---

## 6. Key Findings

### ✅ What's Working
1. **MeshVariable**: min, max, sum, mean, norm methods with proper num_components handling
2. **UnitAwareArray**: Complete numpy-compatible reduction API with unit preservation
3. **Global MPI operations**: global_max, global_min, global_sum, global_mean with units
4. **Stats dispatch**: stats() method dispatches to _scalar_stats, _vector_stats, _tensor_stats

### ⚠️ Gaps & Inconsistencies
1. **Array View Classes**: No reduction operations defined
   - SimpleMeshArrayView (line 1401-1527) - missing all reductions
   - TensorMeshArrayView (line 1534-1659) - missing all reductions
   
2. **MeshVariable Missing Methods**:
   - `.std()` - only in UnitAwareArray
   - `.var()` - only in UnitAwareArray
   - `.global_*()` methods - only in UnitAwareArray
   - No axis parameter support (unlike numpy)

3. **Inconsistent Return Types**:
   - MeshVariable returns tuples for multi-component
   - numpy.max() returns arrays when axis is specified
   - Should align one approach

### 🎯 Implementation Priority
1. **Add to Array View Classes** (lines 1401-1659):
   - min(), max(), sum(), mean(), std(), var()
   - Support axis and keepdims parameters
   - Handle num_components properly
   - Preserve units if available

2. **Extend MeshVariable**:
   - Add std() and var() methods
   - Add axis parameter support
   - Consider adding global_* variants

3. **Maintain backward compatibility**:
   - Keep existing num_components → tuple pattern
   - Make axis parameter optional (default None for global reduction)

