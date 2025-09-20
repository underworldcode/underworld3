# Mathematical Objects Plan: Variables & Expressions as SymPy Citizens

## Project Overview

This plan outlines the transformation of Underworld3 variables into mathematical citizens that work seamlessly with SymPy through direct arithmetic operations. The goal is to enable natural mathematical notation (e.g., `v1 = -1 * v2`) while preserving the existing computational infrastructure and JIT compilation system.

## Current State Analysis

### Problems with Current Architecture

1. **Dual Identity Crisis**: Objects exist in two forms - computational (`velocity`) and mathematical (`velocity.sym`)
2. **Cognitive Load**: Users must constantly switch between object and symbol domains
3. **Inconsistent API**: Sometimes `.sym` required, sometimes not
4. **Mathematical Intuition Broken**: Code doesn't look like mathematics

### Current Usage Patterns

```python
# Current patterns requiring explicit .sym access
velocity = MeshVariable("velocity", mesh, 2)
temperature = MeshVariable("temperature", mesh, 1)

# Need .sym for mathematical operations
strain_rate = sympy.sqrt(
    velocity.sym[0].diff(x)**2 +  # .sym required
    velocity.sym[1].diff(y)**2 +  # .sym required
    0.5 * (velocity.sym[0].diff(y) + velocity.sym[1].diff(x))**2
)

# Component access requires .sym
momentum_x = density * velocity.sym[0]  # .sym required for components

# Notebook display shows data visualization instead of mathematical form
velocity  # Shows data plot/array info instead of mathematical symbol
```

## Proposed Architecture

### Key Insight: Minimal Mixin Approach

After analysis of the JIT compilation system and expression unwrapping, the safest and most effective approach is a simple mixin class that adds mathematical behavior to existing variables without disrupting the computational infrastructure.

### 1. Mathematical Mixin Class

```python
# underworld3/function/mathematical_mixin.py
import sympy
from typing import Any

class MathematicalMixin:
    """
    Mixin class that makes variables work directly in mathematical expressions.
    
    Key principle: Objects behave like their symbolic form in mathematical contexts
    while preserving computational data storage and access.
    
    This mixin enables:
    - Direct arithmetic: v1 = -1 * v2 (instead of v1 = -1 * v2.sym)
    - Component access: v[0] (instead of v.sym[0])
    - Mathematical display in notebooks
    - JIT compilation compatibility
    """
    
    def _sympify_(self):
        """
        SymPy protocol: Tell SymPy to use the symbolic form.
        
        This is the key method that enables direct arithmetic operations.
        When SymPy encounters this object, it automatically calls this method
        to get the SymPy representation.
        
        Example:
            velocity = MeshVariable("velocity", mesh, 2)
            momentum = density * velocity  # Calls velocity._sympify_()
            # Result: density * velocity.sym (pure SymPy expression)
        """
        return self.sym
    
    def __getitem__(self, index):
        """
        Component access for vector/tensor fields.
        
        Enables: velocity[0] instead of velocity.sym[0]
        Returns: Pure SymPy expression (SymPy Function)
        """
        return self.sym[index]
    
    def __repr__(self):
        """
        String representation shows mathematical form.
        
        In notebooks, this makes variables display as mathematical symbols
        instead of data visualizations.
        """
        return str(self.sym)
    
    def _repr_latex_(self):
        """
        Jupyter LaTeX display of mathematical form.
        """
        return f"${sympy.latex(self.sym)}$"
    
    def diff(self, *args, **kwargs):
        """
        Direct differentiation.
        
        Returns: Pure SymPy expression suitable for JIT compilation
        """
        return self.sym.diff(*args, **kwargs)
    
    def view(self):
        """
        Explicit access to object details and data visualization.
        
        Preserves the old notebook display behavior for when users
        want to see computational data rather than mathematical form.
        """
        # Call the original __repr__ from the parent class
        return super().__repr__()

# Note: Arithmetic operations return pure SymPy objects, not wrapped objects
# This ensures JIT compatibility and prevents wrapper proliferation
#
# Example flow:
# velocity = MeshVariable("velocity", mesh, 2)  # Has MathematicalMixin
# momentum = density * velocity                  # Returns sympy.Matrix
# strain = velocity.diff(x)                      # Returns sympy.Matrix
# 
# The JIT system receives pure SymPy expressions and can identify
# the original SymPy Functions for PETSc substitution
```

### 2. Enhanced Variable Classes

```python
# Enhanced MeshVariable with mathematical behavior
class MeshVariable(MathematicalMixin, _MeshVariable):
    """
    MeshVariable enhanced with direct mathematical operations.
    
    Inherits all existing functionality from _MeshVariable and adds:
    - Direct arithmetic: velocity * density
    - Component access: velocity[0] 
    - Mathematical display in notebooks
    - JIT compilation compatibility
    """
    pass  # All functionality comes from mixin and parent class

# Enhanced SwarmVariable with mathematical behavior  
class SwarmVariable(MathematicalMixin, _SwarmVariable):
    """
    SwarmVariable enhanced with direct mathematical operations.
    
    Inherits all existing functionality from _SwarmVariable and adds
    the same mathematical interface as MeshVariable.
    """
    pass  # All functionality comes from mixin and parent class
```

### 3. JIT Compatibility Analysis

**Critical Insight**: The JIT compilation system works by:
1. **Atom Identification**: Finding SymPy Functions in expressions
2. **PETSc Mapping**: Mapping `variable.fn` to PETSc sub-vectors 
3. **Code Generation**: Substituting `petsc_u[i]` for each variable

**Why `_sympify_` is Safe**:
```python
# Both patterns produce identical SymPy atoms:
old_expr = density * velocity.sym     # Contains velocity's SymPy Function
new_expr = density * velocity         # _sympify_() returns same SymPy Function

# JIT atom identification works identically:
# - Finds same SymPy Function atoms
# - Maps to same variable.fn objects  
# - Generates same PETSc substitutions
```

**Expression Flow**:
```python
velocity = MeshVariable("velocity", mesh, 2)
expr = -1 * velocity

# Step 1: _sympify_() called
# Returns: -1 * velocity.sym (pure SymPy Matrix)

# Step 2: JIT compilation
# Finds SymPy Function atoms: V_x(x,y,z), V_y(x,y,z)
# Maps to velocity.fn from varlist
# Generates: -1 * petsc_u[0], -1 * petsc_u[1]
```

## Implementation Strategy

## Implementation Completed ✅

### Phase 1: Mathematical Mixin Implementation (COMPLETED)

**✅ Deliverables Completed:**
1. `MathematicalMixin` class in `utilities/mathematical_mixin.py`
2. Complete `_sympify_` protocol implementation
3. Component access (`__getitem__`) method
4. Full arithmetic operations (`__add__`, `__mul__`, `__sub__`, `__truediv__`, `__pow__`, `__neg__`, etc.)
5. Right-hand arithmetic operations (`__radd__`, `__rmul__`, etc.)
6. Complete SymPy Matrix API delegation via `__getattr__`
7. Preserved computational display behavior in `__repr__`
8. Mathematical representation via `sym_repr()` method
9. Jupyter LaTeX display via `_repr_latex_`

**✅ Key Files Created/Modified:**
- `utilities/mathematical_mixin.py`: Complete mixin implementation
- `test_mathematical_mixin.py`: Comprehensive test suite

### Phase 2: Integration with Variable Classes (COMPLETED)

**✅ Deliverables Completed:**
1. Added `MathematicalMixin` to `_MeshVariable` inheritance chain
2. Added `MathematicalMixin` to `SwarmVariable` inheritance chain
3. Fixed circular import issues by moving mixin to `utilities/`
4. Verified JIT compilation compatibility
5. All existing functionality preserved

**✅ Key Files Modified:**
- `discretisation/discretisation_mesh_variables.py`: Added mixin to `_MeshVariable`
- `swarm.py`: Added mixin to `SwarmVariable`
- Import statements updated to use `utilities.mathematical_mixin`

### Phase 3: Testing and Validation (COMPLETED)

**✅ Validation Results:**
1. ✅ Direct arithmetic operations working: `var * 2`, `2 * var`, `var + 1`, `-var`
2. ✅ Component access working: `var[0]` equivalent to `var.sym[0]`
3. ✅ Full SymPy Matrix API available: `var.T`, `var.dot()`, `var.norm()`, etc.
4. ✅ JIT compilation compatibility verified
5. ✅ Backward compatibility maintained: `.sym` property still available
6. ✅ Original computational display preserved

### Phase 4: Key Lessons Learned

**Critical Implementation Insights:**
1. **Arithmetic Operation Requirements**: Both explicit methods (`__mul__`, `__add__`) and `_sympify_` needed
   - `_sympify_()` handles SymPy-initiated operations  
   - Explicit methods handle Python-initiated operations
   - Right-hand methods (`__rmul__`) essential for operations like `2 * var`

2. **SymPy API Delegation**: `__getattr__` provides complete SymPy Matrix functionality
   - Automatically delegates missing methods to `self.sym`
   - Scales to hundreds of SymPy methods without individual implementation
   - Future-proof for new SymPy methods

3. **Display Behavior Balance**: Preserve computational view by default
   - Users want to see data/mesh info, not symbolic form by default
   - `sym_repr()` method available for mathematical display when needed
   - Jupyter LaTeX display for mathematical contexts

4. **Circular Import Resolution**: Module organization matters
   - Moving shared code to `utilities/` breaks dependency cycles
   - Avoid imports between `discretisation` and `function` modules

5. **JIT Compatibility Verification**: `_sympify_` is completely safe
   - Returns identical SymPy objects as `.sym` property
   - JIT atom identification works identically
   - No performance or compilation impact

## Usage Examples

### Before (Original Pattern)
```python
velocity = MeshVariable("velocity", mesh, 2)
pressure = MeshVariable("pressure", mesh, 1)
density = UWexpression(r'\rho', sym=1000)

# Mathematical operations require .sym
momentum = density * velocity.sym
strain_rate = velocity.sym[0].diff(x) + velocity.sym[1].diff(y)
divergence = velocity.sym[0].diff(x) + velocity.sym[1].diff(y)

# Vector operations require .sym
velocity_magnitude = velocity.sym.norm()
velocity_transposed = velocity.sym.T

# Notebook display shows data visualization
velocity  # Shows plot/array instead of mathematical symbol
```

### After (Enhanced Pattern) ✅
```python
velocity = MeshVariable("velocity", mesh, 2)  # Now has MathematicalMixin
pressure = MeshVariable("pressure", mesh, 1)  # Now has MathematicalMixin  
density = UWexpression(r'\rho', sym=1000)

# Natural mathematical operations
momentum = density * velocity              # Direct arithmetic
strain_rate = velocity[0].diff(x) + velocity[1].diff(y)  # Component access
divergence = velocity[0].diff(x) + velocity[1].diff(y)   # No .sym needed

# Full SymPy Matrix API available
velocity_magnitude = velocity.norm()       # Direct method access
velocity_transposed = velocity.T           # Direct property access
velocity_dot_product = velocity.dot(other) # Matrix operations

# Computational display preserved
velocity         # Shows data visualization (original behavior)
velocity.sym_repr()  # Shows: Matrix([[V_0(x, y, z)], [V_1(x, y, z)]])
velocity.array   # Access computational arrays
```

### Backward Compatibility ✅
```python
# Old patterns continue to work perfectly
old_momentum = density * velocity.sym       # Still works
new_momentum = density * velocity           # Also works
old_norm = velocity.sym.norm()              # Still works  
new_norm = velocity.norm()                  # Also works

# All produce identical SymPy expressions
assert old_momentum.equals(new_momentum)   # True
assert old_norm.equals(new_norm)           # True
```

### Complete SymPy Matrix Integration ✅
```python
# All SymPy Matrix methods now available directly:
velocity.T                    # Transpose
velocity.dot(other)          # Dot product  
velocity.cross(other)        # Cross product
velocity.norm()              # Vector norm
velocity.diff(x)             # Differentiation
velocity.subs(x, 1)          # Substitution
velocity.applyfunc(func)     # Apply function to elements
velocity.reshape(1, 2)       # Reshape matrix
# And hundreds more SymPy Matrix methods...
```

## Cross-References to Other Plans

### Units System Integration
**See: `units_system_plan.md`**
- Mathematical objects will integrate with unit-aware expressions
- Differential operations must handle unit transformations
- Natural arithmetic preserves unit relationships

### Material Properties Integration  
**See: `material_properties_plan.md`**
- Material property expressions work with mathematical objects
- Parameter system integration via direct arithmetic
- Multi-material constitutive models use natural notation

### Multi-Material System
**See: `MultiMaterial_ConstitutiveModel_Plan.md`**
- Level-set averaging works with enhanced field objects
- IndexSwarmVariable becomes mathematical object
- Flux computation uses direct mathematical expressions

## Benefits

### For Users
1. **Natural Mathematical Expressions**: `v1 = -1 * v2` instead of `v1 = -1 * v2.sym`
2. **Component Access**: `velocity[0]` instead of `velocity.sym[0]`
3. **Mathematical Display**: Variables show as mathematical symbols in notebooks
4. **Consistent Interface**: All variables work the same way in mathematical contexts

### For Developers  
1. **Minimal Changes**: Simple mixin addition to existing classes
2. **JIT Compatibility**: Preserves existing compilation system
3. **Pure SymPy Results**: Arithmetic operations return SymPy objects
4. **No Breaking Changes**: All existing code continues to work

### For the Ecosystem
1. **Mathematical Software Standards**: Follows patterns from NumPy, SymPy
2. **Educational Value**: Code that looks like mathematical equations
3. **Research Integration**: Natural interface for mathematical modeling
4. **Backward Compatibility**: Smooth migration path from current patterns

## Success Criteria ✅ ACHIEVED

### Functional Requirements ✅
1. ✅ **Direct Arithmetic**: `v1 = -1 * v2`, `momentum = density * velocity` - Working
2. ✅ **Component Access**: `velocity[0]` returns `velocity.sym[0]` - Working
3. ✅ **Complete SymPy API**: `velocity.T`, `velocity.dot()`, `velocity.norm()` - Working
4. ✅ **JIT Compatibility**: `uw.function.expression.unwrap()` works unchanged - Verified
5. ✅ **Backward Compatibility**: `.sym` property remains available - Preserved
6. ✅ **Computational Display**: Variables show data visualization by default - Preserved

### Performance Requirements ✅
1. ✅ **No Regression**: Mathematical operations use identical SymPy paths
2. ✅ **Memory Efficiency**: No additional memory overhead (delegation only)
3. ✅ **SymPy Integration**: Direct delegation to existing SymPy objects

### User Experience Requirements ✅
1. ✅ **Intuitive**: Variables work exactly like SymPy matrices in mathematical contexts
2. ✅ **Migration Path**: Old `.sym` usage continues to work alongside new patterns
3. ✅ **Error Handling**: Standard Python/SymPy error messages via delegation
4. ✅ **Complete API**: Full SymPy Matrix functionality available

## Implementation Success Summary ✅

### Technical Achievements
- ✅ **Complete SymPy Integration**: `__getattr__` delegation provides full Matrix API
- ✅ **Zero Performance Impact**: Direct delegation to existing SymPy objects
- ✅ **Minimal Memory Footprint**: No additional data storage, only method delegation
- ✅ **JIT Compatibility Verified**: Identical SymPy atoms and compilation paths

### User Experience Achievements  
- ✅ **Natural Mathematical Syntax**: Variables work exactly like SymPy matrices
- ✅ **Zero Breaking Changes**: All existing code continues to work
- ✅ **Progressive Enhancement**: Users can adopt new patterns gradually
- ✅ **Complete Feature Parity**: All SymPy Matrix functionality available

### Architecture Success
- ✅ **Minimal Implementation**: Simple mixin with maximum functionality
- ✅ **Scalable Design**: Automatically supports future SymPy Matrix methods
- ✅ **Clean Separation**: Mathematical behavior separate from computational infrastructure
- ✅ **Conservative Approach**: No changes to solvers or core compilation system

---

*Document Version: 2.0*  
*Created: 2025-01-19*  
*Updated: 2025-09-20*  
*Cross-References: units_system_plan.md, material_properties_plan.md, MultiMaterial_ConstitutiveModel_Plan.md*  
*Status: ✅ IMPLEMENTATION COMPLETED*

**Implementation Summary:**
- Mathematical objects working with natural syntax: `var * 2`, `var[0]`, `var.T`, `var.dot()`
- Complete SymPy Matrix API available via `__getattr__` delegation
- Full backward compatibility maintained
- JIT compilation compatibility verified
- Zero performance impact, minimal code changes
