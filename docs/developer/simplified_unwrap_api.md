# Simplified Unwrap and Scaling API

## Overview

The unwrap and scaling API has been simplified to provide a single, consistent user interface with minimal functions and clear usage patterns.

## Single Public Function

**`uw.unwrap(expr, keep_constants=True, return_self=True, apply_scaling=False)`**

This is the only function users need to know for unwrapping expressions.

### Usage Patterns

**Normal unwrapping:**
```python
result = uw.unwrap(expr)
```

**Unwrapping with automatic scaling:**
```python
scaled_result = uw.unwrap(expr, apply_scaling=True)
```

**With other parameters:**
```python
result = uw.unwrap(expr, keep_constants=False, apply_scaling=True)
```

## What Was Removed

### Object Methods (Removed)
- ~~`uw.unwrap(expr)`~~ → Use `uw.unwrap(expr)`
- ~~`uw.unwrap(derivative_expr)`~~ → Use `uw.unwrap(derivative_expr)`

### Context Managers (Hidden)
- ~~`uw.apply_scaling()`~~ → Use `uw.unwrap(expr, apply_scaling=True)`
- ~~`uw.scaled_symbols`~~ → Deprecated alias, use parameter approach

## Benefits

### Consistency
- **Single function**: Always `uw.unwrap(anything)`
- **Universal**: Works on all expression types
- **Clear parameters**: Boolean flags are self-documenting

### Simplicity
- **Two patterns**: Normal vs scaled unwrapping
- **No choice paralysis**: One way to do each operation
- **Fewer concepts**: No need to learn multiple interfaces

### Documentation
- **Single point**: Only one function to document
- **Clear examples**: Consistent usage across all docs
- **Less confusion**: No "multiple ways to do the same thing"

## Implementation Details (Hidden from Users)

The simplified API uses internal implementation details that users don't need to know about:

- `_apply_scaling()` - Internal context manager
- `UWDerivativeExpression` special handling
- Symbol matching and substitution logic

## Migration Summary

✅ **Removed**: 16 files updated to remove `.unwrap()` method calls
✅ **Standardized**: All code now uses `uw.unwrap(expr)` function
✅ **Simplified**: Single function with clear boolean parameters
✅ **Tested**: All existing functionality preserved

## User Guidelines

**Always use:**
```python
uw.unwrap(expr)                    # Normal unwrapping
uw.unwrap(expr, apply_scaling=True)  # Scaled unwrapping
```

**Never use:**
```python
uw.unwrap(expr)                      # Removed - inconsistent
with uw.apply_scaling(): ...       # Hidden - use parameter instead
```

This simplified API provides a clean, consistent interface while hiding implementation complexity from users.