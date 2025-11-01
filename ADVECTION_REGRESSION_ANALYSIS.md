# PETSc Advection Error - Regression Analysis

**Issue**: Particles traced outside mesh domain (`Point not located in mesh`)
**Error Value**: `Point 0: -1.42085e+10 1.40692e+10 0. not located in mesh`
**Root Cause**: Units/Dimensional Mismatch in Semi-Lagrangian Backward Tracing
**Status**: IDENTIFIED - Requiring Fix

## Problem Location

File: `src/underworld3/systems/ddt.py`
Method: `SemiLagrangian.update_pre_solve()`
Lines: 704-749 (semi-Lagrangian backward tracing)

## Root Cause Analysis

### The Calculation (ddt.py lines 704-749)

```python
# Line 704: Convert input dt to non-dimensional
dt = model.to_model_magnitude(dt)
# Result: dt is now a scalar number (non-dimensional time)
# e.g., if dt was uw.quantity(1e7, "year"), it becomes something like 0.001

# Line 715: Get coordinate array
coords = self.psi_star[i].coords
# Result: If reference quantities are set, coords has units attached

# Lines 716-724: Extract magnitude
if hasattr(coords, "magnitude"):
    coords_magnitude = coords.magnitude
# BUG: If coords has units (km), coords_magnitude is in PHYSICAL units (km), not mesh coords!
else:
    coords_magnitude = coords

# Line 709-712: Evaluate velocity at current position
v_at_node_pts = uw.function.evaluate(self.V_fn, node_coords)

# Lines 727-732: Extract velocity magnitude
if hasattr(v_at_node_pts, "magnitude"):
    v_at_node_pts_magnitude = v_at_node_pts.magnitude
# Result: Velocity magnitude in non-dimensional units (since evaluate returns non-dimensional)
else:
    v_at_node_pts_magnitude = v_at_node_pts

# Line 734: DIMENSIONAL MISMATCH!
mid_pt_coords = coords_magnitude - 0.5 * dt * v_at_node_pts_magnitude
# [physical_units_km] - [non-dimensional] * [non-dimensional]
# = [km] - [unitless]
# INCOMPATIBLE ARITHMETIC!
```

### Why It's Wrong

**Coordinate System Mismatch**:
- `coords_magnitude` when extracted from unit-aware coordinates is in PHYSICAL units (km, m)
- Global mesh coordinates range from -1 to 1 or -0.5 to 0.5 (non-dimensional)
- Physical domain coordinates range from 0 to 2900 km (for a 2900 km deep mantle)

**Equation**:
```
mid_pt_coords = coords_magnitude - 0.5 * dt * v_at_node_pts_magnitude
```

With actual values from Notebook 14:
- `coords_magnitude`: ~1000 km (physical coordinate in domain)
- `dt`: ~0.001 (non-dimensional time after `to_model_magnitude()`)
- `v_at_node_pts_magnitude`: ~0.0001 (non-dimensional velocity in model space)
- **Result**: 1000 - 0.5 × 0.001 × 0.0001 = 1000 - 5e-8 ≈ 1000 km ✓ (seems OK)

Wait, let me recalculate. Looking at the error value -1.42085e+10, that's ~14 billion km!

This suggests:
- If `v_at_node_pts_magnitude` is NOT non-dimensional but in physical units
- Example: velocity in m/s or cm/year
- Then: 1000 km - 0.5 × [large_velocity_number] × [large_distance]
- Could produce huge numbers!

### The Real Issue: Coordinate Unit Patching

Recent commit `0bd538ad` added coordinate unit patching. This means:
- `mesh.X[0]` now returns coordinates WITH units (km)
- When you call `.magnitude` on these, you get the physical coordinate value
- But the semi-Lagrangian code assumes coordinates are in mesh space (-0.5 to 0.5)

**The 0.5 factor in backward tracing**:
- Semi-Lagrangian uses: `coords - 0.5*dt*velocity` and `coords - dt*velocity`
- This assumes coords and (dt×velocity) are in the SAME coordinate system
- If coords are in km and velocity is in m/s, they're incompatible!

## Solution Approaches

### **Option 1: Use Non-Dimensional Coordinates (RECOMMENDED)**

**Problem**: The code should use mesh coordinates (non-dimensional), not physical coordinates

**Fix**:
```python
# Instead of:
coords = self.psi_star[i].coords
if hasattr(coords, "magnitude"):
    coords_magnitude = coords.magnitude  # WRONG: gets physical coords
else:
    coords_magnitude = coords

# Use:
coords = self.psi_star[i].coords
# Use the raw mesh coordinates without units
if isinstance(coords, np.ndarray):
    coords_nd = coords  # Already non-dimensional
else:
    # If it's a unit-aware array, get the underlying non-dimensional array
    coords_nd = coords.magnitude if hasattr(coords, 'magnitude') else coords
```

**But wait**: This assumes coords exist as non-dimensional arrays. Need to check what `.coords` actually returns.

### **Option 2: Ensure Dimensional Consistency**

Make sure velocity is in non-dimensional form before arithmetic:
```python
# After extracting velocity, convert to non-dimensional
v_at_node_pts_nd = model.to_model_magnitude(v_at_node_pts)
# Then use non-dimensional velocity with non-dimensional coords and time
mid_pt_coords = coords_nd - 0.5 * dt * v_at_node_pts_nd
```

### **Option 3: Transform Between Coordinate Systems**

If coordinates are in physical units:
```python
# Convert coords from physical to non-dimensional mesh space
length_scale = model.get_length_scale()  # Get reference length
coords_nd = coords_magnitude / length_scale

# Then do non-dimensional backward tracing
mid_pt_coords_nd = coords_nd - 0.5 * dt * v_at_node_pts_magnitude

# Convert back if needed
mid_pt_coords = mid_pt_coords_nd * length_scale
```

## Why This is a Regression

**Timeline**:
- Semi-Lagrangian backward tracing was working
- Recent coordinate units patching (`0bd538ad`) added unit information to mesh coordinates
- Code that extracts `.magnitude` from coordinates now gets PHYSICAL coordinates
- Old code assumed coordinates were non-dimensional
- Result: Dimensional mismatch introduced by the units patch

## Recommended Fix Strategy

1. **Identify coordinate system**: What does `.coords` return on `psi_star[i]`?
   - Is it mesh coordinates (-0.5 to 0.5) or physical coordinates?
   - Is it unit-aware or plain arrays?

2. **Determine intended system**: Semi-Lagrangian should work in non-dimensional mesh space
   - All coordinates should be mesh coords
   - All velocities should be non-dimensional
   - Time step should be non-dimensional

3. **Apply appropriate fix**:
   - **If `.coords` is physical**: Convert back to non-dimensional before arithmetic
   - **If `.coords` is non-dimensional**: Don't extract `.magnitude`, use as-is
   - **Mixed case**: Ensure all quantities converted to same system

4. **Add defensive checks**:
   - Verify backward-traced coordinates are within mesh bounds before evaluation
   - Add warning if traced coordinates exceed reasonable values

## Testing

Create test that:
1. Sets up thermal convection with reference quantities (like Notebook 14)
2. Runs one advection-diffusion step
3. Verifies backward-traced coordinates stay within mesh bounds
4. Verifies particle values updated correctly

## Impact Assessment

**Severity**: HIGH - blocks thermal convection simulations with units
**Affected code**: Any semi-Lagrangian advection with units-aware coordinates
**Scope**: `SemiLagrangian.update_pre_solve()` method
**Risk**: Low - fix is localized to coordinate extraction logic
