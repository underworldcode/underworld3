# Simplified DMInterpolation Caching: Just Cache Cell Hints

**Date**: 2025-11-16
**Insight**: Cache only the geometric mapping (coords → cells), not the full DMInterpolationInfo structure

## The Key Insight

The DMInterpolationInfo structure at line 631 depends on **total DOF count** (which changes when variables are added/removed):

```cython
# Lines 624-631: DOF count from current variable configuration
dofcount = 0
for var in vars:
    var_start_index[var] = dofcount
    dofcount += var.num_components
ierr = DMInterpolationSetDof(ipInfo, dofcount);  # ← Depends on current variables!
```

**We can't cache this structure** - it would break when variables change.

But the **expensive part** is separate and cacheable:

```cython
# Line 651: THIS is the expensive KDTree query!
cells = mesh.get_closest_cells(coords)
# Returns: numpy array of cell indices (shape: n_coords)
```

## What get_closest_cells() Does

```python
def get_closest_cells(self, coords):
    """Uses KDTree to find closest cell for each coordinate."""
    self._build_kd_tree_index()  # Build once (already cached)
    dist, closest_points = self._index.query(coords, k=1)  # ← EXPENSIVE!
    return self._indexMap[closest_points]  # Map to cell indices
```

**Returns**: Simple numpy array `[cell_id_0, cell_id_1, ..., cell_id_n]`

**Cost**: KDTree query is O(N log M) where N=points, M=cells

## Simplified Caching Strategy

### Cache Structure

```python
# On mesh object:
mesh._cell_hints_cache = {}  # {coord_hash: cell_indices_array}
```

**Cache key**: Hash of coordinates (xxhash)
**Cache value**: Cell indices array from `get_closest_cells()`
**Memory**: Tiny! Just integer array, ~400 bytes for 100 points

### Modified petsc_interpolate()

```cython
def interpolate_vars_on_mesh(varfns, coords):
    mesh = varfns[0].meshvar().mesh

    # 1. CHECK CACHE for cell hints
    coord_hash = _hash_coords(coords)

    if coord_hash in mesh._cell_hints_cache:
        cells = mesh._cell_hints_cache[coord_hash]  # ← CACHE HIT!
    else:
        cells = mesh.get_closest_cells(coords)  # ← Expensive KDTree query
        mesh._cell_hints_cache[coord_hash] = cells.copy()  # Cache for next time

    # 2. BUILD DMInterpolation with current variable configuration
    # (This is cheap compared to the KDTree query!)
    cdef DMInterpolationInfo ipInfo
    ierr = DMInterpolationCreate(MPI_COMM_SELF, &ipInfo)
    ierr = DMInterpolationSetDim(ipInfo, mesh.dim)

    # Use CURRENT DOF count (adapts to variable changes)
    dofcount = sum(var.num_components for var in mesh.vars.values())
    ierr = DMInterpolationSetDof(ipInfo, dofcount)

    ierr = DMInterpolationAddPoints(ipInfo, coords.shape[0], coords_buff)

    # 3. USE CACHED CELLS as hints
    ierr = DMInterpolationSetUp_UW(ipInfo, dm.dm, 0, 0, <size_t*> cells.data)

    # 4. EVALUATE with current values
    mesh.update_lvec()  # Fresh values!
    ierr = DMInterpolationEvaluate_UW(ipInfo, dm.dm, mesh.lvec.vec, outvec.vec)

    # 5. DESTROY structure (rebuilt next time, but cells reused!)
    ierr = DMInterpolationDestroy(&ipInfo)
```

## Why This Works

### ✅ Handles Variable Changes
```python
# Add variable after caching cells:
mesh._cell_hints_cache = {hash1: cells_array}  # Cached

# Add new variable:
new_var = uw.discretisation.MeshVariable("P", mesh, 1)

# Next evaluate():
# - Reuses cached cells ✓
# - Rebuilds DMInterpolation with NEW dofcount ✓
# - Everything works!
```

### ✅ Handles Value Changes
```python
# Cell cache persists across value changes:
T.array[...] = new_values  # Values change

# Next evaluate():
# - Reuses cached cells ✓
# - Rebuilds DMInterpolation (cheap)
# - Gets fresh values from update_lvec() ✓
```

### ✅ Simple Invalidation
Only invalidate when:
1. **Coordinates change** → Different hash → Automatic cache miss
2. **Mesh topology changes** → Clear entire cache

```python
def clear_cell_hints_cache(self):
    """Call when mesh topology changes (remeshing, refinement)."""
    self._cell_hints_cache.clear()
```

## Performance Analysis

### What's Expensive?

From diagnostic results (11 calls = 10 + warmup):

**Option 1: No Caching** (current):
- `get_closest_cells()`: Called 11 times
- KDTree queries: 11 × ~2ms = **22ms**
- DMInterpolationSetUp: 11 × ~8ms = **88ms**
- **Total overhead**: ~110ms for 11 calls

**Option 2: Cache Cell Hints** (proposed):
- `get_closest_cells()`: Called 1 time (first call)
- KDTree queries: 1 × ~2ms = **2ms**
- DMInterpolationSetUp: 11 × ~8ms = **88ms** (still needed, but uses cached cells)
- **Total overhead**: ~90ms for 11 calls
- **Savings**: ~20ms (18% reduction)

Hmm, that's not huge. Let me reconsider...

Actually, looking at the PETSc log, DMInterpolationSetUp creates **404 Index Sets per call**. Those Index Sets are probably created INSIDE SetUp, not by get_closest_cells().

Let me check what actually happens in DMInterpolationSetUp...

Actually, the user might be asking: Does caching cell hints even help? Or is the expensive part something else inside DMInterpolationSetUp_UW?

## Wait - Let Me Reconsider

Looking at line 653:
```cython
ierr = DMInterpolationSetUp_UW(ipInfo, dm.dm, 0, 0, <size_t*> cells_buff)
```

The cells are passed as **hints** to SetUp. But SetUp still has to:
1. Verify points are actually in hinted cells
2. Compute local (reference) coordinates within cells
3. Build interpolation structure (creates Index Sets!)

So caching cells might only save the "hint generation" step, not the expensive SetUp itself.

**Question**: Can we cache MORE than just cells? Can we cache the result of DMInterpolationSetUp somehow?

But the user said "the DMInterpolation structure depends on the dm.lvec" - implying we can't cache the full structure.

Let me ask for clarification...

Actually, looking again at line 662:
```cython
ierr = DMInterpolationEvaluate_UW(ipInfo, dm.dm, pyfieldvec.vec, outvec.vec)
```

The **vector** (pyfieldvec) is passed to **Evaluate**, not SetUp!

So SetUp doesn't depend on the vector values, only on:
- DM structure (mesh topology + field configuration)
- Point coordinates
- Cell hints

But it DOES depend on the DOF count (line 631) which changes with variables.

So the question is: Is the "SetDof with different count, then SetUp" process cheaper than full SetUp from scratch if we provide cached cells?

I think I need to just ask the user what they meant...
