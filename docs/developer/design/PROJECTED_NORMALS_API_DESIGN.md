# Projected Surface Normals API Design

**Date:** 2025-01-18
**Status:** Approved - Implementation by LM
**Context:** Should we provide projected normals as a general interface?

---

## Background

Investigation showed that for curved boundaries:
- `mesh.Gamma` (raw PETSc normals) gives ~27% error vs analytical
- Projected normals give ~0.06% error (99.8% improvement)
- Raw normals ARE unit vectors, but their DIRECTION is facet-based

The question: should we provide projected normals as a built-in API?

---

## Current State

```python
# mesh.Gamma is symbolic, maps to petsc_n in boundary integrals
Gamma = mesh.Gamma
stokes.add_natural_bc(penalty * Gamma.dot(v.sym) * Gamma, "Boundary")

# Users must manually project for accuracy on curved boundaries
n_proj = uw.discretisation.MeshVariable("n", mesh, 2, degree=2)
projection = uw.systems.Vector_Projection(mesh, n_proj)
# ... setup and solve ...
stokes.add_natural_bc(penalty * n_proj.sym.dot(v.sym) * n_proj.sym, "Boundary")
```

---

## Options Considered

### Option A: Replace mesh.Gamma with projected normals

**Against:**
- Silent behavior change (breaks existing code semantics)
- Adds computation cost even for straight-edged meshes
- Hides the projection (magic behavior)
- Raw normals are correct for boxes/circles

**Verdict:** ❌ Not recommended

### Option B: Add mesh.boundary_normals(projected=True)

```python
n = mesh.boundary_normals(projected=True)  # Returns MeshVariable
n = mesh.boundary_normals(projected=False)  # Returns symbolic Gamma
```

**Against:**
- Return type changes based on argument (confusing)
- Not clear it's a projection solve happening

**Verdict:** ❌ Not recommended

### Option C: Add mesh.Gamma_projected property

```python
mesh.Gamma           # Raw symbolic (current)
mesh.Gamma_projected # Pre-computed projected MeshVariable
```

**Against:**
- When is it computed? (lazy? eager?)
- What boundaries? All of them?
- Memory cost for meshes that don't need it

**Verdict:** ⚠️ Possible but has issues

### Option D: Add mesh.project_surface_normals() method

```python
# Explicit method that returns a MeshVariable
n_proj = mesh.project_surface_normals(
    surfaces=["Outer", "Inner"],  # Which surfaces (boundaries or internal)
    degree=2,                      # FE degree
    normalize=True,                # Normalize to unit vectors
)

# Use like any MeshVariable
stokes.add_natural_bc(penalty * n_proj.sym.dot(v.sym) * n_proj.sym, "Outer")
```

**For:**
- Explicit about what it does (projection solve)
- Returns clear type (MeshVariable)
- User controls which surfaces and degree
- Discoverable as mesh method
- No hidden computation or memory
- Keeps `mesh.Gamma` for advanced/direct usage
- Name `project_surface_normals` applies to both boundaries AND internal surfaces

**Verdict:** ✅ **Approved**

### Option E: Keep current API, improve documentation only

```python
# Just document the pattern well
```

**For:**
- No code changes
- No API expansion

**Against:**
- Users must copy boilerplate code
- Easy to get wrong (orientation, normalization)

**Verdict:** ⚠️ Acceptable but not ideal

---

## Recommendation: Option D

Add a method to the Mesh class:

```python
def project_boundary_normals(
    self,
    boundaries: list[str] = None,  # None = all boundaries
    degree: int = 2,
    normalize: bool = True,
    name: str = "boundary_normals",
) -> MeshVariable:
    """
    Project boundary normals onto a continuous mesh variable.

    This creates smooth, interpolated boundary normals that are more
    accurate than the raw mesh.Gamma for curved boundaries.

    Parameters
    ----------
    boundaries : list of str, optional
        Boundary names to project normals for. If None, projects all boundaries.
    degree : int, default 2
        Finite element degree for the normal variable.
    normalize : bool, default True
        Whether to normalize the projected normals to unit vectors.
    name : str, default "boundary_normals"
        Name for the created MeshVariable.

    Returns
    -------
    MeshVariable
        A vector mesh variable containing the projected boundary normals.
        Use .sym for symbolic access in boundary conditions.

    Examples
    --------
    >>> n = mesh.project_boundary_normals(["Outer", "Inner"])
    >>> stokes.add_natural_bc(10000 * n.sym.dot(v.sym) * n.sym, "Outer")

    Notes
    -----
    For straight-edged boundaries (boxes), mesh.Gamma is already exact.
    Use this method for curved or elliptical boundaries where the
    mesh facet normals don't represent the true surface direction.
    """
    import sympy

    # Create variable
    n_proj = MeshVariable(name, self, self.dim, degree=degree)

    # Set up projection
    projection = uw.systems.Vector_Projection(self, n_proj)
    projection.uw_function = sympy.Matrix([[0] * self.dim])

    # Radial direction for orientation (works for most cases)
    x, y = self.X[:2]
    r = sympy.sqrt(x**2 + y**2)
    unit_r = sympy.Matrix([x/r, y/r] + ([self.X[2]/r] if self.dim == 3 else [])).T

    # Orientation correction
    orientation = sympy.sign(unit_r.dot(self.Gamma))

    # Add boundary conditions
    if boundaries is None:
        boundaries = [b.name for b in self.boundaries]

    for boundary in boundaries:
        projection.add_natural_bc(self.Gamma * orientation, boundary)

    projection.solve()

    # Normalize if requested
    if normalize:
        import numpy as np
        mag = np.sqrt(np.sum(n_proj.data**2, axis=1, keepdims=True))
        mag = np.maximum(mag, 1e-12)  # Avoid division by zero
        n_proj.data[:] /= mag

    return n_proj
```

---

## Implementation Location

File: `src/underworld3/discretisation/discretisation_mesh.py`

Add the method to the `Mesh` class, after the `Gamma` property definition.

---

## Documentation Updates

1. Add to `mesh.Gamma` docstring:
   > For curved boundaries, see `project_boundary_normals()` for improved accuracy.

2. Update `curved-boundary-conditions.md` to show the new API.

3. Add to API reference.

---

## Migration Path

- `mesh.Gamma` remains unchanged (no breaking changes)
- New method provides the recommended approach for curved boundaries
- Documentation guides users to the right choice

---

## Orientation Handling

**Boundary surfaces**: PETSc handles orientation correctly for surfaces that lie on the domain boundary. No special handling needed.

**Internal surfaces**: Orientation is more problematic. PETSc may not provide consistent orientation for internal surfaces (e.g., fault planes, material interfaces).

### Future Work: Internal Surface Orientation

This is a known limitation that may require discussion with the PETSc team. Options:
1. User-specified orientation vector for internal surfaces
2. Request PETSc enhancement for consistent internal surface orientation
3. Post-processing heuristics based on geometry

**Status**: Added to high-level planning. See `docs/developer/design/mesh-geometry-audit.md` section "Future Work: Internal Surface Normal Orientation".

---

## Implementation

**Assigned to**: Louis Moresi
**Method name**: `mesh.project_surface_normals()`
**Status**: Approved for implementation

---

*Updated 2025-01-18: Approved with feedback on naming and internal surface orientation.*
