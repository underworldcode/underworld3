# Prescribed pressure jumps

`Stokes.pressure_split` separates total pressure into a continuous P1 unknown
and a prescribed discontinuous field: `p = q + p_J`. It accepts scalar jump
expressions, continuous scalar mesh fields, constants or coordinate callables.
The interface must already be a named set of interior facets of an affine
triangle or tetrahedron mesh. The method constructs a DG P1 lifting and its
DG P0 negative broken gradient. It does not require an analytical Stokes solution.

## Use

For a mesh fitted to an interface named `Internal`, with the positive side
above it:

```python
split = stokes.pressure_split(
    jumps={"Internal": jump_values},
    normals={"Internal": (0, 1)},
)
stokes.bodyforce = regular_bodyforce + split.regular_force.sym
# Add this correction to the original UW3 natural datum.
stokes.add_natural_bc(original_natural_datum + split.natural_datum, "Top")
stokes.solve()
total_pressure = split.physical_pressure
```

Here `jump_values` represents `p_plus - p_minus`; the supplied normal chooses
which side is positive. The sheet being extracted is the **normal** load
`[p] n delta_interface`, where `n` is the fitted facet normal. Remove that sheet
from the original body force when adding the returned regular force. Tangential
sheet forces and velocity jumps are not removed by a pressure lift.
Fully prescribed velocity boundaries have zero test trace and need no traction
correction. On mixed boundaries, register the datum on the natural components
only, using the same component selection as the original natural condition.

For supplied numerical data, fill a continuous scalar mesh field whose trace
contains the prescribed jump, then pass that field as `jump_values`. Alternatively,
pass a callable accepting `(N, dim)` coordinates and returning `N` scalar values,
including an empty result when `N=0`. Values are sampled at interface vertices
and interpolated P1; this is an approximation of a non-linear jump profile.
The data are prescribed inputs, not values inferred from the computed pressure.

Use `split.physical_pressure` for pressure errors and stresses.
For a zero-mean total-pressure gauge, subtract
`(integral(q) + split.pressure_jump_integral) / volume` from the regular pressure.
The returned `pressure_jump_field` can be saved alongside the regular pressure.
Reconstruct the split after changing mesh geometry, topology or jump values.

## Finite and intersecting interfaces

Supply all interfaces together in the `jumps` and `normals` mappings. For each
mesh vertex, the construction solves for cell-side nodal values: their difference
is the prescribed jump on an interface facet and zero on every other facet.
Closed cycles must sum to zero. A common continuous P1 component is arbitrary;
the unknown `q` absorbs it. No harmonic extension is assumed.

The same construction handles closed interfaces, interfaces reaching the
external boundary, finite interior segments/patches and intersections. For this
bounded nodal lifting, the jump must vanish at a finite tip/rim and sector data
must satisfy junction compatibility. Incompatible cycles are rejected on all
ranks. Nonzero tip singularities require a different enrichment; unknown jumps
require coupled interface equations. This API keeps velocity continuous and
does not configure the split-node fault/slip machinery.

## Relation to the circular benchmark

`Stokes.cylindrical_pressure_lifting` remains the specialised exact-circle
method. It subtracts curved segments from regular-load quadrature and stores
moment-equivalent DG P2 forces. The general API instead uses the **fitted
polygon/polyhedron geometry** and P1 jump interpolation. It does not silently
apply exact-circle geometry corrections or replace existing benchmark results.

## Reference

[Chang and Kwak (2009), *Extraction method for Stokes Flow with jumps in the
pressure*](https://arxiv.org/abs/0911.4769), Sections 2--3, describe prescribed
pressure extraction and a numerical extension. UW3's vertex-star constraint
lift is a separate construction, not their harmonic extension or
Crouzeix--Raviart discretisation; their convergence claims do not establish a
rate for this implementation.
