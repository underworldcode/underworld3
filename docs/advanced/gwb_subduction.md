# Importing a GWB subduction model into Underworld3

This example initializes a fixed two-dimensional Underworld3 Stokes model
from the Geodynamic World Builder (GWB) v1.1.0
`simple_subduction_2d_cartesian` cookbook. It demonstrates the transfer of
GWB temperatures and compositions to a UW3 mesh and particle swarm, then
compares a coupled slab with an illustrative weak slab-top band.

The example source is in `examples/gwb_subduction/`. The underlying numerical
workflow was tested as a single-rank proof of concept. It is not a transient
subduction simulation, an actual fault-contact formulation, or a demonstration
of finite-element mesh convergence.

## Dependencies and provenance

Underworld3's normal Pixi environment provides the Python-side solver and
numerical dependencies. GWB is a separate, optional external application.
The validated setup used GWB **v1.1.0**, checked out at its release tag, with
the standalone **`gwb-dat`** executable. No GWB Python wrapper is needed.

The starting geology and temperature model are taken from GWB's official
`cookbooks/simple_subduction_2d_cartesian/simple_subduction_2d_cartesian.wb`.
The included `four_material_subduction.wb` is the derived input, not a
replacement for that original cookbook file. The example checks that the
model generated from the cookbook matches the included reference exactly.

From the UW3 repository root, configure the GWB checkout and output directory:

```bash
export GWB_ROOT="$HOME/REPOSITORIES/WorldBuilder"
export GWB_DAT="$GWB_ROOT/build-uw3py/bin/gwb-dat"
export GWB_UW3_OUTPUT_DIR="$PWD/outputs/gwb_subduction"
```

The `GWB_SUBDUCTION_SOURCE` environment variable may instead specify the
location of the original GWB cookbook. The script checks that the GWB Git
checkout is at the v1.1.0 release tag. To use a release archive without a Git
checkout, that provenance check must be adapted and revalidated first.

## Verification sequence

Run the fast geometry and I/O regression tests first:

```bash
pixi run python -m pytest examples/gwb_subduction/tests -q
```

Next, run the actual GWB queries and UW3 Stokes solves (one MPI rank):

```bash
pixi run python examples/gwb_subduction/gwb_uw3_subduction_demo.py
```

Finally, audit the recorded fields, solver decisions, and scientific outputs
against the fixed reference:

```bash
pixi run python examples/gwb_subduction/verify_workflow.py "$GWB_UW3_OUTPUT_DIR"
```

The run writes `verification.json`, `demo_fields.npz`, the GWB query files,
a slab-top CSV, and three PNG figures. The post-run audit checks the actual
saved arrays, not just success messages printed by the solver. The numerical
reference comparison allows a 1% relative change in the reported maximum
speeds and full-field difference; exact integer counts and field-transfer
invariants remain required.

## Modeling choices

The model covers an 8,000 km by 1,600 km Cartesian region. Its base structured
mesh is 80 by 40 elements with one refinement level, giving a final 160 by 80
mesh and 13,041 P1 temperature degrees of freedom. The reference population
contains 115,200 particles, with four geological material labels. The GWB
temperature field is nondimensionalized as `(T - 1573 K) / 1300 K`.

The incoming plate, descending slab, and overriding plate are assigned distinct
composition IDs in a copy of the cookbook input. In the **specific verified
GWB v1.1.0 two-dimensional `gwb-dat` output**, these IDs are read from the
columns labeled `c2`, `c3`, and `c4`. This indexing is a version- and
query-format-specific workaround. Do not assume it applies to every GWB
version or interface: the query function checks the complete expected header
and record dimensions.

The slab top is reconstructed from cookbook dip segments with an increment
of at most 0.25 km along the arc. The separate weak band is 40 km wide and
limited to the upper 200 km depth. It is defined using distances to sampled
slab-top points, not a discontinuous internal boundary condition. The band
serves only to replace viscosity in the illustrative weak-interface case.

The nondimensional bulk viscosities are 1, 5, 10, and 5, relative to
`1e21 Pa s`. The weak zone uses `1e20 Pa s`. The slab density anomaly is
`50 kg/m³`; buoyancy also depends on the transferred temperature. Two
instantaneous Stokes systems use identical geometry and boundary conditions:
rotated free-slip on left, right, and top; no-slip at the base. The solution
requires a converged KSP and the custom FMG preconditioner without fallback.

The reported strain-rate values come from finite differences of the solved
velocity on the structured Q2 nodes. They are a **nodal kinematic proxy**, not
finite-element strain integration, a prescribed plate convergence rate, or
fault slip. This distinction matters when interpreting a stronger weak-band
signal.

## Archived reference

The included `reference_verification.json` records these historical outcomes
for the fixed demonstration:

| Quantity | Coupled | Weak interface |
|:--|--:|--:|
| Maximum velocity (cm/year) | 21.7031 | 24.6207 |
| Band RMS strain proxy (1/s) | 1.1839e-14 | 8.9500e-14 |
| Solver convergence | Yes | Yes |
| Velocity preconditioner | Custom FMG | Custom FMG |

The weak-band strain proxy increases by approximately 7.56 times. Temperature
transfer and tag-preservation differences were zero in the archived run, and
all material mask comparisons matched. The full velocity-field relative
change was approximately 0.3354.

These are single-configuration numerical results. The existing geometry
assessment also identified resolution dependence of the finite-width weak
band and requested visual review. Neither mesh convergence nor robustness to
alternative rheologies follows from this example.

## References

Fraters, M. et al. (2019), *The Geodynamic World Builder: a solution for
complex initial conditions in numerical modelling*, Solid Earth,
<https://doi.org/10.5194/se-10-1785-2019>.

Fraters, M. et al. (2024), *The Geodynamic World Builder: A planetary
structure creator for the geosciences*, Journal of Open Source Software,
<https://doi.org/10.21105/joss.06671>.

Geodynamic World Builder v1.1.0 (2026),
<https://doi.org/10.5281/zenodo.19222670>.
