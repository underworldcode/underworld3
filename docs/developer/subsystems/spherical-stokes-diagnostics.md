# Spherical Stokes diagnostics and observers

Reusable finite-element diagnostics live in `uw.diagnostics`. Benchmark drivers
supply analytical reference callables, physical radii and quadrature orders;
they retain campaign policy, metric naming, provenance and output files.

- `spherical_stokes_errors` integrates velocity/pressure errors, wall flux and
  stress errors on the physical shell, including continued inner traces.
- `spherical_stokes_field_differences` compares finite-element fields using those
  same integration rules.
- `spherical_stokes_reactions` integrates scalar or mapped reaction fields against
  `HarmonicTarget` values. Reaction signs and pressure-gauge conventions remain
  explicit inputs.
- `runtime_info` and `collective_file_sha256` provide library/runtime facts and
  collective shared-input hashing. They do not select a benchmark source checkout.

Use solver methods for solver state and residual verification:

```python
snapshot = stokes.rotated_stokes_diagnostics(rtol=1e-9)
original_residual = stokes.verify_linear_stokes_residual(rtol=1e-9)
```

These methods verify the original equations after finalization. A projected or
preconditioned residual is not substituted for the full original residual.
Requested tolerances and observed errors remain distinct quantities.

## Optional observations

Subclass `uw.diagnostics.SphericalStokesObserver` and register it for one solve:

```python
class ResidualObserver(uw.diagnostics.SphericalStokesObserver):
    def solve_finished(self, ksp, vector, state):
        physical = state.gauged_copy(vector)
        try:
            self.last_residual = state.residual(physical)
        finally:
            physical.destroy()

observer = ResidualObserver()
with stokes.observe_stokes(observer):
    result = stokes.solve_spherical_gap(...)  # Existing explicit solve arguments.
```

Callbacks are collective and borrow PETSc objects until the solve returns. They
may inspect those objects or retain owned copies; they must not mutate/destroy
operators, RHS vectors, settings or iterates. `SphericalSolveState` owns the native
pressure/rotation gauge formulas, original residual block norms, layout metadata
and field conversion. Callers destroy vectors returned by `build_iterate` and
`gauged_copy`. `finalized(result)` runs after native residual checks and field
assignment; intermediate checkpoints remain algebraic iterates, not accepted fields.

The optional phase and preconditioner callbacks support application-owned timing
and checkpoint files. Registration restores the prior observer attribute even on
exceptions. Only the observer's own Python KSP monitor is removed; existing Python
and native PETSc monitors remain. The observer checks native function identity and
source bytes without parsing, rewriting or executing generated solver source.
Ordinary solves incur no additional matrix copies, assembly or residual evaluation.

## Validation scope

Tests 1119 and 1120 cover closed-form volume/moment/harmonic integrals, affine FE
fields, reaction signs/gauges, singular algebraic original residuals, monitor and
registration ownership, shared hashes and platform memory units. Test 1104 attaches
the public observer to actual free-slip and zero-slip hydrostatic solves. All are
serial/MPI gates; none constitutes an inf-sup or campaign acceptance claim.

## Internal ownership after the shared Stokes cleanup

The solver implementations no longer depend on `uw.diagnostics`. Shared FE
numbering/scatter operations live in `utilities/_field_layout.py`. Shared borrowed
mixed state and residual measurements live in `_stokes_state.py`; physical gauges
and field assignment live in `_cylindrical_stokes_state.py` and
`_spherical_stokes_state.py`. The diagnostics namespace re-exports the historical
spherical state/replay spellings for compatibility; it does not own their solver
implementation.

`Stokes.observe_stokes` and `uw.diagnostics.StokesObserver` provide the same scoped
solve-start/iteration/solve-finish/finalized contract for both gap solvers. Additional
physical/background/PC/phase events are backend-specific. The older spherical
registration spelling and source-integrity observer remain supported. Nested
registration across either spelling is rejected, and existing PETSc monitors
survive exceptions.

Annulus basis/cap/volume rules are separated into `_cylindrical_gap_geometry.py`;
distributed graph allocation and insertion live in `_cylindrical_gap_assembly.py`.
The solve module orchestrates assembly, physical gauges, verification and field
finalization. Spherical cut-cell/trace/curtain/preconditioner mathematics retain
their separate geometry modules. This aligns responsibilities without combining
2D circle and 3D sphere formulas or changing integration/convergence settings.

Tests 1100 and 1104 attach the common API to actual free-slip and zero-slip solves
in both geometries. Test 1120_stokes_architecture checks exact mixed gauges,
registration ownership and the solver-to-diagnostics dependency boundary.
