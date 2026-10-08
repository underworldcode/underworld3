# Coupled volume fields in Stokes: adversarial review record

Branch `feature/shear-band-length-scale`, commits `003d3b95` (`add_coupled_field`) and
`6f0627ce` (`add_coupled_dirichlet_bc`, the shared essential-BC helper, the Cosserat
shear-layer test), written overnight on Hyperion (Linux, conda-forge PETSc). Reviewed on
the Mac (`runtime` environment) against the installed build of those commits.

## What the branch does

`SNES_Stokes_SaddlePt.add_coupled_field(field, F0, F1, momentum_flux)` adds an extra
volume field to the Stokes DM. Its weak form is `∫ F0 ψ + F1·∇ψ = 0`, and it is solved
in the same Newton system as velocity and pressure. Every Jacobian block touching the
field is derived from the residuals, through `_jacobian_source`. `momentum_flux` is added
to the momentum flux (a Cosserat skew stress). Every new path is guarded on
`_coupled_fields`, so an ordinary Stokes solve registers and compiles exactly what it did
before.

The driver behind it is the notch length scale: a micromorphic (gradient-plasticity)
field and a 2-D Cosserat micro-rotation. Both converge on the notch at ref 3 with no rate
strengthening; the campaign report is `notch_length_scale/REPORT.md` in uw3-campaigns.

## Checks that hold

- **Ordinary Stokes C is byte-identical.** The fingerprint script
  (`notch_length_scale/drivers/stokes_c_md5.py`) hashes the header of three ordinary
  solvers: isoviscous Picard, Drucker-Prager Newton, and block-constrained. On the Mac it
  reproduces Hyperion's branch and base values exactly
  (`655ff38a…`, `d5a5ba4b…`, `4fc089ca…`), including after the fixes below.
- **The generated C indexes the coupled field correctly.** `petsc_u[3]` is χ and
  `petsc_u[2]` is p, against DS components `[2, 1, 1]` (offsets 0, 2, 3).
- **Residual slots are wired to the matching compiled functions.**
- **test_1070 passes at np = 3 on every rank.**

## Findings and their state

1. **The two-way Taylor test failed on the Mac (fixed: test).** Remainder slopes were
   0.11 and 0.26. The χ and pressure rows had slope 2; the velocity row did not respond
   to pressure at all. The cause is the fixture, not the plumbing:
   - The box is closed on all four walls, so the pressure is defined only up to a
     constant, and no null space was declared.
   - MUMPS LU of the singular system let that constant drift to −2.3e14 over the Newton
     steps. At that magnitude adjacent doubles are about 0.03 apart, so the residual cannot
     be resolved below |F| ≈ 2.
   - The solve then stopped on step size (`SNES_CONVERGED_SNORM_RELATIVE`) without
     converging, and the test never checked.
   - Hyperion's MUMPS kept the constant small, so the test passed there.

   Fix: the fixture declares `petsc_use_pressure_nullspace`, and the test asserts
   |F| < 1e-8 before the Taylor steps. The negative control (frozen tangent) still fails,
   at slope 1.00.
2. **Rotated free-slip with a coupled field stopped at iteration 0 with no
   explanation (fixed: refusal).** The rotated driver splits velocity and pressure by field
   number, so a coupled field lies outside its split; this is the #464 situation the
   multipliers are already refused for. `_reject_rotated_with_coupled_fields` now refuses
   the pair in `add_coupled_field`, `add_rotated_freeslip_bc`, `add_fault_bc` and
   `solve`. Tested in both orders.
3. **`momentum_flux` was silently dropped by solvers that build their own `F1`
   (fixed: refusal).** Only the `SNES_Stokes` template adds it. `SNES_NavierStokes` and
   `SNES_NavierStokes_Composed` do not. The class flag `_F1_carries_coupled_momentum_flux`
   (True on `SNES_Stokes`, False on the composed Navier-Stokes) makes `add_coupled_field`
   raise instead. Tested.
4. **The same field could be coupled twice (fixed: refusal).** That would have registered
   two DM fields for one variable. Tested.
5. **The default solver path was never exercised: every test and run used exact LU
   (fixed: test).** It converges for two-way coupling in 6 Newton steps to |F| = 8e-9.
   Its split 0 is velocity plus coupled fields under the velocity multigrid, and split 1
   is pressure. Pinned by a test. Hyperion has since run FMG on the notch: same Newton
   counts as LU when every linear solve is tight.
6. **The stale strict xfail in test_1064 is pre-existing** (XPASS on the base
   `f57c2a27` too). The branch's `TODO(BUG)` comment there is reverted: it is outside this
   change and needs its own decision.
7. **`add_natural_bc`'s value is minus the traction (pre-existing; docs).** It enters
   PETSc's boundary f0 as given, and the docstring says "traction". The Cosserat test
   passes the negated traction, with a comment. Not changed here.

8. **OPEN: the default fieldsplit solve with a coupled field HANGS at np = 3.**
   `test_two_way_coupling_converges_with_the_default_fieldsplit_solver` passes serially
   (6 Newton steps) and was still running after 120 s at np = 3. The other seven
   test_1070 tests pass at np = 3, including the two-way exact-LU solve, so the hang
   lies in the parallel fieldsplit/multigrid path with velocity and the coupled field
   in split 0. Not yet localised. CI runs test_1070 serially, so it does not hang CI;
   it must be fixed or refused before coupled fields are used in parallel.

## Not covered

- Coupled fields at np ≥ 3 on the notch.
- FMG with a multi-component coupled field (3-D Cosserat).
- `add_coupled_dirichlet_bc` with the units model active: numbers are taken as
  non-dimensional, as documented.
