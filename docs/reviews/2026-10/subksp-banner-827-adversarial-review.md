# Quiet sub-solver query (#827): adversarial review record

Branch `bugfix/subksp-banner-827`, off `development`. One independent reviewer,
run against the installed build of the branch, with an older feature branch
carrying the same unguarded call sites as the unfixed reference.

## What the branch does

`_fieldsplit_sub_ksps(pc)` (`utilities/_petsc_tools.py`) returns a field split's
block solvers, or `[]` when the split is not set up, with PETSc's error banner
silenced for that one call (the `"return"` handler: the error is still raised, not
printed; popped on every path). Two call sites use it:

| Site | Path that printed the banner |
|---|---|
| `SNES_Stokes._check_velocity_preconditioner` | the viscoelastic solve ran the check before solving (the #827 reproducer); a plain warm solve that converged without a linear solve after a `consistent_jacobian` change rebuilt the solver |
| `custom_mg._pcmg_still_installed` | the pre-solve "still installed?" test on a solver rebuilt by a `consistent_jacobian` change; every notch run with a non-nested FMG tail printed it |

The viscoelastic check now runs after the solve, as on the plain path.

## Findings and their state

- **The plain-path site had no test** (medium). A loud helper in `systems.solvers`
  passed every test. Fixed: `test_warm_solve_without_a_linear_solve_prints_no_banner`;
  12 banner lines unfixed, 0 fixed.
- **Viscoelastic solves now warn and record a fallback** (behaviour change, kept).
  Before, the pre-solve check always returned early, so a viscoelastic solve on a
  mesh without a hierarchy never reported running GAMG. It now warns (once per
  process under the default filter) and records `velocity.fell_back_from_fmg`, as a
  plain solve does. Separately and pre-existing: `stacklevel=3` resolves to
  `memprobe.py`, so every solve shares one warning location.
- **The helper reached the `uw` namespace** through `from .utilities._petsc_tools
  import *`. Fixed: renamed `_fieldsplit_sub_ksps`.
- **Second copy of the pattern** in `petsc_generic_snes_solvers.pyx`
  (`_discard_preconditioner_setup`). Left: the solver core is not changed in a
  bugfix; a follow-up can route it through the helper.
- **Docstring** told the debugging history and lacked Parameters/Returns. Fixed.

## Attacks that failed

- Handler semantics: on an un-set-up Schur split the helper returns `[]` silently;
  a direct call afterwards still raises and prints, so the stack is restored; a
  `KeyboardInterrupt` inside propagates with the stack restored; 1000 calls leave it
  intact; after set-up it returns 2 blocks.
- Parallel (np 2 and 4): helper before/after set-up, the viscoelastic reproducer,
  3 viscoelastic steps, the custom-FMG rewire. Exit 0, no banner, no hang. Set-up
  state follows `PCSetUp`, which every rank runs; the not-set-up error is a local
  check, so no rank takes a different branch.
- `_pcmg_still_installed` returns `False` for an un-set-up split as before (it used
  to raise and be caught); no extra re-installs.
- Other `getFieldSplitSubKSP` calls (`custom_mg.py` transfer install,
  `rotated_bc.py`, `solver_health.py`, the solve-config dump) all follow a
  `setUp`, run inside a solve, or are already guarded.
- No import cycle: `_petsc_tools` imports only petsc4py.
- Tests: all three regression tests fail on the unfixed code on the banner
  assertion and pass fixed; `test_107*` is in `scripts/test.sh`;
  `check_test_coverage.py` reports every file reachable.

## Suites

Related suites on the fixed build (multigrid, custom-MG hierarchy, rotated
free-slip, MG option bundle, restore/continuation, viscoelastic): 77 passed, no
banner. The reviewer's wider run (388 tests, `not tier_c`): 387 passed, 1 failed:
`test_0017_describe_and_render.py::test_every_family_describes_itself_at_the_class_level`,
order-dependent and unrelated. `test_1057_ddt_transport_plugin.py` defines a
solver subclass `TensorTransport` with no summary, and the class-level description
test enumerates every subclass, so it fails whenever test_1057 ran first in the
same process.

Underworld development team with AI support from Claude Code
