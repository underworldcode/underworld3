# Underworld3 AI Assistant Context

> **MANDATORY**: Read `docs/developer/UW3_STYLE_CHARTER.md` before writing any code.
> It is the normative style contract for every session, it is two pages, and it WINS
> over the surrounding code and over any other style document on conflict.
> Core clause: match the Charter, not the code next door — and flag deviations you find.

This file is a router, not a manual. It carries what a session needs *before* it
knows where to look; everything else is one hop away through the authority map in
`docs/developer/index.md`.

---

## Session bootstrap

**External planning.** If `UW_AI_TOOLS_PATH` is set (colon-separated directories),
check each for `.md` files — `underworld.md` especially — and report relevant
Active/Bugs items briefly. If it is unset, proceed with repo-local context. Set it
via `./uw setup`.

When you complete a task from an external planning file, annotate it in place:

```markdown
<!-- PROJECT RESPONSE (YYYY-MM-DD underworld3):
What was done, and which files changed.
-->
```

Add newly discovered work to the external planning file under the appropriate
section with a `<!-- project:underworld3/subsystem -->` tag — not to a local TODO
file, and not here. Don't rewrite strategic paragraphs, move items between
sections, or restructure the document; those carry cross-project context and are
handled by the planning tools.

**Inline TODOs** mark the *place*; the planning file tracks the *work*:

```python
# TODO(BUG): add_natural_bc() causes PETSc error 73
# The Stokes solver works; issue is specific to scalar Poisson setup.
# See planning file: underworld.md (Bugs section, 2026-01-19)
```

---

## Hard constraints

**Rebuild after every source change.** `./uw build`. Underworld3 is installed into
the environment, not imported from `src/`; verify with `uw.__file__`.

**Never use an editable install.** `pip install -e .` contaminates every pixi
environment sharing the source tree and persists after uninstall. No exceptions.

**Never move `/Users/lmoresi/+Underworld/underworld-pixi-2/petsc/`.** PETSc
hardcodes paths at configure time; moving it costs a full rebuild.

**Worktrees are never on `development` or `main`.** All work happens on a side
branch (`feature/…`, `bugfix/…`, `docs/…`) and merges via PR. Use a worktree for
any multi-file change — concurrent sessions sharing one checkout overwrite each
other. Each worktree builds into its own environment, so enter it *first*, then
build.

Detail: [`guides/development-setup.md`](docs/developer/guides/development-setup.md),
[`guides/branching-strategy.md`](docs/developer/guides/branching-strategy.md).

**Attribution.** End commit messages and PR bodies with:

```
Underworld development team with AI support from Claude Code
```

(PR bodies may link it.) Do not use `Co-Authored-By:` with a noreply address, and
no emoji in PR descriptions.

**Solver stability is paramount.** The PETSc-based solvers in
`petsc_generic_snes_solvers` are carefully tuned and validated. No changes
without extensive benchmarking.

**The JOSS paper (`publications/joss-paper/`) is frozen.** It is the publication of
record; do not modify it.

---

## Where things go

| Content | Location |
|---|---|
| Subsystem documentation | `docs/developer/subsystems/` |
| Architecture and design decisions | `docs/developer/design/` |
| How-to guides | `docs/developer/guides/` |
| User tutorials | `docs/beginner/tutorials/` |
| Advanced user guides | `docs/advanced/` |
| Session scripts, benchmarks, profilers | `scripts/sessions/` |
| Simulation output worth keeping | `~/+Simulations/` |

**All documentation goes under `docs/`** — never the repository root, `src/`, or
`tests/`. MyST Markdown for Sphinx: ` ```python ` blocks, `{note}`/`{warning}`
admonitions, `$inline$` and `$$display$$` math. Verify with `pixi run docs-build`.

Plan files in `~/.claude/plans/` take descriptive kebab-case names that say what
they are about (`mesh-adaptation-architecture.md`), never whimsical ones.

---

## Rulings a session needs in hand

**Free-slip: prefer rotated strong free-slip.**
`solver.add_rotated_freeslip_bc(conds, boundary, normal=None)`, value first
(`conds=0` is free-slip). It enforces `v·n̂ = 0` to machine precision where
Nitsche/penalty leaks ~1e-3, is correct on curved and deformed boundaries, works
inside the nonlinear SNES and with geometric FMG, and its reaction is the boundary
normal traction. Leave `normal=None` unless the constraint must follow the true
surface rather than the mesh. Reserve `add_nitsche_bc` for BCs that must *evolve in
time* — a hard rotated constraint cannot morph.
Governing doc: [`subsystems/rotated-freeslip.md`](docs/developer/subsystems/rotated-freeslip.md).

**Data access.** New code uses `.array` with three-index shapes — scalars
`(N,1,1)`, vectors `(N,1,dim)`, tensors `(N,dim,dim)` — and `mesh.X.coords` for
coordinates. `with mesh.access(...)`, `with swarm.access(...)`, `mesh.data` and
`mesh.points` exist only so old code keeps running. The flat `.data` has exactly
one sanctioned use: a raw variable-to-variable copy inside the
non-dimensionalisation boundary. Solver internals use `vec`.
Governing doc: [`subsystems/data-access.md`](docs/developer/subsystems/data-access.md).

**Unwrap before extracting atoms.** UWexpressions hide coordinates, so order
matters:

```python
expr = _unwrap_for_compilation(expr, keep_constants=False, return_self=False)
symbols = expr.atoms(...)      # only now
```

**Never name a variable `model`.** Use `constitutive_model` (material behaviour) or
`orchestration_model` / `uw.get_default_model()` (serialization). Two unrelated
concepts share the word.

**Units.** Accept strings, store and return Pint objects: `uw.quantity(1e21, "Pa*s")`.
`.units` returns a Pint *Unit*; call `.to("m")` on the Quantity, not on `.units`.
Governing doc: [`design/UNITS_SIMPLIFIED_DESIGN_2025-11.md`](docs/developer/design/UNITS_SIMPLIFIED_DESIGN_2025-11.md).

**Parallelism.** PETSc handles synchronisation — avoid direct mpi4py. Use
`uw.pprint()` and `uw.selective_ranks()`; a user should never see an MPI call.

**Swarms.** Migration moves particles between processors automatically; batch with
`migration_disabled()`. Swarm variables with `proxy_degree > 0` carry a proxy mesh
variable that must be refreshed (`swarmVar._update()`) when data or positions move.

**Prefer Glob and Grep over `find`/`grep` in Bash** — safer, faster, no approval
prompt.

**Desktop notification** from a background monitor — be quiet unless something
needs attention:

```bash
osascript -e 'display notification "msg" with title "title" sound name "Glass"'   # macOS
notify-send "title" "msg"                                                         # Linux
```

---

## Tests

Files are `tests/test_NNNN_description.py` and carry both a level
(`level_1`/`level_2`/`level_3`) and a tier (`tier_a`/`tier_b`/`tier_c`).

```bash
pytest -m "level_1 and tier_a"     # quick
pytest -m "tier_a or tier_b"       # full validation
```

Tier A is hardened and reviewed — safe to build code around. Tier C is not mature
enough to drive coding. Every bug fix ships the regression test that would have
caught it, written first and shown to fail. Validate a new test's own correctness
before changing library code to satisfy it.

Every test file must be reachable from a glob in `scripts/test.sh`;
`scripts/check_test_coverage.py` enforces it.

---

## Finding the rest

`docs/developer/index.md` carries the **authority map** — one governing document
per topic. Read it rather than guessing which of several documents on a topic is
current. Development history and completed migrations are in
`docs/developer/ai-notes/historical-notes.md`.

Key implementation files:

| | |
|---|---|
| `src/underworld3/mpi.py` | parallel-safe output |
| `src/underworld3/scaling/` | units system |
| `src/underworld3/function/expressions.py` | UWexpression, lazy evaluation |
| `src/underworld3/function/_function.pyx` | mesh-variable symbols |
| `src/underworld3/utilities/mathematical_mixin.py` | mathematical objects |
| `src/underworld3/utilities/rotated_bc.py` | rotated free-slip |
| `src/underworld3/discretisation/enhanced_variables.py` | units, math ops, persistence |
