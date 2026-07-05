# UW3 Style Charter

**Status**: Normative. Every development session — human or AI — loads this document
and follows it. It is deliberately short; there is no excuse for not reading all of it.

## 1. The Founding Rule

Anyone — specifically, any working geodynamicist — must be able to read this code and
understand it. Over-complication and inconsistent patterns are the enemies of that
rule: a clever construction that saves ten lines but costs the reader a detour has
made the code worse. When in doubt, write the plain version.

Don't forget this: Underworld3 is a python api for building mathematical models, particularly for geoscience. The founding rule applies as much to the python scripts and Jupyter notebook codes as it does to the code we write under-the-hood. We write code that makes models easy to read. We always hide complex and irrelevant detail at least one level below the notebooks / scripts. There should never be inner-workings (private _method_xxx() ) visible to the user. 

Consistent patterns, common choices are the defaults, uncommon choices are harder to find. Documentation is great: the docstrings turn into the documentation and are jupyter friendly. We do not shy away from equations in docstrings, and we do provide good, simple examples. We don't overdo it and we don't force feed our users with emoji, nor pat ourselves on the back for our cleverness. We also don't care how complicated something has been to code, or debug, we just report how it works.

## 2. Match the Charter, Not the Surrounding Code

**This is a load-bearing clause.** A month of drift means existing code is not a
reliable pattern to copy — drifted code copied becomes drift squared. You MUST follow
this Charter even where nearby code disagrees with it. When you find such a
disagreement, do not silently imitate it and do not silently "fix" it either: follow
the Charter in the code you write, and flag the existing deviation with a
`# TODO(BUG):` marker or a note to the maintainer. NEVER treat "the file next door
does it this way" as authority.

## 3. Naming

- Names state what a thing IS or DOES. No hedging prefixes — `maybe_`, `try_`, `do_`
  are banned (this rule has been enforced twice already; it keeps coming back).
- Private helpers and attributes take a single underscore prefix; everything without
  one is public API and must behave like it.
- Never name a variable `model`: use `constitutive_model` (material behaviour) or
  `orchestration_model` / `uw.get_default_model()` (serialization system).

```python
# BAD:  def _maybe_install_snes_update(self): ...
# GOOD: def _attach_snes_update_dispatcher(self): ...
```

- Usually default to the mathematical naming - surface_flux_recovery is better than compute_topography. Geographic meshes might be the main exception. 

## 4. Comments

- Comments state the physical or mathematical intent and the constraints the code
  cannot show: why this term is in the weak form, why this barrier must precede that
  collective, why a tolerance has this value.
- Never narrate mechanics (`# increment i`), never address the reviewer
  (`# as requested`), never leave editorial musings (`# why is this here??`) — file
  those as `# TODO(DESIGN):` or delete them.
- Delete commented-out code. Git remembers; the reader should not have to. Leave a TODO if there is reason to look up the deleted code for another time. 
- Every intentional exception swallow states its sanctioned failure mode on the same
  block. A bare `except Exception: pass` with no comment is a defect.
- Refer to methods by name, never by line number — line references rot within weeks.

## 5. Structure

- DRY after the 2nd occurrence: the moment you would paste a block a second time,
  extract it instead. Copy-paste-as-reuse is the single worst drift pattern found in
  the 2026-07 audit (2–6 diverging copies of the same machinery per hotspot file).
- No speculative generality: no parameters "for forward-compatibility", no reserved
  branches that `pass`, no dead flags. Build it when it is needed.
- No defensive bloat: do not guard states that cannot occur (e.g. `try/except` around
  importing a hard dependency). Guards imply the state is reachable; false guards lie.
- Before writing a helper, look for the existing one: `uw.function`, `uw.maths`, and
  `src/underworld3/utilities/` are the discovery paths. Prefer a battle-tested
  package over a hand-rolled implementation.

## 6. API Conventions

These settle the June-2026 drift (see `docs/reviews/2026-07/API-CONSISTENCY-REVIEW.md`).

| Topic | Rule |
|---|---|
| BC argument order | `add_<kind>_bc(value, boundary, ...)` — value first, matching the original trio, the published examples/benchmarks, and previous major versions (maintainer decision 2026-07-04). The newer boundary-first methods (`add_nitsche_bc`, `add_rotated_freeslip_bc`, `add_constraint_bc`) migrate to it; their current signatures are shimmed. |
| BC value parameter | ONE name across all BC methods ⟨pending maintainer confirmation: `conds` (implied by the original-order decision) vs `value`⟩. Until decided, do not introduce a third spelling. |
| Direction selection | Component masking via `None`/`sympy.oo` entries in the value vector; `direction=` only for a scalar constraint along a vector (defaults to outward normal); `normal=` strictly overrides the geometric surface-normal source. `components=` is deprecated — never in new code. |
| Solver capabilities | Anything that configures or reads one solver is a METHOD on that solver, lazily importing its `utilities/*` implementation (the `boundary_flux` pattern) — never a free function as the documented entry point. |
| Namespaces | Every user-facing module is exported from its subpackage `__init__`/`__all__` in the PR that creates it. No deep-import-only features. |
| Docstrings | NumPy/Sphinx style with RST `:math:`. This SUPERSEDES the Markdown-for-pdoc prescription still printed in `UW3_Style_and_Patterns_Guide.md` — that section is wrong; do not follow it. |

## 7. Data Access

- Variable data in new code uses the `array` property with the three-index shapes:
  scalars `(N, 1, 1)`, vectors `(N, 1, dim)`, tensors `(N, dim, dim)`.
- Mesh coordinates are `mesh.X.coords`.
- NEVER in new code: `with mesh.access(...)` / `with swarm.access(...)`, `mesh.data`,
  `mesh.points`, or the flat `.data` compatibility property. They exist only so old
  code keeps running.
- .data bypasses all units evaluation and all re-packing. It CAN be used for a raw variable to variable copy. It is likely to be working within the non-dimensionalisation boundary.

```python
temperature.array[:, 0, 0] = values      # GOOD — scalar, three indices
temperature.data[:, 0] = values          # BAD  — compatibility layer in new code
```

## 8. Tests

- Every bug fix ships the regression test that would have caught the bug — written
  first, shown to fail, then fixed.
- Test files follow `tests/test_NNNN_description.py` numbering and carry both markers:
  a level (`level_1`/`level_2`/`level_3`) and a tier (`tier_a`/`tier_b`/`tier_c`).
- Validate a new test's own correctness before changing library code to satisfy it.
- NOTE: test tiers A,B,C ... A are the hardened tests that have been explicitly reviewed. You can build code around tier A tests, but tier C are tests that are not mature enough to drive coding.  

## 9. Scope Discipline for AI Sessions

Fix what you were asked to fix. No drive-by refactors, renames, or "while I was here"
cleanups — they turn a reviewable diff into an unreviewable one and have been the
main vector of drift. When you find a real problem outside your task, mark the exact
location with the project's `# TODO(BUG): description` format (see CLAUDE.md) and
mention it in your report; do not fix it silently.

## 10. Authority

This Charter wins on any conflict. One governing document per topic:

| Topic | Governing document |
|---|---|
| Data access | `docs/developer/subsystems/data-access.md` |
| Units | `docs/developer/design/UNITS_SIMPLIFIED_DESIGN_2025-11.md` |
| Testing tiers | `docs/developer/TESTING-RELIABILITY-SYSTEM.md` |
| Branching & releases | `docs/developer/guides/branching-strategy.md` |

`UW3_Style_and_Patterns_Guide.md` remains as the detailed reference, but where it and
this Charter disagree (notably docstring format and coordinate access), the Charter wins.

## 11. Things to Remember

Underworld is a parallel code — no feature is complete if it only works in serial.

Underworld functions are a better choice than PETSc, PETSc is a better choice than MPI,
and serial libraries like scipy are usually a poor choice, generally only for
prototypes. Do not leave them in code without checking.

Examples:

- `uw.print` or `uw.timing` are better than PETSc because they wrap the best of the
  PETSc tools in a uw-styled interface. Avoid `.npz` or `.csv` logging when we have
  good parallel output that is properly flushed.
- The user should never see parallel / mpi calls. `uw.mpi.rank` or `mpi.rank` is a
  mistake — there should be a wrapper already and it will work better. Guide the users!
- Scripts and drivers written during development sessions use uw's own machinery, not
  hand-rolled equivalents. Option parsing is the canonical example: use `uw.Params` /
  `Param` (notebook-editable defaults, units-aware, `-uw_name value` CLI overrides via
  PETSc options) — not `argparse`, not ad-hoc `sys.argv` handling, not a config dict at
  the top of the file. A session script is a model a reader should be able to run and
  read; the founding rule applies to it in full.
