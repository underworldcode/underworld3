# Style gates (CI)

The cheapest-to-check rules of the [UW3 Style Charter](../UW3_STYLE_CHARTER.md)
are machine-enforced on every pull request to `development` and `main` by
`.github/workflows/style-gates.yml`. The job is fast (no PETSc build) and runs
a single stdlib-only scanner: `scripts/check_deprecated_patterns.py`.

## What the gate checks

| Pattern id | Charter | Flags | Use instead |
|---|---|---|---|
| `access-context` | §7 | `with mesh.access(...)` / `with swarm.access(...)` | the variable's `.array` property directly |
| `mesh-data` | §7 | `mesh.data` (coordinate read) | `mesh.X.coords` |
| `hedging-name` | §3 | `def maybe_*` / `def try_*` / `def do_*` (and `_`-private forms) | a name stating what the function DOES |
| `except-pass` | §4 | `except ...:` + bare `pass` with no comment nearby | a comment stating the sanctioned failure mode |

Detection is line-based and deliberately simple; the exact heuristics (and the
documented exclusions, e.g. commented-out lines and `self.data` inside Mesh
methods) are in the scanner's module docstring. A related regression net —
`tests/test_0641_wave_c_api_shims.py` in the `level_1 + tier_a` CI gate —
covers the deprecation *warnings* the shimmed APIs must keep emitting.

## The allowlist only shrinks

`scripts/deprecated_pattern_allowlist.txt` records the legacy hits that
existed when the gate was introduced (baseline: `development` @ `9f5ed9e9`,
2026-07), one `path:pattern-id` per line. The gate fails on any hit **not**
listed there.

```{warning}
The allowlist may only **shrink**. If your PR trips the gate, fix the code —
do not add allowlist lines. Only a maintainer adds entries, and only for code
that predates the gate. When you clean up a listed file, delete its entries;
the scanner reminds you by reporting stale ones.
```

## Running locally

```bash
pixi run -e amr-dev python scripts/check_deprecated_patterns.py                # the CI gate
pixi run -e amr-dev python scripts/check_deprecated_patterns.py --include-docs # + docs/ report
pixi run -e amr-dev python scripts/check_deprecated_patterns.py --no-allowlist # full legacy inventory
```

The scanner needs nothing beyond the Python standard library, so plain
`python scripts/check_deprecated_patterns.py` works in any environment.

## When your PR trips the gate

1. Read the flagged line(s) in the job log — each hit names its pattern id and
   the Charter rule it violates.
2. Fix the code (the table above gives the replacement for each pattern).
3. If you believe a hit is a false positive of the line-based heuristic,
   say so in the PR and ask a maintainer to rule — do not edit the allowlist
   yourself.

## Formatting (not gated)

The tree is not currently `black`-clean (81 of 95 files at baseline), so there
is **no** formatting gate — adding one would fail every PR that touches
library code. Do not reformat files wholesale to "fix" this: bulk reformatting
destroys `git blame` and turns reviewable diffs into unreviewable ones. If the
maintainers later adopt a formatter, it will arrive as a single dedicated
reformat commit plus a gate, not piecemeal.

## See also

- [UW3 Style Charter](../UW3_STYLE_CHARTER.md) — the normative rules
- [Release process](release-process.md) — style gates must be green on the release branch
- `/check-patterns` (Claude command) — interactive audit of docs, notebooks and tests
