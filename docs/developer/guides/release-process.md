# Release Process: Maturity-Gated Quarterly Promotion

**Status**: Active
**Date**: 2026-06

This guide describes how finished work is promoted from `development` to `main`
*with confidence*. It complements `branching-strategy.md` (which covers how work
lands on `development` in the first place) and `version-management.md` (tags and
versioning).

## Overview — the "we can be sure it works" guarantee

`development` is a busy integration trunk — at any time it is hundreds of commits
ahead of `main` and carries many features at different stages of maturity. The
goal is **not** to surgically isolate one feature's commits onto `main`. That
fights the grain: a feature like the MMPDE mover depends on the Winslow smoother,
boundary-slip surfaces, FMG infrastructure, and field-transfer — its dependency
closure is most of `development` anyway.

Instead:

> `main` accumulates whatever stable work comes over from `development` each
> quarter. The **release notes** — not branch surgery — distinguish features that
> are *validated and guaranteed* from features that are merely *present but not
> guaranteed to work*.

So the mover's code can be on `main` while the release simply does **not** claim
it works. FMG can be announced as supported while the mover rides along as
preview. This matches how the project actually develops.

## Maturity definitions

Every shippable feature is announced at one of three maturities:

| Maturity | Meaning | Release-notes section |
|----------|---------|-----------------------|
| **supported** | `tier_a`/`tier_b` tests exist and pass on the release candidate. Guaranteed to work. | *Supported (validated)* |
| **preview** | Code is on `main`, but the release does **not** guarantee it works. | *Preview (present, unguaranteed)* |
| **experimental** | Present in the tree, not announced as working. | *Preview* (tagged `experimental`) |

The announced maturity is `min(claim, tests_maturity)`:

- An owner may be deliberately **cautious** — claim `preview` even though the
  tests pass (this is the mover today).
- The gate may **downgrade** — claim `supported` but a test fails, or there is no
  validation suite, so it drops to `preview`/`experimental`.

The gate **never blocks the `development → main` merge**. The code ships
regardless; only the announcement changes.

## The feature manifest

Features are declared in **`docs/release-notes/feature-manifest.yaml`**. Each
entry scopes *which tests belong to a feature*; the existing tier markers scope
*which of those tests are trustworthy*. There is **no per-feature pytest marker**
to maintain.

```yaml
features:
  - key: units-system
    title: "Units and Scaling System"
    owner: "@lmoresi"
    claim: supported                 # what the owner wants to announce
    summary: >
      Pint-backed dimensional quantities and unit-aware arithmetic.
    validation:
      paths:                          # test files/globs that scope the feature
        - tests/test_0700_units_system.py
        - tests/test_0710_units_utilities.py
      select: "..."                   # optional pytest -k to narrow
      markers: "tier_a or tier_b"     # default; ties "supported" to validated tiers
      levels: "1,2,3"                 # optional level filter (cost control)
    docs:
      - docs/developer/design/UNITS_SIMPLIFIED_DESIGN_2025-11.md
```

### Worked example: FMG vs the mover

- **FMG** claims `supported` but has **no dedicated `tier_a` test** yet. The gate
  finds an empty selection and downgrades it to `experimental`. That is the
  signal you want: *FMG cannot be announced as supported until it has a
  validation suite.* Add a `tier_a` test and the row goes green.
- **The MMPDE mover** has passing `tier_a` tests, but its owner claims `preview`
  because it is not yet trusted across production problems. It announces as
  `preview` — code on `main`, not guaranteed. This is the canonical case.

## The validation gate

`scripts/release_gate.py` reads the manifest and, per feature, builds a pytest
invocation equivalent to:

```bash
pytest --config-file=tests/pytest.ini <expanded paths> \
       -m "(tier_a or tier_b) and (level_1 or level_2 or level_3)" \
       -k "<select>"
```

It then resolves `tests_maturity` from the result:

- non-empty selection under `tier_a or tier_b` **and** all pass → `supported`
- tests ran but some failed → `preview`
- empty selection (no `tier_a`/`tier_b` tests — e.g. only `tier_c`), all selected
  tests skipped, no tests at all, or a pytest execution error → `experimental`

"Supported" is therefore tied to the validated tiers *by construction* — you
cannot announce a feature as supported without `tier_a`/`tier_b` tests that pass.

Run it any time (intended on `development`):

```bash
./uw dev release-check          # quick dry-run, levels 1,2
./uw dev release-check 1,2,3    # full
```

## Quarterly checklist

Run `./uw dev release` from the **main checkout on `development`**. Every
state-changing step is confirmed; the tool stops **before** the production tag so
`main`'s branch protection (PR + review) stays intact.

| # | Step | Automated? |
|---|------|------------|
| 1 | Preflight: on `development`, clean tree, fetched, ahead of `main` | auto |
| 2 | Pick the next version (`vX.Y.0`) | confirm |
| 3 | Tag + push the RC on `development` (`vX.Y.0rcN`) — RC tags are allowed here | confirm |
| 4 | Full validation: `test_levels.sh 1,2,3 --isolation` + the per-feature gate | auto |
| 5 | Aggregate results → achieved maturity per feature (results to `$TMPDIR`) | auto |
| 6 | Scaffold `docs/release-notes/vX.Y.0.md` with auto Supported / Preview sections | confirm |
| 7 | Open the `development → main` PR with the rendered notes | confirm |
| 8 | **After the PR merges**: `git tag vX.Y.0 && git push` → CI publishes the release | manual |

Step 8 is deliberately left to a human so the review gate on `main` is never
bypassed. CI (`.github/workflows/release.yml`) publishes the GitHub Release from
the committed `docs/release-notes/vX.Y.0.md`.

The CI style gates must also be green on the release branch — the
`development → main` PR (step 7) runs `.github/workflows/style-gates.yml`
automatically; see [style-gates.md](style-gates.md) for the checks and the
shrink-only allowlist policy.

### Documentation freshness sweeps (before step 6)

Two "pull" documents go stale silently between releases; refresh both as part
of every release cycle:

1. **Docstring review queue** — regenerate and commit
   `docs/docstrings/review_queue.md` (+ `inventory.json`):

   ```bash
   python scripts/docstring_sweep.py 'src/underworld3/**/*.py' 'src/underworld3/**/*.pyx'
   ```

   A stale queue misdirects documentation work both ways: it re-flags items
   that have since been documented and omits API added since the last sweep.

2. **Quarterly changelog** — sweep the merge history since the changelog's
   last entry and backfill `docs/developer/CHANGELOG.md` at its conceptual
   granularity (grouped by subsystem, not per PR):

   ```bash
   git log --first-parent --oneline <last-changelog-commit>..development
   ```

Both documents rotted over the January–June 2026 development burst (findings
DOC-02 / DOC-03 in `docs/reviews/2026-07/DOCS-STANDARDS-COHERENCE.md`); this
checklist item exists so that cannot happen unnoticed again.

## Release notes generation

Steps 4–6 auto-populate two sections of the notes from the gate:

- **Supported (validated)** — features whose achieved maturity is `supported`.
- **Preview (present, unguaranteed)** — everything else that is announced
  (`preview`, and `experimental` tagged as such).

You then hand-write **Highlights** and **Contributors**. The template lives at
`docs/release-notes/TEMPLATE.md`.

## Branch hygiene

The other half of staying organised is not letting merged branches pile up. Run
before each release (or whenever the branch list feels tangled):

```bash
./uw dev branch-hygiene            # report: merged / open-PR / stale
./uw dev branch-hygiene --prune    # delete merged branches (confirms each)
```

It lists branches merged into `development` (safe to delete), branches with open
PRs (never pruned), and stale branches (unmerged, no PR, >90 days). `--prune`
deletes only merged-and-no-open-PR branches, confirming each.

## Knowing what's available to release

Before a release, find out which features have actually landed on `development`
since the last one — the pool you can choose to announce:

```bash
./uw dev feature-audit              # feature-branch PR merges since last release
./uw dev feature-audit --all        # also include bugfix/ and docs/ merges
./uw dev feature-audit --since v3.0.1
```

It lists each merged `feature/*` PR (date, number, branch) and reminds you how
many features the manifest currently declares. Use it to decide which merges
become manifest entries (and at what maturity) for the upcoming release. It
detects PR-*merge* commits only — squash-merged PRs won't appear.

This is the feature-level counterpart to the test-level drift check below:
`feature-audit` answers "what merged that we might announce", while
`check_manifest_coverage.py` answers "what validated test isn't claimed".

## Keeping the manifest honest

The main risk is **drift** — a `tier_a` test file that no feature references, so
its coverage is invisible to the gate. `scripts/check_manifest_coverage.py` (run
in `development` CI) warns when a `tier_a`-marked test file is not referenced by
any feature's `validation.paths`. Start as a warning; promote to an error once
coverage is established.

## See also

- `docs/developer/guides/branching-strategy.md` — how work lands on `development`
- `docs/developer/guides/version-management.md` — tags and `setuptools-scm`
- `docs/developer/TESTING-RELIABILITY-SYSTEM.md` — the tier A/B/C system
