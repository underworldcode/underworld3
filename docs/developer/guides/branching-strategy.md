# Branching and Release Strategy

**Status**: Active
**Date**: 2026-02

## Principles

This strategy optimises for a small team (humans + AI assistants) working on parallel features that share a stable core. The goal is predictable releases, independent feature work, and clean integration — without bureaucratic overhead.

The key expectations are:

1. **`main` is always releasable.** It represents the current stable version.
2. **`development` is always buildable.** It may have rough edges, but tests pass.
3. **Feature branches are independent.** They don't depend on each other's uncommitted work.
4. **API changes are separable from feature implementations.** This is the discipline that makes everything else work.

## Branch Structure

```
main ────────●──────────●──────────●──── tagged releases (v3.0.0, v3.1.0, ...)
             ↑          ↑          ↑
             │          │  merge   │
             │ cherry   │          │
             │ -pick    │          │
development ─●──●──●──●─●──●──●──●──── integration (fixes, API stubs, merges)
               ↑     ↑        ↑
               │     │        │
feature/X ─────┘     │        │
feature/Y ───────────┘        │
feature/Z ────────────────────┘
```

### `main` — stable releases

- Receives merges from `development` at release time (quarterly, or when ready).
- Critical bug fixes are cherry-picked from `development` between releases.
- Every merge is tagged: `v3.0.0`, `v3.0.1` (patch), `v3.1.0` (quarterly).
- Binder launcher tracks the latest release tag.
- Protected: requires PR with passing CI.

### `development` — integration

- Bug fixes and small improvements land here via direct commit or small PR.
- Feature branches merge here via reviewed PR.
- API interface additions (stubs, new signatures) land here so all features can access them.
- CI must pass. If it doesn't, fixing CI is the top priority.

### `feature/*` — long-lived feature work

- Branch from `development`. PR back to `development` when ready.
- Reviewed by human + Copilot. CI runs on the PR.
- Periodically incorporate changes from `development` (merge or rebase, developer's choice).
- No direct API changes — see next section.

### Retired: `uw3-release-candidate`

This branch added a staging layer between `development` and `main` that wasn't being used as intended. Release candidates are now handled with tags on `development` (e.g. `v3.1.0rc1`) — setuptools-scm produces the correct version automatically.

## API Changes and Feature Independence

The hardest problem with parallel feature branches is API coupling. If `feature/darcy` adds `mesh.boundary_flux()` and `feature/faults` needs it too, the branches become entangled.

**The discipline**: API surface changes (new methods, new parameters, changed signatures) must be separable from the feature implementation that uses them.

### How this works in practice

**Before or during feature work**, when you realise an API change is needed:

1. **Design the interface** — method signature, parameters, return type, docstring.
2. **Put it on `development`** — either as a stub (`raise NotImplementedError`) or a minimal working implementation. Small PR, quick review focused on API design.
3. **All feature branches pick it up** — via their normal sync with `development`.
4. **The feature branch implements the full behaviour** behind that interface.

**After feature work**, if the API change emerged organically during implementation:

1. **Extract the interface change** into a separate commit or short-lived branch.
2. **Merge the interface to `development` first**, then rebase/merge the feature branch so its PR only contains the implementation.

The point is not to prescribe a rigid sequence of branches. It's that **when a feature PR arrives for review, the API changes it introduces should already be on `development`**. This keeps the PR focused on implementation, makes review easier, and ensures other features can access the same interfaces.

### Decision guide

| Situation | Approach |
|-----------|----------|
| Adding a parameter or changing a default | Direct commit to `development` |
| New method with clear design | Stub or minimal implementation → `development` |
| New subsystem or uncertain interface | Short-lived `api-update/X` branch for design iteration, then merge to `development` |
| Interface emerged during feature work | Extract after the fact, merge to `development` separately |

### Cross-pollination

When a fix or API change lands on `development`, the AI assistant (underworld-claude) cherry-picks it to active feature branches. This keeps feature branches current without requiring developers to manually track what changed upstream.

## Bug Fix Flow

```
Bug found
  │
  ├─ Fix on development (commit or small PR)
  │
  ├─ Critical for stable users?
  │     → cherry-pick to main
  │     → tag patch release (v3.0.1)
  │     → update binder if needed
  │
  └─ Affects active feature branches?
        → cherry-pick to each (automated by underworld-claude)
```

## Release Cadence

| Event | Frequency | Action |
|-------|-----------|--------|
| Quarterly release | ~Every 3 months | Merge `development` → `main`, tag `v3.X.0` |
| Patch release | As needed | Cherry-pick fix to `main`, tag `v3.X.Y` |
| Pre-release | Before quarterly | Tag `v3.X.0rc1` on `development` for testing |
| Binder update | Each release | Binder workflow triggers on tag push |

Version numbers are managed by setuptools-scm from git tags — see `version-management.md`.

## CI Requirements

For this strategy to work, CI must be reliable:

- **`development`**: Tests must pass. Broken CI blocks all feature merges.
- **`main`**: Tests must pass. This is the release gate.
- **Feature PRs**: CI runs automatically. Failures are the feature author's problem.

The test suite uses a tiered system (A/B/C reliability). CI runs Tier A tests as the minimum gate. See `TESTING-RELIABILITY-SYSTEM.md`.

## Branch Protection (GitHub)

### `main`
- Require PR (no direct push)
- Require CI to pass
- Require at least one review (human or Copilot)

### `development`
- Allow direct push for small fixes (trusted committers)
- PRs required for feature branch merges
- Require CI to pass on PRs

## Git Worktrees

Worktrees provide isolated working copies for parallel feature work, documentation
cleanup, or any multi-file change.  Each worktree has its own checkout of the source
tree but shares the main repo's pixi environment and PETSc build — so there is only
one set of dependencies and one (expensive) PETSc compilation.

### Why worktrees?

- **Session isolation**: Multiple Claude sessions or human editors sharing one
  working directory will overwrite each other's work.
- **Quick context switching**: Jump between features without stashing or committing
  half-finished work.
- **Clean PRs**: Each worktree has its own branch, so commits stay focused.

### Lifecycle

```bash
# Create — sets up symlinks, resets to development, names the branch
./uw worktree create viscoelasticity          # → feature/viscoelasticity
./uw worktree create mesh-fix bugfix          # → bugfix/mesh-fix

# Work — start a shell in the worktree directory
./uw worktree shell viscoelasticity
# Now you're cd'd into the worktree with pixi activated:
./uw build           # builds from THIS source into the shared env
./uw test            # runs tests
pixi run python ...  # uses the shared env
exit                 # leave worktree shell

# List — see all worktrees with branch, link status, dirty files
./uw worktree list

# Remove — cleans up worktree directory and branch
./uw worktree remove viscoelasticity
```

### How sharing works

The `./uw worktree create` command sets up two symlinks:

| Symlink | Target | Purpose |
|---------|--------|---------|
| `.pixi/` | main repo's `.pixi/` | Shared conda/pip packages |
| `petsc-custom/petsc` | main repo's PETSc build | Shared PETSc (not relocatable) |

It also copies `.pixi-env` so `./uw` knows which environment to use.

Because there is **one shared environment**, `./uw build` from any worktree installs
that worktree's source.  When you switch worktrees, rebuild to pick up the new source:

```bash
./uw worktree shell other-feature
./uw build    # now the shared env has other-feature's code
```

### Worktrees and branches

Worktrees follow the same branching conventions as regular branches:

| Prefix | Use |
|--------|-----|
| `feature/<name>` | New functionality (default) |
| `bugfix/<name>` | Bug fixes |
| `docs/<name>` | Documentation changes |

Merge to `development` via PR when ready.  The `./uw worktree remove` command
handles deleting both the worktree directory and its branch.

## For AI Assistants

When working on Underworld3:

- **Bug fixes**: Commit to `development`. If critical, note that it should be cherry-picked to `main`.
- **Feature work**: Work on a `feature/*` branch. Keep API changes in separate commits that can be extracted.
- **Worktrees**: Use `./uw worktree create` for any multi-file change. Always build and run from inside the worktree.
- **Cross-pollination**: When you see a fix on `development` that affects a feature branch you're working on, cherry-pick it.
- **Don't push to `main` directly.** Always go through `development` or a PR.
- **If CI is broken on `development`**, fixing it takes priority over feature work.

## Summary

The strategy is simple: `main` is stable, `development` integrates, features are independent, and API changes are shared infrastructure. The discipline of separating interface from implementation is what makes parallel feature development tractable. Worktrees provide the isolation needed for parallel work without duplicating the expensive build environment. Everything else follows from that.
