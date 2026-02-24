---
title: "Contributing to Underworld3"
---

# Contributing Guidelines

## Development Workflow

### Branching and Releases

See the [Branching Strategy](branching-strategy.md) for the full guide. Summary:

- **Bug fixes**: Commit to `development` (or small PR).
- **Features**: Create a `feature/*` branch from `development`, work there, submit a PR back to `development` when ready.
- **API changes**: Separate interface from implementation. Merge the interface to `development` first so all feature branches can access it.
- **Releases**: `development` merges to `main` quarterly and gets tagged.

### Setting Up for Development

See [Development Setup](development-setup.md) for environment configuration.

```bash
# Clone and build
git clone https://github.com/underworldcode/underworld3.git
cd underworld3
./uw setup

# Or manually with pixi
pixi run underworld-build
pixi run underworld-test
```

### Making Changes

1. Create a branch from `development`:
   ```bash
   git checkout development
   git pull
   git checkout -b feature/my-feature
   ```

2. Make your changes. Rebuild after modifying source:
   ```bash
   pixi run underworld-build
   ```

3. Run tests:
   ```bash
   pixi run underworld-test          # Quick (Tier A, Level 1)
   pixi run underworld-test-all      # Full suite
   ```

4. Submit a PR to `development`.

### Code Style

See the [Style Guide](../UW3_Style_and_Patterns_Guide.md) for coding conventions, naming patterns, and documentation standards.

### Commit Messages

Write clear commit messages focused on "why" not "what". For AI-assisted work:

```
Fix stale .data cache after DM rebuild with self-validating mechanism

The .data property cached a view that became invalid after DM rebuild.
Now tracks id(self._lvec) and auto-rebuilds when stale.

Underworld development team with AI support from Claude Code
```

## Pull Request Guidelines

- **Target `development`**, not `main`.
- Keep PRs focused — one feature or fix per PR.
- If your feature required API changes, those should already be on `development` (see branching strategy).
- CI must pass before merge.
- Include a brief description of what changed and why.

## Code Review

- All feature PRs are reviewed by a human and/or GitHub Copilot.
- Bug fixes on `development` may be committed directly by trusted contributors.
- Architectural changes require more thorough review — see [Code Review Process](CODE-REVIEW-PROCESS.md).

## Reporting Issues

Use GitHub Issues with the appropriate template:
- **Bug reports**: Include reproduction steps and environment details
- **Feature requests**: Describe the use case, not just the desired implementation
- **Documentation issues**: Point to what's wrong or missing

## Areas Needing Contribution

Based on architecture analysis, these areas need attention:

1. **Expressions & Functions** — User-facing documentation gap
2. **Cython Bridge** — No documentation for Python-C interface
3. **Meshing Parameters** — Geometric setup guidance missing
4. **Test Coverage** — Tier assignment for units tests (79 pending)
