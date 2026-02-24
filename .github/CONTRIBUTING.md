# Contributing to Underworld3

For contributions of Underworld models, see https://github.com/underworld-community

---

We welcome contributions in the form of:

- Code improvements or new functionality
- Bug reports and bug fixes
- Suggestions and feature requests
- Documentation improvements

## Quick Start

1. Fork the repository and clone it
2. Create a branch from `development`
3. Make your changes, add tests if applicable
4. Push and submit a PR to the `development` branch

## Full Guidelines

See the [Contributing Guide](docs/developer/guides/contributing.md) for:

- Development workflow and branching strategy
- Code style and commit conventions
- Pull request and code review process
- Setting up your development environment

## Branching Strategy

See the [Branching Strategy](docs/developer/guides/branching-strategy.md) for how `main`, `development`, and feature branches interact.

**Key points:**
- PRs target `development`, not `main`
- Bug fixes go on `development`; critical ones are cherry-picked to `main`
- Feature work happens on `feature/*` branches
- API changes are merged to `development` separately from feature implementations

## Reporting Issues

Submit issues at https://github.com/underworldcode/underworld3/issues using the appropriate template.
