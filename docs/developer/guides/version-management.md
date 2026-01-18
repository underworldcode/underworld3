# Version Management with Git Tags

**Date**: 2025-01-19
**Status**: IMPLEMENTED

## Overview

Underworld3 uses **git tag-based versioning** via `setuptools-scm`. The version is derived automatically from git tags at build time, eliminating merge conflicts on version files.

## How It Works

1. **Git tags define releases**: `v3.1.0`, `v3.1.0b1`, `v3.2.0-rc1`
2. **Version derived at build time**: No hardcoded version file to conflict
3. **Automatic dev versions**: Commits after a tag get `.devN` suffix

### Version Examples

| Git State | Resulting Version |
|-----------|-------------------|
| On tag `v3.1.0` | `3.1.0` |
| 5 commits after `v3.1.0` | `3.1.1.dev5` |
| On tag `v3.1.0b1` | `3.1.0b1` |
| 3 commits after `v3.1.0b1` | `3.1.0b2.dev3` |
| Dirty working tree | `3.1.1.dev5` (no local version suffix) |

## Creating Releases

### Stable Release

```bash
# On main branch, after all changes merged
git tag -a v3.1.0 -m "Release 3.1.0"
git push origin v3.1.0
```

### Beta/Pre-release

```bash
# For beta releases
git tag -a v3.1.0b1 -m "Beta 1 for 3.1.0"
git push origin v3.1.0b1

# For release candidates
git tag -a v3.1.0rc1 -m "Release candidate 1 for 3.1.0"
git push origin v3.1.0rc1
```

### Alpha Releases (for development branches)

```bash
# On development branch
git tag -a v3.2.0a1 -m "Alpha 1 for 3.2.0 development"
git push origin v3.2.0a1
```

## Tag Naming Convention

Use these patterns for proper version ordering:

| Type | Format | Example | PEP 440 Version |
|------|--------|---------|-----------------|
| Stable | `vX.Y.Z` | `v3.1.0` | `3.1.0` |
| Beta | `vX.Y.ZbN` | `v3.1.0b1` | `3.1.0b1` |
| Alpha | `vX.Y.ZaN` | `v3.2.0a1` | `3.2.0a1` |
| Release Candidate | `vX.Y.ZrcN` | `v3.1.0rc1` | `3.1.0rc1` |

**Important**: The `v` prefix is optional but recommended for clarity.

## Workflow for Branches

### Main Branch (Releases)

```bash
# After release v3.1.0, commits on main get:
# 3.1.1.dev1, 3.1.1.dev2, ...

# When ready for next release:
git tag -a v3.1.1 -m "Patch release 3.1.1"
```

### Development Branch

```bash
# Start development cycle with alpha tag
git checkout development
git tag -a v3.2.0a1 -m "Start 3.2.0 development"

# Commits get: 3.2.0a2.dev1, 3.2.0a2.dev2, ...

# As features mature, release betas
git tag -a v3.2.0b1 -m "Beta 1"
# Commits get: 3.2.0b2.dev1, ...
```

### Feature Branches

Feature branches inherit the version from their base branch plus `.devN`:

```bash
# On feature branch from main (v3.1.0):
# Version: 3.1.1.dev15 (if 15 commits since v3.1.0)
```

## Checking the Version

### At Runtime

```python
import underworld3 as uw
print(uw.__version__)
```

### During Development

```bash
# See what version would be assigned
pixi run python -c "from setuptools_scm import get_version; print(get_version())"

# Or after building
pixi run python -c "import underworld3; print(underworld3.__version__)"
```

## Files Involved

| File | Purpose |
|------|---------|
| `pyproject.toml` | setuptools-scm configuration |
| `src/underworld3/_version.py` | Auto-generated at build (gitignored) |
| `src/underworld3/__init__.py` | Version import with fallbacks |

## Handling Edge Cases

### Non-Git Installation (tarballs)

If someone installs from a tarball without git history:
1. Build process uses `fallback_version = "0.0.0+unknown"`
2. Recommendation: Include version in tarball name

### Editable Installs

For `pip install -e .`:
- setuptools-scm queries git for version on every import
- Version updates automatically as you make commits

### Missing Tags

If no tags exist:
```bash
# Create an initial tag
git tag -a v0.99.0 -m "Initial version tag"
```

## Migration from Hardcoded Version

The old `_version.py` with hardcoded `__version__` has been replaced. To complete the migration:

```bash
# 1. Remove _version.py from git tracking (already in .gitignore)
git rm --cached src/underworld3/_version.py

# 2. Commit the changes
git add pyproject.toml setup.py src/underworld3/__init__.py .gitignore
git commit -m "Switch to git tag-based versioning with setuptools-scm"

# 3. Create a tag for the current version
git tag -a v0.99.0 -m "Current release version"

# 4. Rebuild to generate new _version.py
pixi run underworld-build
```

## Troubleshooting

### "Unknown" or "0.0.0+unknown" Version

**Cause**: Not in a git repository or no tags exist.

**Solution**:
```bash
git tag -a v0.99.0 -m "Set current version"
pixi run underworld-build
```

### Version Not Updating After Commits

**Cause**: Stale build or cached _version.py.

**Solution**:
```bash
rm -f src/underworld3/_version.py
pixi run underworld-build
```

### "LookupError: setuptools-scm was unable to detect version"

**Cause**: Git history not available or corrupted.

**Solution**:
```bash
# Verify git is working
git describe --tags

# If shallow clone, unshallow it
git fetch --unshallow
```

## References

- [setuptools-scm documentation](https://setuptools-scm.readthedocs.io/)
- [PEP 440 - Version Identification](https://peps.python.org/pep-0440/)
