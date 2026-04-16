# Underworld3 Development Changelog

This log tracks significant development work at a conceptual level, suitable for quarterly reporting to CIG and stakeholders. For detailed commit history, see git log.

---

## 2026 Q1 (January – March)

### v3.0.0 Release (March 2026)

**Underworld3 v3.0.0 released**: Merged 398 commits from development to main. Major release incorporating 18 months of work since the JOSS v0.99 publication, including units system overhaul, symbol disambiguation, boundary integrals, mathematical mixin, platform-conditional MPI, and comprehensive CI/CD automation.

- Tagged `v0.99` at previous main HEAD (pixi-compatible JOSS snapshot) for binder compatibility
- Deleted obsolete `uw3-release-candidate` branch
- Cleaned up 10 merged feature/bugfix branches

### Binder Infrastructure Overhaul (March 2026)

**Versioned binder links** with full CI automation for tag-based releases.

- Four launcher branches: `v0.99`, `v3.0.0`, `main`, `development` — each with frozen Dockerfile
- CI workflow handles `v*` tag builds with automatic launcher branch creation via `repository_dispatch`
- Manual dispatch overrides (`uw3_branch`, `image_tag`) for building images from old tags
- Dockerfile made version-resilient: versioned lib subdirectories (vtk-X.Y, openvino-X.Y.Z) use wildcards instead of hardcoded paths
- Launcher dispatch payload fixed: field names now match target workflow (`branch`/`ref_type`)
- README badges updated with three versioned binder launch links

**Files**: `binder-image.yml`, `Dockerfile.base.optimized`, `binder_wizard.py`, `containers.md`

### Checkpoint XDMF Fix (March 2026)

**`petsc_save_checkpoint()` now uses modern XDMF output** (fixes #80). Previously used legacy `generateXdmf()` which missed vertex/cell compatibility groups, field projection (P2→P1), and tensor repacking for ParaView.

- Refactored as thin wrapper around `write_timestep()` — single checkpoint code path
- Output file layout changes from single HDF5 to per-variable files (consistent with `write_timestep()`)

**Files**: `discretisation_mesh.py`

### Boundary Integral Support (March 2026)

**New `uw.maths.BdIntegral` class** for boundary and surface integrals (closes #47). Wraps PETSc's `DMPlexComputeBdIntegral` with MPI Allreduce and units support. Works on external boundaries and internal boundaries (e.g. `AnnulusInternalBoundary`). Integrands can reference the outward unit normal via `mesh.Gamma`.

- PETSc patch (`plexfem-internal-boundary-ownership-fix.patch`): fixes ghost facet ownership and part-consistent assembly in boundary residual, integral, and Jacobian paths. Resolves rank-dependent L2 norms for internal boundary natural BCs (fixes #77). Contributed by gthyagi.
- C wrapper simplified: ghost filtering delegated to PETSc patch, wrapper retains MPI Allreduce only
- 20 tests across external/internal boundaries, normal vectors, mesh variables
- MPI regression test for internal boundary circumference

**Files**: `petsc_compat.h`, `petsc_maths.pyx`, `petsc_extras.pxi`, `maths/__init__.py`, `petsc-custom/patches/`

### Binder Image Fix (March 2026)

**Fixed Dockerfile building from stale branch** (fixes #71). The binder Dockerfile hardcoded `uw3-release-candidate` as the clone branch, but the CI workflow triggers on `main` and `development` pushes. The image was missing recent dependencies (e.g. `python-xxhash`).

- Dockerfile now uses `ARG UW3_BRANCH=development` instead of hardcoded branch
- CI workflow passes the triggering branch name via `--build-arg`
- Binder wizard script default updated to `development`

### Worktree Symlink Safety (March 2026)

**Prevented worktree symlinks from being accidentally committed**. The `./uw worktree create` command creates `.pixi` and `petsc-custom/petsc` symlinks that could be picked up by `git add -A`, breaking CI.

- `.gitignore` patterns now match both directories and symlinks (removed trailing `/`)
- `./uw worktree create` writes exclusions to the worktree's `.git/info/exclude`

### MeshVariable Data Cache Bug Fix (February 2026)

**Self-validating `.data` cache**: Fixed a critical bug where the `.data` property could return stale (zero) values after PETSc DM rebuilds. When new MeshVariables are added to a mesh, PETSc requires a new DM — destroying and recreating all existing variables' local vectors (`_lvec`). The cached `_canonical_data` array (a NumPy view into the old `_lvec`) would silently read freed memory, returning zeros even though the solver correctly wrote results to the new vector.

- Root cause: Early `.data` access cached a view that became invalid after DM rebuild
- Fix: `.data` property now tracks `id(self._lvec)` and auto-rebuilds when stale
- Self-healing design: no code path that replaces `_lvec` needs to manually invalidate the cache
- Eager invalidation in DM rebuild loop and `mesh.adapt()` preserved as performance optimization

**Files**: `discretisation_mesh_variables.py` (`.data` property), `discretisation_mesh.py` (`mesh.adapt()`)

### Binder/Docker CI Automation (January 2026)

**Automated container build pipeline**: Implemented full GitHub Actions automation for Docker image builds and mybinder.org integration.

- **Binder images** (`binder-image.yml`): Builds to GHCR on push to main/development
  - Triggers on Dockerfile, pixi.toml, Cython, or setup.py changes
  - Pushes to `ghcr.io/underworldcode/uw3-base:<branch>-slim`
  - Cross-repo dispatch updates launcher repository automatically

- **Command-line images** (`docker-image.yml`): Separate workflow for GHCR (micromamba-based)

- **Launcher auto-update**: `uw3-binder-launcher` receives `repository_dispatch` events and updates its Dockerfile reference automatically

- **Container consolidation**: All container files now in `container/` directory with comprehensive README

**Key infrastructure**:
- `LAUNCHER_PAT` secret enables cross-repo communication
- Branch-specific image tags enable testing different versions
- nbgitpuller allows any repository to use pre-built images

**Documentation**: `docs/developer/subsystems/containers.md` — comprehensive guide covering both binder and command-line container strategies.

---

## 2025 Q4 (October – December)

### Symbol Disambiguation (December 2025)

**Clean multi-mesh symbol identity**: Replaced the invisible whitespace hack (`\hspace{}`) with SymPy-native symbol disambiguation using `_uw_id` in `_hashable_content()`. This follows the `sympy.Dummy` pattern.

- Variables on different meshes with same name are now symbolically distinct without display name pollution
- Clean LaTeX rendering — no more invisible whitespace artifacts
- Proper serialization/pickling support
- Coordinate symbols (`mesh.N.x`, etc.) also isolated per-mesh to prevent cache pollution bugs

**Key technical insight**: SymPy's `Symbol.__new__` has an internal cache that runs *before* `_hashable_content()`. Solution: use `Symbol.__xnew__()` to bypass the cache, same as `sympy.Dummy` does.

**Expression rename capability**: Added `UWexpression.rename()` method to customize display names without changing symbolic identity. Uses SymPy's custom printing protocol (`_latex()`, `_sympystr()`) to separate display from identity. Useful for multi-material models where parameters need distinct LaTeX labels:
```python
viscosity.rename(r"\eta_{\mathrm{mantle}}")  # Custom LaTeX display
```

**Files**: `expressions.py`, `_function.pyx`, `discretisation_mesh_variables.py`
**Design doc**: `docs/developer/design/SYMBOL_DISAMBIGUATION_2025-12.md`

### Units System Overhaul (November 2025)

**Gateway pattern implementation**: Units are now handled at system boundaries (user input/output) rather than propagating through internal symbolic operations. This eliminates unit-related errors during solver execution while preserving dimensional correctness for users.

- `UWQuantity` provides lightweight Pint-backed quantities
- `UWexpression` wraps symbolic expressions with lazy unit evaluation
- Linear algebra dimensional analysis replaces fragile pattern-matching
- Proper non-dimensional scaling throughout advection-diffusion solvers
- **Pint-only arithmetic policy**: All unit conversions delegated to Pint — no manual fallbacks that could lose scale factors

**Key fixes:**
- `delta_t` setter correctly converts between unit systems (Pint's `.to_reduced_units()`)
- `estimate_dt()` properly non-dimensionalizes diffusivity parameters
- Data cache invalidation after PETSc solves (buffer pointer changes)
- JIT compilation unwrapping respects `keep_constants` parameter
- Subtraction chain unit propagation fixed (chained operations preserve correct units)

### Automatic Expression Optimisation (November 2025)

**Lambdification for pure sympy expressions**: `uw.function.evaluate()` now automatically detects pure sympy expressions (no UW3 MeshVariables) and uses cached lambdified functions for dramatic performance improvements.

- 10,000x+ speedups for analytical solutions — no code changes required
- Automatic detection: UW3 variables use RBF interpolation, pure sympy uses lambdify
- Cached compilation: repeated evaluations reuse compiled functions
- Transparent fallback: mixed expressions still work correctly

### Timing System (November 2025)

**Unified PETSc timing integration**: Refactored timing system to route all profiling through PETSc's event system, eliminating environment variable complexity.

- `uw.timing.start()` / `uw.timing.print_summary()` API for simple profiling
- Filters PETSc internals to show only UW3-relevant operations
- Now Jupyter-friendly — no environment variables needed
- Programmatic access via `uw.timing.get_summary()`

### Solver Robustness (November 2025)

**Quad mesh boundary interpolation**: Fixed Semi-Lagrangian advection scheme failing on `StructuredQuadBox` meshes. The point location algorithm was receiving coordinates exactly on element boundaries. Solution: use pre-computed centroid-shifted coordinates for evaluation.

### Test Infrastructure (November 2025)

- Strict units mode enforcement in test collection
- All advection-diffusion tests now pass across mesh types (StructuredQuadBox, UnstructuredSimplex regular/irregular)
- **Dual test classification system**: Levels (0000-9999 complexity prefixes) + Tiers (A/B/C reliability markers)
  - Tier A: Production-ready, trusted for TDD
  - Tier B: Validated but recent, use with caution
  - Tier C: Experimental, development only

### Build System & Developer Experience (December 2025)

**`./uw` wrapper script**: Unified command-line interface for all underworld3 operations. Replaces fragmented pixi/mamba instructions with a single entry point.

- `./uw setup` — Interactive wizard installs pixi, configures environment, builds underworld3
- `./uw build` — Smart rebuild with automatic dependency chain handling
- `./uw test` / `./uw test-all` — Tiered test execution
- `./uw doctor` — Diagnoses configuration issues (PETSc mismatches, missing deps)
- `./uw status` — Check for updates on GitHub without pulling
- `./uw update` — Pull latest changes and rebuild

**Documentation overhaul**: Rewrote installation docs to focus on `./uw` workflow. The 3-line install now appears on the landing page. Removed outdated mamba/conda instructions; Docker and system PETSc kept as alternatives for specific use cases.

### Documentation & Planning (November 2025)

- Reorganised `planning/` → `docs/developer/design/` to distinguish from strategic planning
- Hub-spoke planning system integration for cross-project coordination
- This changelog established for quarterly reporting

---

## Format Guide

Each quarter should capture:

1. **Major features or capabilities** — What can users do now that they couldn't before?
2. **Architectural improvements** — What's better about the system design?
3. **Significant bug fixes** — Only those affecting correctness of results
4. **Infrastructure changes** — Testing, documentation, build system

Keep entries conceptual. Technical details belong in design documents or commit messages.
