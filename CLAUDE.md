# Underworld3 AI Assistant Context

> **Note**: Human-readable developer documentation is in `docs/developer/` (Sphinx/MyST format).
> For development history and completed migrations, see `docs/developer/ai-notes/historical-notes.md`

---

## Extended AI Context (Optional)

You can configure additional AI instruction files that live outside the Underworld repository. This is useful for:
- Personal coding style preferences
- Project planning and coordination systems
- Private instructions not shared with other contributors

**Setup**: If the environment variable `UW_AI_TOOLS_PATH` is set (colon-separated directories, like `PATH`), check each directory for `.md` files containing additional context. Configure this via `./uw setup` or manually in your shell profile.

**At conversation start**: If `UW_AI_TOOLS_PATH` is set, look for `underworld.md` (or other relevant files) in those directories. If found, report relevant Active/Bugs items briefly. If the variable is unset or files aren't found, proceed normally with repo-local context only.

### Responding to External Planning Items

When completing a task from an external planning file, add an annotation directly below the item:

```markdown
<!-- PROJECT RESPONSE (YYYY-MM-DD underworld3):
Brief summary of what was done.
Reference to any files created/modified.
-->
```

### Adding New Items to External Planning

If you discover bugs, identify new tasks, or have items that should be tracked in the external planning file:
- **Don't create local TODO files or add to CLAUDE.md**
- **Do add to the external planning file** under the appropriate section (Bugs, Active, Nice to Have)
- Use the project tag: `<!-- project:underworld3/subsystem -->`

### Inline TODO Comments in Code

Use inline `TODO` comments to mark **problem locations** in the source code. These provide self-documentation and help future developers (human or AI) find the relevant code quickly.

**Format:**
```python
# TODO(BUG): Brief description of the issue
# More context if needed
# See planning file: underworld.md (section, date)
```

**When to use:**
- Mark the exact location of a known bug
- Flag code that needs enhancement or refactoring
- Note incomplete implementations

**Example:**
```python
# TODO(BUG): add_natural_bc() causes PETSc error 73
# The Stokes solver works; issue is specific to scalar Poisson setup.
# See planning file: underworld.md (Bugs section, 2026-01-19)
self.natural_bcs = []
```

This complements the planning file — the planning file tracks *what* needs doing, inline TODOs mark *where* in the code.

### What Not to Do

- Don't rewrite strategic paragraphs — they contain cross-project context
- Don't move items between sections (Active → Done) — planning-claude handles that
- Don't restructure the document

If something needs more than an annotation or simple addition, mention it in conversation for the user to handle.

---

## Pending Release Actions

**⚠️ REMINDER: Tag v3.0.0 when `uw3-release-candidate` merges to `main`**

The AI-friendly codebase with improved documentation, patterns, and tooling should be released as Underworld3 version 3.0.0. After merging the release candidate branch to main:

```bash
git tag -a v3.0.0 -m "Underworld3 Release 3.0.0"
git push origin v3.0.0
```

See `docs/developer/guides/version-management.md` for details.

---

## Documentation Requests

**⚠️ MANDATORY - READ BEFORE WRITING ANY DOCUMENTATION ⚠️**

- **ALL documentation MUST go in `docs/` directory** - NO exceptions
- **NEVER create .md files in the repository root, src/, tests/, or anywhere else**
- **NEVER create planning/design documents outside `docs/developer/design/`**
- If you're tempted to create a file like `SOME-FEATURE-NOTES.md` in the repo root - **DON'T**. Put it in `docs/developer/` instead.
- This applies to: design docs, how-to guides, technical notes, implementation plans, reviews, audits - EVERYTHING goes in `docs/`

**Where to put documentation:**

| Content Type | Location |
|--------------|----------|
| System documentation (meshing, solvers, swarms) | `docs/developer/subsystems/` |
| Architecture and design decisions | `docs/developer/design/` |
| How-to guides and best practices | `docs/developer/guides/` |
| User tutorials | `docs/beginner/tutorials/` |
| Advanced user guides | `docs/advanced/` |

**Format** - Use MyST Markdown (`.md` files) compatible with Sphinx:
- Standard markdown with MyST extensions
- Use ```` ```python ```` for code blocks (not `{python}`)
- Use `{note}`, `{warning}`, `{tip}` for admonitions
- Math: `$inline$` and `$$display$$`

**Style** - Concise, helpful, standalone:
- Self-contained explanations (don't assume reader has context)
- Include practical code examples
- Link to related documentation where appropriate
- Focus on "why" and "how to use", not just "what"
- Follow the notebook style guide for tutorials

**Integration** - Link into the documentation system:
- Add to appropriate toctree in parent `index.md`
- Cross-reference related docs with `:doc:` or relative links
- Build and verify: `pixi run docs-build`

**Style references**:
- Notebook writing: `docs/developer/guides/notebook-style-guide.md`
- Code patterns: `docs/developer/UW3_Style_and_Patterns_Guide.md`

---

## Git and Branching Strategy

**Full guide**: `docs/developer/guides/branching-strategy.md`

### Branch Roles
- **`main`** — stable releases (tagged quarterly). No direct pushes.
- **`development`** — integration branch. Bug fixes land here. Features merge here via PR.
- **`feature/*`** — long-lived feature work. Branch from and PR back to `development`.

### Key Discipline: Separate API from Implementation
Feature branches must not introduce API changes (new methods, changed signatures) that other branches can't access. When a feature needs an API change:
1. Extract the interface (stub or minimal implementation) into a separate commit.
2. Merge that to `development` first (or extract after the fact).
3. The feature PR should only contain implementation behind already-merged interfaces.

This keeps feature branches independent and makes cross-pollination of fixes straightforward.

### Bug Fix Flow
- Fix on `development` (commit or small PR)
- Cherry-pick to `main` if critical → tag patch release
- Cherry-pick to active feature branches (underworld-claude handles this)

### Git Worktrees for Session Isolation
**Use a worktree for any multi-file change** (docs cleanup, refactoring, features).
Multiple Claude sessions sharing one working directory will overwrite each other's work.

Worktrees share the main repo's pixi environment and PETSc build via symlinks —
there is one set of dependencies, not one per worktree. `./uw build` from inside
a worktree installs that worktree's source into the shared environment.

**Full documentation**: `docs/developer/guides/branching-strategy.md` (Git Worktrees section)

#### Creating and using a worktree

```bash
# Create — resets to development, sets up symlinks, names the branch
./uw worktree create <name>              # → feature/<name>
./uw worktree create <name> bugfix       # → bugfix/<name>

# Work — drops you into a shell cd'd to the worktree
./uw worktree shell <name>
./uw build           # builds from THIS source into the shared env
./uw test            # runs tests
exit                 # leave

# List worktrees with branch and status
./uw worktree list

# Bring files from other branches without switching:
git checkout origin/<branch> -- path/to/file
```

#### Cleanup

```bash
# Removes worktree directory and deletes the branch
./uw worktree remove <name>
```

#### Important: always build and run from inside the worktree

Because there is one shared environment, `./uw build` installs whichever source
tree you run it from. If you build from the main repo then run code expecting
worktree changes, the worktree edits will not be active. Always:

1. `./uw worktree shell <name>` (or `cd` into the worktree)
2. `./uw build`
3. Run your code / tests from there

### AI-Assisted Attribution (Commits and PRs)
When committing code or creating pull requests with AI assistance, end the
message/body with:

```
Underworld development team with AI support from [Claude Code](https://claude.com/claude-code)
```

(In commit messages, use the plain-text form without the markdown link.)

**Do NOT use**:
- `Co-Authored-By:` with a noreply email (useless for soliciting responses)
- Generic AI attribution without team context
- Emoji in PR descriptions

---

## CRITICAL BUILD CONSTRAINTS

### PETSc Directory (DO NOT MOVE)
**WARNING**: `/Users/lmoresi/+Underworld/underworld-pixi-2/petsc/` MUST NOT be moved.
- PETSc is NOT relocatable after compilation (hardcoded paths)
- Moving breaks petsc4py bindings and all pixi tasks
- Requires complete rebuild (~1 hour) if relocated

### Rebuild After Source Changes
**After modifying source files, always run `./uw build`!**
- Underworld3 is installed as a package in the pixi environment
- Changes go to `.pixi/envs/default/lib/python3.12/site-packages/underworld3/`
- Verify with `uw.model.__file__`

**Note**: `./uw build` uses `--no-cache-dir` to prevent pip from reusing stale
wheels (UW3 is always version `0.0.0`). If you still suspect stale code, clean
the build directory: `rm -rf build/lib.* build/bdist.*` then rebuild.

### Test Quality Principles
**New tests must be validated before making code changes to fix them!**
- Validate test correctness before changing main code
- If core tests (0000-0599) pass, the system is working correctly
- Disable problematic new tests, validate core functionality, then fix test structure

### JOSS Paper (FROZEN)
**Location**: `publications/joss-paper/` - Publication of record, DO NOT modify.

---

## Design Documents Reference

**Location**: `docs/developer/design/`

| Document | Status | Purpose |
|----------|--------|---------|
| `UNITS_SIMPLIFIED_DESIGN_2025-11.md` | **AUTHORITATIVE** | Current units architecture |
| `PARALLEL_PRINT_SIMPLIFIED.md` | Implemented | `uw.pprint()` and `selective_ranks()` |
| `RANK_SELECTION_SPECIFICATION.md` | Implemented | Rank selection syntax |
| `mathematical_objects_plan.md` | Implemented | Mathematical objects design |

---

## Units System Principles

### String Input, Pint Object Storage
**Accept strings for convenience, store/return Pint objects internally.**

```python
# User creates with string (convenience)
viscosity = uw.quantity(1e21, "Pa*s")

# Internally stored as Pint object
# .units returns Pint Unit object (NOT string!)
viscosity.units  # <Unit('pascal * second')>

# Arithmetic works correctly
Ra = (rho0 * alpha * g * DeltaT * L**3) / (eta0 * kappa)
```

### Unit vs Quantity Distinction
```python
# Pint Quantity = value + units (can convert)
qty = uw.quantity(2900, "km")
qty.to("m")              # Returns new UWQuantity
qty.to_base_units()      # Returns new UWQuantity

# Pint Unit = just the unit (cannot convert!)
qty.units                # <Unit('kilometer')>
qty.units.to("m")        # AttributeError! Use qty.to("m") instead
```

### Transparent Container Principle
**UWexpression is a container that derives properties from its contents.**
- Atomic (UWQuantity): `.units` comes from stored value
- Composite (SymPy tree): `.units` derived via `get_units(self._sym)`
- No cached state on composites - eliminates sync issues

---

## Parallel Computing Patterns

### Key Understanding
**Underworld3 rarely uses MPI directly - PETSc handles all parallel synchronization.**

- PETSc manages parallelism for mesh operations, solvers, vector updates
- UW3 API wraps PETSc collective operations correctly
- Avoid direct mpi4py usage unless absolutely necessary

### Current Parallel Safety API

```python
# OLD (deprecated) - DANGEROUS if stats() is collective
if uw.mpi.rank == 0:
    print(f"Stats: {var.stats()}")

# NEW (safe) - All ranks execute, only selected ranks print
uw.pprint(0, f"Stats: {var.stats()}")

# For code blocks (visualization, etc.)
with uw.selective_ranks(0) as should_execute:
    if should_execute:
        import pyvista as pv
        plotter = pv.Plotter()
```

**Implementation**: `src/underworld3/mpi.py`
**Documentation**: `docs/advanced/parallel-computing.qmd`

---

## Architecture Priorities

### Solver Stability is Paramount
The PETSc-based solvers are carefully optimized and validated. **NO CHANGES without extensive benchmarking.**

### Module Boundaries

| Module | Purpose | Access Pattern |
|--------|---------|----------------|
| **Solvers** (`petsc_generic_snes_solvers`) | High-performance PETSc solving | Direct `vec` property |
| **Mesh Variables** | User-facing field data | `array` property (new) |
| **Swarm Variables** | Particle data with mesh proxies | `data` property |

### Conservative Migration Strategy
- **User-facing code**: Use `array` property with automatic sync
- **Solver internals**: Keep using `vec` property with direct PETSc access
- **Gradual transition**: Only change when driven by actual needs

---

## Data Access Patterns

**Authoritative Reference**: `docs/developer/UW3_Style_and_Patterns_Guide.md`
**Pattern Checker**: Use `/check-patterns` to scan for deprecated patterns

### Quick Summary
| Pattern | Status | Use Instead |
|---------|--------|-------------|
| `with mesh.access(var):` | **Deprecated** | Direct: `var.data[...]` |
| `with swarm.access(var):` | **Deprecated** | Direct: `var.data[...]` |
| `mesh.data` (coordinates) | **Deprecated** | `mesh.X.coords` |

### Current Patterns
```python
# Single variable - direct access
var.data[...] = values
var.array[:, 0, 0] = scalar_values   # Scalar
var.array[:, 0, :] = vector_values   # Vector

# Multiple variables - batch synchronization
with uw.synchronised_array_update():
    var1.data[...] = values1
    var2.data[...] = values2

# Coordinates
mesh.X.coords    # Mesh vertex coordinates
var.coords       # Variable DOF coordinates
swarm.data       # Swarm particle positions
```

### Array Shapes
- **array**: `(N, a, b)` where scalar=`(N,1,1)`, vector=`(N,1,dim)`, tensor=`(N,dim,dim)`
- **data**: `(-1, num_components)` flat format for backward compatibility

### Data Cache Safety
The `.data` property caches an `NDArray_With_Callback` view into the PETSc local vector. This cache self-validates via `id(self._lvec)` tracking — if the underlying vector is replaced (DM rebuild, mesh adaptation), the cache auto-rebuilds on next access. See `docs/developer/subsystems/data-access.md` for details.

---

## Expression Processing

### Unwrap Before Extracting Atoms
When extracting `.atoms()` or `.free_symbols` from expressions before compilation:

```python
# CORRECT ORDER:
# 1. First unwrap UWexpressions to reveal hidden coordinates
if any_uwexpressions_in_expression:
    expr = _unwrap_for_compilation(expr, keep_constants=False, return_self=False)
# 2. Then extract atoms/symbols from the FULLY PROCESSED expression
symbols = expr.atoms(...)
```

**Safe locations**: JIT Compiler (`_jitextension.py`), `extract_expressions()`
**Check if issues**: `is_pure_sympy_expression()`, `nondimensional.py`

---

## Swarm Concepts

### Migration
Migration moves particles between processors based on spatial location.
- Happens automatically when particles move
- Use `migration_disabled()` context for batch operations
- Essential for parallel correctness

### Proxy Mesh Variables
Swarm variables with `proxy_degree > 0` create proxy mesh variables using RBF interpolation.
- Used for integration and derivative calculations
- Must be updated when swarm data/positions change
- Update happens automatically via `swarmVar._update()`

---

## Mathematical Objects

Variables support natural mathematical syntax:

```python
# Direct arithmetic (no .sym needed)
momentum = density * velocity
strain_rate = velocity[0].diff(x) + velocity[1].diff(y)

# Full SymPy Matrix API available
velocity.T              # Transpose
velocity.dot(other)     # Dot product
velocity.norm()         # Magnitude
```

**Implementation**: `MathematicalMixin` in `utilities/mathematical_mixin.py`

---

## Coding Conventions

### Prefer Glob and Grep Over find
**Use the Glob and Grep tools instead of `find` in Bash.**
- `Glob` handles file pattern matching (e.g., `**/*.py`, `src/**/*.pyx`)
- `Grep` handles content search (e.g., searching for class definitions, imports)
- Both are faster, safer, and give the user better visibility than shell `find`
- `find` with `-exec`, `-execdir`, or `-delete` can execute arbitrary commands — avoid it
- Only fall back to `find` via Bash if Glob/Grep genuinely cannot express the query

### Desktop Notifications for Background Monitoring
When using CronCreate for background monitoring (CI status, issues, etc.), use
platform-appropriate notification commands. Both are in the allowed tools list:
- **macOS**: `osascript -e 'display notification "message" with title "title" sound name "Glass"'`
- **Linux**: `notify-send "title" "message"`

Be quiet when everything is fine — only notify when something needs attention.

### Plan File Naming Policy
**Plan files must have descriptive names that indicate their content.**

```
# GOOD - Descriptive names
mesh-adaptation-architecture.md
gradient-evaluation-p2-fix.md
units-system-refactor-plan.md

# BAD - Random/whimsical names
proud-petting-pretzel.md
happy-dancing-dolphin.md
```

When creating plan files in `~/.claude/plans/`, use kebab-case names that describe:
- The feature or subsystem being worked on
- The type of work (architecture, fix, refactor, feature)

### Avoid Ambiguous 'model'
Two different "model" concepts exist:
- `uw.Model`: Serialization/orchestration system
- Constitutive models: Material behavior (ViscousFlowModel, etc.)

```python
# GOOD - Clear and unambiguous
constitutive_model = stokes.constitutive_model
orchestration_model = uw.get_default_model()

# AVOID - Ambiguous
model = stokes.constitutive_model
```

---

## Test Classification

### By Complexity Level (pytest markers)
- `@pytest.mark.level_1`: Quick core tests (seconds)
- `@pytest.mark.level_2`: Intermediate tests (minutes)
- `@pytest.mark.level_3`: Physics/solver tests (minutes to hours)

### By Reliability Tier
- `@pytest.mark.tier_a`: Production-ready (TDD-safe)
- `@pytest.mark.tier_b`: Validated (use with caution)
- `@pytest.mark.tier_c`: Experimental (development only)

```bash
# Quick validation
pytest -m "level_1 and tier_a"

# Full validation
pytest -m "tier_a or tier_b"
```

**Details**: `docs/developer/TESTING-RELIABILITY-SYSTEM.md`

---

## On-Demand Documentation References

When working on specific subsystems, these documents provide detailed guidance.
**Read them on demand using the Read tool** — do NOT load them all at conversation start.

> **AI Assistant Protocol**: When you need deeper context, explicitly tell the user
> what you're reading and why. Use the Read tool to load the specific file.
> Example: "Let me check the units design doc for this..."

### Units & Scaling
- `docs/developer/design/UNITS_SIMPLIFIED_DESIGN_2025-11.md` - **Authoritative** units architecture
- `docs/developer/ai-notes/COORDINATE-UNITS-TECHNICAL-NOTE.md` - Coordinate unit handling
- `docs/developer/design/WHY_UNITS_NOT_DIMENSIONALITY.md` - Design rationale

### Testing
- `docs/developer/TESTING-RELIABILITY-SYSTEM.md` - Test tier classification (A/B/C)
- `docs/developer/ai-notes/TEST-CLASSIFICATION-2025-11-15.md` - Current test status

### Code Style, Workflow & Patterns
- `docs/developer/guides/branching-strategy.md` - Branching, releases, API change discipline
- `docs/developer/UW3_Style_and_Patterns_Guide.md` - Development standards

### Data Access & Variables
- `docs/developer/subsystems/data-access.md` - Data access patterns, self-validating cache
- `docs/developer/UW3_Developers_NDArrays.md` - NDArray_With_Callback internals

### Architecture & Design
- `docs/developer/design/ARCHITECTURE_ANALYSIS.md` - System structure analysis
- `docs/developer/design/MATHEMATICAL_MIXIN_DESIGN.md` - Mathematical objects internals
- `docs/developer/design/GEOGRAPHIC_COORDINATE_SYSTEM_DESIGN.md` - Spherical/planetary meshes
- `docs/developer/design/SYMBOL_DISAMBIGUATION_2025-12.md` - Multi-mesh symbol identity
- `docs/developer/TEMPLATE_EXPRESSION_PATTERN.md` - Solver template expressions

### Coordinates & Mesh
- `docs/developer/design/COORDINATE_MIGRATION_GUIDE.md` - Coordinate system changes
- `docs/developer/design/mesh-geometry-audit.md` - Mesh geometry patterns

### Development History
- `docs/developer/ai-notes/historical-notes.md` - Completed migrations, fixed bugs

---

## Quick Reference

### Build & Test Commands
```bash
./uw build                   # Rebuild after source changes (preferred)
./uw test                    # Run test suite
pixi run -e default python   # Run Python in environment
```

### Key Files
- `src/underworld3/mpi.py` - Parallel safety implementation
- `src/underworld3/scaling/` - Units system
- `src/underworld3/utilities/mathematical_mixin.py` - Mathematical objects
- `src/underworld3/function/expressions.py` - UWexpression (lazy evaluation, symbol disambiguation)
- `src/underworld3/function/_function.pyx` - UnderworldFunction (mesh variable symbols)
- `src/underworld3/discretisation/enhanced_variables.py` - EnhancedMeshVariable (units, math ops, persistence)
- `src/underworld3/discretisation/persistence.py` - Stub for future persistence features

### Historical Notes
For development history, completed migrations, and fixed bugs:
See `docs/developer/ai-notes/historical-notes.md`

---

*Reorganized 2025-12-13: Historical content moved to docs/developer/ai-notes/historical-notes.md*
