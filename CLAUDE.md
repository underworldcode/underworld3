# Underworld3 AI Assistant Context

> **Note**: Human-readable developer documentation is in `docs/developer/` (Sphinx/MyST format).
> For development history and completed migrations, see @docs/developer/historical-notes.md

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
- Notebook writing: @docs/developer/guides/notebook-style-guide.md
- Code patterns: @docs/developer/UW3_Style_and_Patterns_Guide.md

---

## Git Commit Conventions

### AI-Assisted Commit Attribution
When committing code developed with AI assistance, end the commit message with:

```
Underworld development team with AI support from Claude Code
```

**Do NOT use**:
- `Co-Authored-By:` with a noreply email (useless for soliciting responses)
- Generic AI attribution without team context

---

## CRITICAL BUILD CONSTRAINTS

### PETSc Directory (DO NOT MOVE)
**WARNING**: `/Users/lmoresi/+Underworld/underworld-pixi-2/petsc/` MUST NOT be moved.
- PETSc is NOT relocatable after compilation (hardcoded paths)
- Moving breaks petsc4py bindings and all pixi tasks
- Requires complete rebuild (~1 hour) if relocated

### Rebuild After Source Changes
**After modifying source files, always run `pixi run underworld-build`!**
- Underworld3 is installed as a package in the pixi environment
- Changes go to `.pixi/envs/default/lib/python3.12/site-packages/underworld3/`
- Verify with `uw.model.__file__`

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

**Authoritative Reference**: @docs/developer/UW3_Style_and_Patterns_Guide.qmd
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
Read them when you need deeper context beyond what's in this file.

> **AI Assistant Protocol**: When reading any @ referenced document, explicitly tell
> the user what you're checking and why. This confirms you're accessing deeper context
> and prevents redundant prompting. Example: "Let me check the units design doc for this..."

### Units & Scaling
- @docs/developer/design/UNITS_SIMPLIFIED_DESIGN_2025-11.md - **Authoritative** units architecture
- @docs/developer/COORDINATE-UNITS-TECHNICAL-NOTE.md - Coordinate unit handling
- @docs/developer/design/WHY_UNITS_NOT_DIMENSIONALITY.md - Design rationale

### Testing
- @docs/developer/TESTING-RELIABILITY-SYSTEM.md - Test tier classification (A/B/C)
- @docs/developer/TEST-CLASSIFICATION-2025-11-15.md - Current test status

### Code Style & Patterns
- @docs/developer/UW3_Style_and_Patterns_Guide.qmd - Development standards

### Architecture & Design
- @docs/developer/design/ARCHITECTURE_ANALYSIS.md - System structure analysis
- @docs/developer/design/MATHEMATICAL_MIXIN_DESIGN.md - Mathematical objects internals
- @docs/developer/design/GEOGRAPHIC_COORDINATE_SYSTEM_DESIGN.md - Spherical/planetary meshes
- @docs/developer/design/SYMBOL_DISAMBIGUATION_2025-12.md - Multi-mesh symbol identity
- @docs/developer/TEMPLATE_EXPRESSION_PATTERN.md - Solver template expressions

### Coordinates & Mesh
- @docs/developer/design/COORDINATE_MIGRATION_GUIDE.md - Coordinate system changes
- @docs/developer/design/mesh-geometry-audit.md - Mesh geometry patterns

### Development History
- @docs/developer/historical-notes.md - Completed migrations, fixed bugs

---

## Quick Reference

### Pixi Commands
```bash
pixi run underworld-build    # Rebuild after source changes
pixi run underworld-test     # Run test suite
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
See @docs/developer/historical-notes.md

---

*Reorganized 2025-12-13: Historical content moved to docs/developer/historical-notes.md*
