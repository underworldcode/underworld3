---
description: Find deprecated access patterns in code and documentation
---

## Reference Document

**Read the authoritative patterns guide first:**
`docs/developer/UW3_Style_and_Patterns_Guide.qmd`

Key sections:
- Section 4 "Array and Data Management" - array indexing patterns
- Section 5 "Context Managers" - deprecated vs current access patterns
- Section 6 "MPI and Parallel Patterns" - parallel safety

---

## Search for Legacy Patterns

### Source Code
```bash
grep -r "with.*\.access(" src/underworld3/ --include="*.py"
```

### Examples
```bash
grep -r "with.*\.access(" docs/examples/ --include="*.py"
grep -r "mesh\.data\[" docs/examples/ --include="*.py"
```

### Documentation (markdown, quarto)
```bash
grep -r "with.*\.access(" docs/ --include="*.md" --include="*.qmd"
grep -r "mesh\.data\[" docs/ --include="*.md" --include="*.qmd"
```

### Notebooks
```bash
find docs -name "*.ipynb" -exec grep -l "\.access(" {} \;
find docs -name "*.ipynb" -exec grep -l "mesh\.data" {} \;
```

### Tests (IMPORTANT: Tests are a source of truth for AI tools)
```bash
grep -r "with.*\.access(" tests/ --include="*.py"
grep -r "mesh\.data\[" tests/ --include="*.py"
```

---

## Pattern Classification

| Pattern | Status | Replacement |
|---------|--------|-------------|
| `with mesh.access(var):` | **Deprecated** | Direct: `var.data[...]` |
| `with swarm.access(var):` | **Deprecated** | Direct: `var.data[...]` |
| `mesh.data` (coordinates) | **Deprecated** | `mesh.X.coords` |
| `var.array[:, 0, 0]` | Current | Scalar indexing |
| `var.array[:, 0, :]` | Current | Vector indexing |
| `uw.synchronised_array_update()` | Current | Multi-variable batch |

---

## Analysis Required

For each occurrence found, classify as:
- **Safe to remove**: Simple data access, visualization, I/O
- **Preserve for now**: Solver-critical operations needing further analysis
- **Already correct**: Part of the access context implementation itself

## Output Format

- Total occurrences by location (src, examples, docs, notebooks, **tests**)
- Files needing updates grouped by type
- Priority order:
  1. User-facing examples (docs/examples)
  2. **Tests** (source of truth for AI tools writing notebooks)
  3. Documentation (notebooks, markdown)
  4. Source code
- Recommended fixes

---

## Docstring Health Checks

### Regenerate Inventory
```bash
pixi run -e default python scripts/docstring_sweep.py
```

### Quick Statistics
```bash
# Count items by status in review queue
grep -c "none" docs/docstrings/review_queue.md || echo "0"
grep -c "minimal" docs/docstrings/review_queue.md || echo "0"
grep -c "partial" docs/docstrings/review_queue.md || echo "0"
grep -c "complete" docs/docstrings/review_queue.md || echo "0"
```

### Format Inconsistencies
```bash
# Find Markdown math in source (should be RST :math: format)
grep -rn '\$[^$]*\$' src/underworld3/ --include="*.py" | grep -v "test_" | head -20
```

### Reference
- Full inventory: `docs/docstrings/review_queue.md` (with NEEDS_* flags)
- Conversion plan: `docs/plans/docstring-conversion-plan.md`
- Target format: NumPy/Sphinx with RST math (`:math:`, `.. math::`)
