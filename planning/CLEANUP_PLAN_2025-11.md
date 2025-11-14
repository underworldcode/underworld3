# Repository Cleanup Plan

**Date**: 2025-11-14
**Scope**: Root directory has 73 .md files that need organization

## Cleanup Strategy

### 1. KEEP in Root (Essential Documentation)
**Action**: Keep as-is

- `README.md` - Project overview
- `LICENCE.md` / `LICENSE.md` - Legal (check for duplicates)
- `CLAUDE.md` - AI assistant context (critical)
- `CONTRIBUTING.md` - Contribution guidelines
- `CHANGES.md` - Change log
- `SPELLING_CONVENTION.md` - Project conventions
- `TODO.md` - Active task tracking (just created)

### 2. RECENT/CURRENT Work (Keep in Root for Now)
**Action**: Keep - these are actively referenced

- `ARCHITECTURE_ANALYSIS.md` - Just created (2025-11-14)
- `UNWRAPPING_COMPARISON_REPORT.md` - Just created (2025-11-14)
- `UNWRAPPING_UNIFICATION_PROPOSAL.md` - Just created (2025-11-14)

### 3. PLANNING DOCUMENTS → Move to `planning/`
**Action**: `mv [file] planning/`

**Design Documents**:
- `EXPLICIT-MODEL-UNITS-DESIGN.md`
- `MATHEMATICAL_MIXIN_DESIGN.md`
- `UNIFIED_DIMENSIONALITY_MIXIN_DESIGN.md`
- `UNIFIED-UNITS-INTERFACE-DESIGN.md`
- `UNIFIED-UNITS-INTERFACE-FUTURE.md`
- `WHY_BOTH_UNIT_SYSTEMS.md`
- `WHY_UNITS_NOT_DIMENSIONALITY.md`

**Plans**:
- `PHASE_4_COMPLETE_PLAN.md`
- `PHASE_5_CLEANUP_PLAN.md`
- `UNITS_REFACTOR_PLAN.md`
- `CLOSURE_INCONSISTENCY_FIX_PLAN.md`

### 4. SESSION SUMMARIES → Archive or Delete
**Action**: Move to `archive/session_summaries/` or delete

- `SESSION_SUMMARY.md`
- `SESSION_SUMMARY_2025-10-26.md`
- `SESSION_SUMMARY_2025-11-08.md`

**Recommendation**: Delete (git history preserves them)

### 5. COMPLETED BUG FIX REPORTS → Archive or Delete
**Action**: Mine for useful info → Update docs → Delete

- `DIMENSIONALITY_BUG_FIXES.md`
- `POWER_OPERATION_BUG_FIX.md`
- `POWER_UNITS_BUG_FIX.md`
- `DEPRECATION_FIXES_REPORT.md`
- `EVALUATE_FIXES_SUMMARY.md`
- `UNWRAP_AND_SYMBOLIC_FIXES_SUMMARY.md`
- `DERIVATIVE_UNITS_SUMMARY.md`
- `UNIT_ROUNDING_SUMMARY.md`

**Process**:
1. Check if info belongs in TODO.md
2. Check if info belongs in CLAUDE.md
3. Check if info belongs in docs/developer/
4. Delete if already documented elsewhere

### 6. ANALYSIS/INVESTIGATION REPORTS → Archive or Delete
**Action**: Mine for insights → Move to docs/ or delete

- `ADVECTION_REGRESSION_ANALYSIS.md`
- `COMPREHENSIVE_SECONDARY_ISSUES_ANALYSIS.md`
- `SECONDARY_ISSUES_DETAILED_ANALYSIS.md`
- `SECONDARY_ISSUES_INDEX.md`
- `SECONDARY_ISSUES_QUICK_REFERENCE.md`
- `PROJECTION_COMPARATIVE_ANALYSIS.md`
- `PROJECTION_ANALYSIS_FINAL_SUMMARY.md`
- `REDUCTION_OPERATIONS_ANALYSIS.md`
- `UNWRAP_INVESTIGATION.md`
- `FINAL_UNWRAP_INVESTIGATION_REPORT.md`
- `MIXIN_SEPARATION_ANALYSIS.md`
- `UNITS_CLOSURE_ANALYSIS.md`
- `UNIT_TRACKING_COMPARISON.md`
- `ARCHITECTURE_ANALYSIS_UNITS_PROTOCOL.md`

**Process**:
1. Check if insights documented elsewhere
2. If unique info, extract to appropriate doc
3. Delete

### 7. PROGRESS/STATUS REPORTS → Delete
**Action**: Delete (outdated work logs)

- `PHASE_4_PROGRESS_SUMMARY.md`
- `PHASE_4_SUMMARY.md`
- `PHASE_4_FINAL_STATUS.md`
- `PHASE_4_COMPLETE_SUMMARY.md`
- `PARAMETER_ROLLOUT_STATUS.md`
- `PARAMETER_UNITS_PROGRESS.md`
- `UNITS_REFACTOR_PROGRESS.md`
- `FINAL_CLEANUP_SUMMARY.md`
- `MATHEMATICAL_MIXIN_FIX_STATUS.md`
- `MATHEMATICAL_MIXIN_FINAL_SUMMARY.md`
- `ND_SCALING_TEST_REPORT.md`
- `UNITS_TEST_RESULTS_BASELINE.md`
- `DIMENSIONALITY_CHANGES_LOG.md`

**Reasoning**: These are historical work logs. Git history preserves them.

### 8. REFERENCE/GUIDE DOCUMENTS → Consolidate or Move
**Action**: Review and consolidate

- `COMPLETE_ANALYSIS_INDEX.md` - Index of analyses (check if needed)
- `UNIT_AWARE_OBJECTS_METHOD_TABLE.md` - Reference (move to docs/?)
- `UNIT_CONVERSION_QUICK_REFERENCE.md` - Reference (move to docs/?)
- `UNITS_COMPLETE_GUIDE.md` - User guide (move to docs/?)
- `UNITS_PRINCIPLES_ENHANCEMENT.md` - Design principles (planning/?)
- `LIGHTWEIGHT_UNITAWAREARRAY_USAGE.md` - Usage guide (docs/?)
- `DEPRECATED_METHODS_USAGE_AUDIT.md` - Audit (archive/delete)

**Process**:
1. Check if info is in official docs
2. If not, add to official docs
3. Delete original

### 9. SPECIFIC ISSUE REPORTS → Delete
**Action**: Check issue resolved → Delete

- `ANISOTROPIC_TENSOR_PATTERN.md`
- `UNITS_MAX_MIN_ISSUE.md`
- `UNITS_ND_IMPACT_REPORT.md`
- `UNIT_CONSISTENCY_COMPLETE.md`

### 10. TODO CONSOLIDATION
**Action**: Merge into `TODO.md`

- `TODO_NOTEBOOK_13_DIAGRAMS.md` → Extract items to TODO.md
- `TODO_VIEW_METHODS.md` → Extract items to TODO.md

## Directory Structure After Cleanup

```
underworld3/
├── README.md                              # Keep
├── LICENCE.md                             # Keep (merge LICENSE.md if duplicate)
├── CLAUDE.md                              # Keep
├── CONTRIBUTING.md                        # Keep
├── CHANGES.md                             # Keep
├── SPELLING_CONVENTION.md                 # Keep
├── TODO.md                                # Keep (active)
├── ARCHITECTURE_ANALYSIS.md               # Keep (recent)
├── UNWRAPPING_COMPARISON_REPORT.md        # Keep (recent)
├── UNWRAPPING_UNIFICATION_PROPOSAL.md     # Keep (recent)
├── planning/                              # Organized plans
│   ├── [design documents]
│   └── [architectural plans]
└── docs/
    ├── developer/
    │   └── [extracted reference material]
    └── ...
```

## Execution Order

1. **Check for duplicates**: LICENSE.md vs LICENCE.md
2. **Move to planning/**: All design/plan documents
3. **Extract TODOs**: Consolidate TODO items from scattered files
4. **Mine references**: Extract useful info from guides/references → docs/
5. **Delete completed work**: Progress reports, bug fix summaries
6. **Delete analyses**: After mining for insights
7. **Final sweep**: Any remaining files not categorized
8. **Update .gitignore**: Prevent future clutter

## Mining Process for Each Document

Before deleting/archiving any document:
1. **Scan for TODO items** → Add to TODO.md
2. **Scan for design decisions** → Check if in CLAUDE.md or docs/
3. **Scan for useful patterns** → Check if documented
4. **Scan for gotchas/warnings** → Check if captured
5. Only then: Delete or archive

## Post-Cleanup Validation

- [ ] All TODO items consolidated
- [ ] All architectural decisions documented
- [ ] All useful patterns captured
- [ ] Planning documents organized
- [ ] Root directory clean (< 15 files)
- [ ] Git commit: "Major cleanup: Organize 73 .md files"

---

## Additional Cleanup Items

### Check for Other Clutter
- **Python scripts**: `test_*.py`, `debug_*.py`, `reproduce_*.py` in root (DONE - none found)
- **Temporary files**: `*.tmp`, `*.bak`, `*.orig`
- **Empty directories**: Remove if unused
- **Old experimental code**: Check src/ for dead code

### Update .gitignore
Add patterns to prevent future clutter:
```
# Temporary analysis files
*_ANALYSIS.md
*_REPORT.md
*_SUMMARY.md
SESSION_SUMMARY*.md

# Debug scripts (keep in tests/ only)
/debug_*.py
/test_*.py
/reproduce_*.py
/verify_*.py

# Except explicitly versioned ones
!/TODO.md
!/README.md
!/CLAUDE.md
```
