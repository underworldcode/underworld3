# Quarto Documentation Infrastructure: Implementation Assessment

**Date**: 2025-10-25
**Status**: Assessment & Priority Planning
**Based on**: Current Underworld3 structure vs EMSC-QuartoBook-Course patterns

---

## Executive Summary

Underworld3's documentation is **already well-organized** and uses Quarto effectively. The current structure includes:
- ✅ Quarto book format with three-tier documentation (beginner/advanced/developer)
- ✅ Comprehensive table of contents with logical navigation
- ✅ Extensive developer guides covering subsystems and advanced topics
- ✅ Code review process and governance documentation
- ✅ Media assets including logos and diagrams

**Key Gaps** from QuartoBook-Course patterns:
- ⚠️ SCSS not split into layered components (all in single `theme.scss`)
- ⚠️ No `_extensions/` directory for custom Quarto filters
- ⚠️ No automated build scripts integrated with pixi
- ⚠️ No GitHub Actions workflows for CI/CD
- ⚠️ Media files not organized by category
- ⚠️ No formal contributor guide for documentation

**Recommendation**: Incremental enhancement focused on **infrastructure automation** rather than restructuring. The content organization is already sound.

---

## Part 1: Current State Analysis

### What's Working Well ✅

| Feature | Status | Notes |
|---------|--------|-------|
| **Quarto Format** | ✅ | Type: book, properly configured |
| **Navigation** | ✅ | Sidebar + navbar with search enabled |
| **Three-tier Structure** | ✅ | Beginner → Advanced → Developer (clear progression) |
| **Table of Contents** | ✅ | 60+ documents organized in logical chapters |
| **Branding** | ✅ | Logo (MansoursNightmare.png), colors defined, assets present |
| **Content Depth** | ✅ | Developer subsystems well-documented (19+ files) |
| **Code Organization** | ✅ | Code reviews, architecture docs, examples |
| **Media Assets** | ✅ | Logo, diagrams, screenshots present |

### Current File Organization

```
docs/
├── _quarto.yml                 # ✅ Main config - well structured
├── _variables.yml              # ✅ Shared variables
├── index.qmd                   # ✅ Landing page
├── assets/
│   ├── theme.scss              # ⚠️ Single monolithic file
│   └── (images, fonts)         # ✅ Present
├── media/
│   ├── MansoursNightmare.png   # ✅ Logo
│   ├── pyvista/                # ✅ Organized
│   └── (various images)        # ✅ Present but not categorized
├── beginner/                   # ✅ Getting Started (well-organized)
├── advanced/                   # ✅ Advanced Usage (10+ docs)
├── developer/                  # ✅ Developer Guide (20+ docs)
├── examples/                   # ✅ Standalone examples
├── reviews/                    # ✅ Code review archive
└── planning/                   # Historical planning docs
```

### What Needs Enhancement ⚠️

| Area | Current | Needed | Priority |
|------|---------|--------|----------|
| **SCSS Organization** | Single file | Layered system (colors, fonts, components) | Medium |
| **Quarto Extensions** | Not present | Custom filters for callouts, code tabs | Low |
| **Build Automation** | Manual | Pixi tasks + shell scripts | **High** |
| **CI/CD Workflows** | Not present | GitHub Actions for build/deploy | **High** |
| **Documentation Guide** | Partial | Clear "How to Write Docs" guide | Medium |
| **Media Organization** | Flat structure | Organized by type (logos/, diagrams/, screenshots/) | Low |
| **Version Management** | Not implemented | Multi-version docs support | Very Low (future) |
| **Dark Mode** | Not implemented | Theme switching support | Very Low (future) |

---

## Part 2: Priority Assessment

### Tier 1: Infrastructure Automation (High Priority)

**Why**: Enables contributors to work efficiently and sets up CI/CD for production-ready documentation.

#### 1.1 Create Build Scripts
**Files to create**:
- `docs/scripts/build-local.sh` - Local development build
- `docs/scripts/build-production.sh` - Production-optimized build
- `docs/scripts/watch-docs.sh` - Auto-rebuild on file changes

**Integration**: Pixi tasks for easy access
```bash
pixi run docs-build      # Build once
pixi run docs-watch      # Local preview
pixi run docs-validate   # Check quality
```

**Effort**: 1-2 hours
**Impact**: High (enables contributor workflow)

#### 1.2 GitHub Actions Workflows
**Files to create**:
- `.github/workflows/docs-build.yml` - Build & validate on push
- `.github/workflows/docs-deploy.yml` - Deploy to GitHub Pages
- `.github/workflows/docs-pr-check.yml` - Check PRs for doc issues

**Benefits**:
- Automatic build on every commit
- Early detection of broken documentation
- Automated deployment when merged to main

**Effort**: 2-3 hours
**Impact**: High (ensures quality and deployment)

#### 1.3 Update pixi.toml
**Add documentation tasks**:
```toml
[tasks]
docs-build = "cd docs && quarto render . --to html"
docs-watch = "cd docs && quarto preview --port 4173"
docs-validate = "python docs/scripts/validate-docs.py"
docs-clean = "rm -rf docs/_build docs/_freeze"
docs-check-links = "cd docs && quarto render . --check-links"
```

**Effort**: 30 minutes
**Impact**: High (standardizes workflow)

### Tier 2: Styling & Extensibility (Medium Priority)

**Why**: Makes documentation maintainable and allows custom styling patterns similar to QuartoBook.

#### 2.1 Split SCSS into Layers
**Create**:
- `docs/assets/_colors.scss` - Color palette (maroon, gold, green, etc.)
- `docs/assets/_typography.scss` - Font definitions and text styles
- `docs/assets/_components.scss` - Custom callout, card, button styles
- `docs/assets/theme.scss` - Main entry point importing all

**Benefits**:
- Easier to maintain colors across documentation
- Support for future dark mode
- Matches QuartoBook patterns (familiar to contributors)

**Effort**: 2 hours
**Impact**: Medium (maintainability + familiarity)

#### 2.2 Create Quarto Extensions (Optional)
**Examples**:
- Custom callout blocks (Info, Warning, Tip for geodynamics topics)
- Code language tabs (show same example in Python/Julia)
- Performance tip boxes
- API reference styling

**Status**: Optional - only if specific needs arise
**Effort**: 3-4 hours per extension
**Impact**: Low-Medium (nice-to-have)

### Tier 3: Documentation & Organization (Medium Priority)

**Why**: Guides contributors on how to work with the documentation system.

#### 3.1 Contributor Documentation Guide
**Create**: `docs/developer/DOCUMENTATION-GUIDE.md`

**Content**:
- Quick start for contributors
- Where to add different types of content
- File naming conventions
- Formatting standards (YAML front matter, cross-references, code examples)
- Available custom styles

**Effort**: 2-3 hours
**Impact**: Medium (enables contributions)

#### 3.2 Media Organization
**Reorganize** `docs/media/`:
```
media/
├── logos/           # Brand assets
├── diagrams/        # Architecture, workflow diagrams
├── screenshots/     # UI/output screenshots
├── animations/      # GIFs, WebP animations
└── pyvista/        # 3D visualization outputs
```

**Effort**: 1 hour
**Impact**: Low (organizational)

#### 3.3 Validation Script
**Create**: `docs/scripts/validate-docs.py`

**Checks**:
- Internal links are valid
- Code examples compile
- Missing cross-references
- Broken image links
- Docstring references up-to-date

**Effort**: 2-3 hours
**Impact**: Medium (quality assurance)

### Tier 4: Future Enhancements (Very Low Priority)

**Status**: Plan for future, not needed now

- Version management (docs for different UW3 versions)
- Dark mode support
- Multilingual documentation
- API reference generation from docstrings

---

## Part 3: Recommended Implementation Plan

### Phase 1: Automation Foundation (Week 1) - START HERE
- [ ] Create `docs/scripts/build-local.sh` and related scripts
- [ ] Update `pixi.toml` with doc tasks
- [ ] Setup basic GitHub Actions workflows
- [ ] Create `docs/scripts/validate-docs.py`

**Time**: 3-4 hours
**Impact**: Unblocks contributor workflow, enables CI/CD

### Phase 2: Documentation (Week 2)
- [ ] Write `docs/developer/DOCUMENTATION-GUIDE.md`
- [ ] Split SCSS into layered components
- [ ] Organize media/ directory
- [ ] Update _quarto.yml if needed

**Time**: 4-5 hours
**Impact**: Enables contributions, improves maintainability

### Phase 3: Polish & Testing (Week 3)
- [ ] Test all build scripts locally
- [ ] Verify GitHub Actions workflows
- [ ] Run validation script against docs
- [ ] Create example contribution (test the guide)

**Time**: 2-3 hours
**Impact**: Ensures everything works

### Phase 4: Optional Enhancements (Future)
- [ ] Create custom Quarto extensions if specific needs identified
- [ ] Implement version management system
- [ ] Add dark mode support

**Time**: Deferred
**Impact**: Nice-to-have improvements

---

## Part 4: Implementation Quick-Start

### To Begin (Recommended Order)

1. **First**: Create build scripts (Phase 1.1)
   - Takes ~1 hour
   - Immediately enables contributors
   - Foundation for everything else

2. **Second**: Update pixi.toml (Phase 1.3)
   - Takes ~30 minutes
   - Standardizes workflow
   - Makes scripts discoverable

3. **Third**: Setup GitHub Actions (Phase 1.2)
   - Takes ~2-3 hours
   - Enables automated CI/CD
   - Professional documentation infrastructure

4. **Then**: Documentation guide (Phase 2.1)
   - Takes ~2-3 hours
   - Explains how to use the system
   - Enables community contributions

### Each Phase Can Be Done Independently

- **Phase 1** can be done alone (minimal doc structure already present)
- **Phase 2** doesn't depend on Phase 1 (styling improvements are optional)
- **Phase 3** validates that Phases 1-2 work correctly

---

## Part 5: QuartoBook Patterns Comparison

### What We're Replicating From QuartoBook

| Pattern | QuartoBook | Underworld3 | Status |
|---------|-----------|------------|--------|
| Quarto book format | ✅ | ✅ | Already done |
| Three-tier content | ✅ (tutorial/intermediate/reference) | ✅ (beginner/advanced/developer) | Already matches |
| Build scripts | ✅ | ⚠️ | Needs scripts |
| Pixi integration | ✅ | ⚠️ | Needs tasks |
| GitHub Actions | ✅ | ❌ | Not present |
| Layered SCSS | ✅ | ⚠️ | Needs splitting |
| Contributor guide | ✅ | ⚠️ | Partial only |

**Key Insight**: We're not restructuring - we're adding automation and documentation on top of already-solid content organization.

---

## Part 6: File Creation Checklist

### Phase 1 Files
- [ ] `docs/scripts/build-local.sh`
- [ ] `docs/scripts/build-production.sh`
- [ ] `docs/scripts/watch-docs.sh`
- [ ] `docs/scripts/validate-docs.py`
- [ ] `.github/workflows/docs-build.yml`
- [ ] `.github/workflows/docs-deploy.yml`
- [ ] `.github/workflows/docs-pr-check.yml`
- [ ] Updated `pixi.toml` (docs section)

### Phase 2 Files
- [ ] `docs/assets/_colors.scss`
- [ ] `docs/assets/_typography.scss`
- [ ] `docs/assets/_components.scss`
- [ ] Updated `docs/assets/theme.scss` (imports)
- [ ] `docs/developer/DOCUMENTATION-GUIDE.md`
- [ ] Reorganized `docs/media/` subdirectories

### Total New Files**: 14-15
### Total Modified Files**: 3-4 (theme.scss, pixi.toml, _quarto.yml, media structure)

---

## Part 7: Success Criteria

### Phase 1 Success
- ✅ `pixi run docs-build` builds documentation without errors
- ✅ `pixi run docs-watch` enables local preview
- ✅ GitHub Actions run on push and PR
- ✅ `pixi run docs-validate` identifies common issues

### Phase 2 Success
- ✅ SCSS changes applied without breaking styling
- ✅ Documentation guide is clear and usable
- ✅ Media files organized and linked correctly
- ✅ First-time contributor can follow guide successfully

### Phase 3 Success
- ✅ All build scripts tested and working
- ✅ GitHub Actions workflows validated
- ✅ Test contribution added to verify guide works
- ✅ Documentation site builds and deploys correctly

---

## Summary

**Current Status**: Documentation is well-organized and ready for enhancement.

**Recommendation**: Focus on **Phase 1 (Automation)** first because:
1. It's highest impact for contributor experience
2. It's relatively quick to implement (3-4 hours)
3. It unblocks other improvements
4. It establishes CI/CD pipeline for quality assurance

**Not Needed Now**:
- Complete restructuring (current structure is good)
- Custom Quarto extensions (content doesn't require them yet)
- Version management (single current version is fine)
- Dark mode (can be added later)

**Next Step**: User clarifies if this approach matches their intent, then we proceed with Phase 1 implementation.

---

**Document Version**: 1.0
**Created**: 2025-10-25
**Status**: Ready for user feedback before implementation
