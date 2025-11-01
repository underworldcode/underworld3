# Underworld3 Quarto Documentation Site Proposal

**Date**: 2025-10-25
**Status**: Proposal for Discussion
**Based On**: EMSC-QuartoBook-Course Template Structure
**Target**: Familiar to QuartoBook users, Optimized for Underworld3 Community

---

## Executive Summary

This proposal outlines upgrading the Underworld3 documentation infrastructure to match the maturity and usability of the EMSC-QuartoBook-Course template while preserving Underworld3's existing structure and brand identity.

**Key Goals:**
- Recognizable to users familiar with QuartoBook-Course, Zero-to-Python, EMSC-2010
- Easy for contributors to follow established patterns
- Integrated with pixi task system for automated builds
- Professional styling reflecting Underworld3 geodynamics focus
- Clear governance and contribution guidelines

**Current Status:** ✅ Already using Quarto book format. We're enhancing the infrastructure, not migrating.

---

## Part 1: Current State Analysis

### What's Already Working ✅

| Aspect | Current Status | Notes |
|--------|---|---|
| **Quarto Format** | ✅ Using Quarto book (`type: book`) | Base infrastructure ready |
| **Structure** | ✅ Three-tier (beginner/advanced/developer) | Excellent audience segmentation |
| **Branding** | ✅ Logo, colors, fonts defined | MansoursNightmare.png + theme |
| **Navigation** | ✅ Sidebar + navbar with search | User-friendly structure |
| **Tutorials** | ✅ Jupyter notebooks integrated | Interactive examples present |
| **Chapters** | ✅ Part/chapter organization | Logical progression |

### What Needs Enhancement ⚠️

| Area | Current | Needed |
|------|---------|--------|
| **Build Infrastructure** | Manual build | Pixi integration + scripts |
| **Styling System** | Basic theme.scss | Layered CSS like QuartoBook |
| **_extensions/** | Not present | Quarto extensions for custom blocks |
| **Contributor Guide** | Partial | Clear HOW-TO for adding docs |
| **Build Documentation** | Minimal | How to build locally and deploy |
| **Automated Workflows** | Not present | GitHub Actions for CI/CD |
| **Version Management** | Not implemented | Docs version tracking |
| **Asset Management** | Basic | Organized media/ folder structure |

---

## Part 2: Proposed Directory Structure

### Overview

```
underworld3/
├── docs/                           # Documentation root (Quarto project)
│   ├── _quarto.yml                 # Main Quarto config (ENHANCED)
│   ├── _variables.yml              # Shared variables (EXISTS)
│   ├── index.qmd                   # Landing page (EXISTS)
│   │
│   ├── _extensions/                # NEW: Quarto extensions
│   │   ├── callout-blocks/         # Custom callout styling
│   │   └── code-tabs/              # Tabbed code blocks
│   │
│   ├── assets/                     # NEW: Organized styling
│   │   ├── theme.scss              # MOVED: Main theme (currently in root)
│   │   ├── colors.scss             # NEW: Color palette
│   │   ├── fonts.scss              # NEW: Typography
│   │   ├── components.scss         # NEW: Component styles
│   │   ├── css/                    # Compiled CSS output
│   │   └── images/                 # NEW: SVGs, icons, etc.
│   │
│   ├── media/                      # NEW: Organized media
│   │   ├── logos/                  # Logos and branding
│   │   ├── diagrams/               # Architecture diagrams
│   │   ├── screenshots/            # UI screenshots
│   │   └── animations/             # Animated gifs/webp
│   │
│   ├── beginner/                   # Getting Started (EXISTS)
│   │   ├── index.qmd
│   │   ├── installation.qmd
│   │   ├── quickstart.qmd
│   │   ├── tutorials/              # Interactive notebooks
│   │   │   ├── Notebook_Index.ipynb
│   │   │   ├── 1-Meshes.ipynb
│   │   │   ├── ... (other tutorials)
│   │   │   └── _quarto.yml         # NEW: Notebook-specific config
│   │   └── exercises/              # NEW: Standalone exercises
│   │       └── exercise-1.ipynb
│   │
│   ├── advanced/                   # Advanced Usage (EXISTS)
│   │   ├── index.qmd
│   │   ├── parallel-computing.qmd
│   │   ├── performance.qmd
│   │   ├── complex-rheologies.qmd
│   │   ├── mesh-adaptation.qmd
│   │   ├── troubleshooting.qmd
│   │   ├── api-patterns.qmd
│   │   └── examples/               # NEW: Advanced examples
│   │       └── geothermal-model.ipynb
│   │
│   ├── developer/                  # Developer Guide (EXISTS)
│   │   ├── index.qmd
│   │   ├── CONTRIBUTING.md         # NEW: Contribution guidelines
│   │   ├── contributing.qmd
│   │   ├── development-setup.qmd
│   │   ├── developer-faq.qmd       # NEW: Common questions
│   │   │
│   │   ├── subsystems/             # Architecture deep-dives
│   │   │   ├── _quarto.yml         # NEW: Subsystem navigation
│   │   │   ├── meshing.qmd
│   │   │   ├── discretisation.qmd
│   │   │   ├── solvers.qmd
│   │   │   ├── swarm-system.qmd
│   │   │   ├── variables.qmd
│   │   │   ├── model-orchestration.qmd
│   │   │   └── ... (others)
│   │   │
│   │   ├── guidelines/             # Development standards
│   │   │   ├── _quarto.yml
│   │   │   ├── code-style.qmd      # NEW: Coding conventions
│   │   │   ├── testing-patterns.qmd
│   │   │   ├── documentation.qmd   # NEW: Doc writing guide
│   │   │   └── performance-optimization.qmd
│   │   │
│   │   ├── advanced/               # Advanced developer topics
│   │   │   ├── _quarto.yml
│   │   │   ├── petsc-integration.qmd
│   │   │   ├── solver-development.qmd
│   │   │   ├── architecture.qmd    # NEW: System architecture
│   │   │   └── ... (others)
│   │   │
│   │   ├── reference/              # Reference docs
│   │   │   ├── _quarto.yml
│   │   │   ├── build-system.qmd
│   │   │   ├── glossary.qmd
│   │   │   ├── troubleshooting.qmd
│   │   │   └── faq.qmd             # NEW: Frequently asked questions
│   │   │
│   │   └── review-process/         # NEW: Code review docs
│   │       ├── review-overview.qmd
│   │       ├── reviewers-guide.qmd
│   │       └── authors-guide.qmd
│   │
│   ├── reviews/                    # NEW: Code review archive (from earlier work)
│   │   ├── README.md
│   │   └── 2025-10/
│   │       ├── REDUCTION-OPERATIONS-REVIEW.md
│   │       └── ... (review documents)
│   │
│   ├── examples/                   # NEW: Standalone examples directory
│   │   ├── basic/
│   │   │   ├── 01-simple-poisson.ipynb
│   │   │   └── 02-stokes-flow.ipynb
│   │   ├── advanced/
│   │   │   ├── 01-coupled-thermal-flow.ipynb
│   │   │   └── 02-multi-material-flow.ipynb
│   │   └── benchmarks/
│   │       ├── solcx-benchmark.ipynb
│   │       └── blankenbach-benchmark.ipynb
│   │
│   ├── _build/                     # Quarto build output (gitignored)
│   ├── _freeze/                    # Quarto computation cache
│   ├── _quarto/                    # Quarto metadata
│   │
│   └── scripts/                    # NEW: Build & dev scripts
│       ├── build-local.sh          # Build locally
│       ├── build-production.sh     # Production build
│       ├── watch-docs.sh           # Auto-rebuild on file changes
│       └── validate-docs.py        # NEW: Check doc quality
│
├── .github/workflows/              # NEW: CI/CD workflows
│   ├── docs-build.yml              # Build & test docs
│   ├── docs-deploy.yml             # Deploy to GitHub Pages
│   └── docs-pr-check.yml           # Check PRs for doc issues
│
├── pixi.toml                       # ENHANCED: Add doc tasks
│
└── CONTRIBUTING.md                 # Enhanced with doc contribution section
```

---

## Part 3: Key Improvements from QuartoBook Pattern

### 1. **Build System Integration** (from QuartoBook-Course)

**Create**: `docs/scripts/build-local.sh`
```bash
#!/bin/bash
# Build documentation locally with proper environment

cd docs
quarto render . --to html
echo "✓ Documentation built to _build/"
echo "Open _build/index.html in your browser"
```

**Create**: `pixi.toml` additions
```toml
[tasks]
docs-build = "cd docs && quarto render ."
docs-watch = "cd docs && quarto preview"
docs-validate = "python docs/scripts/validate-docs.py"
```

**Benefits:**
- Users familiar with `pixi run docs-build`
- Consistent with QuartoBook-Course workflow
- Integrates with CI/CD naturally

### 2. **Layered Styling System** (from QuartoBook-Course)

**Replicate QuartoBook's approach** (currently simplified):

**Keep**: `docs/assets/theme.scss` as main entry point
**Split into**:
- `docs/assets/_colors.scss` - Underworld3 color palette
- `docs/assets/_typography.scss` - Fonts and text styles
- `docs/assets/_components.scss` - Custom callouts, cards, buttons
- `docs/assets/theme.scss` - Imports all, applies to Quarto

**Benefits:**
- Matches QuartoBook contributor expectations
- Easy to maintain color scheme centrally
- Supports different themes (dark mode future)

### 3. **Quarto Extensions** (from QuartoBook-Course)

**Add**: `docs/_extensions/` directory with custom filters

Examples to consider:
- **Info/Warning blocks**: `{{< info >}} Content {{< /info >}}`
- **Code language tabs**: Show same example in Python/Julia/MATLAB
- **Emphasis markers**: Important concepts, warnings, pro-tips

**Current QuartoBook-Course uses:**
- Custom code styling
- Special callout blocks
- Language-specific formatting

**For Underworld3:**
- Geodynamics terminology definitions
- Physics concept callouts
- Performance tips boxes
- API reference styling

### 4. **Clear Contributor Documentation**

**Create**: `docs/developer/DOCUMENTATION-GUIDE.md`

```markdown
# How to Contribute Documentation

This guide shows how to add documentation following Underworld3's patterns.

## Quick Start for Contributors

1. **Clone and setup**:
   ```bash
   pixi install
   ```

2. **Write your content** in the appropriate folder:
   - Beginner tutorial: `docs/beginner/tutorials/`
   - Advanced guide: `docs/advanced/`
   - Developer reference: `docs/developer/subsystems/`

3. **Preview locally**:
   ```bash
   pixi run docs-watch
   ```

4. **Build and validate**:
   ```bash
   pixi run docs-build
   pixi run docs-validate
   ```

5. **Submit PR** with docs changes

## File Organization

See directory structure above. Each section has clear purposes.

## Formatting Standards

- Use `.qmd` files (Quarto Markdown)
- Include YAML front matter with title and description
- Cross-reference other docs using `{{< ref "path/to/file" >}}`
- Code examples should be executable or clearly marked
- Mathematical notation: Use LaTeX within `$...$`

## Special Styles

See `docs/assets/components.scss` for available custom styles.
```

### 5. **Automated Validation**

**Create**: `docs/scripts/validate-docs.py`

```python
#!/usr/bin/env python3
"""Validate documentation quality and consistency."""

import re
from pathlib import Path

def check_links(docs_dir):
    """Verify internal links are valid."""
    pass

def check_code_examples(docs_dir):
    """Ensure code examples are syntactically valid."""
    pass

def check_missing_cross_refs(docs_dir):
    """Find dangling references and missing links."""
    pass

if __name__ == "__main__":
    docs = Path("docs")
    print("Validating documentation...")
    # Run checks
```

---

## Part 4: Branding & Personalization

### Colors & Logo (Preserve Underworld3 Identity)

**Keep existing**:
- Logo: `docs/media/logos/MansoursNightmare.png` (or preferred version)
- Primary color: `#883344` (maroon) - very geodynamics appropriate
- Accent colors: Expand the palette (see below)

**Proposed Underworld3 Color Scheme**:
```scss
// Geodynamics-inspired palette
$primary: #883344;           // Maroon (mantle-ish)
$secondary: #C58812;         // Gold (crust/lithosphere)
$accent: #2E7D32;            // Green (fertile, growth)
$info: #0277BD;              // Blue (water, fluids)
$warning: #F57C00;           // Orange (heat, deformation)
$danger: #C62828;            // Red (deep stress)
$success: #388E3C;           // Light green

// Typography
$primary-font: "Jost", sans-serif;
$mono-font: "JetBrains Mono", monospace;
```

### Visual Hierarchy

Match QuartoBook-Course's approach:
- **Hero banner** on landing page (color gradient)
- **Card-based navigation** for main sections
- **Consistent icon usage** (from Quarto's default library)
- **Clear emphasis** on code examples

---

## Part 5: Integration with pixi

### Current pixi.toml additions needed

```toml
[tasks]
# Documentation tasks
docs-build = "cd docs && quarto render . --to html"
docs-watch = "cd docs && quarto preview --port 4173"
docs-validate = "python docs/scripts/validate-docs.py"
docs-clean = "rm -rf docs/_build docs/_freeze"

# CI/CD preparation
docs-check-links = "cd docs && quarto render . --check-links"
docs-github-pages = "cd docs && quarto render . --to html && echo '_build' > _build/.nojekyll"
```

### GitHub Actions Workflows

**Create**: `.github/workflows/docs-build.yml`
```yaml
name: Build Documentation

on:
  push:
    branches: [development, main]
    paths: ['docs/**', 'pixi.toml']
  pull_request:
    paths: ['docs/**']

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: prefix-dev/setup-pixi@v0.5.0
      - name: Build docs
        run: pixi run docs-build
      - name: Validate docs
        run: pixi run docs-validate
```

---

## Part 6: Migration Path (Minimal Changes)

### Phase 1: Infrastructure (Week 1)

- [ ] Create `docs/assets/` structure with split SCSS files
- [ ] Create `docs/_extensions/` with basic custom blocks
- [ ] Add build scripts in `docs/scripts/`
- [ ] Update `pixi.toml` with doc tasks
- [ ] Create GitHub Actions workflows

### Phase 2: Documentation (Week 2)

- [ ] Create `docs/developer/DOCUMENTATION-GUIDE.md`
- [ ] Create `docs/developer/CONTRIBUTING.md`
- [ ] Add missing sections:
  - `docs/developer/guidelines/code-style.qmd`
  - `docs/developer/advanced/architecture.qmd`
  - `docs/developer/reference/faq.qmd`

### Phase 3: Content Organization (Week 3)

- [ ] Organize `docs/examples/` directory
- [ ] Move review documents into `docs/reviews/`
- [ ] Add cross-references between sections
- [ ] Enhance landing page with better navigation

### Phase 4: Validation & Polish (Week 4)

- [ ] Test all build scripts locally
- [ ] Verify GitHub Actions workflows
- [ ] Run `docs-validate` and fix issues
- [ ] User testing with contributors

---

## Part 7: Questions & Clarifications

Before proceeding, I'd like to confirm:

1. **Branding**:
   - Happy with Mansoursnightmare.png logo? Or prefer different image?
   - Color palette suggestions beyond existing maroon?
   - Any specific geodynamics visual themes?

2. **Scope**:
   - Should we include API reference generation (Sphinx-to-Quarto bridge)?
   - Do you want Binder integration updated?
   - GitHub Pages auto-deploy or manual process?

3. **Timeline**:
   - Priority: Just structure now? Or full implementation?
   - Resources: Who will update/maintain docs?

4. **Features**:
   - Version management (docs for different Underworld3 versions)?
   - Dark mode support?
   - Multilingual documentation (future)?

5. **Contributor Experience**:
   - Is local preview (pixi run docs-watch) important?
   - Should validation be strict or warnings-only?

---

## Part 8: Benefits Summary

### For Users
- ✅ Familiar navigation if they know QuartoBook-Course
- ✅ Clear learning progression (beginner → advanced → developer)
- ✅ Beautiful, consistent styling
- ✅ Responsive design for mobile/tablet
- ✅ Integrated search functionality

### For Contributors
- ✅ Clear guidelines for adding documentation
- ✅ Reusable templates and examples
- ✅ Simple build process (`pixi run docs-build`)
- ✅ Validation to catch issues early
- ✅ Professional-looking results without effort

### For Project
- ✅ Production-ready documentation site
- ✅ Automated CI/CD for docs
- ✅ Scalable for growth
- ✅ Maintainable structure
- ✅ Brand consistency

---

## Next Steps

1. **Feedback**: Review this proposal for conflicts with existing plans
2. **Prioritization**: Decide which phases to tackle first
3. **Implementation**: Proceed with structured approach
4. **Testing**: Validate with external contributors

---

**Prepared by**: Claude Code
**Based on**: EMSC-QuartoBook-Course Template
**Relevant to**: docs/developer/ organization and contributor experience
**Status**: Ready for discussion and feedback
