# Documentation Platform Decision - January 2025

**Status**: DECIDED - Moving forward with Sphinx + MyST + ReadTheDocs
**Date**: 2025-01-13

## Summary

Evaluated documentation platforms for Underworld3 API documentation with the following requirements:
- NumPy-style docstrings with mathematics
- ReadTheDocs deployment (requested by stakeholders)
- Familiar syntax (user knows MyST/mystmd)
- Jupyter notebook integration with Binder launch

## Decision: Sphinx + MyST + ReadTheDocs

**Recommended stack:**

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Markup | MyST Markdown (`.md`) | Familiar syntax from mystmd |
| Math | `$...$` and `$$...$$` | Same as mystmd |
| API docs | autodoc + napoleon | NumPy docstrings work perfectly |
| Notebooks | myst-nb | Direct `.ipynb` rendering |
| Theme | Furo | Modern, dark mode, mobile responsive |
| Hosting | ReadTheDocs | Native Sphinx support, versioning |

## Key Findings

### 1. pdoc3 (Current)
- ✅ Simple setup
- ✅ Display math works (`.. math::`)
- ❌ Inline math (`:math:`) does NOT render
- ❌ No cross-references
- ❌ No ReadTheDocs integration

### 2. Sphinx + MyST (Recommended)
- ✅ MyST Markdown syntax (same as mystmd)
- ✅ `$...$` math works everywhere
- ✅ Full NumPy docstring support via Napoleon
- ✅ Cross-references between docs
- ✅ Native ReadTheDocs support
- ✅ Notebook integration with Binder buttons
- ✅ Module docstrings auto-populate overviews
- ⚠️ autodoc directives need `{eval-rst}` wrapper

### 3. Module Docstrings Already Good
Checked key modules - they already have well-written docstrings:
- `underworld3.systems.ddt` - Overview, class list, Notes, See Also
- `underworld3.systems.solvers` - Full description with math, Examples
- `underworld3.coordinates` - Key components, See Also
- `underworld3.discretisation` - Classes, Functions lists

These can be pulled directly into docs via `automodule::`.

## Test Build Location

A working MyST + Sphinx + Furo test exists at:
```
/tmp/sphinx_test_uw3/
├── conf.py          # Sphinx config with MyST enabled
├── index.md         # Landing page (MyST)
├── solvers.md       # API docs with automodule
├── systems_ddt.md   # API docs with automodule
├── coordinates.md   # API docs with automodule
└── _build/html/     # Generated HTML
```

**To view**: `open /tmp/sphinx_test_uw3/_build/html/index.html`

**To remove**: `rm -rf /tmp/sphinx_test_uw3`

## MyST Syntax Example

```markdown
# Solvers

```{eval-rst}
.. automodule:: underworld3.systems.solvers
   :no-members:
```

## Stokes Solver

The momentum equation: $-\nabla \cdot [\boldsymbol{\tau} - p\mathbf{I}] = \mathbf{f}$

```{eval-rst}
.. autoclass:: underworld3.systems.solvers.SNES_Stokes
   :members:
```
```

## conf.py Key Settings

```python
extensions = [
    'myst_nb',                  # MyST + Notebooks (includes myst_parser)
    'sphinx.ext.autodoc',       # Docstring extraction
    'sphinx.ext.napoleon',      # NumPy docstrings
    'sphinx.ext.mathjax',       # Math rendering
    'sphinx.ext.intersphinx',   # External doc links
    'sphinx_math_dollar',       # $...$ math in docstrings (REQUIRED)
]

myst_enable_extensions = [
    "dollarmath",       # $...$ math in MyST Markdown files
    "colon_fence",      # ::: directives
    "deflist",          # Definition lists
]

# sphinx-math-dollar converts $...$ to \(...\), MathJax renders \(...\)
mathjax3_config = {
    'tex': {
        'inlineMath': [['\\(', '\\)']],
        'displayMath': [['\\[', '\\]']],
    }
}

html_theme = 'furo'
nb_execution_mode = 'off'  # Don't re-run notebooks
```

### Math in Docstrings

**Important**: MyST's `dollarmath` extension only handles `.md` files, NOT docstrings.
For `$...$` math in Python docstrings, you need `sphinx-math-dollar` (from SymPy):

```bash
pip install sphinx-math-dollar
```

This allows writing natural LaTeX in docstrings:
```python
def my_function():
    """
    Computes $\nabla \cdot \mathbf{u}$ (divergence of velocity).
    """
```

## Next Steps

1. **Create `docs/` directory** in underworld3 repo with Sphinx config
2. **Add `.readthedocs.yaml`** for ReadTheDocs integration
3. **Migrate existing content** from Quarto docs
4. **Add Binder buttons** for notebook pages
5. **Connect to ReadTheDocs** for automatic builds

## Docstring Improvements Identified

1. **CoordinateSystem class** - Has mathematical symbols not properly marked up
   - Should use `$...$` for `R`, `X`, `N`, rotation matrices (with `sphinx-math-dollar`)

2. **Template/ExpressionDescriptor** - Now have physics-aware documentation
   - Added `self.__doc__ = description` to expose instance-specific docs
   - `F0`, `F1`, `PF0` in Stokes solver now show what they represent physically
   - Uses `$...$` math syntax for readable source code

## Related Work

- Docstring refactoring completed for: SNES solvers, constitutive_models, meshing, systems/ddt
- Commits pushed to `uw3-release-candidate` branch
- See inventory at `docs/docstrings/inventory.json`
