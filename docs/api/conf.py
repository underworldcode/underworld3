# Sphinx configuration for Underworld3 API Documentation
# Uses MyST Markdown for familiar syntax
# NOTE: underworld3 is installed via 'pixi run build' before docs build
# Do NOT add source to sys.path - it causes circular import issues

# =============================================================================
# Project Information
# =============================================================================
project = 'Underworld3'
copyright = '2025, Underworld Team'
author = 'Underworld Team'
version = '0.9'
release = '0.9b'

# =============================================================================
# Extensions
# NOTE: myst_nb includes myst_parser, don't include both!
# =============================================================================
extensions = [
    'myst_nb',                  # MyST Markdown + Notebooks (includes myst_parser)
    'sphinx.ext.autodoc',       # Extract docstrings
    'sphinx.ext.napoleon',      # NumPy/Google style docstrings
    'sphinx.ext.mathjax',       # LaTeX math rendering
    'sphinx.ext.viewcode',      # [source] links
    'sphinx.ext.intersphinx',   # Link to external docs
    'sphinx_math_dollar',       # $...$ math in docstrings
]

# =============================================================================
# MyST Configuration - familiar syntax from mystmd
# =============================================================================
myst_enable_extensions = [
    "dollarmath",       # $...$ and $$...$$ in .md files
    "colon_fence",      # ::: for directives
    "deflist",          # Definition lists
    "fieldlist",        # Field lists
    "tasklist",         # - [ ] task lists
    "attrs_inline",     # {#id .class}
]

# Allow dollar signs for math (like mystmd)
myst_dmath_double_inline = True

# Source file types
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'myst-nb',
    '.ipynb': 'myst-nb',
}

# =============================================================================
# Napoleon - NumPy docstrings
# =============================================================================
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# =============================================================================
# Autodoc
# =============================================================================
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': False,
    'show-inheritance': True,
}
autodoc_typehints = 'description'
autodoc_class_signature = 'separated'

# =============================================================================
# Intersphinx - link to external docs
# =============================================================================
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'sympy': ('https://docs.sympy.org/latest/', None),
    'petsc4py': ('https://petsc.org/release/petsc4py/', None),
}

# =============================================================================
# HTML - Furo theme
# =============================================================================
html_theme = 'furo'
html_static_path = []
html_title = "Underworld3 API"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2962ff",
        "color-brand-content": "#2962ff",
    },
    "dark_css_variables": {
        "color-brand-primary": "#5c8aff",
        "color-brand-content": "#5c8aff",
    },
}

# =============================================================================
# Math - sphinx-math-dollar handles $...$ conversion, MathJax renders \(...\)
# =============================================================================
mathjax3_config = {
    'tex': {
        'inlineMath': [['\\(', '\\)']],
        'displayMath': [['\\[', '\\]']],
    }
}

# =============================================================================
# Other Settings
# =============================================================================
# Don't fail on missing references during development
nitpicky = False
suppress_warnings = ['ref.python', 'myst.header']

# Don't execute notebooks (they may require PETSc/MPI)
nb_execution_mode = 'off'

# Exclude patterns
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
