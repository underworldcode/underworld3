# Sphinx configuration for Underworld3 Documentation
# Unified build: User guides, tutorials, and API reference
# NOTE: underworld3 is installed via 'pixi run build' before docs build

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
    'sphinx_design',            # Cards, grids, tabs for modern layouts
    'sphinxcontrib.mermaid',    # Mermaid diagrams
]

# =============================================================================
# MyST Configuration
# =============================================================================
myst_enable_extensions = [
    "dollarmath",       # $...$ and $$...$$ in .md files
    "colon_fence",      # ::: for directives
    "deflist",          # Definition lists
    "fieldlist",        # Field lists
    "tasklist",         # - [ ] task lists
    "attrs_inline",     # {#id .class}
]

# Allow dollar signs for math
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
html_static_path = ['_static']
html_css_files = ['custom.css']  # Custom styling (smaller headings)
html_js_files = ['report-issue.js']  # Page-specific issue reporting
html_extra_path = ['media']  # Copy media folder (including pyvista HTML embeds) to build root
html_title = "Underworld3"
html_logo = "assets/MansoursNightmare.png"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2962ff",
        "color-brand-content": "#2962ff",
        # Font sizing - smaller than Furo defaults
        "font-size--normal": "14px",
        "font-size--small": "12.5px",
        "font-size--small--2": "11.5px",
        "font-size--small--3": "10.5px",
        "font-size--small--4": "9.5px",
    },
    "dark_css_variables": {
        "color-brand-primary": "#5c8aff",
        "color-brand-content": "#5c8aff",
        # Same font sizing for dark mode
        "font-size--normal": "14px",
        "font-size--small": "12.5px",
        "font-size--small--2": "11.5px",
        "font-size--small--3": "10.5px",
        "font-size--small--4": "9.5px",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    # GitHub integration - Edit/View Source buttons
    "source_repository": "https://github.com/underworldcode/underworld3",
    "source_branch": "development",
    "source_directory": "docs/",
    # Footer icons
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/underworldcode/underworld3",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
        {
            "name": "Report Issue",
            "url": "https://github.com/underworldcode/underworld3/issues/new?labels=documentation&template=docs-issue.md",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path d="M8 9.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Z"></path>
                    <path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM1.5 8a6.5 6.5 0 1 0 13 0 6.5 6.5 0 0 0-13 0Z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

# =============================================================================
# Math - sphinx-math-dollar handles $...$ conversion, MathJax renders
# =============================================================================
mathjax3_config = {
    'tex': {
        'inlineMath': [['\\(', '\\)']],
        'displayMath': [['\\[', '\\]']],
    }
}

# =============================================================================
# Notebook execution
# =============================================================================
# Don't execute notebooks (they may require PETSc/MPI)
# NOTE: Notebook outputs are stripped for documentation builds to avoid
# rendering issues. If you want to show example output, add it to markdown cells.
nb_execution_mode = 'off'

# =============================================================================
# Exclude patterns
# =============================================================================
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '.quarto',
    'api/_build',
    # Skip Quarto-specific files
    '_quarto.yml',
    '_variables.yml',
    '*.qmd',
    # Skip presentations (handled separately)
    'presentations/**',
    'slides/**',
    # Skip planning/implementation docs
    'planning/**',
    'plans/**',
    'QUARTO-*.md',
    # Skip reviews and internal docs
    'reviews/**',
    'docstrings/**',
    # Skip examples (not needed for main docs)
    'examples/**',
]

# =============================================================================
# Other Settings
# =============================================================================
nitpicky = False
suppress_warnings = ['ref.python', 'myst.header', 'myst.xref_missing']
