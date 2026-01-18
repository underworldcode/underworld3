# Docstring Conversion Plan: Markdown to NumPy/Sphinx

## Goals

1. Convert existing markdown docstrings to NumPy/Sphinx format
2. Maintain pretty-printed output in Jupyter via `.view()` method
3. Work with pdoc3 for mathematical documentation generation
4. Support intelligent format detection for `.view()` output

## Current State

- Docstrings use markdown with LaTeX math (`$...$` and `$$...$$`)
- `.view()` method uses `IPython.display.Markdown()` directly
- Located in `src/underworld3/utilities/_api_tools.py` (base implementation)
- Individual classes override `_object_viewer()` for custom display

## NumPy Docstring Format

Standard sections:
```python
def function(param1, param2):
    """
    Short summary line.

    Extended description with math support using :math:`LaTeX`.

    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type, optional
        Description of param2.

    Returns
    -------
    type
        Description of return value.

    Examples
    --------
    >>> function(1, 2)
    3

    Notes
    -----
    Mathematical formulation:

    .. math::

        \\nabla \\cdot \\mathbf{u} = 0

    See Also
    --------
    related_function : Description.
    """
```

## Architecture

### 1. Docstring Parser Module

Create `src/underworld3/utilities/docstring_utils.py`:

```python
"""
Docstring parsing and rendering utilities.

Supports both markdown and NumPy/Sphinx formats with automatic detection
and conversion for different output contexts (Jupyter, terminal, pdoc3).
"""

import re
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class DocstringFormat(Enum):
    """Detected docstring format."""
    NUMPY = "numpy"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


@dataclass
class ParsedDocstring:
    """Parsed docstring components."""
    summary: str
    extended_description: str
    parameters: Dict[str, Tuple[str, str]]  # name -> (type, description)
    returns: Optional[Tuple[str, str]]  # (type, description)
    notes: str
    examples: str
    math_blocks: List[str]
    see_also: List[str]
    format: DocstringFormat


def detect_format(docstring: str) -> DocstringFormat:
    """
    Detect whether docstring is NumPy/Sphinx or Markdown format.

    Heuristics:
    - NumPy: Contains "Parameters\n----------" or "Returns\n-------"
    - Markdown: Contains "$...$" math or "```" code blocks
    """
    if docstring is None:
        return DocstringFormat.UNKNOWN

    # NumPy format indicators
    numpy_patterns = [
        r"Parameters\s*\n\s*-{3,}",
        r"Returns\s*\n\s*-{3,}",
        r"Notes\s*\n\s*-{3,}",
        r":math:`",
        r"\.\. math::",
    ]

    for pattern in numpy_patterns:
        if re.search(pattern, docstring):
            return DocstringFormat.NUMPY

    # Markdown format indicators
    markdown_patterns = [
        r"\$[^$]+\$",  # Inline math
        r"\$\$[^$]+\$\$",  # Display math
        r"```",  # Code blocks
        r"^\s*[-*]\s+",  # Bullet lists
    ]

    for pattern in markdown_patterns:
        if re.search(pattern, docstring, re.MULTILINE):
            return DocstringFormat.MARKDOWN

    return DocstringFormat.UNKNOWN


def parse_numpy_docstring(docstring: str) -> ParsedDocstring:
    """Parse a NumPy-format docstring into components."""
    # Implementation details...
    pass


def parse_markdown_docstring(docstring: str) -> ParsedDocstring:
    """Parse a Markdown-format docstring into components."""
    # Implementation details...
    pass


def numpy_to_markdown(parsed: ParsedDocstring) -> str:
    """
    Convert parsed NumPy docstring to Markdown for Jupyter display.

    Transforms:
    - :math:`x` -> $x$
    - .. math:: blocks -> $$...$$ blocks
    - Parameters section -> **Parameters** with formatted list
    """
    pass


def numpy_to_rst(parsed: ParsedDocstring) -> str:
    """
    Convert parsed NumPy docstring to RST for Sphinx/pdoc3.
    """
    pass
```

### 2. Enhanced View Mixin

Update `src/underworld3/utilities/_api_tools.py`:

```python
from .docstring_utils import (
    detect_format,
    parse_numpy_docstring,
    parse_markdown_docstring,
    numpy_to_markdown,
    DocstringFormat,
)


class ViewableMixin:
    """
    Mixin providing intelligent .view() method for documentation display.

    Features:
    - Auto-detects docstring format (NumPy or Markdown)
    - Converts to appropriate output format based on context
    - Renders math properly in Jupyter
    - Falls back gracefully in terminal
    """

    @class_or_instance_method
    def view(self_or_cls, class_documentation=False):
        """
        Display documentation with proper rendering.

        In Jupyter: Renders as rich Markdown with LaTeX math
        In terminal: Prints formatted text
        """
        docstring = self_or_cls.__doc__
        if docstring is None:
            return

        # Detect format
        fmt = detect_format(docstring)

        # Parse based on format
        if fmt == DocstringFormat.NUMPY:
            parsed = parse_numpy_docstring(docstring)
            display_text = numpy_to_markdown(parsed)
        else:
            # Already markdown or unknown - use as-is
            display_text = docstring

        # Detect environment and display appropriately
        if _in_jupyter():
            from IPython.display import Markdown, display
            display(Markdown(display_text))
        else:
            # Terminal - strip markdown formatting
            print(_markdown_to_plain(display_text))

        # Instance-specific details
        if not inspect.isclass(self_or_cls):
            self_or_cls._object_viewer()


def _in_jupyter() -> bool:
    """Detect if running in Jupyter environment."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell in ('ZMQInteractiveShell', 'TerminalInteractiveShell')
    except (ImportError, AttributeError):
        return False


def _markdown_to_plain(text: str) -> str:
    """Convert markdown to plain text for terminal display."""
    # Strip markdown formatting but preserve structure
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
    text = re.sub(r'\$([^$]+)\$', r'\1', text)  # Inline math
    text = re.sub(r'\$\$([^$]+)\$\$', r'\1', text)  # Display math
    return text
```

### 3. Math Rendering

For pdoc3 compatibility, math needs to be in RST format:
- Inline: `:math:\`x^2\``
- Display: `.. math::\n\n   x^2 + y^2 = r^2`

For Jupyter display via `.view()`:
- Inline: `$x^2$`
- Display: `$$x^2 + y^2 = r^2$$`

Conversion functions:

```python
def rst_math_to_latex(text: str) -> str:
    """Convert RST math notation to LaTeX for Jupyter."""
    # :math:`x` -> $x$
    text = re.sub(r':math:`([^`]+)`', r'$\1$', text)

    # .. math:: block -> $$...$$ block
    def convert_math_block(match):
        content = match.group(1).strip()
        # Remove leading whitespace from each line
        lines = [line.strip() for line in content.split('\n')]
        return '$$\n' + '\n'.join(lines) + '\n$$'

    text = re.sub(
        r'\.\. math::\s*\n((?:\s+.+\n?)+)',
        convert_math_block,
        text
    )
    return text


def latex_to_rst_math(text: str) -> str:
    """Convert LaTeX notation to RST for pdoc3."""
    # $x$ -> :math:`x`
    text = re.sub(r'\$([^$]+)\$', r':math:`\1`', text)

    # $$...$$ -> .. math:: block
    def convert_display_math(match):
        content = match.group(1).strip()
        indented = '\n'.join('   ' + line for line in content.split('\n'))
        return f'.. math::\n\n{indented}\n'

    text = re.sub(r'\$\$([^$]+)\$\$', convert_display_math, text)
    return text
```

## Migration Strategy

### Phase 1: Infrastructure (Week 1)

1. Create `docstring_utils.py` with parsing and conversion functions
2. Add unit tests for format detection and conversion
3. Update `_api_tools.py` with enhanced `ViewableMixin`

### Phase 2: Core Classes (Week 2)

Convert docstrings for high-visibility classes:
1. `Model` class
2. `Mesh` classes
3. `MeshVariable` / `SwarmVariable`
4. Constitutive models

### Phase 3: Solvers and Functions (Week 3)

1. All solver classes (Stokes, Poisson, Advection, etc.)
2. Function evaluation utilities
3. Swarm classes

### Phase 4: Validation and Documentation (Week 4)

1. Run pdoc3 and verify output
2. Test `.view()` in Jupyter for all converted classes
3. Update any remaining markdown-only docstrings
4. Add documentation for the docstring format guidelines

## Example Conversion

### Before (Markdown)

```python
class DiffusionSolver:
    """
    Solves the diffusion equation:

    $$\\frac{\\partial T}{\\partial t} = \\kappa \\nabla^2 T$$

    where $T$ is temperature and $\\kappa$ is thermal diffusivity.

    **Arguments:**
    - `mesh`: The computational mesh
    - `T`: Temperature field (MeshVariable)
    - `kappa`: Diffusivity (scalar or function)
    """
```

### After (NumPy/Sphinx)

```python
class DiffusionSolver:
    """
    Solves the diffusion equation.

    This solver implements the heat diffusion equation for temperature
    evolution in a domain.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh.
    T : MeshVariable
        Temperature field to solve for.
    kappa : float or sympy.Expr
        Thermal diffusivity coefficient.

    Notes
    -----
    The governing equation is:

    .. math::

        \\frac{\\partial T}{\\partial t} = \\kappa \\nabla^2 T

    where :math:`T` is temperature and :math:`\\kappa` is thermal diffusivity.

    Examples
    --------
    >>> solver = DiffusionSolver(mesh, T, kappa=1.0)
    >>> solver.solve(dt=0.01)
    """
```

## pdoc3 Configuration

Create `pdoc_config.py` or use command line:

```bash
pdoc --html --math --output-dir docs/api src/underworld3
```

The `--math` flag enables MathJax rendering of `:math:` directives.

## Testing

```python
# tests/test_docstring_utils.py

def test_detect_numpy_format():
    docstring = '''
    Summary.

    Parameters
    ----------
    x : int
        Description.
    '''
    assert detect_format(docstring) == DocstringFormat.NUMPY


def test_detect_markdown_format():
    docstring = '''
    Summary.

    The equation is $x^2 + y^2 = r^2$.
    '''
    assert detect_format(docstring) == DocstringFormat.MARKDOWN


def test_numpy_to_markdown_math():
    docstring = 'The value is :math:`x^2`.'
    parsed = parse_numpy_docstring(docstring)
    md = numpy_to_markdown(parsed)
    assert '$x^2$' in md
```

## Backward Compatibility

The system supports both formats during transition:
- Existing markdown docstrings continue to work
- New/updated docstrings use NumPy format
- `.view()` handles both transparently
- pdoc3 works best with NumPy format but tolerates markdown

## Files to Modify

1. **New**: `src/underworld3/utilities/docstring_utils.py`
2. **Update**: `src/underworld3/utilities/_api_tools.py`
3. **Update**: All class docstrings (gradual migration)
4. **New**: `tests/test_docstring_utils.py`
5. **New**: `docs/developer/docstring_guidelines.md`
