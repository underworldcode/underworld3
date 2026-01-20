"""
Docstring parsing and rendering utilities.

Supports both markdown and NumPy/Sphinx formats with automatic detection
and conversion for different output contexts (Jupyter, terminal, pdoc3).
"""

import re
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from textwrap import dedent, indent


class DocstringFormat(Enum):
    """Detected docstring format."""
    NUMPY = "numpy"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


@dataclass
class ParsedDocstring:
    """Parsed docstring components."""
    summary: str = ""
    extended_description: str = ""
    parameters: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    returns: Optional[Tuple[str, str]] = None
    notes: str = ""
    examples: str = ""
    see_also: List[str] = field(default_factory=list)
    format: DocstringFormat = DocstringFormat.UNKNOWN


def detect_format(docstring: str) -> DocstringFormat:
    """
    Detect whether docstring is NumPy/Sphinx or Markdown format.

    Parameters
    ----------
    docstring : str
        The docstring to analyze.

    Returns
    -------
    DocstringFormat
        The detected format.
    """
    if docstring is None or not docstring.strip():
        return DocstringFormat.UNKNOWN

    # NumPy format indicators (check these first - more specific)
    numpy_patterns = [
        r"Parameters\s*\n\s*-{3,}",
        r"Returns\s*\n\s*-{3,}",
        r"Notes\s*\n\s*-{3,}",
        r"Examples\s*\n\s*-{3,}",
        r"Attributes\s*\n\s*-{3,}",
        r":math:`",
        r"\.\. math::",
    ]

    for pattern in numpy_patterns:
        if re.search(pattern, docstring):
            return DocstringFormat.NUMPY

    # Markdown format indicators
    markdown_patterns = [
        r"(?<!\$)\$(?!\$)[^$\n]+\$(?!\$)",  # Inline math $...$ (not $$)
        r"\$\$[^$]+\$\$",  # Display math $$...$$
        r"```",  # Code blocks
        r"^\s*\*\*[^*]+\*\*\s*:",  # **Bold**: style headers
    ]

    for pattern in markdown_patterns:
        if re.search(pattern, docstring, re.MULTILINE):
            return DocstringFormat.MARKDOWN

    return DocstringFormat.UNKNOWN


def parse_numpy_docstring(docstring: str) -> ParsedDocstring:
    """
    Parse a NumPy-format docstring into components.

    Parameters
    ----------
    docstring : str
        NumPy-formatted docstring.

    Returns
    -------
    ParsedDocstring
        Parsed components.
    """
    if not docstring:
        return ParsedDocstring(format=DocstringFormat.NUMPY)

    docstring = dedent(docstring).strip()
    result = ParsedDocstring(format=DocstringFormat.NUMPY)

    # Split into sections
    # Pattern matches section headers like "Parameters\n----------"
    section_pattern = r'^(\w+)\s*\n\s*-{3,}\s*\n'
    sections = re.split(section_pattern, docstring, flags=re.MULTILINE)

    # First part is summary + extended description
    preamble = sections[0].strip()
    if preamble:
        lines = preamble.split('\n\n', 1)
        result.summary = lines[0].strip()
        if len(lines) > 1:
            result.extended_description = lines[1].strip()

    # Process named sections
    i = 1
    while i < len(sections) - 1:
        section_name = sections[i].strip()
        section_content = sections[i + 1].strip() if i + 1 < len(sections) else ""
        i += 2

        if section_name.lower() == "parameters":
            result.parameters = _parse_parameter_section(section_content)
        elif section_name.lower() == "returns":
            returns_dict = _parse_parameter_section(section_content)
            if returns_dict:
                first_key = list(returns_dict.keys())[0]
                result.returns = (first_key, returns_dict[first_key][1])
        elif section_name.lower() == "notes":
            result.notes = section_content
        elif section_name.lower() == "examples":
            result.examples = section_content
        elif section_name.lower() == "see also":
            result.see_also = [line.strip() for line in section_content.split('\n') if line.strip()]

    return result


def _parse_parameter_section(content: str) -> Dict[str, Tuple[str, str]]:
    """
    Parse a Parameters or Returns section.

    Format:
        param_name : type
            Description that may span
            multiple lines.
    """
    params = {}
    if not content:
        return params

    # Pattern: name : type\n    description
    current_name = None
    current_type = None
    current_desc_lines = []

    for line in content.split('\n'):
        # Check if this is a new parameter definition
        match = re.match(r'^(\w+)\s*:\s*(.*)$', line)
        if match and not line.startswith(' '):
            # Save previous parameter
            if current_name:
                params[current_name] = (current_type, '\n'.join(current_desc_lines).strip())

            current_name = match.group(1)
            current_type = match.group(2).strip()
            current_desc_lines = []
        elif current_name and line.strip():
            # Continuation of description
            current_desc_lines.append(line.strip())

    # Save last parameter
    if current_name:
        params[current_name] = (current_type, '\n'.join(current_desc_lines).strip())

    return params


def parse_markdown_docstring(docstring: str) -> ParsedDocstring:
    """
    Parse a Markdown-format docstring into components.

    Parameters
    ----------
    docstring : str
        Markdown-formatted docstring.

    Returns
    -------
    ParsedDocstring
        Parsed components.
    """
    if not docstring:
        return ParsedDocstring(format=DocstringFormat.MARKDOWN)

    docstring = dedent(docstring).strip()
    result = ParsedDocstring(format=DocstringFormat.MARKDOWN)

    # For markdown, we primarily preserve structure
    # Split on **Section**: style headers
    lines = docstring.split('\n')

    # First non-empty line is summary
    for i, line in enumerate(lines):
        if line.strip():
            result.summary = line.strip()
            remaining = '\n'.join(lines[i+1:]).strip()
            break
    else:
        return result

    # Look for **Arguments**: or **Parameters**: sections
    param_match = re.search(
        r'\*\*(?:Arguments|Parameters)\*\*:\s*\n((?:[-*]\s+.+\n?)+)',
        remaining,
        re.IGNORECASE
    )
    if param_match:
        param_text = param_match.group(1)
        for line in param_text.split('\n'):
            # Parse "- `name`: description" or "- name: description"
            match = re.match(r'[-*]\s+`?(\w+)`?\s*:\s*(.+)', line)
            if match:
                result.parameters[match.group(1)] = ("", match.group(2).strip())

    # Everything else goes to extended description
    result.extended_description = remaining

    return result


def numpy_to_markdown(parsed: ParsedDocstring) -> str:
    """
    Convert parsed NumPy docstring to Markdown for Jupyter display.

    Transforms:
    - :math:`x` -> $x$
    - .. math:: blocks -> $$...$$ blocks
    - Parameters section -> formatted list

    Parameters
    ----------
    parsed : ParsedDocstring
        Parsed docstring components.

    Returns
    -------
    str
        Markdown-formatted string.
    """
    parts = []

    # Summary
    if parsed.summary:
        parts.append(f"**{_rst_math_to_latex(parsed.summary)}**\n")

    # Extended description
    if parsed.extended_description:
        parts.append(_rst_math_to_latex(parsed.extended_description))

    # Parameters
    if parsed.parameters:
        parts.append("\n**Parameters**\n")
        for name, (ptype, desc) in parsed.parameters.items():
            type_str = f" : `{ptype}`" if ptype else ""
            desc_converted = _rst_math_to_latex(desc)
            parts.append(f"- **{name}**{type_str} - {desc_converted}")

    # Returns
    if parsed.returns:
        rtype, rdesc = parsed.returns
        parts.append("\n**Returns**\n")
        rdesc_converted = _rst_math_to_latex(rdesc)
        parts.append(f"- `{rtype}` - {rdesc_converted}")

    # Notes
    if parsed.notes:
        parts.append("\n**Notes**\n")
        parts.append(_rst_math_to_latex(parsed.notes))

    # Examples
    if parsed.examples:
        parts.append("\n**Examples**\n")
        parts.append(f"```python\n{parsed.examples}\n```")

    # See Also
    if parsed.see_also:
        parts.append("\n**See Also**\n")
        for item in parsed.see_also:
            parts.append(f"- {item}")

    return '\n'.join(parts)


def markdown_to_numpy(text: str) -> str:
    """
    Convert Markdown docstring to NumPy format.

    Transforms:
    - $x$ -> :math:`x`
    - $$...$$ -> .. math:: blocks

    Parameters
    ----------
    text : str
        Markdown-formatted docstring.

    Returns
    -------
    str
        NumPy/RST-formatted docstring.
    """
    return _latex_to_rst_math(text)


def _rst_math_to_latex(text: str) -> str:
    """
    Convert RST math notation to LaTeX for Jupyter display.

    Parameters
    ----------
    text : str
        Text with RST math notation.

    Returns
    -------
    str
        Text with LaTeX math notation.
    """
    if not text:
        return text

    # :math:`x` -> $x$
    text = re.sub(r':math:`([^`]+)`', r'$\1$', text)

    # .. math:: block -> $$...$$ block
    def convert_math_block(match):
        content = match.group(1)
        # Remove common leading whitespace
        lines = content.split('\n')
        # Find minimum indentation (excluding empty lines)
        non_empty = [line for line in lines if line.strip()]
        if non_empty:
            min_indent = min(len(line) - len(line.lstrip()) for line in non_empty)
            lines = [line[min_indent:] if len(line) > min_indent else line for line in lines]
        content = '\n'.join(line.strip() for line in lines if line.strip())
        return f'\n$$\n{content}\n$$\n'

    text = re.sub(
        r'\.\.\s*math::\s*\n((?:[ \t]+.+\n?)+)',
        convert_math_block,
        text
    )

    return text


def _latex_to_rst_math(text: str) -> str:
    """
    Convert LaTeX notation to RST for pdoc3/Sphinx.

    Parameters
    ----------
    text : str
        Text with LaTeX math notation.

    Returns
    -------
    str
        Text with RST math notation.
    """
    if not text:
        return text

    # $$...$$ -> .. math:: block (do this first to avoid conflicts)
    def convert_display_math(match):
        content = match.group(1).strip()
        indented = '\n'.join('   ' + line for line in content.split('\n'))
        return f'\n.. math::\n\n{indented}\n'

    text = re.sub(r'\$\$([^$]+)\$\$', convert_display_math, text, flags=re.DOTALL)

    # $x$ -> :math:`x` (but not $$)
    text = re.sub(r'(?<!\$)\$([^$\n]+)\$(?!\$)', r':math:`\1`', text)

    return text


def in_jupyter() -> bool:
    """
    Detect if running in Jupyter environment.

    Returns
    -------
    bool
        True if in Jupyter, False otherwise.
    """
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        shell_name = shell.__class__.__name__
        return shell_name in ('ZMQInteractiveShell',)
    except (ImportError, AttributeError):
        return False


def markdown_to_plain(text: str) -> str:
    """
    Convert markdown to plain text for terminal display.

    Parameters
    ----------
    text : str
        Markdown-formatted text.

    Returns
    -------
    str
        Plain text with formatting stripped.
    """
    if not text:
        return text

    # Remove bold/italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)

    # Convert math to plain
    text = re.sub(r'\$\$([^$]+)\$\$', r'\n\1\n', text, flags=re.DOTALL)
    text = re.sub(r'\$([^$]+)\$', r'\1', text)
    text = re.sub(r':math:`([^`]+)`', r'\1', text)

    # Remove code block markers
    text = re.sub(r'```\w*\n?', '', text)

    # Convert bullet points
    text = re.sub(r'^- ', '  * ', text, flags=re.MULTILINE)

    return text


def render_docstring(docstring: str, target: str = "auto") -> str:
    """
    Render a docstring for the specified target.

    Parameters
    ----------
    docstring : str
        The docstring to render.
    target : str
        Output target: "jupyter", "terminal", "rst", or "auto".
        "auto" detects the environment.

    Returns
    -------
    str
        Rendered docstring.
    """
    if not docstring:
        return ""

    docstring = dedent(docstring).strip()

    # Auto-detect target
    if target == "auto":
        target = "jupyter" if in_jupyter() else "terminal"

    # Detect source format
    fmt = detect_format(docstring)

    if target == "jupyter":
        if fmt == DocstringFormat.NUMPY:
            parsed = parse_numpy_docstring(docstring)
            return numpy_to_markdown(parsed)
        else:
            # Already markdown or unknown - return as-is
            return docstring

    elif target == "terminal":
        if fmt == DocstringFormat.NUMPY:
            parsed = parse_numpy_docstring(docstring)
            md = numpy_to_markdown(parsed)
            return markdown_to_plain(md)
        else:
            return markdown_to_plain(docstring)

    elif target == "rst":
        if fmt == DocstringFormat.MARKDOWN:
            return markdown_to_numpy(docstring)
        else:
            # Already NumPy/RST format
            return docstring

    return docstring
