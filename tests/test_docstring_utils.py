"""
Tests for the docstring parsing and conversion utilities.

These tests verify that:
1. Format detection works correctly for NumPy and Markdown docstrings
2. Math notation conversion works both ways:
   - :math:`...` and .. math:: (RST) -> $...$ and $$...$$ (LaTeX/Markdown)
   - $...$ and $$...$$ (LaTeX) -> :math:`...` and .. math:: (RST)
3. Parsed docstring components are extracted correctly
"""

import pytest
from underworld3.utilities.docstring_utils import (
    detect_format,
    DocstringFormat,
    parse_numpy_docstring,
    parse_markdown_docstring,
    numpy_to_markdown,
    markdown_to_numpy,
    render_docstring,
    in_jupyter,
    markdown_to_plain,
)


class TestFormatDetection:
    """Tests for docstring format detection."""

    def test_detect_numpy_with_parameters(self):
        """NumPy format detected by Parameters section."""
        doc = """
        Summary.

        Parameters
        ----------
        x : int
            Description.
        """
        assert detect_format(doc) == DocstringFormat.NUMPY

    def test_detect_numpy_with_returns(self):
        """NumPy format detected by Returns section."""
        doc = """
        Summary.

        Returns
        -------
        int
            The result.
        """
        assert detect_format(doc) == DocstringFormat.NUMPY

    def test_detect_numpy_with_rst_math(self):
        """NumPy format detected by :math: notation."""
        doc = "The value is :math:`x^2`."
        assert detect_format(doc) == DocstringFormat.NUMPY

    def test_detect_numpy_with_math_block(self):
        """NumPy format detected by .. math:: block."""
        doc = """
        Summary.

        .. math::

            x^2 + y^2 = r^2
        """
        assert detect_format(doc) == DocstringFormat.NUMPY

    def test_detect_markdown_with_inline_math(self):
        """Markdown format detected by $...$ notation."""
        doc = "The value is $x^2$."
        assert detect_format(doc) == DocstringFormat.MARKDOWN

    def test_detect_markdown_with_display_math(self):
        """Markdown format detected by $$...$$ notation."""
        doc = """
        Summary.

        $$x^2 + y^2 = r^2$$
        """
        assert detect_format(doc) == DocstringFormat.MARKDOWN

    def test_detect_markdown_with_code_block(self):
        """Markdown format detected by ``` code blocks."""
        doc = """
        Summary.

        ```python
        print("hello")
        ```
        """
        assert detect_format(doc) == DocstringFormat.MARKDOWN

    def test_detect_unknown_plain_text(self):
        """Plain text without markers returns UNKNOWN."""
        doc = "Just a plain description without any special formatting."
        assert detect_format(doc) == DocstringFormat.UNKNOWN

    def test_detect_empty_or_none(self):
        """Empty or None docstrings return UNKNOWN."""
        assert detect_format("") == DocstringFormat.UNKNOWN
        assert detect_format(None) == DocstringFormat.UNKNOWN
        assert detect_format("   ") == DocstringFormat.UNKNOWN


class TestNumpyParsing:
    """Tests for NumPy docstring parsing."""

    def test_parse_summary(self):
        """Summary is extracted correctly."""
        doc = """
        This is the summary line.

        Extended description here.
        """
        parsed = parse_numpy_docstring(doc)
        assert parsed.summary == "This is the summary line."

    def test_parse_extended_description(self):
        """Extended description is extracted."""
        doc = """
        Summary.

        This is the extended description
        that spans multiple lines.
        """
        parsed = parse_numpy_docstring(doc)
        assert "extended description" in parsed.extended_description

    def test_parse_parameters(self):
        """Parameters section is parsed correctly."""
        doc = """
        Summary.

        Parameters
        ----------
        x : float
            The x coordinate.
        y : int, optional
            The y coordinate.
        """
        parsed = parse_numpy_docstring(doc)
        assert "x" in parsed.parameters
        assert "y" in parsed.parameters
        assert parsed.parameters["x"][0] == "float"
        assert "x coordinate" in parsed.parameters["x"][1]

    def test_parse_notes(self):
        """Notes section is extracted."""
        doc = """
        Summary.

        Notes
        -----
        Some important notes here.
        """
        parsed = parse_numpy_docstring(doc)
        assert "important notes" in parsed.notes

    def test_parse_examples(self):
        """Examples section is extracted."""
        doc = """
        Summary.

        Examples
        --------
        >>> func(1, 2)
        3
        """
        parsed = parse_numpy_docstring(doc)
        assert "func(1, 2)" in parsed.examples


class TestMathConversion:
    """Tests for math notation conversion."""

    def test_rst_inline_to_latex(self):
        """Convert :math:`...` to $...$."""
        parsed = parse_numpy_docstring("The value is :math:`x^2`.")
        md = numpy_to_markdown(parsed)
        assert "$x^2$" in md
        assert ":math:" not in md

    def test_rst_block_to_latex(self):
        """Convert .. math:: block to $$...$$."""
        doc = """
        Summary.

        Notes
        -----
        The equation:

        .. math::

            E = mc^2
        """
        parsed = parse_numpy_docstring(doc)
        md = numpy_to_markdown(parsed)
        assert "$$" in md
        assert "E = mc^2" in md

    def test_latex_inline_to_rst(self):
        """Convert $...$ to :math:`...`."""
        md = "The value is $x^2$."
        rst = markdown_to_numpy(md)
        assert ":math:`x^2`" in rst
        # Check that $ signs are gone (but not $$ which would be display math)
        assert "$x^2$" not in rst

    def test_latex_block_to_rst(self):
        """Convert $$...$$ to .. math:: block."""
        md = "Summary.\n\n$$E = mc^2$$"
        rst = markdown_to_numpy(md)
        assert ".. math::" in rst
        assert "E = mc^2" in rst


class TestRenderDocstring:
    """Tests for the render_docstring function."""

    def test_render_jupyter_converts_numpy(self):
        """NumPy docstring converted to markdown for Jupyter."""
        doc = "Summary with :math:`x^2`."
        rendered = render_docstring(doc, target="jupyter")
        assert "$x^2$" in rendered

    def test_render_rst_converts_markdown(self):
        """Markdown docstring converted to RST for pdoc3."""
        doc = "Summary with $x^2$."
        rendered = render_docstring(doc, target="rst")
        assert ":math:`x^2`" in rendered

    def test_render_terminal_strips_formatting(self):
        """Terminal output strips markdown formatting."""
        doc = "**Bold** and *italic* with $x^2$."
        rendered = render_docstring(doc, target="terminal")
        assert "**" not in rendered
        assert "*" not in rendered

    def test_render_empty_returns_empty(self):
        """Empty docstring returns empty string."""
        assert render_docstring("", target="jupyter") == ""
        assert render_docstring(None, target="jupyter") == ""


class TestMarkdownToPlain:
    """Tests for markdown stripping."""

    def test_strip_bold(self):
        """Bold markers removed."""
        assert markdown_to_plain("**bold**") == "bold"

    def test_strip_italic(self):
        """Italic markers removed."""
        assert markdown_to_plain("*italic*") == "italic"

    def test_strip_inline_math(self):
        """Math delimiters removed."""
        assert markdown_to_plain("$x^2$") == "x^2"

    def test_strip_code_blocks(self):
        """Code block markers removed."""
        text = "```python\ncode\n```"
        result = markdown_to_plain(text)
        assert "```" not in result
