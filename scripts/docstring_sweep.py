#!/usr/bin/env python3
"""
Docstring Sweep Tool for Underworld3

Scans Python/Cython files to:
1. Inventory all functions, classes, and methods
2. Extract existing docstrings and inline comments
3. Generate skeleton docstrings in NumPy format
4. Output a review queue for interactive enrichment

Usage:
    python scripts/docstring_sweep.py [files...]

    # Pilot run on specific files:
    python scripts/docstring_sweep.py \
        src/underworld3/systems/solvers.py \
        src/underworld3/discretisation/*.py \
        src/underworld3/ckdtree.pyx
"""

import ast
import re
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum


class DocStatus(Enum):
    """Documentation status for a code element."""
    NONE = "none"           # No docstring
    MINIMAL = "minimal"     # Has docstring but very short
    PARTIAL = "partial"     # Has some sections, missing others
    COMPLETE = "complete"   # Has all expected sections


class NeedsFlag(Enum):
    """Flags for what human input is needed."""
    OVERVIEW = "NEEDS_OVERVIEW"
    MATH = "NEEDS_MATH"
    EXAMPLE = "NEEDS_EXAMPLE"
    REFERENCE = "NEEDS_REFERENCE"
    PARAMETERS = "NEEDS_PARAMETERS"
    RETURNS = "NEEDS_RETURNS"


@dataclass
class Parameter:
    """Represents a function parameter."""
    name: str
    type_hint: Optional[str] = None
    default: Optional[str] = None
    description: str = ""


@dataclass
class CodeElement:
    """Represents a function, class, or method."""
    name: str
    kind: str  # 'function', 'class', 'method', 'property'
    file: str
    line: int
    signature: str
    parameters: List[Parameter] = field(default_factory=list)
    returns: Optional[str] = None
    existing_docstring: Optional[str] = None
    harvested_comments: List[str] = field(default_factory=list)
    status: DocStatus = DocStatus.NONE
    needs: List[str] = field(default_factory=list)
    parent_class: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    is_public: bool = True

    def priority(self) -> int:
        """Return priority score (lower = higher priority)."""
        score = 0
        if not self.is_public:
            score += 100  # Private/internal = lower priority
        if self.kind == 'class':
            score -= 20  # Classes are important
        if self.kind == 'method' and self.name == '__init__':
            score -= 10  # Constructors are important
        if 'Solver' in self.name or 'solver' in self.file:
            score -= 15  # Solvers are core API
        if self.status == DocStatus.NONE:
            score -= 5  # Undocumented = higher priority
        return score


def parse_type_annotation(node) -> Optional[str]:
    """Convert AST annotation node to string."""
    if node is None:
        return None
    return ast.unparse(node)


def extract_parameters(func_node: ast.FunctionDef) -> List[Parameter]:
    """Extract parameters from a function definition."""
    params = []
    args = func_node.args

    # Positional args
    defaults_offset = len(args.args) - len(args.defaults)
    for i, arg in enumerate(args.args):
        if arg.arg == 'self' or arg.arg == 'cls':
            continue
        default_idx = i - defaults_offset
        default = None
        if default_idx >= 0 and default_idx < len(args.defaults):
            try:
                default = ast.unparse(args.defaults[default_idx])
            except:
                default = "..."
        params.append(Parameter(
            name=arg.arg,
            type_hint=parse_type_annotation(arg.annotation),
            default=default
        ))

    # *args
    if args.vararg:
        params.append(Parameter(
            name=f"*{args.vararg.arg}",
            type_hint=parse_type_annotation(args.vararg.annotation)
        ))

    # **kwargs
    if args.kwarg:
        params.append(Parameter(
            name=f"**{args.kwarg.arg}",
            type_hint=parse_type_annotation(args.kwarg.annotation)
        ))

    return params


def extract_return_type(func_node: ast.FunctionDef) -> Optional[str]:
    """Extract return type annotation."""
    return parse_type_annotation(func_node.returns)


def get_function_signature(func_node: ast.FunctionDef) -> str:
    """Generate a clean function signature string."""
    params = []
    args = func_node.args

    defaults_offset = len(args.args) - len(args.defaults)
    for i, arg in enumerate(args.args):
        param_str = arg.arg
        if arg.annotation:
            param_str += f": {ast.unparse(arg.annotation)}"
        default_idx = i - defaults_offset
        if default_idx >= 0 and default_idx < len(args.defaults):
            try:
                param_str += f" = {ast.unparse(args.defaults[default_idx])}"
            except:
                param_str += " = ..."
        params.append(param_str)

    if args.vararg:
        params.append(f"*{args.vararg.arg}")
    if args.kwarg:
        params.append(f"**{args.kwarg.arg}")

    sig = f"({', '.join(params)})"
    if func_node.returns:
        sig += f" -> {ast.unparse(func_node.returns)}"

    return sig


def analyze_docstring(docstring: Optional[str]) -> tuple[DocStatus, List[str]]:
    """Analyze a docstring and determine what's missing."""
    if not docstring:
        return DocStatus.NONE, [NeedsFlag.OVERVIEW.value, NeedsFlag.PARAMETERS.value]

    needs = []
    docstring_lower = docstring.lower()

    # Check for key sections
    has_params = 'parameters' in docstring_lower or 'args:' in docstring_lower
    has_returns = 'returns' in docstring_lower or 'return:' in docstring_lower
    has_math = '.. math::' in docstring or '$$' in docstring or r'\(' in docstring
    has_example = 'example' in docstring_lower or '>>>' in docstring
    has_notes = 'notes' in docstring_lower or 'note:' in docstring_lower

    # Short docstrings are minimal
    lines = [l for l in docstring.split('\n') if l.strip()]
    if len(lines) <= 2 and not has_params:
        if not has_params:
            needs.append(NeedsFlag.PARAMETERS.value)
        return DocStatus.MINIMAL, needs

    # Check what's missing
    if not has_params:
        needs.append(NeedsFlag.PARAMETERS.value)
    if not has_returns:
        needs.append(NeedsFlag.RETURNS.value)

    if needs:
        return DocStatus.PARTIAL, needs

    return DocStatus.COMPLETE, []


def extract_inline_comments(source: str, start_line: int, end_line: int) -> List[str]:
    """Extract inline comments from a code region."""
    lines = source.split('\n')[start_line-1:end_line]
    comments = []
    for line in lines:
        # Match inline comments (not docstrings)
        match = re.search(r'#\s*(.+)$', line)
        if match:
            comment = match.group(1).strip()
            # Skip trivial comments
            if len(comment) > 10 and not comment.startswith('---'):
                comments.append(comment)
    return comments


def parse_python_file(filepath: Path) -> List[CodeElement]:
    """Parse a Python file and extract all code elements."""
    elements = []

    try:
        source = filepath.read_text()
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"  Syntax error in {filepath}: {e}")
        return []

    def process_node(node, parent_class=None):
        if isinstance(node, ast.ClassDef):
            # Extract class docstring
            docstring = ast.get_docstring(node)
            status, needs = analyze_docstring(docstring)

            decorators = [ast.unparse(d) for d in node.decorator_list]

            elem = CodeElement(
                name=node.name,
                kind='class',
                file=str(filepath),
                line=node.lineno,
                signature=f"class {node.name}",
                existing_docstring=docstring,
                status=status,
                needs=needs,
                decorators=decorators,
                is_public=not node.name.startswith('_')
            )
            elements.append(elem)

            # Process methods
            for child in ast.iter_child_nodes(node):
                process_node(child, parent_class=node.name)

        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            docstring = ast.get_docstring(node)
            status, needs = analyze_docstring(docstring)

            # Determine kind
            decorators = [ast.unparse(d) for d in node.decorator_list]
            if 'property' in decorators or 'cached_property' in decorators:
                kind = 'property'
            elif parent_class:
                kind = 'method'
            else:
                kind = 'function'

            # Get end line for comment extraction
            end_line = node.end_lineno if hasattr(node, 'end_lineno') else node.lineno + 20
            comments = extract_inline_comments(source, node.lineno, end_line)

            elem = CodeElement(
                name=node.name,
                kind=kind,
                file=str(filepath),
                line=node.lineno,
                signature=get_function_signature(node),
                parameters=extract_parameters(node),
                returns=extract_return_type(node),
                existing_docstring=docstring,
                harvested_comments=comments[:5],  # Limit to 5
                status=status,
                needs=needs,
                parent_class=parent_class,
                decorators=decorators,
                is_public=not node.name.startswith('_')
            )
            elements.append(elem)

    for node in ast.iter_child_nodes(tree):
        process_node(node)

    return elements


def parse_cython_file(filepath: Path) -> List[CodeElement]:
    """Parse a Cython .pyx file using regex (AST doesn't work for Cython)."""
    elements = []
    source = filepath.read_text()

    # Pattern for function/method definitions
    func_pattern = re.compile(
        r'^(\s*)((?:cdef|cpdef|def)\s+\w+\s*\([^)]*\)[^:]*:)\s*$',
        re.MULTILINE
    )

    # Pattern for class definitions
    class_pattern = re.compile(
        r'^(\s*)((?:cdef\s+)?class\s+(\w+)[^:]*:)\s*$',
        re.MULTILINE
    )

    lines = source.split('\n')

    # Find classes
    for match in class_pattern.finditer(source):
        indent = len(match.group(1))
        class_line = match.group(2)
        class_name = match.group(3)
        line_no = source[:match.start()].count('\n') + 1

        # Try to extract docstring (next non-empty line with quotes)
        docstring = None
        for i in range(line_no, min(line_no + 5, len(lines))):
            line = lines[i].strip()
            if line.startswith('"""') or line.startswith("'''"):
                # Find end of docstring
                doc_lines = []
                quote = line[:3]
                if line.endswith(quote) and len(line) > 6:
                    docstring = line[3:-3]
                else:
                    doc_lines.append(line[3:])
                    for j in range(i+1, len(lines)):
                        if lines[j].strip().endswith(quote):
                            doc_lines.append(lines[j].strip()[:-3])
                            break
                        doc_lines.append(lines[j])
                    docstring = '\n'.join(doc_lines)
                break
            elif line and not line.startswith('#'):
                break

        status, needs = analyze_docstring(docstring)

        elem = CodeElement(
            name=class_name,
            kind='class',
            file=str(filepath),
            line=line_no,
            signature=class_line.strip(),
            existing_docstring=docstring,
            status=status,
            needs=needs,
            is_public=not class_name.startswith('_')
        )
        elements.append(elem)

    # Find functions (simplified - doesn't track parent class well)
    for match in func_pattern.finditer(source):
        func_line = match.group(2)
        line_no = source[:match.start()].count('\n') + 1

        # Extract function name
        name_match = re.search(r'(?:cdef|cpdef|def)\s+(\w+)', func_line)
        if not name_match:
            continue
        func_name = name_match.group(1)

        # Skip if it's a method inside a class (indented)
        indent = len(match.group(1))

        # Try to extract docstring
        docstring = None
        for i in range(line_no, min(line_no + 5, len(lines))):
            if i >= len(lines):
                break
            line = lines[i].strip()
            if line.startswith('"""') or line.startswith("'''"):
                quote = line[:3]
                if line.endswith(quote) and len(line) > 6:
                    docstring = line[3:-3]
                else:
                    doc_lines = [line[3:]]
                    for j in range(i+1, len(lines)):
                        if lines[j].strip().endswith(quote):
                            doc_lines.append(lines[j].strip()[:-3])
                            break
                        doc_lines.append(lines[j])
                    docstring = '\n'.join(doc_lines)
                break
            elif line and not line.startswith('#'):
                break

        status, needs = analyze_docstring(docstring)

        kind = 'method' if indent > 0 else 'function'

        elem = CodeElement(
            name=func_name,
            kind=kind,
            file=str(filepath),
            line=line_no,
            signature=func_line.strip(),
            existing_docstring=docstring,
            status=status,
            needs=needs,
            is_public=not func_name.startswith('_')
        )
        elements.append(elem)

    return elements


def generate_skeleton_docstring(elem: CodeElement) -> str:
    """Generate a skeleton NumPy-style docstring."""
    lines = ['"""']

    # One-line summary
    if elem.existing_docstring:
        # Use first line of existing docstring
        first_line = elem.existing_docstring.split('\n')[0].strip()
        if first_line.startswith('#'):
            first_line = first_line.lstrip('#').strip()
        lines.append(first_line if first_line else f"[AUTO] {elem.name.replace('_', ' ').title()}.")
    else:
        lines.append(f"[AUTO] {elem.name.replace('_', ' ').title()}.")

    lines.append("")

    # Overview placeholder
    lines.append("[NEEDS_OVERVIEW]")
    lines.append("")

    # Parameters
    if elem.parameters:
        lines.append("Parameters")
        lines.append("----------")
        for param in elem.parameters:
            type_str = param.type_hint or "TYPE"
            if param.default:
                lines.append(f"{param.name} : {type_str}, optional")
                lines.append(f"    [AUTO] Default: {param.default}")
            else:
                lines.append(f"{param.name} : {type_str}")
                lines.append(f"    [AUTO] Description needed.")
        lines.append("")

    # Returns
    if elem.returns and elem.returns != 'None':
        lines.append("Returns")
        lines.append("-------")
        lines.append(elem.returns)
        lines.append("    [AUTO] Description needed.")
        lines.append("")

    # Harvested comments as notes
    if elem.harvested_comments:
        lines.append("Notes")
        lines.append("-----")
        lines.append("[HARVESTED from source]:")
        for comment in elem.harvested_comments:
            lines.append(f"- {comment}")
        lines.append("")

    lines.append('"""')
    return '\n'.join(lines)


def generate_review_queue(elements: List[CodeElement], output_path: Path):
    """Generate a markdown review queue."""
    # Sort by priority
    elements.sort(key=lambda e: (e.priority(), e.file, e.line))

    lines = ["# Docstring Review Queue", ""]
    lines.append("Generated by `docstring_sweep.py`")
    lines.append("")

    # Summary stats
    by_status = {}
    for elem in elements:
        by_status.setdefault(elem.status.value, []).append(elem)

    lines.append("## Summary")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|--------|-------|")
    for status in DocStatus:
        count = len(by_status.get(status.value, []))
        lines.append(f"| {status.value} | {count} |")
    lines.append("")

    # Group by file
    by_file = {}
    for elem in elements:
        by_file.setdefault(elem.file, []).append(elem)

    lines.append("## By File")
    lines.append("")

    for filepath, file_elements in by_file.items():
        rel_path = Path(filepath).relative_to(Path.cwd()) if Path(filepath).is_absolute() else filepath
        lines.append(f"### `{rel_path}`")
        lines.append("")

        # Public items first
        public = [e for e in file_elements if e.is_public]
        private = [e for e in file_elements if not e.is_public]

        if public:
            lines.append("**Public API:**")
            lines.append("")
            for elem in public:
                status_emoji = {
                    DocStatus.NONE: "❌",
                    DocStatus.MINIMAL: "⚠️",
                    DocStatus.PARTIAL: "📝",
                    DocStatus.COMPLETE: "✅"
                }.get(elem.status, "❓")

                needs_str = ", ".join(elem.needs) if elem.needs else ""
                parent = f" ({elem.parent_class})" if elem.parent_class else ""

                lines.append(f"- {status_emoji} `{elem.name}`{parent} [{elem.kind}] L{elem.line}")
                if needs_str:
                    lines.append(f"  - {needs_str}")
            lines.append("")

        if private:
            lines.append("<details>")
            lines.append("<summary>Private/Internal ({} items)</summary>".format(len(private)))
            lines.append("")
            for elem in private:
                lines.append(f"- `{elem.name}` [{elem.kind}] L{elem.line} - {elem.status.value}")
            lines.append("")
            lines.append("</details>")
            lines.append("")

    output_path.write_text('\n'.join(lines))
    print(f"  Review queue written to: {output_path}")


def generate_inventory_json(elements: List[CodeElement], output_path: Path):
    """Generate JSON inventory for further processing."""
    data = []
    for elem in elements:
        d = {
            'name': elem.name,
            'kind': elem.kind,
            'file': elem.file,
            'line': elem.line,
            'signature': elem.signature,
            'parameters': [asdict(p) for p in elem.parameters],
            'returns': elem.returns,
            'existing_docstring': elem.existing_docstring,
            'harvested_comments': elem.harvested_comments,
            'status': elem.status.value,
            'needs': elem.needs,
            'parent_class': elem.parent_class,
            'is_public': elem.is_public,
        }
        data.append(d)

    output_path.write_text(json.dumps(data, indent=2))
    print(f"  Inventory written to: {output_path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Collect files
    files = []
    for pattern in sys.argv[1:]:
        path = Path(pattern)
        if path.exists():
            files.append(path)
        else:
            # Try glob
            files.extend(Path('.').glob(pattern))

    if not files:
        print("No files found!")
        sys.exit(1)

    print(f"Scanning {len(files)} files...")

    all_elements = []

    for filepath in files:
        print(f"  Parsing: {filepath}")

        if filepath.suffix == '.pyx':
            elements = parse_cython_file(filepath)
        elif filepath.suffix == '.py':
            elements = parse_python_file(filepath)
        else:
            print(f"    Skipping unknown file type: {filepath.suffix}")
            continue

        print(f"    Found {len(elements)} elements")
        all_elements.extend(elements)

    print(f"\nTotal elements: {len(all_elements)}")

    # Generate outputs
    output_dir = Path('docs/docstrings')
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_review_queue(all_elements, output_dir / 'review_queue.md')
    generate_inventory_json(all_elements, output_dir / 'inventory.json')

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    by_status = {}
    for elem in all_elements:
        by_status.setdefault(elem.status.value, []).append(elem)

    for status in DocStatus:
        items = by_status.get(status.value, [])
        print(f"  {status.value:12}: {len(items):4} items")

    print("\nNext steps:")
    print("  1. Review: docs/docstrings/review_queue.md")
    print("  2. Interactive session to enrich docstrings")
    print("  3. Commit changes")


if __name__ == '__main__':
    main()
