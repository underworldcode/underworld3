#!/usr/bin/env python3
"""
Convert Quarto (.qmd) files to MyST Markdown (.md) for Sphinx.

This script handles:
- Callout syntax conversion (::: {.callout-*} → ```{directive})
- Link extension changes (.qmd → .md)
- YAML frontmatter preservation
- Variable substitution ({{< var ... >}} → relative paths)
"""

import re
import os
import sys
import yaml
from pathlib import Path


def load_variables(docs_dir):
    """Load Quarto variables from _variables.yml"""
    var_file = docs_dir / "_variables.yml"
    if not var_file.exists():
        return {}

    with open(var_file, 'r') as f:
        return yaml.safe_load(f) or {}


def resolve_variable(var_path, variables):
    """Resolve a Quarto variable like links.beginner.quickstart to its value"""
    parts = var_path.split('.')
    value = variables
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None
    return value


def convert_variable_refs(content, variables, current_file):
    """Convert {{< var links.xxx >}} to relative paths"""
    def replace_var(match):
        var_path = match.group(1).strip()
        value = resolve_variable(var_path, variables)
        if value:
            # Convert .qmd to .md
            value = value.replace('.qmd', '.md')
            # Make relative to current file
            return value
        return match.group(0)  # Keep original if not found

    return re.sub(r'\{\{<\s*var\s+([^>]+)\s*>\}\}', replace_var, content)


def convert_callouts(content):
    """Convert Quarto callout syntax to MyST admonitions"""
    # Pattern for callout blocks
    # ::: {.callout-TYPE}
    # ## Title (optional)
    # Content
    # :::

    callout_map = {
        'note': 'note',
        'tip': 'tip',
        'warning': 'warning',
        'important': 'important',
        'caution': 'caution',
    }

    def replace_callout(match):
        callout_type = match.group(1)
        content_block = match.group(2)

        myst_type = callout_map.get(callout_type, 'note')

        # Check if there's a title (## Title at start)
        title_match = re.match(r'\s*##\s*(.+?)\n(.*)', content_block, re.DOTALL)
        if title_match:
            title = title_match.group(1).strip()
            body = title_match.group(2).strip()
            return f'```{{{myst_type}}} {title}\n{body}\n```'
        else:
            return f'```{{{myst_type}}}\n{content_block.strip()}\n```'

    # Match ::: {.callout-TYPE} ... :::
    pattern = r':::\s*\{\.callout-(\w+)\}\n(.*?):::'
    content = re.sub(pattern, replace_callout, content, flags=re.DOTALL)

    return content


def convert_links(content):
    """Convert .qmd links to .md"""
    # Pattern for markdown links
    content = re.sub(r'\[([^\]]+)\]\(([^)]+)\.qmd\)', r'[\1](\2.md)', content)
    return content


def convert_btn_classes(content):
    """Convert Quarto button classes to simple links"""
    # [Text](link){.btn .btn-primary} → **[Text](link)**
    content = re.sub(r'\[([^\]]+)\]\(([^)]+)\)\{\.btn[^}]*\}', r'**[\1](\2)**', content)
    return content


def convert_frontmatter(content):
    """Ensure frontmatter is MyST compatible"""
    # Just keep YAML frontmatter as-is, MyST handles it
    return content


def convert_qmd_to_myst(input_file, output_file, variables, docs_dir):
    """Convert a single QMD file to MyST Markdown"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try with latin-1 encoding as fallback
        with open(input_file, 'r', encoding='latin-1') as f:
            content = f.read()

    # Apply conversions
    content = convert_variable_refs(content, variables, input_file)
    content = convert_callouts(content)
    content = convert_links(content)
    content = convert_btn_classes(content)
    content = convert_frontmatter(content)

    # Write output
    os.makedirs(output_file.parent, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

    return True


def main():
    """Main conversion function"""
    docs_dir = Path(__file__).parent.parent

    # Load variables
    variables = load_variables(docs_dir)

    # Find all .qmd files (excluding presentations and build directories)
    qmd_files = []
    for qmd_file in docs_dir.rglob('*.qmd'):
        # Skip presentations
        if 'slides' in str(qmd_file) or 'presentations' in str(qmd_file):
            print(f"Skipping presentation: {qmd_file}")
            continue
        # Skip build directories
        if '_build' in str(qmd_file):
            print(f"Skipping build artifact: {qmd_file}")
            continue
        qmd_files.append(qmd_file)

    print(f"Found {len(qmd_files)} QMD files to convert")

    # Convert each file
    for qmd_file in qmd_files:
        # Output file with .md extension
        md_file = qmd_file.with_suffix('.md')

        print(f"Converting: {qmd_file.relative_to(docs_dir)} -> {md_file.relative_to(docs_dir)}")
        convert_qmd_to_myst(qmd_file, md_file, variables, docs_dir)

    print(f"\nConverted {len(qmd_files)} files")
    print("\nNext steps:")
    print("1. Review converted .md files")
    print("2. Delete original .qmd files (after verification)")
    print("3. Update conf.py to include new directories")


if __name__ == '__main__':
    main()
