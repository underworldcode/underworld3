#!/usr/bin/env python3
"""
Audit underworld3 API documentation coverage.

Compares what's exported in the package vs what's documented in docs/api/*.md
to identify gaps in documentation coverage.

Usage:
    pixi run docs-audit
    python scripts/docs_audit.py
"""

import inspect
import re
import sys
from pathlib import Path


def get_documented_items(docs_dir: Path) -> dict[str, set[str]]:
    """Extract all documented items from .md files."""
    documented = {}

    for md_file in docs_dir.glob("*.md"):
        if md_file.name == "index.md":
            continue

        content = md_file.read_text()
        module_name = md_file.stem

        # Find all autofunction, autoclass, automodule directives
        patterns = [
            r'\.\. autofunction:: ([\w.]+)',
            r'\.\. autoclass:: ([\w.]+)',
            r'\.\. automodule:: ([\w.]+)',
        ]

        items = set()
        for pattern in patterns:
            matches = re.findall(pattern, content)
            items.update(matches)

        documented[module_name] = items

    return documented


def get_exported_items() -> dict[str, dict[str, list]]:
    """Get all public items exported by underworld3."""
    import underworld3 as uw

    exports = {}

    # Key modules to audit
    modules_to_check = [
        ('meshing', uw.meshing),
        ('discretisation', uw.discretisation),
        ('swarm', uw.swarm),
        ('function', uw.function),
        ('systems.solvers', uw.systems.solvers),
        ('constitutive_models', uw.constitutive_models),
        ('coordinates', uw.coordinates),
        ('maths', uw.maths),
        ('visualisation', uw.visualisation),
        ('utilities', uw.utilities),
        ('scaling', uw.scaling),
        ('systems.ddt', uw.systems.ddt),
        ('adaptivity', uw.adaptivity),
        ('materials', uw.materials),
        ('model', uw.model),
    ]

    for mod_name, mod in modules_to_check:
        classes = []
        functions = []

        for name in dir(mod):
            if name.startswith('_'):
                continue

            obj = getattr(mod, name)

            # Check if it's defined in underworld3
            obj_module = getattr(obj, '__module__', '')
            if not obj_module.startswith('underworld3'):
                continue

            full_name = f"underworld3.{mod_name}.{name}"

            if inspect.isclass(obj):
                classes.append((name, full_name))
            elif inspect.isfunction(obj):
                functions.append((name, full_name))

        exports[mod_name] = {
            'classes': classes,
            'functions': functions,
        }

    return exports


def audit_documentation():
    """Run the documentation audit."""
    docs_dir = Path("docs/api")

    if not docs_dir.exists():
        print(f"ERROR: {docs_dir} not found. Run from repository root.")
        sys.exit(1)

    print("=" * 70)
    print("UNDERWORLD3 DOCUMENTATION AUDIT")
    print("=" * 70)

    # Get what's documented
    documented = get_documented_items(docs_dir)

    # Get what's exported
    try:
        exports = get_exported_items()
    except ImportError as e:
        print(f"ERROR: Could not import underworld3: {e}")
        print("Make sure underworld3 is installed (pixi run build)")
        sys.exit(1)

    # Compare
    total_documented = 0
    total_missing = 0
    missing_items = []

    for mod_name, items in exports.items():
        # Find corresponding doc file
        doc_key = mod_name.replace('.', '_').replace('systems_', '')
        if doc_key == 'solvers':
            doc_key = 'solvers'
        elif doc_key == 'ddt':
            doc_key = 'systems_ddt'

        doc_items = set()
        for key, doc_set in documented.items():
            doc_items.update(doc_set)

        mod_missing = []
        mod_documented = 0

        for name, full_name in items['classes'] + items['functions']:
            # Check various possible documentation paths
            possible_names = [
                full_name,
                full_name.replace('systems.', ''),
                f"underworld3.{name}",
            ]

            if any(pn in doc_items for pn in possible_names):
                mod_documented += 1
            else:
                mod_missing.append((name, full_name, 'class' if (name, full_name) in items['classes'] else 'function'))

        total_documented += mod_documented
        total_missing += len(mod_missing)

        if mod_missing:
            missing_items.append((mod_name, mod_missing))

    # Print results
    print(f"\nSummary: {total_documented} documented, {total_missing} missing")
    print("=" * 70)

    if missing_items:
        print("\nMISSING DOCUMENTATION:")
        print("-" * 70)

        for mod_name, items in missing_items:
            print(f"\n{mod_name}:")
            for name, full_name, item_type in items:
                print(f"  [{item_type:8}] {name}")
                print(f"             -> .. auto{item_type}:: {full_name}")
    else:
        print("\n✓ All public items are documented!")

    # Print doc file coverage
    print("\n" + "=" * 70)
    print("DOCUMENTATION FILES:")
    print("-" * 70)

    for md_file in sorted(docs_dir.glob("*.md")):
        if md_file.name == "index.md":
            continue
        item_count = len(documented.get(md_file.stem, set()))
        print(f"  {md_file.name}: {item_count} items")

    print("\n" + "=" * 70)

    # Return exit code based on missing items
    return 1 if total_missing > 0 else 0


if __name__ == "__main__":
    sys.exit(audit_documentation())
