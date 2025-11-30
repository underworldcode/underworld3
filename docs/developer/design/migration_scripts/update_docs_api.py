#!/usr/bin/env python3
"""
Update documentation files for new pprint and unwrap API patterns.
"""

import re
import glob
import os

def update_doc_file(file_path):
    """Update patterns in a single documentation file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Could not read {file_path}: {e}")
        return False

    original_content = content

    # 1. Update old pprint calls: uw.pprint(0, ...) → uw.pprint(...)
    content = re.sub(r'uw\.pprint\(\s*0\s*,\s*', 'uw.pprint(', content)

    # 2. Update object.unwrap() → uw.unwrap(object)
    # Simple cases: obj.unwrap()
    content = re.sub(r'([a-zA-Z_][a-zA-Z0-9_\.]*)\s*\.\s*unwrap\(\s*\)', r'uw.unwrap(\1)', content)

    # With arguments: obj.unwrap(args) → uw.unwrap(obj, args)
    content = re.sub(r'([a-zA-Z_][a-zA-Z0-9_\.]*)\s*\.\s*unwrap\(\s*([^)]+)\s*\)', r'uw.unwrap(\1, \2)', content)

    # 3. Fix specific module path calls
    content = re.sub(r'uw\.function\.expression\.unwrap\(', 'uw.unwrap(', content)
    content = re.sub(r'uw\.function\.expressions\.unwrap\(', 'uw.unwrap(', content)

    # 4. Update context manager to parameter usage (basic cases)
    # with uw.apply_scaling(): result = uw.unwrap(expr) → result = uw.unwrap(expr, apply_scaling=True)
    content = re.sub(
        r'with\s+uw\.apply_scaling\(\s*\):\s*\n\s*([^=\n]+)\s*=\s*uw\.unwrap\(([^)]+)\)',
        r'\1 = uw.unwrap(\2, apply_scaling=True)',
        content,
        flags=re.MULTILINE
    )

    # 5. Update documentation examples that still show old API
    # Update table entries and examples
    content = re.sub(r'`uw\.pprint\(0,\s*"[^"]*"\)`', lambda m: m.group(0).replace('uw.pprint(0,', 'uw.pprint('), content)
    content = re.sub(r'uw\.pprint\(0,\s*"([^"]*)"\)', r'uw.pprint("\1")', content)

    # 6. Fix malformed unwrap calls from previous regex (uw.unwrap(uw, uw, ...))
    content = re.sub(r'uw\.unwrap\(uw,\s*uw,?\s*', 'uw.unwrap(', content)
    content = re.sub(r'uw\.unwrap\(uw,\s*([^,\)]+)', r'uw.unwrap(\1', content)

    if content != original_content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated: {file_path}")
            return True
        except Exception as e:
            print(f"Could not write {file_path}: {e}")
            return False

    return False

def main():
    """Update all documentation files."""
    print("🔄 UPDATING DOCUMENTATION FOR NEW API PATTERNS")
    print("=" * 50)

    # Find all documentation files
    patterns = [
        "docs/**/*.md",
        "docs/**/*.qmd",
        "docs/**/*.ipynb",
        "docs/**/*.py"
    ]

    updated_files = []

    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        for file_path in files:
            if update_doc_file(file_path):
                updated_files.append(file_path)

    print(f"\n✅ UPDATED {len(updated_files)} DOCUMENTATION FILES")
    for file_path in updated_files:
        print(f"  - {file_path}")

    print(f"\n💡 Updated uw.pprint(0, ...) → uw.pprint(...)")
    print(f"💡 Updated object.unwrap() → uw.unwrap(object)")
    print(f"💡 Updated apply_scaling() context → apply_scaling=True parameter")
    print(f"💡 Fixed malformed unwrap calls")

if __name__ == "__main__":
    main()