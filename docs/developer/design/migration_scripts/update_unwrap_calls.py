#!/usr/bin/env python3
"""
Replace uw.unwrap(uw, all) method calls with uw.unwrap(uw, uw) function calls.
"""

import re
import glob
import os

def update_unwrap_calls(file_path):
    """Update unwrap calls in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Could not read {file_path}: {e}")
        return False

    original_content = content

    # Pattern 1: uw.unwrap(uw, Simple) calls
    # uw.unwrap(uw, object) → uw.unwrap(uw, object)
    pattern1 = r'([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*|[a-zA-Z_][a-zA-Z0-9_]*(?:\[[^\]]+\])?(?:\([^)]*\))?(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\.\s*unwrap\(\s*\)'

    def replace_simple(match):
        obj = match.group(1)
        return f'uw.unwrap(uw, {obj})'

    content = re.sub(pattern1, replace_simple, content)

    # Pattern 2: .unwrap() with arguments
    # uw.unwrap(object, args) → uw.unwrap(uw, object, args)
    pattern2 = r'([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*|[a-zA-Z_][a-zA-Z0-9_]*(?:\[[^\]]+\])?(?:\([^)]*\))?(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\.\s*unwrap\(\s*([^)]+)\s*\)'

    def replace_with_args(match):
        obj = match.group(1)
        args = match.group(2)
        return f'uw.unwrap(uw, {obj}, {args})'

    content = re.sub(pattern2, replace_with_args, content)

    # Also update old apply_scaling() context manager calls to use new API
    pattern3 = r'with\s+uw\.apply_scaling\(\s*\):\s*\n\s*([^=\n]+)\s*=\s*uw\.unwrap\(([^)]+)\)'

    def replace_context(match):
        var_assign = match.group(1)
        unwrap_args = match.group(2)
        return f'{var_assign} = uw.unwrap(uw, {unwrap_args}, apply_scaling=True)'

    content = re.sub(pattern3, replace_context, content, flags=re.MULTILINE)

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
    """Update all Python files in the project."""
    print("🔄 uw.unwrap(uw, UPDATING) CALLS TO uw.unwrap(uw, uw)")
    print("=" * 45)

    # Find all Python files in the project
    patterns = [
        "src/**/*.py",
        "tests/**/*.py",
        "docs/**/*.py",
        "**/*.py"
    ]

    updated_files = []

    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        for file_path in files:
            if update_unwrap_calls(file_path):
                updated_files.append(file_path)

    print(f"\n✅ UPDATED {len(updated_files)} FILES")
    for file_path in updated_files:
        print(f"  - {file_path}")

    print(f"\n💡 Updated uw.unwrap(uw, object) → uw.unwrap(uw, object)")
    print(f"💡 Updated apply_scaling() context manager → apply_scaling=True parameter")
    print(f"💡 Standardized on single uw.unwrap(uw, uw) function")

if __name__ == "__main__":
    main()