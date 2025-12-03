#!/usr/bin/env python3
"""
Update all uw.pprint(...) calls to new uw.pprint(...) API.
"""

import re
import glob
import os

def update_pprint_calls(file_path):
    """Update pprint calls in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Could not read {file_path}: {e}")
        return False

    # Pattern to match uw.pprint(...) calls
    # This handles various whitespace and argument patterns
    pattern = r'uw\.pprint\(\s*0\s*,\s*'
    replacement = 'uw.pprint('

    new_content = re.sub(pattern, replacement, content)

    if new_content != content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated: {file_path}")
            return True
        except Exception as e:
            print(f"Could not write {file_path}: {e}")
            return False

    return False

def main():
    """Update all Python files in the project."""
    print("🔄 UPDATING PPRINT CALLS TO NEW API")
    print("=" * 40)

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
            if update_pprint_calls(file_path):
                updated_files.append(file_path)

    print(f"\n✅ UPDATED {len(updated_files)} FILES")
    for file_path in updated_files:
        print(f"  - {file_path}")

    print(f"\n💡 Updated uw.pprint(...) → uw.pprint(...)")
    print(f"💡 New API defaults to proc=0, so behavior is identical")

if __name__ == "__main__":
    main()