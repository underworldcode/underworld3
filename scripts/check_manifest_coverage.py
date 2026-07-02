#!/usr/bin/env python3
"""Manifest coverage check for the release gate.

Guards against *manifest drift*: a tier_a-marked test file that no feature in
docs/release-notes/feature-manifest.yaml references, so its coverage is invisible
to the release gate (scripts/release_gate.py).

Every tier_a test file should belong to some feature's validation.paths. This
script reports any that do not.

Usage:
    python scripts/check_manifest_coverage.py          # warn-only (exit 0)
    python scripts/check_manifest_coverage.py --strict # exit 1 if any uncovered

Intended to run in development CI as a warning first, promoted to --strict once
coverage is established. Standard library + PyYAML only.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

try:
    import yaml
except ImportError:
    sys.stderr.write("error: PyYAML not found. Run via the pixi env.\n")
    sys.exit(2)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO_ROOT, "docs", "release-notes", "feature-manifest.yaml")
TESTS_DIR = os.path.join(REPO_ROOT, "tests")

# Match @pytest.mark.tier_a or pytestmark = [... tier_a ...]
TIER_A_RE = re.compile(r"\btier_a\b")


def tier_a_files() -> set[str]:
    """Repo-relative paths of test files that reference tier_a."""
    found = set()
    for path in glob.glob(os.path.join(TESTS_DIR, "**", "test_*.py"), recursive=True):
        try:
            with open(path, encoding="utf-8") as fh:
                if TIER_A_RE.search(fh.read()):
                    found.add(os.path.relpath(path, REPO_ROOT))
        except OSError:
            continue
    return found


def manifest_covered_files() -> set[str]:
    """Repo-relative test paths referenced by any feature's validation.paths."""
    with open(MANIFEST) as fh:
        data = yaml.safe_load(fh) or {}
    covered = set()
    for feat in data.get("features", []):
        for pat in (feat.get("validation", {}) or {}).get("paths", []) or []:
            for hit in glob.glob(os.path.join(REPO_ROOT, pat), recursive=True):
                covered.add(os.path.relpath(hit, REPO_ROOT))
    return covered


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Check release-manifest coverage of tier_a tests")
    ap.add_argument("--strict", action="store_true", help="exit non-zero if any tier_a file is uncovered")
    args = ap.parse_args(argv)

    tier_a = tier_a_files()
    covered = manifest_covered_files()
    uncovered = sorted(tier_a - covered)

    print(f"tier_a test files:        {len(tier_a)}")
    print(f"referenced by manifest:   {len(tier_a & covered)}")
    print(f"uncovered:                {len(uncovered)}")

    if uncovered:
        print("\nThe following tier_a test files are not referenced by any feature")
        print("in docs/release-notes/feature-manifest.yaml — their coverage is")
        print("invisible to the release gate. Add them to a feature's validation.paths:")
        for f in uncovered:
            print(f"  - {f}")
        if args.strict:
            return 1
        print("\n(warning only; re-run with --strict to fail CI)")
    else:
        print("\nAll tier_a test files are covered by the manifest.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
