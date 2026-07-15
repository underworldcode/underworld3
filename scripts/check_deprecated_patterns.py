#!/usr/bin/env python3
"""Deprecated-pattern scanner for the UW3 style gates (stdlib only).

Enforces the cheapest-to-check rules of ``docs/developer/UW3_STYLE_CHARTER.md``
on ``src/``:

- ``access-context`` (Charter S7): ``with mesh.access(...)`` / ``with
  swarm.access(...)`` context-manager data access. New code uses the ``array``
  property directly.
- ``mesh-data`` (Charter S7): ``mesh.data`` as a coordinate read. New code uses
  ``mesh.X.coords``.
- ``hedging-name`` (Charter S3): ``def maybe_*`` / ``def try_*`` / ``def do_*``
  (and single-underscore-private forms). Names state what a thing IS or DOES.
- ``except-pass`` (Charter S4): an ``except ...:`` block whose body is ``pass``
  with no comment nearby. Every intentional swallow states its sanctioned
  failure mode.

Detection is line-based and deliberately simple so any contributor can read
this file and predict what it flags:

- Lines whose first non-space character is ``#`` are skipped entirely
  (commented-out legacy code does not trip the gate; Charter S4 says to delete
  it, but that is a review matter, not a machine gate).
- ``except-pass`` heuristic: an ``except`` header line followed (skipping
  blank lines) by a bare ``pass``, with no ``#`` comment on the line before
  the ``except``, the ``except`` line, the ``pass`` line, or the line after
  the ``pass``. A comment anywhere in that window counts as a stated failure
  mode.
- NOT scanned (documented exclusions): ``self.data`` inside Mesh methods
  (cannot be distinguished reliably from legitimate variable ``.data`` by a
  line scanner) and ``.data`` / ``.access`` mentions inside strings or
  docstrings other than those matching the patterns above.

Allowlist ratchet
-----------------
Existing legacy hits are recorded in ``scripts/deprecated_pattern_allowlist.txt``
as ``path:pattern-id`` lines. A hit is allowed if its file+pattern pair is
listed. The allowlist may only SHRINK: fix the code your PR adds; only a
maintainer adds entries. When a file is cleaned up, remove its entries — the
scanner warns about stale (unused) entries so they are not forgotten.

Usage
-----
    python scripts/check_deprecated_patterns.py                # gate src/
    python scripts/check_deprecated_patterns.py --include-docs # + docs report
    python scripts/check_deprecated_patterns.py --no-allowlist # raw inventory
    python scripts/check_deprecated_patterns.py PATH [PATH...] # custom roots

Exit status: 0 when every hit under the gated roots is allowlisted,
1 otherwise. The docs report never affects the exit status.
"""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ALLOWLIST = Path(__file__).resolve().parent / "deprecated_pattern_allowlist.txt"

SOURCE_SUFFIXES = {".py", ".pyx"}
DOCS_SUFFIXES = {".py", ".md", ".ipynb"}

CHARTER = "docs/developer/UW3_STYLE_CHARTER.md"

# pattern-id -> (compiled regex, Charter section, replacement guidance)
LINE_PATTERNS = {
    "access-context": (
        re.compile(r"\bwith\s+[\w.]+\.access\("),
        "S7",
        "use the variable's .array / .data property directly",
    ),
    "mesh-data": (
        re.compile(r"\bmesh\.data\b"),
        "S7",
        "use mesh.X.coords for coordinates",
    ),
    "hedging-name": (
        re.compile(r"\bdef\s+_?(maybe|try|do)_\w+"),
        "S3",
        "name the function for what it DOES (no maybe_/try_/do_ prefixes)",
    ),
}

EXCEPT_RE = re.compile(r"^\s*except(\s+[^:]+)?:\s*(#.*)?$")
PASS_RE = re.compile(r"^\s*pass\s*(#.*)?$")


def is_comment_line(line):
    stripped = line.lstrip()
    return stripped.startswith("#")


def scan_file(path):
    """Return a list of (pattern_id, line_number, line_text) hits in one file."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        print(f"WARNING: could not read {path}: {exc}", file=sys.stderr)
        return []

    hits = []
    for lineno, line in enumerate(lines, start=1):
        if is_comment_line(line):
            continue
        for pattern_id, (regex, _section, _fix) in LINE_PATTERNS.items():
            if regex.search(line):
                hits.append((pattern_id, lineno, line.strip()))

    hits.extend(scan_except_pass(lines))
    return hits


def scan_except_pass(lines):
    """Find `except ...:` blocks whose body is a bare `pass` with no comment
    on the line before the except, the except line, the pass line, or the
    line after the pass."""
    hits = []
    for i, line in enumerate(lines):
        if not EXCEPT_RE.match(line):
            continue
        # Find the first non-blank line after the except header.
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines) or not PASS_RE.match(lines[j]):
            continue
        window = [lines[i - 1] if i > 0 else "", line, lines[j],
                  lines[j + 1] if j + 1 < len(lines) else ""]
        if any("#" in w for w in window):
            continue
        hits.append(("except-pass", i + 1, line.strip() + " ... pass"))
    return hits


def collect_files(roots, suffixes):
    files = []
    for root in roots:
        if root.is_file():
            if root.suffix in suffixes:
                files.append(root)
            continue
        files.extend(p for p in sorted(root.rglob("*")) if p.suffix in suffixes)
    return files


def load_allowlist(path):
    """Return the set of allowed `path:pattern-id` keys."""
    allowed = set()
    if not path.exists():
        return allowed
    for raw in path.read_text(encoding="utf-8").splitlines():
        entry = raw.strip()
        if not entry or entry.startswith("#"):
            continue
        allowed.add(entry)
    return allowed


def relative_key(path):
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def report_hits(all_hits, header):
    print(header)
    for file_key, pattern_id, lineno, text in all_hits:
        _regex, section, fix = LINE_PATTERNS.get(
            pattern_id, (None, "S4", "state the sanctioned failure mode in a comment")
        )
        print(f"  {file_key}:{lineno}: [{pattern_id}] {text}")
        print(f"      Charter {section}: {fix}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Scan for Charter-banned deprecated patterns (see module docstring)."
    )
    parser.add_argument(
        "paths", nargs="*", type=Path,
        help="Roots to gate (default: src/). Files or directories.",
    )
    parser.add_argument(
        "--allowlist", type=Path, default=DEFAULT_ALLOWLIST,
        help="Allowlist file of path:pattern-id lines (shrink-only).",
    )
    parser.add_argument(
        "--no-allowlist", action="store_true",
        help="Ignore the allowlist: print the raw inventory (used to seed it).",
    )
    parser.add_argument(
        "--include-docs", action="store_true",
        help="Also scan docs/ and report hits there (report only, never fails).",
    )
    args = parser.parse_args(argv)

    roots = args.paths or [REPO_ROOT / "src"]
    files = collect_files(roots, SOURCE_SUFFIXES)

    hits = []  # (file_key, pattern_id, lineno, text)
    for path in files:
        file_key = relative_key(path)
        for pattern_id, lineno, text in scan_file(path):
            hits.append((file_key, pattern_id, lineno, text))

    allowed = set() if args.no_allowlist else load_allowlist(args.allowlist)
    blocked = [h for h in hits if f"{h[0]}:{h[1]}" not in allowed]
    # Stale-entry accounting only makes sense for the default full-src scan;
    # a partial scan would misreport everything unscanned as stale.
    if args.paths:
        stale = []
    else:
        used_keys = {f"{h[0]}:{h[1]}" for h in hits}
        stale = sorted(allowed - used_keys)

    if blocked:
        report_hits(blocked, "Deprecated patterns found (not in allowlist):")
        print()
        print(f"FAIL: {len(blocked)} hit(s) across {len({h[0] for h in blocked})} file(s).")
        print(f"These patterns are banned by {CHARTER}.")
        print("Fix the code rather than extending the allowlist:")
        print(f"  {args.allowlist.name} records pre-existing legacy hits only")
        print("  and may only SHRINK (maintainer-approved entries only).")
    else:
        n_allowed = len(hits)
        print(f"OK: no new deprecated patterns ({n_allowed} known legacy hit(s) allowlisted).")

    if stale:
        print()
        print("NOTE: stale allowlist entries (no longer any hits) — please remove:")
        for entry in stale:
            print(f"  {entry}")

    if args.include_docs:
        docs_files = collect_files([REPO_ROOT / "docs"], DOCS_SUFFIXES)
        docs_hits = []
        for path in docs_files:
            file_key = relative_key(path)
            for pattern_id, lineno, text in scan_file(path):
                docs_hits.append((file_key, pattern_id, lineno, text))
        print()
        if docs_hits:
            report_hits(docs_hits, f"docs/ report (informational only, {len(docs_hits)} hit(s)):")
        else:
            print("docs/ report: clean.")

    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())
