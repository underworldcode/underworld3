#!/usr/bin/env python3
"""Release validation gate for Underworld3.

Reads the feature manifest (docs/release-notes/feature-manifest.yaml) and, for
each shippable feature, runs its declared test selection against the *current*
checkout. The result decides whether the feature can be *announced* at the
maturity its owner claims.

The model (see docs/developer/guides/release-process.md):

  * `main` accumulates whatever stable work comes over from `development`.
  * A release ANNOUNCES features at one of three maturities:
        supported    — validated; tier_a/b tests exist and pass. Guaranteed.
        preview      — present on main, but not guaranteed to work.
        experimental — present, not announced as working.
  * The announced maturity is min(claim, tests_maturity): an owner may be
    cautious (claim preview even though tests pass), and the gate may downgrade
    (claim supported but a test fails -> preview). It NEVER blocks the merge —
    the code ships regardless; only the announcement changes.

"Supported" is tied to the existing tier markers by construction: a feature's
selection is filtered to `tier_a or tier_b`, so passing == validated.

Subcommands:
  report        Run the gate and print a human-readable table (dry-run).
  run           Run the gate and emit results as JSON (for tooling).
  render-notes  Emit the Supported/Preview release-notes sections for a version.

This script uses only the standard library plus PyYAML (already a dependency).
It shells out to `python -m pytest`, so it must run inside the pixi env, e.g.
    pixi run -e <env> python scripts/release_gate.py report
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

try:
    import yaml
except ImportError:  # pragma: no cover - guidance for the rare missing-dep case
    sys.stderr.write(
        "error: PyYAML not found. Run this via the pixi env:\n"
        "  pixi run -e <env> python scripts/release_gate.py ...\n"
    )
    sys.exit(2)

# Repo root is the parent of scripts/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(REPO_ROOT, "docs", "release-notes", "feature-manifest.yaml")
PYTEST_CONFIG = os.path.join("tests", "pytest.ini")

# Maturity ordering (higher == more trustworthy)
MATURITY = {"experimental": 0, "preview": 1, "supported": 2}
MATURITY_NAME = {v: k for k, v in MATURITY.items()}


def _min_maturity(a: str, b: str) -> str:
    return MATURITY_NAME[min(MATURITY[a], MATURITY[b])]


def load_manifest(path: str) -> list[dict]:
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    features = data.get("features", [])
    if not isinstance(features, list):
        raise ValueError(f"{path}: 'features' must be a list")
    return features


def _expand_paths(paths: list[str]) -> list[str]:
    """Expand globs relative to the repo root. Returns repo-relative paths."""
    matched: list[str] = []
    for pat in paths:
        hits = glob.glob(os.path.join(REPO_ROOT, pat), recursive=True)
        for h in sorted(hits):
            matched.append(os.path.relpath(h, REPO_ROOT))
    return matched


def _marker_expr(markers: str, levels: str | None) -> str:
    """Combine the feature's marker expression with an optional level filter.

    `levels` is a comma list like "1,2,3" -> "(level_1 or level_2 or level_3)".
    The marker expression is always applied, so "supported" stays tied to the
    tier_a/b markers no matter how levels narrow the run.
    """
    expr = f"({markers})"
    if levels:
        level_terms = " or ".join(f"level_{n.strip()}" for n in levels.split(",") if n.strip())
        if level_terms:
            expr = f"{expr} and ({level_terms})"
    return expr


def _run_feature(feature: dict, cli_levels: str | None) -> dict:
    """Run one feature's test selection and resolve its maturity."""
    key = feature.get("key", "?")
    claim = feature.get("claim", "experimental")
    if claim not in MATURITY:
        raise ValueError(f"feature '{key}': invalid claim '{claim}'")

    val = feature.get("validation", {}) or {}
    paths = _expand_paths(val.get("paths", []) or [])
    markers = val.get("markers", "tier_a or tier_b")
    select = val.get("select")
    # Per-feature levels override the CLI default; both are optional.
    levels = val.get("levels", cli_levels)

    result = {
        "key": key,
        "title": feature.get("title", key),
        "owner": feature.get("owner", ""),
        "claim": claim,
        "summary": (feature.get("summary") or "").strip(),
        "n_selected": 0,
        "n_passed": 0,
        "n_failed": 0,
        "n_skipped": 0,
        "tests_maturity": "experimental",
        "announced": "experimental",
        "verdict": "ok",
        "note": "",
    }

    if not paths:
        result["note"] = "no test files matched validation.paths"
        result["announced"] = _min_maturity(claim, "experimental")
        result["verdict"] = "downgraded" if claim != "experimental" else "ok"
        return result

    marker_expr = _marker_expr(markers, levels)

    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tf:
        junit_path = tf.name
    try:
        cmd = [
            sys.executable, "-m", "pytest",
            f"--config-file={PYTEST_CONFIG}",
            "-q", "-p", "no:cacheprovider",
            "--timeout=300",
            f"--junit-xml={junit_path}",
            "-m", marker_expr,
        ]
        if select:
            cmd += ["-k", select]
        cmd += paths

        env = dict(os.environ)
        env.setdefault("UW_ENABLE_TELEMETRY", "0")
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                              capture_output=True, text=True)

        # pytest exit codes: 0=passed, 1=failed, 2=interrupted, 3=internal
        # error, 4=usage error, 5=no tests collected.
        if proc.returncode == 5:
            result["note"] = "selection is empty under tier_a/b markers"
            result["tests_maturity"] = "experimental"
        elif proc.returncode not in (0, 1):
            # The gate could not run this feature's tests, so we cannot validate
            # it — never call it supported, and flag the error loudly rather than
            # silently reporting "experimental/skipped".
            result["tests_maturity"] = "experimental"
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            hint = f": {tail[-1]}" if tail else ""
            result["note"] = f"pytest execution error (exit {proc.returncode}){hint}"
        else:
            counts = _parse_junit(junit_path)
            result.update(counts)
            n_real = counts["n_selected"] - counts["n_skipped"]
            if n_real <= 0:
                result["tests_maturity"] = "experimental"
                result["note"] = "all selected tests skipped"
            elif counts["n_failed"] > 0:
                result["tests_maturity"] = "preview"
                result["note"] = f"{counts['n_failed']} test(s) failed"
            else:
                result["tests_maturity"] = "supported"
    finally:
        try:
            os.unlink(junit_path)
        except OSError:
            pass

    result["announced"] = _min_maturity(claim, result["tests_maturity"])
    # PASS if the tests at least met the owner's claim; otherwise downgraded.
    if MATURITY[result["tests_maturity"]] < MATURITY[claim]:
        result["verdict"] = "downgraded"
    return result


def _parse_junit(path: str) -> dict:
    """Aggregate counts from a JUnit XML file across all <testsuite> nodes."""
    tests = failures = errors = skipped = 0
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, FileNotFoundError):
        return {"n_selected": 0, "n_passed": 0, "n_failed": 0, "n_skipped": 0}
    suites = root.iter("testsuite") if root.tag != "testsuite" else [root]
    for s in suites:
        tests += int(s.get("tests", 0))
        failures += int(s.get("failures", 0))
        errors += int(s.get("errors", 0))
        skipped += int(s.get("skipped", 0))
    bad = failures + errors
    return {
        "n_selected": tests,
        "n_passed": tests - bad - skipped,
        "n_failed": bad,
        "n_skipped": skipped,
    }


def run_gate(manifest_path: str, cli_levels: str | None) -> list[dict]:
    features = load_manifest(manifest_path)
    return [_run_feature(f, cli_levels) for f in features]


# ---------------------------------------------------------------------------
# Output renderers
# ---------------------------------------------------------------------------

ICON = {"ok": "✓", "downgraded": "↓"}  # check / down-arrow


def cmd_report(results: list[dict]) -> int:
    title_w = max([len(r["title"]) for r in results] + [5])
    tests_w = 20
    print()
    print(f"  {'FEATURE':<{title_w}}  {'CLAIM':<12}  {'ACHIEVED':<12}  {'TESTS':<{tests_w}}  VERDICT")
    print(f"  {'-' * title_w}  {'-' * 12}  {'-' * 12}  {'-' * tests_w}  {'-' * 10}")
    for r in results:
        tests = f"{r['n_passed']}/{max(r['n_selected'] - r['n_skipped'], 0)} pass"
        if r["n_skipped"]:
            tests += f" ({r['n_skipped']} skip)"
        verdict = "ok" if r["verdict"] == "ok" else "DOWNGRADED"
        mark = ICON.get(r["verdict"], "?")
        print(f"  {r['title']:<{title_w}}  {r['claim']:<12}  "
              f"{r['announced']:<12}  {tests:<{tests_w}}  {mark} {verdict}")
        if r["note"]:
            print(f"  {' ' * title_w}  -> {r['note']}")
    print()
    downgrades = [r for r in results if r["verdict"] == "downgraded"]
    if downgrades:
        keys = ", ".join(r["key"] for r in downgrades)
        print(f"  Note: {len(downgrades)} feature(s) downgraded below their claim: {keys}")
        print("  Their code still ships; the release notes simply will not announce")
        print("  them as supported. This does not block the merge.")
        print()
    return 0


def cmd_run(results: list[dict]) -> int:
    json.dump({"features": results}, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def cmd_render_notes(results: list[dict], version: str) -> int:
    supported = [r for r in results if r["announced"] == "supported"]
    preview = [r for r in results if r["announced"] != "supported"]

    out = [f"# Underworld3 {version}", ""]
    out += [
        "<!-- The Supported / Preview sections below are generated by",
        "     scripts/release_gate.py from docs/release-notes/feature-manifest.yaml.",
        "     Edit the Highlights and Contributors by hand. -->",
        "",
        "## Highlights",
        "",
        "- ",
        "",
        "## Supported (validated)",
        "",
        "<!-- Features whose tier_a/b validation passed on this release. Guaranteed. -->",
        "",
    ]
    if supported:
        for r in supported:
            line = f"- **{r['title']}**"
            if r["summary"]:
                line += f" — {r['summary']}"
            out.append(line)
    else:
        out.append("- _(none validated for this release)_")
    out += [
        "",
        "## Preview (present, unguaranteed)",
        "",
        "<!-- Code is on main but NOT guaranteed to work. Use at your own risk. -->",
        "",
    ]
    if preview:
        for r in preview:
            tag = "" if r["announced"] == "preview" else f" _({r['announced']})_"
            line = f"- **{r['title']}**{tag}"
            if r["summary"]:
                line += f" — {r['summary']}"
            out.append(line)
    else:
        out.append("- _(none)_")
    out += [
        "",
        "## Bug Fixes",
        "",
        "- ",
        "",
        "## Contributors",
        "",
        "Thanks to everyone who contributed to this release.",
        "",
    ]
    print("\n".join(out))
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Underworld3 release validation gate")
    p.add_argument("--manifest", default=DEFAULT_MANIFEST,
                   help="path to feature-manifest.yaml")
    p.add_argument("--levels", default=None,
                   help="default level filter (e.g. '1,2'); per-feature levels override")
    p.add_argument("--results", default=None,
                   help="render-notes: read results JSON instead of re-running the gate")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("report", help="run the gate and print a table")
    sub.add_parser("run", help="run the gate and emit JSON")
    rn = sub.add_parser("render-notes", help="emit Supported/Preview release-notes body")
    rn.add_argument("version", help="release version, e.g. v3.1.0")

    args = p.parse_args(argv)

    # --results lets report/render-notes reuse a prior run's JSON instead of
    # re-running the (slow) test selection. `run` always executes.
    if args.results and args.command != "run":
        with open(args.results) as fh:
            results = json.load(fh).get("features", [])
    else:
        results = run_gate(args.manifest, args.levels)

    if args.command == "report":
        return cmd_report(results)
    if args.command == "run":
        return cmd_run(results)
    if args.command == "render-notes":
        return cmd_render_notes(results, args.version)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
