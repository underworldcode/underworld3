"""Fail if a test file matches none of the globs scripts/test.sh actually runs.

A batch glob that names a numeric range (``test_006[0-1]*``) stops covering the
suite the moment someone adds ``test_0062``. That is how the whole
integration-point suite (#703, #707) and the swarm repopulation tests (#713)
shipped green without CI ever running them. The globs are still ranges — some
of them have to be — so this check is what keeps them honest.

Files deferred on purpose are listed in DEFERRED with the reason and the issue.
The list may only shrink: adding to it needs a maintainer decision, exactly like
the allowlist in check_deprecated_patterns.py.
"""

import glob
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Deferred by maintainer decision, not by accident.
DEFERRED = {
    # No issue: this band was disabled in scripts/test.sh as "potentially
    # problematic" without one being filed. Recorded as it stands rather than
    # dressed up — #721 follow-up work re-enables it and removes this entry.
    "test_06[0-9][0-9]_*": "regression suite disabled in test.sh, no issue filed",
    # Narrow: test_1072 is pulled out of this band and run by name, so a broad
    # test_107* would list a covered file as deferred.
    "test_1070_*": "level_2/level_3 + tier_b/tier_c, awaiting triage (#504)",
    "test_1071_*": "level_2/level_3 + tier_b/tier_c, awaiting triage (#504)",
    "test_1073_*": "level_2/level_3 + tier_b/tier_c, awaiting triage (#504)",
    "test_106*": "level_2/level_3 + tier_b/tier_c, awaiting triage (#504)",
}


def globs_run_by(script):
    """Every tests/... pattern passed to $PYTEST on a line that is not commented."""
    text = script.read_text().replace("\\\n", " ")
    patterns = set()
    for line in text.splitlines():
        if "$PYTEST" in line and not line.lstrip().startswith("#"):
            patterns.update(re.findall(r"tests/[A-Za-z0-9_\[\]*.\-]+", line))
    return patterns


def main():
    covered = set()
    for pattern in globs_run_by(REPO / "scripts" / "test.sh"):
        covered.update(glob.glob(str(REPO / pattern)))
    covered = {Path(p).name for p in covered}

    every = {p.name for p in (REPO / "tests").glob("test_*.py")}

    # A deferred entry that matches nothing, or matches only files a glob
    # already runs, is stale. Without this the list only ever grows: an entry
    # keeps reporting a deferral that stopped being true, which is how the
    # shrink-only invariant quietly becomes a fiction.
    deferred, stale = set(), []
    for pattern, reason in DEFERRED.items():
        matched = {Path(p).name for p in glob.glob(str(REPO / "tests" / pattern))}
        if not matched:
            stale.append(f"{pattern!r} matches no test file")
        elif matched <= covered:
            stale.append(f"{pattern!r} matches only files that already run")
        deferred.update(matched)

    dark = sorted(every - covered - deferred)

    if dark:
        print(f"{len(dark)} test file(s) match no glob in scripts/test.sh:")
        for name in dark:
            print(f"    tests/{name}")
        print("\nWiden the batch glob that should have caught them, or add an entry")
        print("to DEFERRED in this file saying who deferred it and why.")
        return 1

    if stale:
        print("DEFERRED has stale entries — the list may only shrink:")
        for problem in stale:
            print(f"    {problem}")
        print("\nRemove them: the files they name are running, or no longer exist.")
        return 1

    print(f"OK: all {len(every)} test files are reachable from scripts/test.sh "
          f"({len(deferred)} deferred by decision).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
