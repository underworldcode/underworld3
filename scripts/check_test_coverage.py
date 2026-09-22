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
    # Empty, and that is the point. Every band that used to sit here — the
    # test_06NN regression suite and test_106*/test_107* — is now batched in
    # scripts/test.sh. Nothing is excluded from CI by its number any more; a
    # test that should not gate says so with @pytest.mark.tier_c, which is a
    # property of the test rather than of where it sits in the numbering.
}


def globs_run_by(script):
    """Every tests/... pattern passed to $PYTEST on a line that is not commented."""
    text = script.read_text().replace("\\\n", " ")
    patterns = set()
    for line in text.splitlines():
        # Both spellings: the runner is a bash ARRAY, so call sites read
        # "${PYTEST[@]}", but a plain $PYTEST is still worth matching in case
        # one is left behind or reintroduced.
        if re.search(r"\$\{?PYTEST", line) and not line.lstrip().startswith("#"):
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
