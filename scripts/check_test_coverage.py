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


# tests/parallel/ptest_* files CI does not run, each with why (#900). Shrink-only:
# a new parallel test is named test_*, which CI picks up by itself.
SCRIPT_STYLE = "a script with no test functions: needs a harness to report to CI"
DEFERRED_PTESTS = {
    "ptest_0004_checkpoint_fmg_hierarchy.py": SCRIPT_STYLE,
    "ptest_0007_snapshot_inmemory.py": SCRIPT_STYLE,
    "ptest_0008_mesh_radii_accessors.py": SCRIPT_STYLE,
    "ptest_000_petsc4py.py": SCRIPT_STYLE,
    "ptest_000a_uw.py": SCRIPT_STYLE,
    "ptest_000b_uw_petsc4py.py": SCRIPT_STYLE,
    "ptest_000c_uw_petsc4py_h5.py": SCRIPT_STYLE + "; reads a mesh file nothing makes",
    "ptest_000d_uw_petsc4py_swarm.py": SCRIPT_STYLE + "; reads a mesh file nothing makes",
    "ptest_0010_snapshot_disk.py": SCRIPT_STYLE,
    "ptest_0011_dm_rebuild_held_handles.py": SCRIPT_STYLE,
    "ptest_001_start_stop.py": SCRIPT_STYLE,
    "ptest_001a_start_stop_petsc4py.py": SCRIPT_STYLE,
    "ptest_002_projection.py": SCRIPT_STYLE,
    "ptest_003_swarm_projection.py": SCRIPT_STYLE,
    "ptest_0202_warmstart_homotopy_parallel.py": SCRIPT_STYLE,
    "ptest_0203_wallclock_guard_parallel.py": SCRIPT_STYLE,
    "ptest_0848_fault_split_3d_parallel.py": SCRIPT_STYLE,
    "ptest_1020_fmg_single_field_parallel.py": SCRIPT_STYLE,
    "ptest_1079_diffusion_disk_restart.py": SCRIPT_STYLE,
    "ptest_0851_fault_network_3d_parallel.py": "fails at np 2 and 4 (#900)",
    "ptest_0863_fault_network_3d_width_parallel.py": "fails at np 2 and 4 (#956)",
    "ptest_0763_mesh_adapt_transfer.py": "its one test is skipped unconditionally: adaptMetric fails on N>1",
}


def dark_ptests(script_text, every):
    """The ptest_* files in neither test.sh's PTEST arrays nor DEFERRED_PTESTS, and the
    stale entries. Only the array bodies count: a path in a comment or an echo runs nothing."""
    run = set()
    for body in re.findall(r"^PTEST_\w+=\(\n(.*?)^\)", script_text, re.M | re.S):
        for line in body.splitlines():
            line = line.split("#")[0].strip()
            match = re.fullmatch(r"tests/parallel/(ptest_[A-Za-z0-9_]+\.py)", line)
            if match:
                run.add(match.group(1))
    dark = sorted(every - run - set(DEFERRED_PTESTS))
    stale = sorted(set(DEFERRED_PTESTS) - (every - run))
    return dark, stale


def globs_run_by(script):
    """Every tests/... pattern the script hands to pytest, on an uncommented line.

    Three spellings reach pytest, and missing one of them makes this check
    report the whole suite dark — loudly, which is the right direction, but it
    still has to be kept in step with the script:

    * ``batch <shard> tests/...`` — the sharded call site (#730). The shard
      number decides WHICH CI job runs it, not whether it runs, so coverage
      does not care which one it is.
    * ``"${PYTEST[@]}" tests/...`` — the runner is a bash array.
    * ``$PYTEST tests/...`` — in case a plain one is left behind.
    """
    text = script.read_text().replace("\\\n", " ")
    patterns = set()
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        if re.search(r"\$\{?PYTEST", line) or re.match(r"\s*batch\s+\d+\s", line):
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

    script_text = (REPO / "scripts" / "test.sh").read_text()
    if not re.search(r"^\s*PARALLEL_FILES=\(tests/parallel/test_\*\.py\)", script_text, re.M):
        print("    scripts/test.sh no longer enumerates tests/parallel/test_*.py")
        return 1
    ptests = {p.name for p in (REPO / "tests" / "parallel").glob("ptest_*.py")}
    dark, stale = dark_ptests(script_text, ptests)
    if dark or stale:
        for name in dark:
            print(f"    tests/parallel/{name} is in neither scripts/test.sh nor DEFERRED_PTESTS")
        for name in stale:
            print(f"    DEFERRED_PTESTS: {name} is running, or no longer exists; remove it")
        return 1

    print(f"OK: all {len(every)} test files are reachable from scripts/test.sh "
          f"({len(deferred)} deferred by decision); "
          f"{len(ptests) - len(DEFERRED_PTESTS)} of {len(ptests)} ptest files run "
          f"({len(DEFERRED_PTESTS)} deferred with a reason).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
