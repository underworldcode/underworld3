#!/usr/bin/env python3
"""What is actually blocking the issue and PR queues, in one command.

Written because a 133-issue backlog and a 30-PR queue turned out to be two
different failures wearing the same clothes, and neither was visible without
cross-referencing by hand:

* Fifteen issues were already fixed on ``development`` and nobody had closed
  them. The signal is cheap -- an issue referenced by a merged PR -- but the
  reference is often to a NEIGHBOUR, so this reports candidates to probe and
  never a verdict.
* Four PRs sat red for ten days on two tests that ``development`` had already
  fixed and renamed. Their failures named tests that no longer existed. That
  is the ``phantom`` column, and it is the one worth looking at first.

Read-only: it calls ``gh`` and ``git`` and writes nothing. Run it often; the
cost of the sweep is why it was not being done.

    python scripts/triage.py              # the summary
    python scripts/triage.py --verbose    # every row behind the counts
"""

import argparse
import collections
import json
import re
import subprocess
import sys
import time

CLOSES = re.compile(r"(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)\s+#(\d{2,4})", re.I)
REF = re.compile(r"#(\d{2,4})")
#: A fenced code block. Stripped before scanning for closing keywords, because
#: a PR that PASTES this tool's own output has "closes #640, #747" in its body
#: as sample text. Scanned naively, this script reported its own PR as closing
#: seven issues it has nothing to do with.
FENCE = re.compile(r"^```.*?^```", re.M | re.S)


def prose(body):
    """A PR body with its fenced code blocks removed."""
    return FENCE.sub("", body or "")
#: A failing test pytest named, e.g. "tests/test_1060_x.py::TestC::test_y".
FAILED = re.compile(r"^FAILED\s+(\S+?)::(\S+?)(?:\[|\s|$)")

BASE = "development"
#: Two labels, by where the fix has got to. An issue whose fix is in an OPEN PR
#: carries `fixed-in-PR`; once that PR merges to `development` it is relabelled
#: `fixed-in-development` and stays open until the release, because GitHub's own
#: `Closes #N` fires on a DEFAULT-branch merge and that is `main`.
#:
#: Neither state is visible without a label. An unmerged PR can hold the only
#: fix there is -- #785 carried #783 and #784 and could not merge, its base
#: branch having no PR of its own -- and a merged one has not closed anything.
#:
#: Which PR, and whether it merged, belongs in the issue and in this tool's
#: output, where it can be kept true. A label cannot be.
LABEL_IN_PR = "fixed-in-PR"
LABEL_IN_DEV = "fixed-in-development"


def sh(*args, check=True, tries=1):
    """Run a command, optionally retrying. Returns stdout.

    ``tries`` exists for the ``gh`` calls: the GraphQL API returns a bare 502
    often enough that a tool meant to be run several times a session cannot
    treat one as fatal.
    """
    for attempt in range(1, tries + 1):
        out = subprocess.run(args, capture_output=True, text=True)
        if out.returncode == 0:
            return out.stdout
        # "unexpected end of JSON input" is gh's own report of a short read
        # from the API: a transient, and one it exits non-zero on, so it has to
        # be listed here rather than caught at the parse.
        transient = any(code in out.stderr for code in
                        ("502", "503", "504", "timeout", "rate limit",
                         "unexpected end of JSON input", "connection reset",
                         "EOF"))
        if attempt < tries and transient:
            wait = 2 ** attempt
            print(f"triage: {out.stderr.strip().splitlines()[-1][:70]} "
                  f"-- retrying in {wait}s ({attempt}/{tries - 1})", file=sys.stderr)
            time.sleep(wait)
            continue
        if check:
            print(f"triage: `{' '.join(args[:3])}…` failed:\n{out.stderr}", file=sys.stderr)
            sys.exit(1)
        return out.stdout
    return ""


def gh_json(*args, tries=4):
    """``gh ... --json`` as parsed JSON, retrying a truncated response.

    A short read gives valid-looking output that fails to parse ("unexpected
    end of JSON input"), which `sh` cannot see because the exit status is 0.
    That is the same class of transient as the bare 502s this endpoint also
    returns, so it is retried the same way rather than killing a tool meant to
    be run several times a session.
    """
    for attempt in range(1, tries + 1):
        raw = sh("gh", *args, tries=tries) or "[]"
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            if attempt == tries:
                print(f"triage: `gh {' '.join(args[:2])}…` returned unparseable "
                      f"JSON {tries} times ({exc})", file=sys.stderr)
                sys.exit(1)
            wait = 2 ** attempt
            print(f"triage: truncated JSON from gh -- retrying in {wait}s "
                  f"({attempt}/{tries - 1})", file=sys.stderr)
            time.sleep(wait)
    return []


def issues():
    return gh_json("issue", "list", "--limit", "400", "--json",
                   "number,title,labels,author,createdAt,body")


def _reached_base(pr):
    """Whether a merged PR's merge commit is on ``origin/BASE``."""
    oid = (pr.get("mergeCommit") or {}).get("oid")
    if not oid:
        return False
    return f"origin/{BASE}" in sh("git", "branch", "-r", "--contains", oid, check=False)


def prs(state="open"):
    fields = ("number,title,mergeable,additions,deletions,changedFiles,"
              "statusCheckRollup,body,headRefName,baseRefName,mergeCommit,author,updatedAt")
    args = ["pr", "list", "--limit", "400", "--json", fields]
    if state != "open":
        args += ["--state", state]
    return gh_json(*args)


def ci_state(pr):
    outcomes = [c.get("conclusion") for c in (pr.get("statusCheckRollup") or [])]
    if "FAILURE" in outcomes:
        return "red"
    if "SUCCESS" in outcomes:
        return "green"
    return "none"


def failing_test_names(pr):
    """The (file, test) pairs this PR's failing check reported, if we can get them.

    Best-effort: the log is fetched only for red PRs, and a PR whose log has
    aged out of GitHub's retention simply yields nothing.
    """
    for check in pr.get("statusCheckRollup") or []:
        if check.get("conclusion") != "FAILURE":
            continue
        url = check.get("detailsUrl") or ""
        job = url.rsplit("/job/", 1)[-1]
        if not job.isdigit():
            continue
        log = sh("gh", "run", "view", "--log-failed", "--job", job, check=False)
        found = set()
        for line in log.splitlines():
            # The CI log prefixes each line with the step name and a timestamp.
            hit = FAILED.search(line[line.find("FAILED"):]) if "FAILED" in line else None
            if hit:
                found.add((hit.group(1), hit.group(2)))
        if found:
            return found
    return set()


def classify_on_base(path, name):
    """Why a failing test might not reproduce on the base branch.

    Three outcomes, and only one of them is a phantom:

    ``"phantom"``
        The file is on the base but the test is not. It was renamed or
        rewritten there, so the PR is red for a reason that no longer exists
        and a merge clears it. This is the case that cost ten days.
    ``"new"``
        The file is not on the base at all, so it arrived with the PR. The
        failure is the PR's OWN new test and is real -- reporting it as a
        phantom would send someone to merge the base and find nothing fixed.
        (Caught exactly this way on #800's
        ``test_1103_navier_stokes_velocity_transport.py``.)
    ``"live"``
        Both file and test are on the base. The failure stands.
    """
    body = sh("git", "show", f"origin/{BASE}:{path}", check=False)
    if not body:
        return "new"
    leaf = name.split("::")[-1]
    return "live" if f"def {leaf}" in body else "phantom"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="list every row behind the counts")
    ap.add_argument("--no-logs", action="store_true",
                    help="skip CI log fetches (fast, drops the phantom column)")
    args = ap.parse_args()

    sh("git", "fetch", "-q", "origin", BASE, check=False)
    base_sha = sh("git", "rev-parse", "--short", f"origin/{BASE}").strip()

    iss = issues()
    # An issue labelled fixed-in-development has been adjudicated: it stays open
    # only until the release, so it is counted, never probed again.
    fixed = {i["number"] for i in iss
             if any(l["name"] == LABEL_IN_DEV for l in i["labels"])}
    open_numbers = {i["number"] for i in iss} - fixed
    titles = {i["number"]: i["title"] for i in iss}
    opn = prs()
    merged = prs("merged")

    print(f"=== triage against origin/{BASE} ({base_sha}) ===\n")

    # ---- issues -----------------------------------------------------------
    by_author = collections.Counter(i["author"]["login"] for i in iss)
    unlabelled = sum(1 for i in iss if not i["labels"])
    print(f"ISSUES  {len(iss)} open   {unlabelled} unlabelled   "
          f"{len(fixed)} fixed in development, awaiting release")
    print("  by author: " + ", ".join(f"{a} {n}" for a, n in by_author.most_common()))

    # An issue a MERGED PR referenced is a candidate to probe, not a verdict:
    # the reference is often to a neighbouring defect.
    probe = collections.defaultdict(set)
    for p in merged:
        text = prose(p["body"]) + " " + p["title"]
        for n in {int(m) for m in CLOSES.findall(text)}:
            if n in open_numbers:
                probe[n].add(("closes", p["number"]))
        for n in {int(m) for m in REF.findall(text)}:
            if n in open_numbers and n != p["number"]:
                probe[n].add(("mentions", p["number"]))
    strong = {n for n, src in probe.items() if any(k == "closes" for k, _ in src)}
    print(f"  {len(probe)} referenced by a merged PR -- PROBE these, do not close on the reference")
    print(f"    of which {len(strong)} have a merged PR claiming to close them")
    if args.verbose:
        for n in sorted(probe, key=lambda n: (n not in strong, n)):
            kind = "closes " if n in strong else "mention"
            src = ",".join(f"#{p}" for _, p in sorted(probe[n]))
            print(f"      #{n:<5} [{kind}] {src:<18} {titles[n][:60]}")

    # ---- PRs --------------------------------------------------------------
    print(f"\nPULL REQUESTS  {len(opn)} open")
    buckets = collections.defaultdict(list)
    for p in opn:
        buckets[(p["mergeable"], ci_state(p))].append(p)
    for key in sorted(buckets, key=lambda k: (k[0] != "MERGEABLE", k[1] != "green")):
        mergeable, ci = key
        group = buckets[key]
        closes = sorted({n for p in group
                         for n in {int(m) for m in CLOSES.findall(prose(p["body"]))}
                         if n in open_numbers})
        tag = ", ".join(f"#{n}" for n in closes) or "nothing"
        print(f"  {mergeable:<12} CI {ci:<6} {len(group):>3}   closes {tag}")
        if args.verbose:
            for p in sorted(group, key=lambda p: p["additions"] + p["deletions"]):
                behind = sh("git", "rev-list", "--count",
                            f"origin/{p['headRefName']}..origin/{BASE}", check=False).strip()
                size = p["additions"] + p["deletions"]
                print(f"      #{p['number']:<5} {size:>7}L {p['changedFiles']:>3}f "
                      f"behind {behind or '?':>4}  {p['title'][:52]}")

    no_closes = [p for p in opn if not CLOSES.search(prose(p["body"]))]
    print(f"  {len(no_closes)} of {len(opn)} carry no Closes line")

    # ---- issues whose fix is already written ------------------------------
    # Two ways an open issue can already be fixed, and neither is visible
    # anywhere. A PR merged to `development` has not closed it, because GitHub
    # closes on a DEFAULT-branch merge and that is `main` (139 declared closes
    # across the project's life, 5 slipped -- release-gated, not broken). And
    # an UNMERGED PR can hold the only fix: #785 carries #783 and #784 and
    # cannot merge, its base branch having no PR of its own.
    labels = {i["number"]: {l["name"] for l in i["labels"]} for i in iss}
    declared = {}
    for p in merged:
        # A stacked PR merged into its feature base is a written fix that has
        # reached `development` only if that base has since landed there: the
        # merge commit says which.
        base = p.get("baseRefName") or BASE
        state = "merged" if base == BASE or _reached_base(p) else f"merged to {base}"
        for n in {int(m) for m in CLOSES.findall(prose(p["body"]))}:
            if n in open_numbers:
                declared.setdefault(n, []).append((p["number"], state))
    for p in opn:
        for n in {int(m) for m in CLOSES.findall(prose(p["body"]))}:
            if n in open_numbers:
                declared.setdefault(n, []).append((p["number"], "open"))
    limbo = {n: v for n, v in declared.items() if LABEL_IN_PR not in labels.get(n, ())}
    print("\nFIX ALREADY WRITTEN  (a PR declares it; neither label applied)")
    if not limbo:
        print("  none -- every declared fix is closed or labelled")
    for n in sorted(limbo):
        src = ", ".join(f"#{p} ({st})" for p, st in sorted(limbo[n]))
        print(f"  #{n:<5} {src:<26} {titles[n][:52]}")
    if limbo:
        print("\n  Probe each. Three outcomes, and the third is why this is a list")
        print("  of candidates rather than a list of fixes:")
        print("    already in a release on main          -> gh issue close <N>")
        print(f"    fix is in an OPEN pr, or merged to a  -> gh issue edit <N> "
              f"--add-label {LABEL_IN_PR}")
        print("    feature base that has not landed")
        print(f"    its PR has MERGED to {BASE:<13}    -> gh issue edit <N> "
              f"--add-label {LABEL_IN_DEV}")
        print("    NOT fixed -- the PR addressed a neighbour, or papered over it")
        print("                                          -> leave open, say which part is live")
        print("  #611 is the standing example of the third: #656 was credited with")
        print("  closing it, and the hang is handled by a --deselect in scripts/test.sh")
        print("  at the very rank count the issue reports.")
        print("  A PR marked (open) is the commoner case here -- the fix is written")
        print("  and cannot land. Label the issue; the work is on the PR.")
    carrying = [n for n, ls in labels.items() if {LABEL_IN_PR, LABEL_IN_DEV} & ls]
    if carrying:
        print(f"\n  {len(carrying)} issue(s) are flagged as fixed somewhere; the release "
              f"closes the merged ones:")
        print("    " + " ".join(f"#{n}" for n in sorted(carrying)))

    # ---- the one that cost ten days ---------------------------------------
    if args.no_logs:
        return
    red = [p for p in opn if ci_state(p) == "red"]
    if not red:
        return
    print(f"\nRED PR FAILURES  (checked against {BASE})")
    any_phantom = False
    for p in red:
        groups = collections.defaultdict(list)
        for path, test in failing_test_names(p):
            groups[classify_on_base(path, test)].append((path, test))
        if not groups:
            print(f"  #{p['number']}  no test names in the log (aged out, or the "
                  f"failure is not a test)")
            continue
        behind = sh("git", "rev-list", "--count",
                    f"origin/{p['headRefName']}..origin/{BASE}", check=False).strip()
        if groups["phantom"]:
            any_phantom = True
            print(f"  #{p['number']}  behind {behind}  -- MERGE {BASE} IN, "
                  f"{len(groups['phantom'])} failure(s) renamed away there:")
            for path, test in sorted(groups["phantom"]):
                print(f"      phantom  {path}::{test}")
        for kind, note in (("new", "the PR's own new file -- a real failure"),
                           ("live", "still on the base -- a real failure")):
            if groups[kind]:
                if not groups["phantom"]:
                    print(f"  #{p['number']}  behind {behind}")
                for path, test in sorted(groups[kind]):
                    print(f"      {kind:<8} {path}::{test}")
                print(f"               ^ {note}")
    if not any_phantom:
        print(f"  no phantoms -- every red PR fails on something that is really there")


if __name__ == "__main__":
    main()
