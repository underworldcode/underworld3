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
#: Merges to `main` are infrequent, so GitHub's own issue-closing (which fires
#: only on a default-branch merge) lags reality by a release. An issue whose fix
#: is on `development` carries this label until then: closing it would be a lie
#: to anyone running the release, and leaving it bare loses the fact entirely.
FIXED_LABEL = "fixed-in-development"


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
        transient = any(code in out.stderr for code in ("502", "503", "504",
                                                        "timeout", "rate limit"))
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


def gh_json(*args):
    return json.loads(sh("gh", *args, tries=4) or "[]")


def issues():
    return gh_json("issue", "list", "--limit", "400", "--json",
                   "number,title,labels,author,createdAt,body")


def prs(state="open"):
    fields = ("number,title,mergeable,additions,deletions,changedFiles,"
              "statusCheckRollup,body,headRefName,baseRefName,author,updatedAt")
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
    open_numbers = {i["number"] for i in iss}
    titles = {i["number"]: i["title"] for i in iss}
    opn = prs()
    merged = prs("merged")

    print(f"=== triage against origin/{BASE} ({base_sha}) ===\n")

    # ---- issues -----------------------------------------------------------
    by_author = collections.Counter(i["author"]["login"] for i in iss)
    unlabelled = sum(1 for i in iss if not i["labels"])
    print(f"ISSUES  {len(iss)} open   {unlabelled} unlabelled")
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

    # ---- the gap between "merged" and "released" ---------------------------
    # GitHub closes a linked issue when the PR reaches the DEFAULT branch, and
    # this project's default branch is `main` while the work merges to
    # `development`. Across the project's life 139 declared closes produced 5
    # that slipped, so the mechanism does work -- but it works at release time,
    # and releases are rare. Between a merge and a release an issue is fixed
    # and still open, and nothing says so unless somebody labels it.
    labels = {i["number"]: {l["name"] for l in i["labels"]} for i in iss}
    declared = {}
    for p in merged:
        if p.get("baseRefName") not in (None, BASE):
            continue
        for n in {int(m) for m in CLOSES.findall(prose(p["body"]))}:
            if n in open_numbers:
                declared.setdefault(n, []).append(p["number"])
    limbo = {n: v for n, v in declared.items() if FIXED_LABEL not in labels.get(n, ())}
    print(f"\nFIXED BUT STILL OPEN  (a merged PR declared it; `{FIXED_LABEL}` not applied)")
    if not limbo:
        print(f"  none -- every declared close is either closed or labelled")
    for n in sorted(limbo):
        src = ", ".join(f"#{p}" for p in sorted(limbo[n]))
        print(f"  #{n:<5} declared by {src:<14} {titles[n][:58]}")
    if limbo:
        print("\n  Probe each. Three outcomes, and the third is why this is a list")
        print("  of candidates rather than a list of fixes:")
        print(f"    fixed and you want it off the board   -> gh issue close <N>")
        print(f"    fixed, waiting on a release           -> gh issue edit <N> "
              f"--add-label {FIXED_LABEL}")
        print("    NOT fixed -- the PR addressed a neighbour, or papered over it")
        print("                                          -> leave open, say which part is live")
        print("  #611 is the standing example of the third: #656 was credited with")
        print("  closing it, and the hang is handled by a --deselect in scripts/test.sh")
        print("  at the very rank count the issue reports.")
    carrying = [n for n, ls in labels.items() if FIXED_LABEL in ls]
    if carrying:
        print(f"\n  {len(carrying)} issue(s) already carry `{FIXED_LABEL}` and close at "
              f"the next release:")
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
