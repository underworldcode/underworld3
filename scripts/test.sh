#!/bin/bash

## Testing script (runs pytest in batches)
#
# Usage: ./test.sh [OPTIONS]
#   --p N            Run parallel tests with N MPI ranks (default: skip parallel tests)
#   --full-parallel  Add a second parallel pass at 4 ranks (see below; slow on CI)
#   --parallel-only  Run ONLY parallel tests (skip all serial tests)
#   --shard K/N      Run only the batches assigned to shard K of N (see
#                    "Shard assignment" below). Omit it and everything runs,
#                    which is what a local full run wants.
#   --tier-c-report  Run ONLY the tier C characterisations and report them.
#                    Never gates: it always exits 0. CI asks for this on a
#                    push to development, not on every pull request.
#
# Examples:
#   ./test.sh                     # All serial tests only
#   ./test.sh --p 2               # All serial + parallel (2 ranks)
#   ./test.sh --p 2 --full-parallel   # ... and again at 4 ranks
#   ./test.sh --parallel-only --p 2   # Only parallel tests (debugging)
#   ./test.sh --p 2 --shard 1/4   # One CI shard's worth
#
# We do not run one monolithic pytest because tests produce a large number of
# PETSc objects which we cannot always guarantee that PETSc / petsc4py will free
# This makes it possible for individual tests to interact with each other.

status=0

# Parse arguments
PARALLEL_RANKS=0
PARALLEL_ONLY=0
TIER_C_REPORT=0
# Second parallel pass at four ranks. OFF by default: on a 2-core CI runner
# np=4 is oversubscribed and the pass costs far more than the ~3 minutes it
# takes on a workstation — enough to exceed the 120-minute job cap (#573).
FULL_PARALLEL=0
# 0/0 means "no sharding": every batch runs. CI passes --shard K/N.
SHARD=0
SHARD_COUNT=0
PRINT_SHARDS=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --p)
            PARALLEL_RANKS="$2"
            shift 2
            ;;
        --shard)
            if [[ ! "$2" =~ ^[0-9]+/[0-9]+$ ]]; then
                echo "Error: --shard wants K/N, got '$2'"
                exit 1
            fi
            SHARD="${2%/*}"
            SHARD_COUNT="${2#*/}"
            shift 2
            ;;
        --print-shards)
            # The shard count, read out of the batch assignment below, as a JSON
            # array. The CI matrix is built from THIS rather than from a second
            # copy of the number: a matrix of [1,2,3] against an assignment that
            # uses 4 would run green while shard 4's batches -- the largest group
            # -- never ran at all. That is the #721 failure, files going dark
            # with nothing to say so, and the only safe fix is one definition.
            PRINT_SHARDS=1
            shift
            ;;
        --full-parallel)
            FULL_PARALLEL=1
            shift
            ;;
        --parallel-only)
            PARALLEL_ONLY=1
            shift
            ;;
        --tier-c-report)
            TIER_C_REPORT=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--p N] [--full-parallel] [--parallel-only] [--shard K/N] [--tier-c-report]"
            exit 1
            ;;
    esac
done

# Validate arguments
if [ $PARALLEL_ONLY -eq 1 ] && [ $PARALLEL_RANKS -eq 0 ]; then
    echo "Error: --parallel-only requires --p N"
    echo "Usage: $0 --parallel-only --p N"
    exit 1
fi

# ---------------------------------------------------------------------------
# Shard assignment
#
# The batches below differ by two orders of magnitude in cost, so they are
# assigned to shards BY HAND from measurements. Round-robin would be worse than
# useless here: it would put test_106*/test_107* (33 min), test_105* (12 min)
# and test_110* (10 min) in whatever order the list happens to be in.
#
# Measured on run 40623403571 (development, 2026-10-28, SUCCESS, 4 xdist
# workers), per serial batch:
#
#   test_106*,test_107*        33:00     <- sets the floor; see the note below
#   test_105*                  11:42
#   test_110*                  10:23
#   test_08*                    7:14
#   test_101*,test_102*         7:05
#   test_00[0-4]*               5:14
#   test_05*,test_06*,test_07*  3:44
#   test_005[1-9],test_00[6-7]* 3:29
#   test_02*,test_03*           1:13
#   test_01*                    0:56
#   test_100[0-9]*,test_103*    0:54
#   docstring/jit_cache/...     0:48
#   test_1110*,test_1120*       0:25
#   parallel/test_1069          0:14
#   test_0050*                  0:08
#   test_1450*                  0:07
#   the --p N mpirun phase       4:38
#   ----------------------------------
#   total                      91:19   (job wall-clock was 96 min with build)
#
# Giving an ESTIMATED 33 / 19 / 18 / 22 minutes across the four shards.
# MEASURED on the first sharded run (40638600008), including install and build:
# 23.6 / 28.3 / 17.3 / 22.6, wall-clock 28.7 min against 96 before. The
# estimates are conservative and do not rank the same way -- a batch costs less
# in a job running only its own batches. Shard 2 is the long pole, not shard 1.
# Re-measure over several runs before re-assigning: runner variance here is of
# the same order as the imbalance left.
#
# The floor is test_106*/test_107*, and it is a TAIL, not a volume: 48% of its
# 149 tests finish in the first 2 minutes, then the remainder takes 31. Since
# the distribution is --dist loadfile, a file is atomic on a worker, so the
# batch cannot finish before its slowest FILE does. Splitting that band further
# is the next step and wants the per-file numbers, not a guess.
#
# How many shards the assignment below actually uses, read from the `batch`
# call sites in this file. Nothing else states the number, so nothing else can
# disagree with it: add a `batch 5 ...` line and CI grows a fifth shard on the
# next run. The CI matrix comes from --print-shards.
SHARDS_ASSIGNED=$(
    grep -oE '^[[:space:]]*batch[[:space:]]+[0-9]+' "$0" \
        | grep -oE '[0-9]+' | sort -n | tail -1
)
if [ -z "$SHARDS_ASSIGNED" ]; then
    echo "Error: no 'batch <shard> ...' call sites found in $0."
    echo "       Either the assignment was removed or the grep no longer matches it."
    exit 1
fi

if [ "$PRINT_SHARDS" -eq 1 ]; then
    # A JSON array for the CI matrix: 4 -> [1,2,3,4]
    printf '['
    for ((i = 1; i <= SHARDS_ASSIGNED; i++)); do
        [ "$i" -gt 1 ] && printf ','
        printf '%d' "$i"
    done
    printf ']\n'
    exit 0
fi

if [ "$SHARD_COUNT" -gt 0 ]; then
    if [ "$SHARD" -lt 1 ] || [ "$SHARD" -gt "$SHARD_COUNT" ]; then
        echo "Error: shard $SHARD is outside 1..$SHARD_COUNT"
        exit 1
    fi
    if [ "$SHARD_COUNT" -ne "$SHARDS_ASSIGNED" ]; then
        echo "Error: asked for shard $SHARD of $SHARD_COUNT, but the batches below"
        echo "       are assigned to $SHARDS_ASSIGNED shards. Running anyway would"
        echo "       leave whole batches unrun with nothing to say so."
        exit 1
    fi
    echo "Shard $SHARD of $SHARD_COUNT"
fi

# batch <shard> <pytest args...> — run a batch if this shard owns it.
#
# Call sites pass globs unquoted, so bash expands them before we see them and
# "$@" carries the file list intact.
batch() {
    local owner=$1
    shift
    if [ "$SHARD_COUNT" -gt 0 ] && [ "$owner" -ne "$SHARD" ]; then
        return 0
    fi
    "${PYTEST[@]}" "$@" || status=1
}

# shard_owns <shard> — for the phases that are not a single pytest call.
shard_owns() {
    [ "$SHARD_COUNT" -eq 0 ] || [ "$1" -eq "$SHARD" ]
}

export UW_NO_USAGE_METRICS=0
# A hard crash must print a Python stack, not just "Segmentation fault".
export PYTHONFAULTHANDLER=1

# Each worker is a full PETSc/BLAS process. Without pinning the thread pools,
# N workers each start their own and oversubscribe the runner badly enough to
# run slower than serial.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# WORKERS distributes the serial batches, one file at a time per worker. It was
# held back while #567 was open: the worker count decides which files share a
# process, and a units test that switched the units system on at IMPORT time
# then reached a module-scoped fixture in the point-locator suite before
# anything reset it. Fixed at source in tests/conftest.py, so this is now just
# a speed knob. Measured here, ./scripts/test.sh --p 2 on 16 cores:
#
#   serial batches   25.1 min -> 9.6 min at 8 workers
#   end to end       28:04    -> 12:21
#
# Unset (or 1) still runs everything in one process, which is what you want
# when a test passes alone and fails in a full run.
# An ARRAY, not a string. Every call site below expands it unquoted, so a string
# would be word-split by bash: `-m 'not tier_c'` becomes the two words `'not` and
# `tier_c'`, and pytest answers that by silently collecting NOTHING and exiting 0
# — a green run that tested nothing. An array carries its elements intact through
# "${PYTEST[@]}", and globs on the command line still expand normally.
PYTEST=(pytest --config-file=tests/pytest.ini)
# Tier C is a REPORT, not a gate: this mode runs the characterisations, prints
# what changed, and always exits 0.
#
# It is separate from the main run because the tier C set includes the slow
# level_3 cases — test_1064 alone is most of it — costing ~42 min, which took
# the job from ~45 to ~100 minutes against a 120-minute cap. Dropping the slow
# ones would re-hide test_1064, whose two-month absence concealed #734, so the
# COST moves off the pull-request path rather than the COVERAGE being cut.
#
# Since #730 it also runs in its own CI job rather than as a step on the test
# job: the test job is now a shard matrix, and a step on it would pay those
# ~42 minutes once per shard to produce one report.
if [ $TIER_C_REPORT -eq 1 ]; then
  echo "=========================================="
  echo "Tier C characterisations (reported, not gating)"
  echo "=========================================="
  if pytest --config-file=tests/pytest.ini -m tier_c tests/; then
    echo "Tier C: all characterisations still hold."
  else
    echo ""
    echo "⚠️  A tier C characterisation changed. This is NOT a failure."
    echo "    Explain what moved and re-characterise the test — do not revert"
    echo "    code to make it pass. See docs/developer/TESTING-RELIABILITY-SYSTEM.md."
  fi
  exit 0
fi

if [ -n "$WORKERS" ] && [ "$WORKERS" -gt 1 ]; then
    echo "Serial batches: $WORKERS worker process(es)"
    PYTEST+=(--dist loadfile -n "$WORKERS")
else
    echo "Serial batches: in-process (set WORKERS=N to distribute)"
fi

# Tier C never gates. A tier C failure demands an EXPLANATION, not a revert
# (Charter S8, docs/developer/TESTING-RELIABILITY-SYSTEM.md): these are
# characterisations that can fail because the code got BETTER, and comparisons
# whose result moves with the fixture. They are excluded from the batches that
# set $status and run in their own reporting pass at the end, which is read but
# cannot fail the build.
# Tier C never gates. A tier C failure demands an EXPLANATION, not a revert
# (Charter S8, docs/developer/TESTING-RELIABILITY-SYSTEM.md): these are
# characterisations that can fail because the code got BETTER, and comparisons
# whose result moves with the fixture. They are excluded from the batches that
# set $status and run in their own reporting pass at the end, which is read but
# cannot fail the build.
PYTEST+=(-m "not tier_c")

# Run serial tests (unless --parallel-only specified)
if [ $PARALLEL_ONLY -eq 0 ]; then
  echo "Running serial test suite..."
  echo ""

  # Every test file must be reachable from a glob below. #721 was a range that
  # stopped covering its own band; 586224f6 took 006x-007x whole so that one
  # cannot recur, but the other ranges are still hand-maintained. This is what
  # notices when one of them goes stale.
  python3 "$(dirname "$0")/check_test_coverage.py" || status=1

  # Run simple tests (0000-0299: basic functionality, imports, simple operations)
  batch 4 tests/test_00[0-4]*py
  batch 4 tests/test_0050*py
  # test_006[2-9] and test_007x matched NO batch glob and so never ran in CI:
  # the whole integration-point suite (0064-0067), swarm repopulation,
  # mid-time velocity and the VE stress history. They are not covered by the
  # disabled test_06*py line below either - that one is 0600-0699. The band is
  # taken whole (00[6-7]) rather than enumerated, so a test added next to its
  # siblings is not dark again. Verified passing (103 tests) before wiring in.
  batch 4 tests/test_005[1-9]*py tests/test_00[6-7]*py
  batch 4 tests/test_01*py
  batch 4 tests/test_02*py tests/test_03*py

  # Intermediate tests (0500-0799: data structures, transformations, enhanced interfaces)
  # test_06*py was disabled as "potentially problematic" and stayed that way. The
  # whole band was measured 2026-09-12 and passes; a band is not a unit of trust,
  # and holding one back for a suspicion nobody recorded meant its level_1 files
  # never ran at all.
  batch 4 tests/test_05*py tests/test_06*py tests/test_07*py

  # Units system tests (0800-0899: unit-aware functions, arrays, and conversions)
  batch 3 tests/test_08*py

  # Poisson solvers (including Darcy flow)
  batch 4 tests/test_100[0-9]*py tests/test_103*py

  # Solver / system tests (advanced solver problems)
  # test_101* / test_102* include the rotated free-slip suite (test_1018,
  # issue #504) and the MG / boundary-flux suites, which previously matched
  # no batch glob and never ran in CI.
  batch 2 tests/test_101*py tests/test_102*py
  batch 2 tests/test_105*py

  # The boundary-normal guard lives under tests/parallel/ but carries NO
  # mpi(min_size=2) mark, because the defect it guards is present in SERIAL in
  # 3-D as well (#564: the facet-to-DOF routing, up to 5.9 degrees on a uniform
  # spherical shell at np=1). Run it here so the serial job covers that path —
  # every other serial test of the default boundary normal is on a box, where
  # flat walls make the question vacuous. It also runs in the --p N batch below.
  batch 4 tests/parallel/test_1069_boundary_normal_parallel.py
  # test_106*/test_107* were held back pending "a triage/deselect decision". That
  # decision is made: the band is not the unit of exclusion. Of the 243 tests in
  # the deferred bands only 26 are level_3 or tier_c; 55 are level_1 AND tier_a —
  # hardened, fast, and exactly what CI exists to run. test_1064 is the slow one
  # and it is excluded by WHAT IT IS (tier_c) below, not by its number.
  batch 1 tests/test_106*py tests/test_107*py
  # test_108*: thin sheet (objective step, the Newton tangent's closure, thin vs
  # thick, the quadratic benchmark); the two slow sphere tests (1081, 1084) are
  # tier C, so the "not tier_c" filter leaves them out. test_109*: the fault
  # placement and split-level multigrid tests that follow. Shard 3: the lightest
  # measured (17.3 min on run 40638600008); not yet timed in a sharded run.
  batch 3 tests/test_108*py tests/test_109*py
  #
  # test_1072 used to be pulled forward out of test_107* by name, because that
  # band was not batched and it is the ONLY guard on the 3-D free-surface sign
  # and relaxation rate (#496 exists because those regressions were invisible to
  # CI). The band runs as a whole now, so the named line is gone rather than
  # running it twice.

  # Diffusion / Advection tests (test_110*: SUPG/SLCN/swarm adv-diff benchmarks)
  batch 3 tests/test_110*py
  batch 4 tests/test_1110*py tests/test_1120*py  # Annulus + vector SL
  batch 4 tests/test_1115*py  # a failed rotated solve raises (#907)
  batch 4 tests/test_1450*py

  # Named (un-numbered) test files - JIT, docstrings, projections
  batch 4 tests/test_docstring_utils.py tests/test_jit_cache.py \
          tests/test_jit_deterministic_ordering.py \
          tests/test_multicomponent_projection.py \
          tests/test_snes_vector_asymmetric_jacobian.py \
          tests/test_symbol_disambiguation_prototype.py
else
  echo "Skipping serial tests (--parallel-only specified)"
  echo ""
fi

# Parallel tests (run if --p N specified)
#
# The whole mpirun phase belongs to ONE shard (4), not to each of them. It
# costs ~4:38 and is the shard's smallest component, but running it four times
# would quadruple the only part of the suite that holds several MPI ranks at
# once, which is also the part most likely to contend on a shared runner.
if [ $PARALLEL_RANKS -gt 0 ] && shard_owns 4; then
  if command -v mpirun &> /dev/null; then
    echo ""
    echo "=========================================="
    echo "Running parallel tests ($PARALLEL_RANKS ranks)"
    echo "=========================================="

    # Test areas where parallel complexity is likely:
    # - Global statistics (mesh/swarm)
    # - Parallel file I/O
    # - Mesh construction and distribution
    # - Solver operations
    # - Global evaluations

    # Every mpirun goes through the supervisor. A rank blocked in a collective
    # produces nothing and never returns, so an unsupervised batch spends the
    # whole job budget in silence and is cancelled from outside with no
    # diagnosis -- measured at 76 minutes (#675). The supervisor bounds that on
    # silence rather than total runtime, so a legitimately slow batch is not
    # punished, and it dumps and compares the ranks before killing them.
    #
    # It matters more here than it did on the two globs it replaced: this pass
    # covers the directory by enumeration, so files that had never run in CI
    # now do, and a new file is exactly where an unbounded hang comes from.
    SUPERVISE="python $(dirname "$0")/mpi_supervisor.py --silence ${PARALLEL_SILENCE:-300} --"

    # Every file under tests/parallel/, in BATCHES. Two things matter here and
    # they pull in opposite directions.
    #
    # Coverage: the batches used to be `test_075*py` and `test_10*py`, which
    # between them named 14 of the 32 files and left a hole from 0760 to 0999 —
    # test_0760, test_0765..test_0790, test_0855 and test_0873 ran in parallel
    # at NO rank count, in CI or locally. That is how the np=4 hang in #611
    # survived unnoticed, and it is the #570 class: a glob naming ranges grows
    # holes as files are added between them. So the list is ENUMERATED, and
    # covers the directory by construction.
    #
    # Batching: this script does not run one monolithic pytest, for the reason
    # in its header — PETSc objects accumulate across files and tests start
    # interacting. Handing the whole directory to a single mpirun took the CI
    # job past its 120-minute cap while the same set in batches costs a couple
    # of minutes. So the enumeration is chunked rather than passed in one go.
    PARALLEL_BATCH=${PARALLEL_BATCH:-6}
    PARALLEL_FILES=(tests/parallel/test_*.py)
    echo "Testing parallel operations, swarms and solvers"
    echo "  ${#PARALLEL_FILES[@]} files in batches of $PARALLEL_BATCH"
    for ((i = 0; i < ${#PARALLEL_FILES[@]}; i += PARALLEL_BATCH)); do
      $SUPERVISE mpirun -n $PARALLEL_RANKS python -m pytest --with-mpi \
        "${PARALLEL_FILES[@]:i:PARALLEL_BATCH}" || status=1
    done

    # A SECOND pass at four ranks, because two is a special case: the failure
    # this suite exists to catch is a collective entered by some ranks and not
    # others, and with two ranks the mismatched pair often still meets. Both
    # recent instances passed at np=2 and hung at np=4 (#609's conditional
    # collective, and #611).
    #
    # Opt-in via --full-parallel. A CI runner has two cores, so np=4 is
    # oversubscribed there; this belongs in a nightly or a separate job (#573).
    #
    # The deselection is #611: that test passes at np=2 and hangs at np=4 on
    # development. The node id has no `tests/` prefix because tests/pytest.ini
    # puts rootdir at `tests/`, and a deselect that does not match is ignored in
    # silence — confirm "1 deselected" in the output when changing it.
    if [ $FULL_PARALLEL -eq 1 ] && [ "$PARALLEL_RANKS" -ne 4 ]; then
      echo "Testing the same set at 4 ranks (np=2 is a special case)..."
      for ((i = 0; i < ${#PARALLEL_FILES[@]}; i += PARALLEL_BATCH)); do
        $SUPERVISE mpirun -n 4 python -m pytest --with-mpi \
          "${PARALLEL_FILES[@]:i:PARALLEL_BATCH}" \
          --deselect "parallel/test_0760_swarm_cache_migration.py::test_global_evaluate_after_migration" \
          || status=1
      done
    fi

    echo "Parallel tests complete"
    echo "=========================================="
  else
    echo "⚠️  Warning: --p $PARALLEL_RANKS specified but mpirun not available"
    echo "⚠️  Skipping parallel tests"
  fi
elif [ $PARALLEL_RANKS -gt 0 ]; then
  # --p N was given, but this shard does not own the mpirun phase. Say which
  # one does, so a reader of one shard's log does not conclude the parallel
  # tests were skipped for the whole run.
  echo ""
  echo "Parallel tests belong to shard 4; this is shard $SHARD."
else
  echo ""
  echo "⚠️  Skipping parallel tests (use --p N to enable)"
fi

#
if [ $status -ne 0 ]; then
  echo ""
  echo "❌ Some tests failed."
  exit 1
else
  echo ""
  echo "✅ All tests passed!"
  exit 0
fi
