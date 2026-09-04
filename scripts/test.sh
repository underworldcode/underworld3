#!/bin/bash

## Testing script (runs pytest in batches)
#
# Usage: ./test.sh [OPTIONS]
#   --p N            Run parallel tests with N MPI ranks (default: skip parallel tests)
#   --full-parallel  Add a second parallel pass at 4 ranks (see below; slow on CI)
#   --parallel-only  Run ONLY parallel tests (skip all serial tests)
#
# Examples:
#   ./test.sh                     # All serial tests only
#   ./test.sh --p 2               # All serial + parallel (2 ranks)
#   ./test.sh --p 2 --full-parallel   # ... and again at 4 ranks
#   ./test.sh --parallel-only --p 2   # Only parallel tests (debugging)
#
# We do not run one monolithic pytest because tests produce a large number of
# PETSc objects which we cannot always guarantee that PETSc / petsc4py will free
# This makes it possible for individual tests to interact with each other.

status=0

# Parse arguments
PARALLEL_RANKS=0
PARALLEL_ONLY=0
# Second parallel pass at four ranks. OFF by default: on a 2-core CI runner
# np=4 is oversubscribed and the pass costs far more than the ~3 minutes it
# takes on a workstation — enough to exceed the 120-minute job cap (#573).
FULL_PARALLEL=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --p)
            PARALLEL_RANKS="$2"
            shift 2
            ;;
        --full-parallel)
            FULL_PARALLEL=1
            shift
            ;;
        --parallel-only)
            PARALLEL_ONLY=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--p N] [--full-parallel] [--parallel-only]"
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
if [ -n "$WORKERS" ] && [ "$WORKERS" -gt 1 ]; then
    echo "Serial batches: $WORKERS worker process(es)"
    PYTEST="pytest --config-file=tests/pytest.ini --dist loadfile -n $WORKERS"
else
    echo "Serial batches: in-process (set WORKERS=N to distribute)"
    PYTEST="pytest --config-file=tests/pytest.ini"
fi

# Run serial tests (unless --parallel-only specified)
if [ $PARALLEL_ONLY -eq 0 ]; then
  echo "Running serial test suite..."
  echo ""

  # Run simple tests (0000-0299: basic functionality, imports, simple operations)
  $PYTEST tests/test_00[0-4]*py || status=1
  $PYTEST tests/test_0050*py || status=1
  $PYTEST tests/test_005[1-9]*py tests/test_006[0-1]*py || status=1
  $PYTEST tests/test_01*py || status=1
  $PYTEST tests/test_02*py || status=1

  # Intermediate tests (0500-0799: data structures, transformations, enhanced interfaces)
  # NOTE: Temporarily disabling test_06*py regression tests (potentially problematic)
  $PYTEST tests/test_05*py tests/test_07*py || status=1
  # $PYTEST tests/test_06*py || status=1  # DISABLED - regression tests need validation

  # Units system tests (0800-0899: unit-aware functions, arrays, and conversions)
  $PYTEST tests/test_08*py || status=1

  # Poisson solvers (including Darcy flow)
  $PYTEST tests/test_100[0-9]*py || status=1

  # Solver / system tests (advanced solver problems)
  # test_101* / test_102* include the rotated free-slip suite (test_1018,
  # issue #504) and the MG / boundary-flux suites, which previously matched
  # no batch glob and never ran in CI.
  $PYTEST tests/test_101*py tests/test_102*py || status=1
  $PYTEST tests/test_105*py || status=1

  # The boundary-normal guard lives under tests/parallel/ but carries NO
  # mpi(min_size=2) mark, because the defect it guards is present in SERIAL in
  # 3-D as well (#564: the facet-to-DOF routing, up to 5.9 degrees on a uniform
  # spherical shell at np=1). Run it here so the serial job covers that path —
  # every other serial test of the default boundary normal is on a box, where
  # flat walls make the question vacuous. It also runs in the --p N batch below.
  $PYTEST tests/parallel/test_1069_boundary_normal_parallel.py || status=1
  # NOT yet batched (issue #504 audit): test_106*py and test_107*py contain
  # level_2/level_3 + slow + tier_b/tier_c suites (e.g. test_1064) and need
  # a triage/deselect decision before being wired into CI.
  #
  # test_1072 is pulled forward out of that group, the same way test_1069 is
  # above, because leaving it there defeats its purpose: it is the ONLY guard on
  # the 3-D free-surface sign and relaxation rate, and #496 exists precisely
  # because those regressions were invisible to CI. Landing the test into an
  # unbatched file would have closed the issue without closing the gap.
  # level_2/tier_b, ~55s serial; passes at np=1 and np=2.
  $PYTEST tests/test_1072_free_surface_spherical.py || status=1

  # Diffusion / Advection tests
  $PYTEST tests/test_1100*py || status=1
  $PYTEST tests/test_1110*py tests/test_1120*py || status=1  # Annulus + vector SL
  $PYTEST tests/test_1450*py || status=1

  # Named (un-numbered) test files - JIT, docstrings, projections
  $PYTEST tests/test_docstring_utils.py tests/test_jit_cache.py \
          tests/test_jit_deterministic_ordering.py \
          tests/test_multicomponent_projection.py \
          tests/test_snes_vector_asymmetric_jacobian.py \
          tests/test_symbol_disambiguation_prototype.py || status=1
else
  echo "Skipping serial tests (--parallel-only specified)"
  echo ""
fi

# Parallel tests (run if --p N specified)
if [ $PARALLEL_RANKS -gt 0 ]; then
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
