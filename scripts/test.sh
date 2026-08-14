#!/bin/bash

## Testing script (runs pytest in batches)
#
# Usage: ./test.sh [OPTIONS]
#   --p N            Run parallel tests with N MPI ranks (default: skip parallel tests)
#   --parallel-only  Run ONLY parallel tests (skip all serial tests)
#
# Examples:
#   ./test.sh                     # All serial tests only
#   ./test.sh --p 2               # All serial + parallel (2 ranks)
#   ./test.sh --parallel-only --p 2   # Only parallel tests (debugging)
#
# We do not run one monolithic pytest because tests produce a large number of
# PETSc objects which we cannot always guarantee that PETSc / petsc4py will free
# This makes it possible for individual tests to interact with each other.

status=0

# Parse arguments
PARALLEL_RANKS=0
PARALLEL_ONLY=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --p)
            PARALLEL_RANKS="$2"
            shift 2
            ;;
        --parallel-only)
            PARALLEL_ONLY=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--p N] [--parallel-only]"
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
PYTEST="pytest --config-file=tests/pytest.ini"

# Run serial tests (unless --parallel-only specified)
if [ $PARALLEL_ONLY -eq 0 ]; then
  echo "Running serial test suite..."
  echo ""

  # Run simple tests (0000-0299: basic functionality, imports, simple operations)
  $PYTEST tests/test_00[0-4]*py || status=1
  #$PYTEST tests/test_0050*py    || status=1 # disable auditor test for now
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
  # NOT yet batched (issue #504 audit): test_106*py and test_107*py contain
  # level_2/level_3 + slow + tier_b/tier_c suites (e.g. test_1064) and need
  # a triage/deselect decision before being wired into CI.

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

    echo "Testing global statistics and parallel operations..."
    mpirun -n $PARALLEL_RANKS python -m pytest --with-mpi tests/parallel/test_075*py || status=1

    # Parallel SOLVER tests. This line was commented out, so test_1017 and
    # test_1062..test_1068 — the whole rotated / constrained / MG parallel set,
    # including the partition-independence guard for the rotated nodal normal
    # (#560) — executed at NO rank count in CI.
    echo "Testing parallel solvers..."
    mpirun -n $PARALLEL_RANKS python -m pytest --with-mpi tests/parallel/test_10*py || status=1

    # echo "Testing parallel I/O..."
    # mpirun -n $PARALLEL_RANKS python -m pytest --with-mpi tests/parallel/test_io*py || status=1

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
