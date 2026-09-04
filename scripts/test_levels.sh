#!/bin/bash

## Tiered Testing Script for Underworld3
#
# Usage: ./test_levels.sh [OPTIONS] [LEVELS]
#
# This script uses pytest markers to select tests by complexity level.
# Tests are tagged with @pytest.mark.level_1, level_2, or level_3 in the source.
#
# LEVELS:
#   1      = Quick tests only (core functionality, ~2 minutes)
#   2      = Intermediate tests only (~5 minutes)
#   3      = Physics/solver tests only (~10+ minutes)
#   1,2    = Quick + intermediate tests
#   1,3    = Quick + physics tests
#   2,3    = Intermediate + physics tests
#   1,2,3  = All tests (complete suite, ~20-30 minutes)
#   (empty)= All tests (default)
#
# OPTIONS:
#   --parallel          Run parallel (MPI) tests with 2 ranks
#   --parallel-ranks N  Run parallel (MPI) tests with N ranks
#   --full-parallel     Run parallel tests with both 2 and 4 ranks
#   --isolation         Run on ONE worker, still per-file (for pinning down
#                       a pollution failure; slower than the default)
#   --workers N, -j N   Worker processes (default: min(cores, 8))
#   --verbose           Show verbose test output
#   --help              Show this help message
#
# Defaults:
#   Tests run across min(cores, 8) worker processes, one file at a time per
#   worker, and WITHOUT parallel (MPI) tests. Level 1 measured on 16 cores:
#   9:45 serial, 1:32 at 8 workers.
#
# Process Isolation (--isolation):
#   Drops to ONE worker, still one file at a time. Every run is already
#   per-file isolated; this removes the concurrency as well, which is what you
#   want when a test passes alone and fails in a full run.
#
# Examples:
#   ./test_levels.sh 1                       # Quick tests, all workers
#   ./test_levels.sh --isolation 1,2         # Levels 1+2 with process isolation
#   ./test_levels.sh --parallel 2            # Level 2 with MPI tests (2 ranks)
#   ./test_levels.sh --parallel-ranks 4 2    # Level 2 with MPI tests (4 ranks)
#   ./test_levels.sh --full-parallel         # All tests, parallel with 2 and 4 ranks

set -e  # Exit on any error

# Default values (simple/fast mode)
PARALLEL_RANKS=2
RUN_PARALLEL=0      # OFF by default
FULL_PARALLEL=0
RUN_ISOLATION=0     # OFF by default
VERBOSE=""

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            RUN_PARALLEL=1
            shift
            ;;
        --parallel-ranks)
            RUN_PARALLEL=1
            PARALLEL_RANKS="$2"
            shift 2
            ;;
        --full-parallel)
            RUN_PARALLEL=1
            FULL_PARALLEL=1
            shift
            ;;
        --isolation)
            RUN_ISOLATION=1
            shift
            ;;
        --workers|-j)
            WORKERS="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE="-v"
            shift
            ;;
        --help|-h)
            grep "^#" "$0" | sed 's/^# \?//'
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            # This is the test levels argument
            TEST_LEVELS_ARG="$1"
            shift
            ;;
    esac
done

# Default to all levels if not specified
TEST_LEVELS_ARG=${TEST_LEVELS_ARG:-"1,2,3"}

# Convert comma-separated string to array
IFS=',' read -ra TEST_LEVELS <<< "$TEST_LEVELS_ARG"

# Initialize status
status=0

# Configure pytest base command
export UW_NO_USAGE_METRICS=0
# Disable telemetry during tests to prevent race conditions with kdtree
export UW_ENABLE_TELEMETRY=0

# Tests run across several worker processes by default.
#
# --dist loadfile keeps every test in a file on ONE worker, which is the
# granularity the suite is safe at: tests within a file are written to follow
# each other, while global state (Model, units, PETSc contexts) does not
# survive between workers.
#
# Each worker is a full PETSc/BLAS process, so the thread pools have to be
# pinned; without this, N workers each start their own and oversubscribe the
# machine badly enough to run SLOWER than serial.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Worker count. Measured on a 16-core box, level 1: serial 9:45, -n 4 2:17,
# -n 8 1:32, -n 16 1:31 — so throughput saturates around 8 and the last
# doubling buys nothing. It also costs something: at one worker per core the
# file-to-worker grouping changes and three point-locator tests fail on state
# left by whatever shared their process. Default to 8, capped by the machine.
if [ -z "$WORKERS" ]; then
    _cores=$( (command -v nproc >/dev/null && nproc) \
              || sysctl -n hw.ncpu 2>/dev/null || echo 4 )
    WORKERS=$(( _cores < 8 ? _cores : 8 ))
fi

# --timeout=120: prevent tests from hanging indefinitely (2 min per test max)
# Show test configuration
echo "Configuration:"
if [ $RUN_ISOLATION -eq 1 ]; then
    # One worker, still per-file: sequential AND isolated, for pinning down a
    # pollution failure rather than for speed.
    ISOLATION_OPTS="--dist loadfile -n 1"
    echo "  🔒 Process isolation: ON (1 worker, one file at a time)"
else
    ISOLATION_OPTS="--dist loadfile -n $WORKERS"
    echo "  ⚡ Workers: $WORKERS (one file at a time per worker)"
fi
if [ $RUN_PARALLEL -eq 1 ]; then
    echo "  🔀 Parallel (MPI): ON ($PARALLEL_RANKS ranks)"
else
    echo "  🔀 Parallel (MPI): OFF"
fi
echo ""

PYTEST_BASE="pytest --config-file=tests/pytest.ini --timeout=120 $ISOLATION_OPTS $VERBOSE"

# Function to run pytest with error handling
run_tests() {
    local description="$1"
    shift
    echo "=========================================="
    echo "Running: $description"
    echo "=========================================="
    if ! $PYTEST_BASE "$@"; then
        echo "❌ FAILED: $description"
        status=1
    else
        echo "✅ PASSED: $description"
    fi
    echo ""
}

# Functions for each test level using pytest markers
#
# A level selection EXCLUDES the levels above it, and has to: pytest MERGES
# marks rather than overriding them, so a file whose module declares
# `pytestmark = pytest.mark.level_1` and whose heavy test then carries
# `@pytest.mark.level_2` leaves that test marked BOTH. A plain `-m level_1`
# selects it, and the demotion the author wrote does nothing. Nine files rely
# on that demotion; the heaviest of their tests is a 96-second homotopy solve
# that was running in the "quick" tier because of it.
run_level_1() {
    echo "⚡ Running LEVEL 1: Quick Tests (Core Functionality)"
    echo "Using pytest marker: -m 'level_1 and not level_2 and not level_3'"
    echo "Expected runtime: ~2 minutes"
    echo ""

    # Tests marked level_1 and NOT demoted to a higher level (see above)
    run_tests "Level 1 tests (quick core functionality)" \
        tests/ -m "level_1 and not level_2 and not level_3"
}

run_level_2() {
    echo "🔧 Running LEVEL 2: Intermediate Tests"
    echo "Using pytest marker: -m 'level_2 and not level_3'"
    echo "Expected runtime: ~5 minutes"
    echo ""

    # Tests marked level_2 and NOT demoted to level_3 (see above)
    run_tests "Level 2 tests (units, integration, projections)" \
        tests/ -m "level_2 and not level_3"

    # Parallel tests for global statistics (requires MPI)
    if [ $RUN_PARALLEL -eq 1 ]; then
        if command -v mpirun &> /dev/null && command -v pytest &> /dev/null; then
            echo "=========================================="
            echo "Running parallel tests (MPI)"
            echo "=========================================="

            # The WHOLE directory, not `test_07*py`. That glob covered
            # test_0700..test_0790 and so missed test_0005, test_0855,
            # test_0873 and the entire test_10* solver set. `scripts/test.sh`
            # had a different hole (test_075* and test_10*, missing
            # test_0760..test_0790), so between the two scripts three files ran
            # at no rank count anywhere: test_0005, test_0855 and test_0873.
            # A glob naming ranges grows holes as files are added between them
            # (#570, #611).
            echo "Testing with $PARALLEL_RANKS MPI ranks..."
            if mpirun -n $PARALLEL_RANKS python -m pytest --with-mpi tests/parallel/ $VERBOSE; then
                echo "✅ PASSED: Parallel tests ($PARALLEL_RANKS ranks)"
            else
                echo "❌ FAILED: Parallel tests ($PARALLEL_RANKS ranks)"
                status=1
            fi

            # Optional: Test with 4 ranks if --full-parallel specified
            if [ $FULL_PARALLEL -eq 1 ]; then
                echo ""
                # The deselection is #611: this test passes at np=2 and hangs
                # at np=4 on development. The node id carries no `tests/`
                # prefix because tests/pytest.ini puts rootdir at `tests/`, and
                # a deselect that does not match is ignored in silence.
                echo "Running extended parallel tests (4 ranks)..."
                if mpirun -n 4 python -m pytest --with-mpi tests/parallel/ \
                    --deselect "parallel/test_0760_swarm_cache_migration.py::test_global_evaluate_after_migration" \
                    $VERBOSE; then
                    echo "✅ PASSED: Parallel tests (4 ranks)"
                else
                    echo "❌ FAILED: Parallel tests (4 ranks)"
                    status=1
                fi
            fi
            echo ""
        else
            echo "⚠️  Parallel tests requested but mpirun or pytest not available"
            echo ""
        fi
    fi
}

run_level_3() {
    echo "🔬 Running LEVEL 3: Physics and Solver Tests"
    echo "Using pytest marker: -m level_3"
    echo "Expected runtime: ~10-15 minutes"
    echo ""

    # Run all tests marked with level_3
    run_tests "Level 3 tests (physics solvers, time-stepping)" \
        tests/ -m level_3
}

# Validate and run selected test levels
echo "🚀 Running Test Levels: ${TEST_LEVELS[*]}"
echo ""
echo "Test level criteria:"
echo "  Level 1: Quick core tests - imports, setup, no solving"
echo "  Level 2: Intermediate - units, integration, simple projections"
echo "  Level 3: Physics - full solvers, time-stepping, benchmarks"
echo ""

for level in "${TEST_LEVELS[@]}"; do
    case $level in
        1)
            run_level_1
            ;;
        2)
            run_level_2
            ;;
        3)
            run_level_3
            ;;
        *)
            echo "❌ Invalid test level: $level"
            echo "Usage: $0 [LEVELS]"
            echo "  Valid levels: 1, 2, 3"
            echo "  Examples: $0 1, $0 2,3, $0 1,2,3"
            exit 1
            ;;
    esac
done

# Final status report
echo "=========================================="
if [ $status -eq 0 ]; then
    echo "✅ ALL TESTS PASSED for Levels: ${TEST_LEVELS[*]}!"
    echo "=========================================="
    exit 0
else
    echo "❌ SOME TESTS FAILED for Levels: ${TEST_LEVELS[*]}"
    echo "=========================================="
    exit 1
fi
