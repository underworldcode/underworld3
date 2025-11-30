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
#   --parallel-ranks N     Number of MPI ranks for parallel tests (default: 2)
#   --full-parallel        Run parallel tests with both 2 and 4 ranks
#   --no-parallel          Skip parallel tests entirely
#   --verbose              Show verbose test output
#   --help                 Show this help message
#
# Examples:
#   ./test_levels.sh 1                      # Run only quick tests
#   ./test_levels.sh 3                      # Run only physics tests
#   ./test_levels.sh --parallel-ranks 4 2   # Run Level 2 with 4 ranks
#   ./test_levels.sh --full-parallel 1,2,3  # All tests, parallel with 2 and 4 ranks
#   ./test_levels.sh --no-parallel 2        # Level 2 without parallel tests

set -e  # Exit on any error

# Default values
PARALLEL_RANKS=2
FULL_PARALLEL=0
SKIP_PARALLEL=0
VERBOSE=""

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel-ranks)
            PARALLEL_RANKS="$2"
            shift 2
            ;;
        --full-parallel)
            FULL_PARALLEL=1
            shift
            ;;
        --no-parallel)
            SKIP_PARALLEL=1
            shift
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
PYTEST_BASE="pytest --config-file=tests/pytest.ini $VERBOSE"

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
run_level_1() {
    echo "⚡ Running LEVEL 1: Quick Tests (Core Functionality)"
    echo "Using pytest marker: -m level_1"
    echo "Expected runtime: ~2 minutes"
    echo ""

    # Run all tests marked with level_1
    run_tests "Level 1 tests (quick core functionality)" \
        tests/ -m level_1
}

run_level_2() {
    echo "🔧 Running LEVEL 2: Intermediate Tests"
    echo "Using pytest marker: -m level_2"
    echo "Expected runtime: ~5 minutes"
    echo ""

    # Run all tests marked with level_2
    run_tests "Level 2 tests (units, integration, projections)" \
        tests/ -m level_2

    # Parallel tests for global statistics (requires MPI)
    if [ $SKIP_PARALLEL -eq 0 ] && command -v mpirun &> /dev/null && command -v pytest &> /dev/null; then
        echo "=========================================="
        echo "Running parallel tests (MPI required)"
        echo "=========================================="

        # Parallel tests with specified number of ranks
        echo "Testing with $PARALLEL_RANKS MPI ranks..."
        if mpirun -n $PARALLEL_RANKS python -m pytest --with-mpi tests/parallel/test_07*py $VERBOSE; then
            echo "✅ PASSED: Parallel tests ($PARALLEL_RANKS ranks)"
        else
            echo "❌ FAILED: Parallel tests ($PARALLEL_RANKS ranks)"
            status=1
        fi

        # Optional: Test with 4 ranks if --full-parallel specified
        if [ $FULL_PARALLEL -eq 1 ]; then
            echo ""
            echo "Running extended parallel tests (4 ranks)..."
            if mpirun -n 4 python -m pytest --with-mpi tests/parallel/test_07*py $VERBOSE; then
                echo "✅ PASSED: Parallel tests (4 ranks)"
            else
                echo "❌ FAILED: Parallel tests (4 ranks)"
                status=1
            fi
        fi
        echo ""
    elif [ $SKIP_PARALLEL -eq 1 ]; then
        echo "⚠️  Skipping parallel tests (--no-parallel specified)"
        echo ""
    else
        echo "⚠️  Skipping parallel tests (mpirun or pytest not available)"
        echo ""
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
