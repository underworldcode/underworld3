#!/bin/bash

## Tiered Testing Script for Underworld3
#
# Usage: ./test_levels.sh [LEVELS]
#   LEVELS can be:
#     1      = Quick tests only (core functionality, ~2-5 minutes)
#     2      = Intermediate tests only (~5-10 minutes)
#     3      = Physics/solver tests only (~10-15 minutes)
#     1,2    = Quick + intermediate tests
#     1,3    = Quick + physics tests
#     2,3    = Intermediate + physics tests
#     1,2,3  = All tests (complete suite, ~20-30 minutes)
#     (empty)= All tests (default)
#
# Examples:
#   ./test_levels.sh 1     # Run only quick tests
#   ./test_levels.sh 3     # Run only physics tests
#   ./test_levels.sh 1,3   # Run quick + physics tests
#   ./test_levels.sh       # Run all tests

set -e  # Exit on any error

# Parse test levels from argument (default to all levels)
TEST_LEVELS_ARG=${1:-"1,2,3"}

# Convert comma-separated string to array
IFS=',' read -ra TEST_LEVELS <<< "$TEST_LEVELS_ARG"

# Initialize status
status=0

# Configure pytest
export UW_NO_USAGE_METRICS=0
PYTEST="pytest --config-file=tests/pytest.ini"

# Function to run pytest with error handling
run_tests() {
    local description="$1"
    shift
    echo "=========================================="
    echo "Running: $description"
    echo "=========================================="
    if ! $PYTEST "$@"; then
        echo "❌ FAILED: $description"
        status=1
    else
        echo "✅ PASSED: $description"
    fi
    echo ""
}

# Functions for each test level
run_level_1() {
    echo "⚠️ Running LEVEL 1: Quick Tests (Core Functionality)"
    echo "Expected runtime: ~2-5 minutes"
    echo ""

    # Core imports and basic functionality (0000-0199)
    run_tests "Core imports and basic functionality" \
        tests/test_0000*py tests/test_0001*py tests/test_0002*py tests/test_0003*py tests/test_0004*py tests/test_0005*py

    # Essential basic tests
    run_tests "Essential data access tests" \
        tests/test_01*py

    # Critical mathematical objects regression tests
    run_tests "Mathematical objects regression (critical)" \
        tests/test_0725*py tests/test_0521*py

    # Essential units functionality
    run_tests "Core units system" \
        tests/test_0803_simple_workflow_demo.py
}

run_level_2() {
    echo "⚠️ Running LEVEL 2: Intermediate Tests"
    echo "Expected runtime: ~5-10 minutes"
    echo ""

    # Intermediate functionality (0500-0799)
    run_tests "Enhanced array structures and data migration" \
        tests/test_05*py

    # Regression tests (0600-0699) - These are important for stability
    run_tests "Critical regression tests" \
        tests/test_06*py

    # Units system (0700-0799)
    run_tests "Mathematical objects and units integration" \
        tests/test_07*py

    # Unit-aware functionality (0800-0899)
    run_tests "Unit-aware functions and workflows" \
        tests/test_08*py
}

run_level_3() {
    echo "⚠️ Running LEVEL 3: Physics and Solver Tests"
    echo "Expected runtime: ~10-15 minutes"
    echo ""

    # Level 3: Physics and solver tests (1000+)
    run_tests "Poisson solvers (including Darcy flow)" \
        tests/test_100[0-9]*py

    run_tests "Stokes flow solvers" \
        tests/test_1010*py tests/test_1011*py tests/test_1050*py

    run_tests "Advection-diffusion and time-stepping" \
        tests/test_1100*py tests/test_1110*py tests/test_1120*py
}

# Validate and run selected test levels
echo "🚀 Running Test Levels: ${TEST_LEVELS[*]}"
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