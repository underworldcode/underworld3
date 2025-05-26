#!/bin/bash -x

## Testing script (runs pytest in batches)
#
#  We do not run one monolithic pytest because tests produce a large number of
#  PETSc objects which we cannot always guarantee that PETSc / petsc4py will free
#  This makes it possible for individual tests to interact with each other.

status=0

export UW_NO_USAGE_METRICS=0
PYTEST="pytest --config-file=tests/pytest.ini"

# Run simple tests
$PYTEST tests/test_00[0-4]*py || status=1
#$PYTEST tests/test_0050*py    || status=1 # disable auditor test for now


# Spatial / calculation tests
$PYTEST tests/test_01*py tests/test_05*py tests/test_06*py || status=1

# Poisson solvers (including Darcy flow)
$PYTEST tests/test_100[0-9]*py || status=1

# Solver / system tests (advanced solver problems)
$PYTEST tests/test_1010*py tests/test_1011*py tests/test_1050*py || status=1

# Diffusion / Advection tests
# $PYTEST tests/test_1100*py || status=1
# $PYTEST tests/test_1110*py # Annulus version || status=1
#
if [ $status -ne 0 ]; then
  echo "Some test failed."
  exit 1
else
  exit 0
fi
