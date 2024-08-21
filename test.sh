#!/bin/bash -x

## Testing script (runs pytest in batches)
#
#  We do not run one monolithic pytest because tests produce a large number of 
#  PETSc objects which we cannot always guarantee that PETSc / petsc4py will free
#  This makes it possible for individual tests to interact with each other.

PYTEST="pytest -c tests/pytest.ini"

# Run simple tests
$PYTEST tests/test_00[0-4]*py
$PYTEST tests/test_0050*py

# Spatial / calculation tests
$PYTEST tests/test_01*py tests/test_05*py tests/test_06*py

# Poisson solvers (including Darcy flow)
$PYTEST tests/test_100[0-9]*py 

# Solver / system tests (advanced solver problems)
$PYTEST tests/test_1010*py tests/test_1011*py tests/test_1050*py

# Diffusion / Advection tests
# $PYTEST tests/test_1100*py
# $PYTEST tests/test_1110*py # Annulus version 
