#!/bin/bash -x

# Run simple tests
pytest tests/test_00*py

# Spatial / calculation tests
pytest tests/test_01*py tests/test_05*py tests/test_06*py

# Poisson solvers (including Darcy flow)
pytest tests/test_100[0-9]*py 

# Solver / system tests (advanced solver problems)
pytest tests/test_1010*py tests/test_1011*py tests/test_1050*py

# Diffusion / Advection tests
pytest tests/test_1100*py
# pytest tests/test_1110*py # Interpolation issue