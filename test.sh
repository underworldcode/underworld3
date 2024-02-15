#!/bin/bash -x

# Run simple tests
pytest tests/test_00*py

# Spatial / calculation tests
pytest tests/test_01*py tests/test_05*py tests/test_06*py

# Solver / system tests (basic solver problems)
pytest tests/test_1000*py tests/test_1001*py 

# Solver / system tests (advanced solver problems)
pytest tests/test_1002*py 

# Darcy is a mixed solver with projection and poisson
pytest tests/test_1004*py 

# Swarm / Advection tests
pytest tests/test_1003*py tests/test_1010*py
