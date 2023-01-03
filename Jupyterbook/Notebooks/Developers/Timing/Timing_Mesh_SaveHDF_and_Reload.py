# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Mesh save / load from hdf5

# +
import os
os.environ["UW_TIMING_ENABLE"] = "1"

import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import sympy

free_slip_upper = True

import os

os.environ["SYMPY_USE_CACHE"] = "no"
os.environ["UW_TIMING_ENABLE"] = "1"

# Define the problem size
#      1 - ultra low res for automatic checking
#      2 - low res problem to play with this notebook
#      3 - medium resolution (be prepared to wait)
#      4 - highest resolution (benchmark case from Spiegelman et al)

problem_size = 2

# For testing and automatic generation of notebook output,
# over-ride the problem size if the UW_TESTING_LEVEL is set

uw_testing_level = os.environ.get("UW_TESTING_LEVEL")
if uw_testing_level:
    try:
        problem_size = int(uw_testing_level)
    except ValueError:
        # Accept the default value
        pass

r_o = 1.0
r_i = 0.5

if problem_size <= 1:
    res = 0.2
elif problem_size == 2:
    res = 0.1
elif problem_size == 3:
    res = 0.05
elif problem_size == 4:
    res = 0.025
elif problem_size == 5:
    res = 0.01
elif problem_size == 6:
    res = 0.005
elif problem_size == 7:
    res = 0.001

# +
from underworld3 import timing

timing.reset()
timing.start()
# -

mesh1 = uw.meshing.Annulus(radiusOuter=r_o, radiusInner=r_i, cellSize=res)


mesh1.dm.view()

savefile = f"uw_mesh_h5_test_res{problem_size}.h5"
mesh1.save(savefile)

plex = petsc4py.PETSc.DMPlex().createFromFile(savefile)

mesh2 = uw.discretisation.Mesh(plex)

mesh2.dm.view()

timing.print_table(display_fraction=0.999)


