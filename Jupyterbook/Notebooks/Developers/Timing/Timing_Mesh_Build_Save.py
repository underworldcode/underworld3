# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Mesh save / load from hdf5
#
# Build (`gmsh`) meshes for various resolutions and then save them as hdf5 files. After that, reload the mesh. Note that the gmsh part is serial so we are really checking that this is properly managed and that the parallel save does not blow up. 
#
# The script can be run with a negative testing level which will skip the build and re-use the old mesh. This is designed to evaluate the loading of hdf5 dmplex mesh checkpoints for decomposed meshes cleanly.
#
# For most 3D cases, the serial bottleneck is enough that we probably prefer to create the meshes in advance and read them from the h5 checkpoint (we know that because of this timing file !).
#

# +
import os
os.environ["UW_TIMING_ENABLE"] = "1"
os.environ["SYMPY_USE_CACHE"] = "no"

import petsc4py
from petsc4py import PETSc

from underworld3 import timing


import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import sympy

free_slip_upper = True


# Define the problem size
#      1 - ultra low res for automatic checking
#      2 - low res problem to play with this notebook
#      3 - medium resolution (be prepared to wait)
#      4 - highest resolution (benchmark case from Spiegelman et al)

    
# Earth-like ratio of inner to outer
r_o = 1.0
r_i = 0.547

problem_size = uw.options.getInt("problem_size", default=1)
problem_dim = uw.options.getInt("problem_dim", default=3)

if problem_size < 0:
    problem_size = -problem_size
    skip_gmsh = True
else:
    skip_gmsh = False


print(f"problem size {problem_size}")
print(f"problem dim {problem_dim}")



# +
if problem_dim == 2:


    if problem_size <= 1: 
        element_kms = 300
    elif problem_size == 2: 
        element_kms = 150
    elif problem_size == 3: 
        element_kms = 100
    elif problem_size == 4: 
        element_kms = 50
    elif problem_size == 5: 
        element_kms = 25
    elif problem_size == 6: 
        element_kms = 10
    elif problem_size == 7: 
        element_kms = 5
    elif problem_size >= 8: 
        element_kms = 1

    cell_size = element_kms / 6370
    res = cell_size

    element_kms, res



    timing.reset()
    timing.start()

    if uw.mpi.rank == 0:
        print(f"2d mesh generation @ {element_kms}km", flush=True)

    # Annulus in 2D

    tmp_filename = f".meshes/uw_mesh_h5_test2d_res{element_kms}km.msh"

    if not skip_gmsh:
        mesh1 = uw.meshing.Annulus(radiusOuter=r_o, 
                                   radiusInner=r_i, 
                                   cellSize=res,
                                   filename=tmp_filename)

        if uw.mpi.rank == 0:
            print("2d mesh generation complete", flush=True)

        mesh1.dm.view()    

    if uw.mpi.rank == 0:
        print("Read back 2d mesh", flush=True)

    mesh2 = uw.discretisation.Mesh(tmp_filename + ".h5")
    mesh2.dm.view()

    timing.print_table(display_fraction=0.999)


# -


if problem_dim == 3: 
    
    ## The sphere:

    timing.reset()
    timing.start()

    ## "Standard" Resolution Meshes:

    if problem_size <= 1: 
        element_kms = 1000 
    elif problem_size == 2: 
        element_kms = 500 
    elif problem_size == 3: 
        element_kms = 333 
    elif problem_size == 4: 
        element_kms = 200 
    elif problem_size == 5: 
        element_kms = 100
    elif problem_size == 6: 
        element_kms = 50
    elif problem_size == 7: 
        element_kms = 33
    elif problem_size >= 8: 
        element_kms = 25

    cell_size = element_kms / 6370
    res = cell_size

    if uw.mpi.rank == 0:
        print(f"3d mesh generation @ {element_kms}km", flush=True)

    tmp_filename = f".meshes/uw_mesh_h5_test3d_res{element_kms}km.msh"

    if not skip_gmsh:
        mesh3 = uw.meshing.SphericalShell(
            radiusInner=r_i, 
            radiusOuter=r_o, 
            cellSize=res, 
            qdegree=2,
            filename = tmp_filename,
            verbosity=0,
        )

        if uw.mpi.rank == 0:
            print("3d mesh generation complete", flush=True)

        mesh3.dm.view()

    if uw.mpi.rank == 0:
        print("Read back 3d mesh", flush=True)


    mesh4 = uw.discretisation.Mesh(tmp_filename+".h5")
    mesh4.dm.view()

    timing.print_table(display_fraction=1)


