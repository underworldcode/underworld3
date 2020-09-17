#!/usr/bin/env python
# coding: utf-8
# %%
from petsc4py import PETSc
from underworld3 import petsc_gen_xdmf


# %%
options = PETSc.Options()
options["dm_plex_separate_marker"] = None


# %%
dm = PETSc.DMPlex().createFromFile("./input/cubie.dat")


# %%
dm.view()


# %%
outputname = 'cubedphere.h5'
viewer = PETSc.Viewer().createHDF5(outputname, "w")
viewer(dm)
petsc_gen_xdmf.generateXdmf(outputname)

