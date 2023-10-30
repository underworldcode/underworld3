# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Test the  CM updates for stokes & NS
#


import petsc4py
from petsc4py import PETSc

# +
import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function
import numpy as np

import sympy
from sympy import Piecewise
# -

# #### Create the mesh

# +
res = 64
# mesh = uw.meshing.UnstructuredSimplexBox(
#     minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / 50, qdegree=2
# )

mesh  = uw.meshing.StructuredQuadBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), elementRes=(res, res), qdegree=3)
# -


# #### Add some mesh vars

v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

# # Stokes examples

stokes0 = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes0.constitutive_model = uw.constitutive_models.ViscousFlowModel(v, stokes0.DFDt, stokes0.DuDt)


stokes0.constitutive_model.Parameters.shear_viscosity_0

# ## has same memory address
# - when passing in a instance to stokes.constitutive_model

print(stokes0.DuDt)
print(stokes0.constitutive_model.DuDt)

print(stokes0.DFDt)
print(stokes0.constitutive_model.flux_dt)

# ## has different memory address
# - when passing in the class to stokes.constitutive_model

stokes1 = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes1.constitutive_model = uw.constitutive_models.ViscousFlowModel

stokes1.constitutive_model.Parameters.shear_viscosity_0

print(id(stokes1.DFDt))
print(id(stokes1.constitutive_model.flux_dt))

print(stokes1._DFDt)
print(stokes1.constitutive_model.flux_dt)

print(stokes1.DuDt)
print(stokes1.constitutive_model.DuDt)

# # NS examples

NS0 = uw.systems.NavierStokesSLCN(mesh, v, p, order=2)
NS0.constitutive_model = uw.constitutive_models.ViscousFlowModel(v, NS0.DFDt, NS0.DuDt)
NS0.constitutive_model.Parameters.shear_viscosity_0

# +
NS1 = uw.systems.NavierStokesSLCN(mesh, v, p, order=2)

NS1.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
# -

NS1.constitutive_model.Parameters.shear_viscosity_0

# +
NSS0 = uw.systems.NavierStokesSwarm(mesh, v, p, order=2)

NSS0.constitutive_model = uw.constitutive_models.ViscousFlowModel(v, NSS0.DFDt, NSS0.DuDt)
NSS0.constitutive_model.Parameters.shear_viscosity_0
# -

NSS1 = uw.systems.NavierStokesSwarm(mesh, v, p, order=2)
NSS1.constitutive_model = uw.constitutive_models.ViscousFlowModel

NSS1.constitutive_model.Parameters.shear_viscosity_0
