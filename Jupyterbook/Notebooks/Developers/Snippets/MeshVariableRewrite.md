---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Title

```{code-cell} ipython3
import petsc4py
from petsc4py import PETSc
import os
os.environ["UW_TIMING_ENABLE"] = "1"

## --- Underworld

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function
from underworld3 import timing

import numpy as np
import sympy

from underworld3.coordinates import CoordinateSystem, CoordinateSystemType
from underworld3.cython import petsc_discretisation
```

```{code-cell} ipython3
n_els = 4
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / n_els, 
    qdegree=3, refinement=3
)

mesh.dmVars = mesh.dm.clone()
```

```{code-cell} ipython3
uw_mv_1 = uw.discretisation.MeshVariable("uwVar1", mesh, vtype=uw.VarType.SCALAR, degree=1)
uw_mv_2 = uw.discretisation.MeshVariable("uwVar2", mesh, vtype=uw.VarType.VECTOR, degree=2)
uw_mv_3 = uw.discretisation.MeshVariable("uwVar3", mesh, vtype=uw.VarType.VECTOR, degree=3)
```

```{code-cell} ipython3
class meshVarProto(uw._api_tools.uw_object):
    """
    Mesh Variable Rewrite Test Classs
    """

    def __init__(self, mesh, name):

        import re

        # Notes:
        #    holding a local dm for the variable


        self.mesh = mesh
        self.name = name
        self.clean_name = re.sub(r"[^a-zA-Z0-9_]", "", name)

        self.dm = self.mesh.dm.clone()
        self.num_components = 2

        dim = self.mesh.dim

        
        petsc_fe = PETSc.FE().createDefault(
            dim,
            self.num_components,
            self.mesh.isSimplex,
            self.mesh.qdegree,
            name + "_",
            PETSc.COMM_WORLD,
        )

        # global
        # self.field_id = self.mesh.dm.getNumFields()
        # self.mesh.dm.setField(self.field_id, petsc_fe)
        # field, _ = self.mesh.dm.getField(self.field_id)
        # field.setName(self.clean_name)

        # local
        self.field_id = self.dm.getNumFields()
        self.dm.setField(self.field_id, petsc_fe)
        field, _ = self.dm.getField(self.field_id)
        field.setName(self.clean_name)

        self._lvec = self.dm.createLocalVec()
        
        

        return
```

```{code-cell} ipython3
meshVar1 = meshVarProto(mesh, "testVar1")
meshVar2 = meshVarProto(mesh, "testVar2")
meshVar3 = meshVarProto(mesh, "testVar3")
```

```{code-cell} ipython3
mesh.dmVars.addField(meshVar1.dm.getField(meshVar1.field_id)[0], None)
mesh.dmVars.addField(meshVar2.dm.getField(meshVar2.field_id)[0], None)
mesh.dmVars.addField(meshVar3.dm.getField(meshVar3.field_id)[0], None)

mesh.dm_h.addField(meshVar1.dm.getField(meshVar1.field_id)[0], None)
mesh.dm_h.addField(meshVar2.dm.getField(meshVar2.field_id)[0], None)
mesh.dm_h.addField(meshVar3.dm.getField(meshVar3.field_id)[0], None)
```

```{code-cell} ipython3
print(meshVar1._lvec)
print(meshVar2._lvec)
```

```{code-cell} ipython3
meshVar1._lvec.setArray(1)
meshVar2._lvec.setArray(2)
```

```{code-cell} ipython3
mesh.dmVars.createFieldDecomposition()
```

```{code-cell} ipython3
dm0 = mesh.dm_h
for i in range(mesh.dm_h.getRefineLevel()):
        cdm = dm0.getCoarseDM()
        mesh.dm_h.copyFields(cdm)
        dm0 = cdm
```

```{code-cell} ipython3
mesh.dm_h.createFieldDecomposition()
```

```{code-cell} ipython3
mesh.dm.view()
```

```{code-cell} ipython3
with mesh.access(uw_mv_1, uw_mv_2, uw_mv_3):
    uw_mv_1.data[...] = -1.0
    uw_mv_2.data[...] = -2.0
    uw_mv_3.data[...] = -3.0
```

```{code-cell} ipython3
with mesh.access():
    uw_vec1 = uw_mv_1.vec.copy()    
    uw_vec2 = uw_mv_2.vec.copy()    
    uw_vec3 = uw_mv_3.vec.copy()
```

```{code-cell} ipython3
mesh.update_lvec()
```

```{code-cell} ipython3
mesh._lvec.array
```

```{code-cell} ipython3
mesh.vars
```

```{code-cell} ipython3
with mesh.access():
    print(uw_mv_1.data.shape[0] * 6 )
```

```{code-cell} ipython3
mesh.dm.view()
```

```{code-cell} ipython3
vecs = []
with mesh.access():
    for var_key in mesh._vars:
        var = mesh._vars[var_key]
        vecs.append(var.vec)

vn = petsc4py.PETSc.Vec().createNest(vecs)
```

```{code-cell} ipython3
vnc = vn.copy()
```

```{code-cell} ipython3
vnc.getNestSubVecs()
```

```{code-cell} ipython3
concVec = uw.cython.petsc_discretisation.petsc_vec_concatenate(vecs)
```

```{code-cell} ipython3
concVec.array
```

```{code-cell} ipython3
mesh.dm.view()
```

```{code-cell} ipython3
mesh._lvec.array.shape
```

```{code-cell} ipython3
uw_mv_4 = uw.discretisation.MeshVariable("uwVar4", mesh, vtype=uw.VarType.VECTOR, degree=1)
```

```{code-cell} ipython3
mesh.dm.view()
```

```{code-cell} ipython3
with mesh.access():
    print(uw_mv_1.data)
    print(uw_mv_4.data)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
0/0
```

```{code-cell} ipython3
with mesh.access():
    print(uw_mv_4.data)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
mesh.dm.view()
```

```{code-cell} ipython3
mesh._lvec.array.shape
```

```{code-cell} ipython3
with mesh.access():
    print(uw_mv_4.data)
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3
0/0
```

```{code-cell} ipython3
petsc_fe = PETSc.FE().createDefault(
            mesh.dim,
            3,
            mesh.isSimplex,
            mesh.qdegree,
            "AddField_1",
            PETSc.COMM_WORLD,
        )
```

```{code-cell} ipython3
dm1 = mesh.dm.clone()
mesh.dm.copyFields(dm1)
```

```{code-cell} ipython3
dm1.addField(petsc_fe)
dm1.createDS()
```

```{code-cell} ipython3
dm1.view()
```

```{code-cell} ipython3
mdm_is, mdm = dm1.createSubDM([0,1,2])
```

```{code-cell} ipython3
v1 = dm1.createLocalVec()
```

```{code-cell} ipython3
vsub = v1.getSubVector(mdm_is)
# v1.array[...] = oldlvec[...]
```

```{code-cell} ipython3
vsub.array[...] = oldlvec.array[...]
```

```{code-cell} ipython3
v1.restoreSubVector(mdm_is, vsub)
```

```{code-cell} ipython3
v1.array
```

```{code-cell} ipython3

len(mesh.vars)
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
