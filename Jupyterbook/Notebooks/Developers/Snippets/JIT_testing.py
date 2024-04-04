# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Some simple tests for the JIT compiler for pointwise functions

# %%
import pytest
import sympy
import underworld3 as uw
import numpy as np
import os, shutil

import numpy as np
import sympy

from   underworld3.utilities._jitextension import getext


# %%
uw.__file__

# %%
minX = -1.0
maxX = 1.0
minY = -1.0
maxY = 1.0

resX = 4
resY = 4


mesh = uw.meshing.StructuredQuadBox(
        elementRes=(resX, resY), 
        minCoords=(minX, minY), 
        maxCoords=(maxX, maxY), qdegree=3)

x,y = mesh.X

v = uw.discretisation.MeshVariable(
    "v",
    mesh,
    vtype=uw.VarType.VECTOR,
    degree=2,
    varsymbol=r"\mathbf{v}",
)

w = uw.discretisation.MeshVariable(
    "w",
    mesh,
    vtype=uw.VarType.SCALAR,
    degree=1,
    varsymbol=r"\mathbf{w}",
)

# %%
# clear any prior artifacts

shutil.rmtree("/tmp/fn_ptr_ext_TEST_0", ignore_errors=True)
shutil.rmtree("/tmp/fn_ptr_ext_TEST_1", ignore_errors=True)
shutil.rmtree("/tmp/fn_ptr_ext_TEST_2", ignore_errors=True)


# %%
res_fn = sympy.ImmutableDenseMatrix([sympy.sympify(1), sympy.sympify(2)])
jac_fn = sympy.ImmutableDenseMatrix([sympy.sympify(1), sympy.sympify(2)])
bc_fn = sympy.ImmutableDenseMatrix([sympy.sympify(1), sympy.sympify(2)])
bd_res_fn = sympy.ImmutableDenseMatrix([sympy.sympify(1), sympy.sympify(2)])
bd_jac_fn = sympy.ImmutableDenseMatrix([sympy.sympify(1), sympy.sympify(2)])

with uw.utilities.CaptureStdout(split=True) as captured_setup_solver:
    compiled_extns, dictionaries = getext(mesh,
                                          [res_fn, res_fn], 
                                          [jac_fn], 
                                          [bc_fn], 
                                          [bd_res_fn], 
                                          [bd_jac_fn], 
                                          mesh.vars.values(), 
                                          verbose=True,
                                          debug=True,
                                          debug_name="TEST_0",
                                          cache=False)


assert os.path.exists(f"/tmp/fn_ptr_ext_TEST_0")
assert os.path.exists(f"/tmp/fn_ptr_ext_TEST_0/cy_ext.h")
assert "Processing JIT    5 / Matrix([[1], [2]])" in captured_setup_solver



# %%
res_fn = sympy.ImmutableDenseMatrix([x,y])
jac_fn = sympy.ImmutableDenseMatrix([x**2, y**2])
bc_fn = sympy.ImmutableDenseMatrix([sympy.sin(x), sympy.cos(y)])
bd_res_fn = sympy.ImmutableDenseMatrix([sympy.log(x), sympy.exp(y)])
bd_jac_fn = sympy.ImmutableDenseMatrix([sympy.diff(sympy.log(x)), sympy.diff(sympy.exp(y*x),y)])

with uw.utilities.CaptureStdout(split=True) as captured_setup_solver:
    compiled_extns, dictionaries = getext(mesh,
                                          [res_fn, res_fn], 
                                          [jac_fn], 
                                          [bc_fn], 
                                          [bd_res_fn], 
                                          [bd_jac_fn], 
                                          mesh.vars.values(), 
                                          verbose=True,
                                          debug=True,
                                          debug_name="TEST_1",
                                          cache=False)


assert os.path.exists(f"/tmp/fn_ptr_ext_TEST_1")
assert os.path.exists(f"/tmp/fn_ptr_ext_TEST_1/cy_ext.h")
assert "Processing JIT    5 / Matrix([[1/N.x], [N.x*exp(N.x*N.y)]])" in captured_setup_solver


# %%
res_fn = sympy.ImmutableDenseMatrix([v.sym[0],w.sym])
jac_fn = sympy.ImmutableDenseMatrix([x*v.sym[0], y**2])
bc_fn = sympy.ImmutableDenseMatrix([v.sym[1] * sympy.sin(x), sympy.cos(y)])
bd_res_fn = sympy.ImmutableDenseMatrix([sympy.log(v.sym[0]), sympy.exp(w.sym)])
bd_jac_fn = sympy.ImmutableDenseMatrix([sympy.diff(sympy.log(v.sym[0]),y), sympy.diff(sympy.exp(y*x),y)])

with uw.utilities.CaptureStdout(split=True) as captured_setup_solver:
    compiled_extns, dictionaries = getext(mesh,
                                          [res_fn, res_fn], 
                                          [jac_fn], 
                                          [bc_fn], 
                                          [bd_res_fn], 
                                          [bd_jac_fn], 
                                          mesh.vars.values(), 
                                          verbose=True,
                                          debug=True,
                                          debug_name="TEST_2",
                                          cache=False)


assert os.path.exists(f"/tmp/fn_ptr_ext_TEST_2")
assert os.path.exists(f"/tmp/fn_ptr_ext_TEST_2/cy_ext.h")
assert "Processing JIT    5 / Matrix([[\mathbf{v}_{ 0,1}(N.x, N.y)/\mathbf{v}_{ 0 }(N.x, N.y)], [N.x*exp(N.x*N.y)]])" in captured_setup_solver


# %%
# clear up the directories we made

shutil.rmtree("/tmp/fn_ptr_ext_TEST_0", ignore_errors=True)
shutil.rmtree("/tmp/fn_ptr_ext_TEST_1", ignore_errors=True)
shutil.rmtree("/tmp/fn_ptr_ext_TEST_2", ignore_errors=True)

