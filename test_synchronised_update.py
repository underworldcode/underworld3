#!/usr/bin/env python3

import underworld3 as uw
import numpy as np
from underworld3.meshing import UnstructuredSimplexBox

# Create a simple test case to test synchronised_array_update
print("Creating mesh...")
mesh = UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2
)

print("Creating variables...")
u = uw.discretisation.MeshVariable("u", mesh, 2, vtype=uw.VarType.VECTOR, degree=2)
p = uw.discretisation.MeshVariable("p", mesh, 1, vtype=uw.VarType.SCALAR, degree=1)
s = uw.discretisation.MeshVariable("s", mesh, 1, vtype=uw.VarType.SCALAR, degree=1)

print("Testing synchronised_array_update...")
try:
    with uw.synchronised_array_update("test multi-variable update"):
        u.array[...] = np.random.random(u.array.shape)
        p.array[...] = np.random.random(p.array.shape) 
        s.array[...] = np.random.random(s.array.shape)
    print("✓ synchronised_array_update works successfully")
except Exception as e:
    print(f"✗ Failed to use synchronised_array_update: {e}")
    import traceback
    traceback.print_exc()

print("Testing single array access...")
try:
    u.array[...] = np.ones(u.array.shape)
    print("✓ Direct array access works successfully")
except Exception as e:
    print(f"✗ Failed direct array access: {e}")
    import traceback
    traceback.print_exc()