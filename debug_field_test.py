#!/usr/bin/env python3

import underworld3 as uw
import numpy as np
from underworld3.meshing import UnstructuredSimplexBox

# Create a simple test case to debug the field ID issue
print("Creating mesh...")
mesh = UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2
)

print("Creating first variable...")
try:
    u = uw.discretisation.MeshVariable("u", mesh, 2, vtype=uw.VarType.VECTOR, degree=2)
    print(f"✓ Variable u created successfully with field_id={u.field_id}")
except Exception as e:
    print(f"✗ Failed to create variable u: {e}")

print("Creating second variable...")
try:
    p = uw.discretisation.MeshVariable("p", mesh, 1, vtype=uw.VarType.SCALAR, degree=1)
    print(f"✓ Variable p created successfully with field_id={p.field_id}")
except Exception as e:
    print(f"✗ Failed to create variable p: {e}")

print("Creating third variable...")
try:
    s = uw.discretisation.MeshVariable("s", mesh, 1, vtype=uw.VarType.SCALAR, degree=1)
    print(f"✓ Variable s created successfully with field_id={s.field_id}")
except Exception as e:
    print(f"✗ Failed to create variable s: {e}")

print("Testing array access...")
try:
    print("Accessing s.array...")
    s_array = s.array
    print("✓ s.array access successful")
except Exception as e:
    print(f"✗ Failed to access s.array: {e}")
    import traceback
    traceback.print_exc()