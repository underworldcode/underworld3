"""Compile a minimal TI Stokes (tilted director, eta1 != eta0) with JIT verbose,
capture the generated C source dir, and print the residual (f1) and Jacobian
(g3) function bodies so we can compare the director-term value generation.
"""
import os, glob, re
import sympy, math
import underworld3 as uw

mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                         cellSize=0.6, qdegree=2)
v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 3.0
stokes.constitutive_model.Parameters.shear_viscosity_1 = 1.0
stokes.constitutive_model.Parameters.director = sympy.Matrix(
    [math.cos(0.6), math.sin(0.6)])
stokes.add_essential_bc((0.0, 0.0), "Bottom")
stokes.add_essential_bc((0.0, 0.0), "Top")
stokes.add_essential_bc((0.0, 0.0), "Left")
stokes.add_essential_bc((0.0, 0.0), "Right")
stokes.bodyforce = sympy.Matrix([0.0, -1.0])

stokes.petsc_options["snes_max_it"] = 1
stokes._setup_pointwise_functions(verbose=True)   # triggers JIT, prints tmpdir
# also force a solve to be safe
try:
    stokes.solve(verbose=True)
except Exception as e:
    print("solve note:", str(e)[:80])

# find the most recent compiled module dir
cands = sorted(glob.glob("/tmp/**/*.c", recursive=True) +
               glob.glob(os.path.expanduser("~/**/cython_*/*.c"), recursive=True),
               key=lambda f: os.path.getmtime(f))
print("\n=== recent generated .c files ===")
for f in cands[-6:]:
    print(f"  {f}  ({os.path.getsize(f)} bytes)")
