import sympy, underworld3 as uw
uw.reset_default_model()
model = uw.get_default_model()
model.record_every = 1

mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.,0.), maxCoords=(1.,1.), cellSize=0.4, qdegree=2)
T = uw.discretisation.MeshVariable("T_sb", mesh, 1, degree=2)

kappa = uw.expression(r"\kappa", 1.0, "diffusivity")
model.tracker.time, model.tracker.step = 0.0, 0

T.array[...] = 1.0
with model.step(0.1):
    pass
print(f"after step 1:  kappa={float(kappa.sym)}  T={float(T.array[0,0,0])}")

# a run that ramps a parameter -- utterly ordinary
kappa.sym = sympy.Float(7.0)
T.array[...] = 2.0
with model.step(0.1):
    pass
print(f"after step 2:  kappa={float(kappa.sym)}  T={float(T.array[0,0,0])}")

model.rewind(1)
print(f"after rewind:  kappa={float(kappa.sym)}  T={float(T.array[0,0,0])}")
print()
print("T restored to its step-2-start value 2.0 ? ", abs(float(T.array[0,0,0]) - 2.0) < 1e-12)
print("kappa restored to its step-2-start value 7.0 ? ", abs(float(kappa.sym) - 7.0) < 1e-12)

model.rewind(1)
print()
print(f"after rewind to step 1: kappa={float(kappa.sym)} (should be 1.0)  T={float(T.array[0,0,0])} (should be 1.0)")
print("BUG:" if abs(float(kappa.sym) - 1.0) > 1e-12 else "ok:",
      "kappa is", float(kappa.sym), "expected 1.0")
