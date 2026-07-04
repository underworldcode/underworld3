"""Does the block-constrained Stokes solver give a CLEAN Jacobian for the TI
(anisotropic) operator — i.e. does its multiplier free-slip avoid the Nitsche
#239 bug — and with only a moderate (viscosity-weighted) augmentation rather
than a stiff kfs=1e7 penalty?

Compares, via snes_test_jacobian, the same minimal annulus TI Stokes under:
  - Nitsche free-slip            (known bad: ~0.07)
  - penalty free-slip            (known good: ~1e-9, but stiff kfs)
  - CONSTRAINED multiplier slip  (Stokes_Constrained.add_constraint_bc)
Plus an isotropic control for the constrained case.
"""
import os, math
import numpy as np, sympy, underworld3 as uw

CELL = 0.5


def build(kind, noweak=False):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=CELL, qdegree=4)
    v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    X = mesh.CoordinateSystem.X
    unit_r = mesh.CoordinateSystem.unit_e_0
    if kind == "constrained":
        st = uw.systems.Stokes_Constrained(mesh, velocityField=v, pressureField=p)
    else:
        st = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    st.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
    st.constitutive_model.Parameters.shear_viscosity_0 = 3.0
    st.constitutive_model.Parameters.shear_viscosity_1 = (3.0 if noweak else 1.0)
    st.constitutive_model.Parameters.director = sympy.Matrix([math.cos(0.6), math.sin(0.6)])
    st.tolerance = 1e-6
    st.penalty = 0.0
    st.add_essential_bc((0.0, 0.0), "Lower")
    if kind == "nitsche":
        st.add_nitsche_bc(mesh.boundaries.Upper.name, gamma=10.0)
    elif kind == "penalty":
        st.add_natural_bc(1.0e4 * v.sym.dot(unit_r) * unit_r, mesh.boundaries.Upper.name)
    elif kind == "constrained":
        st.add_constraint_bc("Upper")        # multiplier free-slip, moderate augmentation
    st.bodyforce = 1.0e2 * X[1] * unit_r
    st.petsc_options["snes_test_jacobian"] = None
    st.petsc_options["snes_max_it"] = 1
    return st


for kind, noweak in [("nitsche", False), ("penalty", False),
                     ("constrained", False), ("constrained", True)]:
    tag = f"{kind}{' (eta1=eta0 control)' if noweak else ' (TI)'}"
    print(f"\n######## {tag} ########", flush=True)
    try:
        st = build(kind, noweak)
        st.solve(zero_init_guess=True)
    except Exception as e:
        print(f"  note: {str(e)[:90]}", flush=True)
