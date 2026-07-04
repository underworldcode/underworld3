"""Isolate: is the 2% TI Jacobian error POINTWISE (bad differentiation/compile)
or CONFIGURATION (nullspace / Amat-Pmat / Schur / monolithic-vs-fieldsplit /
element)? Minimal TI Stokes on a box, tilted director, eta1 != eta0, driven by
a body force. Run snes_test_jacobian across configurations.

If the ratio MOVES with config  -> configuration (the solver SEES a wrong J).
If the ratio is INVARIANT ~2%    -> pointwise (the assembled tangent is wrong).
A control with eta1=eta0 should be ~1e-8 in every config.
"""
import os, sys, math
import numpy as np, sympy, underworld3 as uw

CELL = float(os.environ.get("CELL", "0.5"))
VDEG = int(os.environ.get("VDEG", "2"))
NWEAK = os.environ.get("NOWEAK", "")          # set -> eta1=eta0 control


GEOM = os.environ.get("GEOM", "box")          # box | annulus
BC = os.environ.get("BC", "dirichlet")        # dirichlet | nitsche
VISC = os.environ.get("VISC", "const")        # const | field
FMGLV = int(os.environ.get("FMGLV", "0"))


def run(tag, configure):
    if GEOM == "annulus":
        ref = FMGLV if FMGLV > 0 else None
        mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5,
                                  cellSize=CELL, qdegree=2 * VDEG, refinement=ref)
    else:
        mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                                 cellSize=CELL, qdegree=2 * VDEG)
    v = uw.discretisation.MeshVariable("U", mesh, 2, degree=VDEG)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=VDEG - 1)
    X = mesh.CoordinateSystem.X
    st = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    st.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
    eta0 = 3.0
    if VISC == "field":
        # spatially-varying viscosity like the fault stack (exp of a coord field)
        eta0 = sympy.exp(1.5 * (X[0] + X[1]))
    st.constitutive_model.Parameters.shear_viscosity_0 = eta0
    st.constitutive_model.Parameters.shear_viscosity_1 = (eta0 if NWEAK else eta0 / 3.0)
    st.constitutive_model.Parameters.director = sympy.Matrix(
        [math.cos(0.6), math.sin(0.6)])
    st.tolerance = 1e-5
    st.penalty = 0.0
    if GEOM == "annulus":
        st.add_essential_bc((0.0, 0.0), "Lower")
        if BC == "nitsche":
            st.add_nitsche_bc(mesh.boundaries.Upper.name, gamma=10.0)
        elif BC == "penalty":
            unit_r0 = mesh.CoordinateSystem.unit_e_0
            st.add_natural_bc(1.0e4 * v.sym.dot(unit_r0) * unit_r0,
                              mesh.boundaries.Upper.name)
        else:
            st.add_essential_bc((0.0, 0.0), "Upper")
        unit_r = mesh.CoordinateSystem.unit_e_0
        st.bodyforce = 1.0e2 * (X[1]) * unit_r
    else:
        for b in ["Bottom", "Top", "Left", "Right"]:
            st.add_essential_bc((0.0, 0.0), b)
        st.bodyforce = sympy.Matrix([sympy.sin(3 * X[1]), sympy.cos(2 * X[0])])
    st.petsc_options["snes_test_jacobian"] = None
    st.petsc_options["snes_max_it"] = 1
    configure(st)
    try:
        st.solve(zero_init_guess=True)
    except Exception as e:
        print(f"[{tag}] solve note: {str(e)[:60]}", flush=True)
    return tag


# --- configurations -------------------------------------------------------
def cfg_default(st):
    pass                                          # UW3 default fieldsplit/schur

def cfg_nullspace_on(st):
    st.petsc_use_pressure_nullspace = True

def cfg_nullspace_off(st):
    st.petsc_use_pressure_nullspace = False

def cfg_monolithic(st):
    st.petsc_options["pc_type"] = "lu"
    st.petsc_options["ksp_type"] = "fgmres"
    st.petsc_options["pc_factor_mat_solver_type"] = "mumps"
    st.petsc_options["pc_fieldsplit_type"] = ""
    st.petsc_use_pressure_nullspace = True

CONFIGS = {
    "default": cfg_default,
    "nullspace_on": cfg_nullspace_on,
    "nullspace_off": cfg_nullspace_off,
    "monolithic_lu": cfg_monolithic,
}

which = sys.argv[1] if len(sys.argv) > 1 else "default"
print(f"### config={which}  VDEG={VDEG} CELL={CELL} "
      f"{'(eta1=eta0 CONTROL)' if NWEAK else '(eta1!=eta0 TI)'} ###", flush=True)
run(which, CONFIGS[which])
