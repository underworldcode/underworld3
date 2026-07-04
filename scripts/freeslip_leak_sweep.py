"""Isolate the outer free-slip leak: ONE Stokes solve on a fixed Ra1e6 buoyancy,
sweeping the free-slip enforcement, measuring v·n leak on the Upper boundary.

If higher gamma / penalty / constrained drives the leak -> 0, it's a TUNING issue
(gamma too weak for the stress). If even strong enforcement leaks, the BC
machinery has regressed.
"""
import numpy as np, sympy, underworld3 as uw

import os
Ra = 1.0e6
theta_FK = np.log(float(os.environ.get("DETA", "1e3")))   # match the failing run
res = 24

def build():
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5,
                              cellSize=1.0/res, qdegree=3)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3, continuous=True)
    V = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2, continuous=True)
    P = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True)
    X = mesh.CoordinateSystem.X
    r = sympy.sqrt(X[0]**2 + X[1]**2)
    th = sympy.atan2(X[1], X[0])
    Tc = sympy.log(r/1.0)/sympy.log(0.5/1.0)
    init = 0.05*sympy.sin(5*th)*sympy.sin(np.pi*(r-0.5)/0.5) + Tc
    T.data[...] = np.asarray(uw.function.evaluate(init, T.coords)).reshape(-1,1)
    return mesh, T, V, P

def leak(mesh, V):
    C = np.asarray(V.coords)[:, :2]; r = np.sqrt((C**2).sum(1))
    Vd = np.asarray(V.data)[:, :2]; vmag = np.sqrt((Vd**2).sum(1))
    nhat = C/r[:,None]; vn = (Vd*nhat).sum(1)
    up = r > r.max()-1e-3
    return np.abs(vn[up]).max(), vmag.max(), np.abs(vn[up]).max()/max(vmag.max(),1e-30)

def run(label, freeslip, gamma=10.0, kfs=1e6):
    mesh, T, V, P = build()
    X = mesh.CoordinateSystem.X
    r = sympy.sqrt(X[0]**2+X[1]**2)
    unit_r = mesh.CoordinateSystem.unit_e_0
    stokes = uw.systems.Stokes(mesh, velocityField=V, pressureField=P)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = sympy.exp(theta_FK*(1-T.sym[0]))
    stokes.tolerance = 1e-6
    stokes.penalty = 0.0
    stokes.petsc_options["snes_type"] = "ksponly"
    stokes.add_essential_bc((0.0,0.0), mesh.boundaries.Lower.name)
    if freeslip == "nitsche":
        stokes.add_nitsche_bc(mesh.boundaries.Upper.name, gamma=gamma)
    else:
        stokes.add_natural_bc(kfs*V.sym.dot(unit_r)*unit_r, mesh.boundaries.Upper.name)
    Tc = sympy.log(r/1.0)/sympy.log(0.5/1.0)
    stokes.bodyforce = Ra*(T.sym[0]-Tc)*unit_r
    stokes.solve(zero_init_guess=True)
    vnmax, vmax, ratio = leak(mesh, V)
    print(f"  {label:28s}: |v|max={vmax:9.2e}  |v·n|max_upper={vnmax:9.2e}  leak={ratio:7.2%}")

print(f"=== free-slip leak sweep, Ra={Ra:g}, res{res} (one Stokes solve each) ===")
run("nitsche g=10", "nitsche", gamma=10)
run("nitsche g=100", "nitsche", gamma=100)
run("nitsche g=1000", "nitsche", gamma=1000)
run("nitsche g=1e4", "nitsche", gamma=1e4)
run("penalty kfs=1e6", "penalty", kfs=1e6)
run("penalty kfs=1e8", "penalty", kfs=1e8)
