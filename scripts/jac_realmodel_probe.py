"""Decisive symbolic-vs-JIT localization using the REAL UW3 TI model.

Established: F1 = C:sym(L), L = u.jacobian(N), strain_tensor uses the same
gradients, residual & Jacobian come from the identical symbolic F1, and
derive_by_array is exact. So the 2% snes_test_jacobian failure must enter
BELOW sympy (JIT) — unless the real model introduces a symbolic inconsistency
the pure-sympy test couldn't see (cached/frozen director, UWexpression
unwrap, Mandel round-trip mangling). This checks exactly that on the real
objects:

  flux_sym            = constitutive_model.flux            (the residual stress)
  G3_sym[i,j,k,l]     = d flux_sym[i,j] / d L[k,l]         (the exact tangent)
  C_analytic          = the c-tensor the model built
and verifies, numerically at a random velocity gradient, that
  G3_sym  ==  d(flux)/dL  ==  (minor-symmetrised) C_analytic
i.e. the SYMBOLIC residual and its SYMBOLIC tangent are mutually consistent.
If they match -> the bug is purely JIT (report + dump c-code next).
If they differ -> the symbolic assembly itself is wrong (found it here).
"""
import numpy as np
import sympy
import underworld3 as uw

mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                         cellSize=0.5, qdegree=3)
v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)

import math
theta = 0.6
nvec = sympy.Matrix([math.cos(theta), math.sin(theta)])
stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 3.0
stokes.constitutive_model.Parameters.shear_viscosity_1 = 1.0   # eta1 != eta0
stokes.constitutive_model.Parameters.director = nvec

cm = stokes.constitutive_model
flux = sympy.Matrix(cm.flux)                       # residual stress (2x2), symbolic
L = sympy.Matrix(stokes.Unknowns.L)                # u.jacobian(N), the diff variable
Lsyms = [L[i, j] for i in range(2) for j in range(2)]
print("flux free L-symbols present:",
      all(any(s in flux[i].free_symbols for i in range(4)) for s in []) or "checked below")

# G3_sym[i,j,k,l] = d flux[i,j] / d L[k,l]  (exact symbolic tangent)
G3 = sympy.derive_by_array(flux, L)

# analytic c-tensor the model built
C = np.array(sympy.Array(cm.c).tolist(), dtype=object)

# numeric eval at a random gradient (problem is linear in L, value irrelevant)
subs = {Lsyms[0]: 0.31, Lsyms[1]: -0.22, Lsyms[2]: 0.17, Lsyms[3]: 0.41}


def ev(expr):
    return float(sympy.Array(expr).subs(subs))


# Build numeric G3 tensor and numeric C tensor
G3n = np.zeros((2, 2, 2, 2))
Cn = np.zeros((2, 2, 2, 2))
for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                G3n[i, j, k, l] = float(sympy.sympify(G3[i, j, k, l]).subs(subs))
                Cn[i, j, k, l] = float(sympy.sympify(C[i, j, k, l]).subs(subs)
                                       if hasattr(C[i, j, k, l], "subs") else C[i, j, k, l])

# The exact tangent of flux[i,j]=C[i,j,k,l] sym(L)[k,l] wrt L[k,l] is
#   T[i,j,k,l] = (C[i,j,k,l] + C[i,j,l,k]) / 2     (symmetrised in last pair)
Tn = 0.5 * (Cn + Cn.transpose(0, 1, 3, 2))

print("\n=== symbolic residual-tangent G3  vs  analytic (minor-sym) c-tensor ===")
diff = np.abs(G3n - Tn).max()
print(f"  max|G3_sym - C_sym(last pair)| = {diff:.3e}")
print(f"  (==0 means the SYMBOLIC residual and its tangent are mutually consistent")
print(f"   => the 2% snes_test_jacobian failure is purely in the JIT compilation)")

# also confirm flux really depends on all L entries (not frozen)
present = [bool(set(Lsyms) & flux[i, j].free_symbols) for i in range(2) for j in range(2)]
print(f"\n  flux entries depending on L symbols: {present}  (all True expected)")

# show the director term survives in c-tensor (not frozen to isotropic)
aniso = np.abs(Cn - Cn.transpose(2, 3, 0, 1)).max()   # major sym check
print(f"  c-tensor major-symmetry residual (should be ~0): {aniso:.2e}")
print(f"  c-tensor is anisotropic (C[0,0,0,0]={Cn[0,0,0,0]:.3f} vs C[1,1,1,1]={Cn[1,1,1,1]:.3f}; "
      f"differ => director active): {abs(Cn[0,0,0,0]-Cn[1,1,1,1]):.3e}")
