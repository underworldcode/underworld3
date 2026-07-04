"""Is Stokes_Constrained actually parallel-safe? Solve the SAME constrained
free-slip annulus problem and report partition-INDEPENDENT diagnostics
(global |v|max via allreduce, and the L2 velocity norm via a mesh integral).
Run at np1 and np2/np4 and compare — if they agree, the rank-local section
reduction is parallel-safe and the serial guard was over-conservative.

  UW_CONSTRAINED_ALLOW_PARALLEL=1 mpirun -np 2 python constrained_parallel_test.py [ti|iso]
"""
import os, sys, math
import numpy as np, sympy, underworld3 as uw

os.environ.setdefault("UW_CONSTRAINED_ALLOW_PARALLEL", "1")
kind = sys.argv[1] if len(sys.argv) > 1 else "iso"
cell = float(sys.argv[2]) if len(sys.argv) > 2 else 0.12

mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=cell, qdegree=4)
v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
X = mesh.CoordinateSystem.X
unit_r = mesh.CoordinateSystem.unit_e_0

st = uw.systems.Stokes_Constrained(mesh, velocityField=v, pressureField=p)
if kind == "ti":
    st.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
    st.constitutive_model.Parameters.shear_viscosity_0 = 3.0
    st.constitutive_model.Parameters.shear_viscosity_1 = 1.0
    st.constitutive_model.Parameters.director = sympy.Matrix([math.cos(0.6), math.sin(0.6)])
else:
    st.constitutive_model = uw.constitutive_models.ViscousFlowModel
    st.constitutive_model.Parameters.shear_viscosity_0 = 1.0
st.tolerance = 1e-8
st.add_essential_bc((0.0, 0.0), "Lower")
h = st.add_constraint_bc("Upper")
st.bodyforce = 1.0e2 * sympy.sin(3 * sympy.atan2(X[1], X[0])) * unit_r
st.solve(zero_init_guess=True)

# Partition-independent diagnostics — uw/petsc parallel-safe integrals ONLY
# (no direct mpi4py reductions).
L2 = float(np.sqrt(uw.maths.Integral(mesh, v.sym.dot(v.sym)).evaluate()))
# free-slip quality: ∫(v·n̂)² on Upper (should be ~0)
vn2 = float(uw.maths.BdIntegral(mesh=mesh, fn=v.sym.dot(unit_r) ** 2,
                                boundary="Upper").evaluate())
# topography boundary trace, and a control: the boundary length itself
# (isolates whether any parallel drift is in BdIntegral or in the h recovery).
topo_bd = float(np.sqrt(uw.maths.BdIntegral(mesh=mesh, fn=h.sym[0] ** 2,
                                            boundary="Upper").evaluate()))
blen = float(uw.maths.BdIntegral(mesh=mesh, fn=sympy.Integer(1),
                                 boundary="Upper").evaluate())
# h is determined only up to the [p,h] gauge constant -> strip its boundary mean.
h_mean = float(uw.maths.BdIntegral(mesh=mesh, fn=h.sym[0],
                                   boundary="Upper").evaluate()) / blen
topo_bd_c = float(np.sqrt(uw.maths.BdIntegral(
    mesh=mesh, fn=(h.sym[0] - h_mean) ** 2, boundary="Upper").evaluate()))

uw.pprint(f"[{kind}] np={uw.mpi.size} cell={cell}  L2(v)={L2:.8f}  "
          f"topo_BD={topo_bd:.8f}  h_mean={h_mean:.6f}  "
          f"topo_BD_centred={topo_bd_c:.8f}  reason={st.snes.getConvergedReason()}")
