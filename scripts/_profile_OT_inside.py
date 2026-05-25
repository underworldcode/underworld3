"""Profile the cost breakdown INSIDE one OT step."""
import os, sys, time
import numpy as np
import sympy
import underworld3 as uw

sys.path.insert(0, os.path.dirname(__file__))
from _test_metric_shapes_analytic_disp import analytic_rho


def build_uniform_mesh():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0),
        cellSize=0.04, qdegree=3)


m = build_uniform_mesh()
rho = analytic_rho(m)
print(f"mesh: {m.dm.getDepthStratum(0)[1]} verts, "
      f"{m.dm.getHeightStratum(0)[1]} cells")

# Set up the same machinery the OT step uses, then time
# individual pieces.
phi = uw.discretisation.MeshVariable(
    "prof_phi", m, vtype=uw.VarType.SCALAR, degree=2,
    continuous=True)
ps = uw.systems.Poisson(m, phi)
ps.constitutive_model = uw.constitutive_models.DiffusionModel
ps.constitutive_model.Parameters.diffusivity = rho
ps.constant_nullspace = True
vol_field = uw.discretisation.MeshVariable(
    "prof_vol", m, vtype=uw.VarType.SCALAR, degree=1,
    continuous=True)
gradphi = uw.discretisation.MeshVariable(
    "prof_gphi", m, vtype=uw.VarType.VECTOR, degree=1,
    continuous=True)
gproj = uw.systems.Vector_Projection(m, gradphi)
gproj.smoothing = 0.0
X = m.CoordinateSystem.X
gproj.uw_function = sympy.Matrix(
    [phi.sym[0].diff(X[i]) for i in range(2)]).T

K_val = 1.0
vol_field.data[:, 0] = 1.0
f_src = rho * sympy.log(rho * vol_field.sym[0] /
                         sympy.Float(K_val))
ps.f = sympy.Matrix([[f_src]])

# Warm-up — JIT compile + first solve setup
t0 = time.time()
ps.solve(zero_init_guess=True)
print(f"ps.solve warm-up: {time.time()-t0:6.2f} s")

t0 = time.time()
gproj.solve()
print(f"gproj.solve warm-up: {time.time()-t0:6.2f} s")

# Now time subsequent solves (no f change, no mesh change)
print("\n-- repeat solves WITHOUT changing ps.f or mesh:")
for k in range(3):
    t0 = time.time()
    ps.solve(zero_init_guess=True)
    print(f"  ps.solve #{k+2}: {time.time()-t0:6.2f} s")
    t0 = time.time()
    gproj.solve()
    print(f"  gproj.solve #{k+2}: {time.time()-t0:6.2f} s")

# Time solves WITH ps.f changed (new K_val each time)
print("\n-- repeat solves WITH ps.f reassigned each time:")
for k in range(3):
    K_val = 1.0 + 0.1 * k
    f_src = rho * sympy.log(rho * vol_field.sym[0] /
                             sympy.Float(K_val))
    t0 = time.time()
    ps.f = sympy.Matrix([[f_src]])
    print(f"  ps.f assign: {time.time()-t0:6.3f} s")
    t0 = time.time()
    ps.solve(zero_init_guess=True)
    print(f"  ps.solve after f change: {time.time()-t0:6.2f} s")

# Time solves WITH mesh deformed (re-assembly cost)
print("\n-- repeat solves AFTER _deform_mesh:")
for k in range(3):
    coords = np.asarray(m.X.coords)
    new = coords.copy()
    new[:] += 1e-5 * np.random.randn(*new.shape)
    t0 = time.time()
    m._deform_mesh(new)
    print(f"  _deform_mesh: {time.time()-t0:6.3f} s")
    t0 = time.time()
    ps.solve(zero_init_guess=True)
    print(f"  ps.solve after deform: {time.time()-t0:6.2f} s")
    t0 = time.time()
    gproj.solve()
    print(f"  gproj.solve after deform: {time.time()-t0:6.2f} s")
