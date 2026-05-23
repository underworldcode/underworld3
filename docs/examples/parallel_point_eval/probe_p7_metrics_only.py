"""P7 SLCN advection metrics only — skip pyvista rendering for speed."""
import time
import numpy as np
import sympy
import underworld3 as uw

T_DEGREE = 7
QDEGREE = T_DEGREE

print(f"Building SphericalManifold(cellSize=0.075, qdegree={QDEGREE}) "
      f"for P{T_DEGREE}")
mesh = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.075, qdegree=QDEGREE)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=T_DEGREE)
coords = np.asarray(T.coords)
print(f"  n_DOFs = {coords.shape[0]}")

sigma = 0.3
T.data[:, 0] = np.exp(
    -((coords[:, 0] - 1.0)**2 + coords[:, 1]**2 + coords[:, 2]**2) / (2 * sigma**2)
)
T_initial = np.asarray(T.data[:, 0]).copy()

x_sym, y_sym, z_sym = mesh.X
V_sym = sympy.Matrix([[-y_sym, x_sym, sympy.sympify(0)]])

adv = uw.systems.AdvDiffusionSLCN(
    mesh, u_Field=T, V_fn=V_sym, order=1, monotone_mode="clamp",
)
adv.constitutive_model = uw.constitutive_models.DiffusionModel
adv.constitutive_model.Parameters.diffusivity = 1.0e-4
adv.f = sympy.Matrix.zeros(1, 1)

N_STEPS = 72
DT = 2 * np.pi / N_STEPS

t_solve_start = time.time()
for step in range(N_STEPS):
    t_step = time.time()
    adv.solve(timestep=DT, _evalf=True)
    if (step + 1) % 8 == 0:
        Tn = np.asarray(T.data[:, 0])
        print(f"  step {step + 1:3d}/{N_STEPS}  "
              f"T: min={float(Tn.min()):+.3f} max={float(Tn.max()):+.3f} "
              f"sum={float(Tn.sum()):.2f}  "
              f"step_time={time.time()-t_step:.1f}s",
              flush=True)

print(f"\nTotal solve time: {time.time() - t_solve_start:.1f} s")

T_final = np.asarray(T.data[:, 0])
err = T_final - T_initial
l2_err = float(np.sqrt((err * err).mean()))
l2_init = float(np.sqrt((T_initial * T_initial).mean()))

print(f"\n=== Final ===")
print(f"  max(T_initial) = {T_initial.max():.4f}")
print(f"  max(T_final)   = {T_final.max():.4f}   "
      f"(amplitude retained: {T_final.max()/T_initial.max()*100:.1f}%)")
print(f"  min(T_final)   = {T_final.min():+.6f}")
print(f"  L2 || T_final − T_initial ||  = {l2_err:.4e}")
print(f"  L2 || T_initial ||            = {l2_init:.4e}")
print(f"  Relative L2 error per rotation = {l2_err/l2_init*100:.2f}%")
