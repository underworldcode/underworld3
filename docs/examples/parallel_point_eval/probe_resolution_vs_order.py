"""SLCN advection on SphericalManifold: P1 at various cellSizes vs Pk
at fixed cellSize — separate the high-order convergence story from the
effective-resolution (DOF-count) story.

Reads degree + cellSize from argv:
    python probe_resolution_vs_order.py <degree> <cellSize>
"""
import sys
import time
import numpy as np
import sympy
import underworld3 as uw


def main():
    if len(sys.argv) not in (3, 4):
        print("usage: probe_resolution_vs_order.py <degree> <cellSize> [qdegree]")
        sys.exit(2)
    T_DEGREE = int(sys.argv[1])
    CELL_SIZE = float(sys.argv[2])
    QDEGREE = int(sys.argv[3]) if len(sys.argv) == 4 else T_DEGREE

    print(f"=== P{T_DEGREE}, cellSize={CELL_SIZE}, qdegree={QDEGREE} ===")
    mesh = uw.meshing.SphericalManifold(
        radius=1.0, cellSize=CELL_SIZE, qdegree=QDEGREE,
    )
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

    t_start = time.time()
    for step in range(N_STEPS):
        adv.solve(timestep=DT, _evalf=True)
    t_total = time.time() - t_start

    T_final = np.asarray(T.data[:, 0])
    err = T_final - T_initial
    l2_err = float(np.sqrt((err * err).mean()))
    l2_init = float(np.sqrt((T_initial * T_initial).mean()))

    print(f"  Total solve time: {t_total:.1f} s")
    print(f"  amplitude retained = {T_final.max()/T_initial.max()*100:.1f}%")
    print(f"  Rel L2 / rotation = {l2_err/l2_init*100:.2f}%")
    print(f"  min(T_final) = {T_final.min():+.4f}")


if __name__ == "__main__":
    main()
