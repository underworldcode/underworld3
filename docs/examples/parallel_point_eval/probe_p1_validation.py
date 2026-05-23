"""Validate the P2-edge-midpoint hypothesis with P1.

Same SLCN setup as the animation but T at degree=1 (vertex DOFs only,
all on r=1). If the antipode anomaly disappears, it confirms the
edge-midpoint chord-interior DOFs are the cause.
"""
import numpy as np
import sympy
import underworld3 as uw

DEGREE = 1  # change to 2 to compare

print(f"=== SLCN advection with T degree={DEGREE} ===")

mesh = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.15)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=DEGREE)
coords = np.asarray(T.coords)

# Classify DOFs by r.
r = np.linalg.norm(coords, axis=1)
on_sphere = r > 0.9999
off_sphere = ~on_sphere
print(f"  Total DOFs: {coords.shape[0]}")
print(f"    on-sphere DOFs (r > 0.9999): {on_sphere.sum()}")
print(f"    off-sphere DOFs:             {off_sphere.sum()}  "
      f"(median r = {np.median(r[off_sphere]):.4f})"
      if off_sphere.any() else "    off-sphere DOFs: 0")

# Initial Gaussian.
sigma = 0.3
T.data[:, 0] = np.exp(
    -((coords[:, 0] - 1.0)**2 + coords[:, 1]**2 + coords[:, 2]**2) / (2 * sigma**2)
)
T_init = np.asarray(T.data[:, 0]).copy()

x_sym, y_sym, z_sym = mesh.X
V_sym = sympy.Matrix([[-y_sym, x_sym, sympy.sympify(0)]])

adv = uw.systems.AdvDiffusionSLCN(mesh, u_Field=T, V_fn=V_sym, order=1)
adv.constitutive_model = uw.constitutive_models.DiffusionModel
adv.constitutive_model.Parameters.diffusivity = 1.0e-4
adv.f = sympy.Matrix.zeros(1, 1)

dt = 2 * np.pi / 72
adv.solve(timestep=dt)
T_after = np.asarray(T.data[:, 0])

# Analytic expected: rotated Gaussian.
end_x = coords[:, 0] + coords[:, 1] * dt
end_y = coords[:, 1] - coords[:, 0] * dt
end_z = coords[:, 2]
T_expected = np.exp(
    -((end_x - 1.0)**2 + end_y**2 + end_z**2) / (2 * sigma**2)
)
abs_diff = np.abs(T_after - T_expected)

print(f"\nAfter one SLCN step (dt={dt:.4f}, {np.degrees(dt):.1f}°):")
print(f"  T_after min={T_after.min():+.4f}  max={T_after.max():+.4f}  "
      f"sum={T_after.sum():.4f}")
print(f"  L2 || T_after - T_expected || / L2 || T_expected || = "
      f"{np.linalg.norm(T_after - T_expected) / np.linalg.norm(T_expected):.4e}")

# Show worst anomalies.
worst = np.argsort(abs_diff)[::-1][:8]
print(f"\n  Top 8 worst-anomaly DOFs:")
print(f"    {'idx':>5} {'r':>7} {'T_init':>9} {'T_after':>9} "
      f"{'T_exp':>9} {'|diff|':>9} {'coord':>22}")
for idx in worst:
    print(
        f"    {int(idx):>5} {r[idx]:>7.4f} "
        f"{T_init[idx]:>+9.4f} {T_after[idx]:>+9.4f} "
        f"{T_expected[idx]:>+9.4f} {abs_diff[idx]:>9.4e} "
        f"[{coords[idx, 0]:+.2f}, {coords[idx, 1]:+.2f}, {coords[idx, 2]:+.2f}]"
    )

# Also: count DOFs with significantly wrong values (where |diff| > 0.1).
n_wild = int((abs_diff > 0.1).sum())
print(f"\n  DOFs with |T_after - T_expected| > 0.1: {n_wild}")
