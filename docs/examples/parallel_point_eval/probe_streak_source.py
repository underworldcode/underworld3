"""Find the DOF source of the P1 advection streak.

Run a few SLCN steps with P1, then locate the DOFs where T differs
most from the analytic rotated Gaussian. Report their coords and the
local mesh neighbourhood (which vertices share a cell with them).
"""
import numpy as np
import sympy
import underworld3 as uw

mesh = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.075)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
coords = np.asarray(T.coords)
n_dof = coords.shape[0]
r = np.linalg.norm(coords, axis=1)

# Initial Gaussian centred at (1, 0, 0).
sigma = 0.3
def gaussian_at(c):
    return np.exp(-((c[..., 0] - 1.0)**2 + c[..., 1]**2 + c[..., 2]**2) / (2 * sigma**2))

T.data[:, 0] = gaussian_at(coords)

x_sym, y_sym, z_sym = mesh.X
V_sym = sympy.Matrix([[-y_sym, x_sym, sympy.sympify(0)]])

adv = uw.systems.AdvDiffusionSLCN(mesh, u_Field=T, V_fn=V_sym, order=1)
adv.constitutive_model = uw.constitutive_models.DiffusionModel
adv.constitutive_model.Parameters.diffusivity = 1.0e-4
adv.f = sympy.Matrix.zeros(1, 1)

# Take 6 steps to match frame 0006.
N_STEPS = 6
dt = 2 * np.pi / 72  # 5° per step
for step in range(N_STEPS):
    adv.solve(timestep=dt)

T_after = np.asarray(T.data[:, 0])

# Analytic: Gaussian rotated by phi = N_STEPS * dt about z.
phi = N_STEPS * dt
# Rotated centre: (cos(phi), sin(phi), 0)
cos_phi = np.cos(phi)
sin_phi = np.sin(phi)
# Distance of each DOF from the rotated centre.
expected_centre = np.array([cos_phi, sin_phi, 0.0])
distance_to_centre = np.linalg.norm(coords - expected_centre, axis=1)
T_expected = np.exp(-distance_to_centre**2 / (2 * sigma**2))

abs_diff = np.abs(T_after - T_expected)

print(f"=== After {N_STEPS} SLCN steps, phi={np.degrees(phi):.1f}° ===")
print(f"  T_after: min={T_after.min():+.4f}  max={T_after.max():+.4f}  sum={T_after.sum():.2f}")
print(f"  T_expected: max={T_expected.max():+.4f}  (centred at {expected_centre})")
print(f"  L2 || T_after - T_expected || = {np.linalg.norm(T_after - T_expected):.4e}")
print()

# Worst-deviation DOFs.
worst = np.argsort(abs_diff)[::-1][:15]
print(f"  Top 15 worst-deviation DOFs (by |T_after - T_expected|):")
print(f"    {'idx':>5} {'T_after':>9} {'T_exp':>9} {'|diff|':>9} {'arc_to_blob':>12} {'coord':>24}")
for idx in worst:
    arc = float(np.degrees(np.arccos(np.clip(np.dot(coords[idx], expected_centre), -1, 1))))
    print(
        f"    {int(idx):>5} {T_after[idx]:>+9.4f} {T_expected[idx]:>+9.4f} "
        f"{abs_diff[idx]:>9.4e} {arc:>10.1f}°  "
        f"[{coords[idx, 0]:+.3f}, {coords[idx, 1]:+.3f}, {coords[idx, 2]:+.3f}]"
    )

# Filter to DOFs where |diff| > FE-interp tolerance AND the analytic
# value is large (so we're seeing the streak through the blob, not
# random small-magnitude noise far from the blob).
in_blob_region = T_expected > 0.1
big_in_blob = np.where((abs_diff > 0.05) & in_blob_region)[0]
print(f"\n  DOFs in blob region (T_exp > 0.1) with |diff| > 0.05: {len(big_in_blob)}")
if len(big_in_blob) > 0:
    # Cluster their coordinates to see if they form a streak.
    # Just look at their arc distance from the blob centre.
    arcs = []
    for idx in big_in_blob:
        arc = float(np.degrees(np.arccos(np.clip(np.dot(coords[idx], expected_centre), -1, 1))))
        arcs.append((idx, arc, coords[idx]))
    arcs.sort(key=lambda x: x[1])
    print(f"  Sorted by arc-distance to blob centre:")
    print(f"    {'idx':>5} {'arc':>7} {'coord':>24} {'T_after':>9} {'T_exp':>9}")
    for idx, arc, c in arcs[:20]:
        print(f"    {idx:>5} {arc:>6.1f}°  "
              f"[{c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f}]  "
              f"{T_after[idx]:>+9.4f} {T_expected[idx]:>+9.4f}")

# Save T for visual cross-reference.
out_path = "/tmp/T_streak.npz"
np.savez(out_path, coords=coords, T_after=T_after, T_expected=T_expected, abs_diff=abs_diff)
print(f"\n  Saved T snapshot to {out_path}")
