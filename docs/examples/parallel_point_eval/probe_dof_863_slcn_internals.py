"""Trace DOF 863 through the SLCN history-initialisation path."""
import numpy as np
import sympy
import underworld3 as uw

mesh = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.15)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
coords = np.asarray(T.coords)

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

# Manual one-shot through update_pre_solve so we can inspect psi_star
# values before / after, without the actual solve.
DuDt = adv.DuDt  # The history-of-u operator
print("=== Inspecting DuDt ===")
print(f"  type: {type(DuDt).__name__}")
print(f"  psi_star levels: {len(DuDt.psi_star)}")

# Snapshot DOF 863's T value before history runs.
print(f"\nDOF 863 BEFORE history:")
print(f"  T.data[863]              = {T.data[863, 0]:+.6f}")
print(f"  global_evaluate(T.sym, [c]) = "
      f"{np.asarray(uw.function.global_evaluate(T.sym, coords[863:864])).flatten()[0]:+.6f}")

# Now actually solve one SLCN step and capture state at multiple points.
dt = 2 * np.pi / 72
adv.solve(timestep=dt)

# After solve, T should reflect the SLCN-advected field.
T_after = np.asarray(T.data[:, 0])
print(f"\n--- After adv.solve(dt={dt:.4f}) ---")
print(f"  T.data[863] AFTER solve = {T_after[863]:+.6f}")

# Inspect ALL state used by the SLCN solver at DOF 863.
DuDt.initialise_history()  # idempotent — but make sure psi_star is populated

# Now psi_star[0] should hold the "T trace-back" values.
psi0 = np.asarray(DuDt.psi_star[0].data)
psi0_coords = np.asarray(DuDt.psi_star[0].coords)

# Look up DOF 863 in psi_star[0]'s coords. Note psi_star may have its
# own DOF ordering, so find the closest matching coord.
target = coords[863]
distances = np.linalg.norm(psi0_coords - target, axis=1)
match_idx = int(np.argmin(distances))
match_dist = distances[match_idx]

print(f"\nAFTER initialise_history:")
print(f"  psi_star[0].coords matching T's DOF 863 coord: idx={match_idx}, dist={match_dist:.4e}")
print(f"  psi_star[0].data[match]  = {psi0.flatten()[match_idx] if psi0.shape[1] == 1 else psi0[match_idx, 0]:+.6f}")

# Also: top anomalous psi_star[0] DOFs at antipode-ish coords.
psi_flat = psi0.flatten() if psi0.shape[1] == 1 else psi0[:, 0]
# Identify DOFs at antipode-ish coords (x < -0.7)
antipode_mask = psi0_coords[:, 0] < -0.7
worst_at_antipode = np.argsort(np.abs(psi_flat[antipode_mask]))[::-1][:6]

# Translate back to global indices
antipode_indices = np.where(antipode_mask)[0]
print(f"\nTop 6 |psi_star[0]| values at antipode coords (x < -0.7):")
for j in worst_at_antipode:
    idx = antipode_indices[j]
    print(f"  idx={idx:5d}  psi*={psi_flat[idx]:+.6f}  "
          f"coord=[{psi0_coords[idx, 0]:+.3f}, {psi0_coords[idx, 1]:+.3f}, {psi0_coords[idx, 2]:+.3f}]")
