"""
Probe: which DOFs are anomalous after 1 SLCN step?

Set up the same advection problem, take ONE step, and identify DOFs
where T deviates significantly from the analytic trace-back result.
For each anomalous DOF, classify it as vertex (r=1) or edge-midpoint
(r<1).
"""
import numpy as np
import sympy
import underworld3 as uw

mesh = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.15)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
coords = np.asarray(T.coords)

sigma = 0.3
def gaussian_at(c):
    dx = c[..., 0] - 1.0
    dy = c[..., 1]
    dz = c[..., 2]
    return np.exp(-(dx**2 + dy**2 + dz**2) / (2 * sigma**2))

T.data[:, 0] = gaussian_at(coords)
T_initial = np.asarray(T.data[:, 0]).copy()

x_sym, y_sym, z_sym = mesh.X
V_sym = sympy.Matrix([[-y_sym, x_sym, sympy.sympify(0)]])

adv = uw.systems.AdvDiffusionSLCN(mesh, u_Field=T, V_fn=V_sym, order=1)
adv.constitutive_model = uw.constitutive_models.DiffusionModel
adv.constitutive_model.Parameters.diffusivity = 1.0e-4
adv.f = sympy.Matrix.zeros(1, 1)

dt = 2 * np.pi / 72  # one step = 5°
adv.solve(timestep=dt)

T_after = np.asarray(T.data[:, 0])

# Analytic expectation: trace-back endpoint = coord - V(coord)*dt,
# then evaluate Gaussian at that endpoint. For solid-body rotation
# V = (-y, x, 0), so endpoint = (x + y*dt, y - x*dt, z) but the
# rotation is small enough that to first order, endpoint stays
# near the original — and the Gaussian value should be close to the
# Gaussian rotated by 5°.
# Build the analytic rotated Gaussian (rotation about z by dt):
cos_dt = np.cos(dt)
sin_dt = np.sin(dt)
# Trace-back endpoint of each DOF coord:
end_x = coords[:, 0] + coords[:, 1] * dt  # = c - V*dt approximated
end_y = coords[:, 1] - coords[:, 0] * dt
end_z = coords[:, 2]
# Approximate Gaussian value at trace-back endpoint:
T_expected = np.exp(
    -((end_x - 1.0)**2 + end_y**2 + end_z**2) / (2 * sigma**2)
)

# Difference between actual and expected after one SL step.
diff = T_after - T_expected
abs_diff = np.abs(diff)

# Radius of each DOF (distinguishes vertex DOFs on r=1 from edge
# midpoints at r<1 in a P2 element).
r = np.linalg.norm(coords, axis=1)
is_vertex = r > 0.999
is_midpt = r < 0.999

print("=== One-step SLCN diagnostic ===")
print(f"  Total DOFs: {coords.shape[0]}")
print(f"    vertex DOFs (r ≈ 1.0):     {is_vertex.sum()}  (median r = {np.median(r[is_vertex]):.4f})")
print(f"    edge-mid DOFs (r ≈ 0.989): {is_midpt.sum()}   (median r = {np.median(r[is_midpt]):.4f})")
print()

# Show distribution of error magnitudes by DOF category.
print("  Per-category abs(T_after - T_expected) percentiles:")
for label, mask in [("vertex", is_vertex), ("edge-mid", is_midpt)]:
    if mask.any():
        v = abs_diff[mask]
        print(f"    {label:>9}: count={mask.sum():5d}  "
              f"p50={np.percentile(v, 50):.4e}  "
              f"p90={np.percentile(v, 90):.4e}  "
              f"p99={np.percentile(v, 99):.4e}  "
              f"max={v.max():.4e}")
print()

# Top 10 worst DOFs and their category.
worst = np.argsort(abs_diff)[::-1][:12]
print("  Top 12 worst-anomaly DOFs:")
print(f"    {'idx':>5} {'r':>7} {'cat':>10} {'T_init':>8} {'T_after':>9} "
      f"{'T_exp':>8} {'|diff|':>8} {'coord':>22}")
for idx in worst:
    cat = "vertex" if is_vertex[idx] else "edge-mid"
    print(
        f"    {int(idx):>5} {r[idx]:>7.4f} {cat:>10} "
        f"{T_initial[idx]:>+8.4f} {T_after[idx]:>+8.4f} "
        f"{T_expected[idx]:>+8.4f} {abs_diff[idx]:>8.4e} "
        f"[{coords[idx, 0]:+.2f}, {coords[idx, 1]:+.2f}, {coords[idx, 2]:+.2f}]"
    )
