"""Probe: mesh_metric_mismatch + skip_threshold kwarg.

1) Compute mismatch for uniform vs adapted snapshot
2) Verify smooth_mesh_interior(skip_threshold=...) skips when
   below threshold, runs when above.
"""
import os
import numpy as np
import underworld3 as uw


SRC_U = os.path.expanduser(
    '~/+Simulations/StagnantLid/uniform_res16_Ra1e7_dEta1e4')
STEM_U = "sl_uniform_res16_Ra1e7_dEta1e4_step00125"
SRC_A = os.path.expanduser(
    '~/+Simulations/StagnantLid/adapted_R15_Ra1e7_dEta1e4')
STEM_A = "adapted"


def load_mesh_and_T(src, stem):
    m = uw.discretisation.Mesh(os.path.join(
        src, f"{stem}.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        f"T_{id(m)}", m, vtype=uw.VarType.SCALAR, degree=3,
        continuous=True)
    T.read_timestep(stem, "T_v2p1", 0, outputPath=src)
    return m, T


print("=" * 64)
print("mesh_metric_mismatch + skip_threshold probe")
print("=" * 64)

for label, (src, stem) in [
    ("uniform", (SRC_U, STEM_U)),
    ("adapted R=1.5", (SRC_A, STEM_A)),
]:
    print(f"\n[{label}]")
    m, T = load_mesh_and_T(src, stem)
    rho = uw.meshing.metric_density_from_gradient(
        m, T, amp=8.0, lo_percentile=50.0, hi_percentile=97.0,
        name=f"probe_{label}")
    mm = uw.meshing.mesh_metric_mismatch(m, rho, resolution_ratio=1.5)
    print(f"  rms={mm['rms']:.4f}  max={mm['max']:.4f}  "
          f"median|δ|={mm['median_abs']:.4f}")
    print(f"  alignment r={mm['alignment']:+.4f}  "
          f"misalignment={mm['misalignment']:.4f}")

# Now demonstrate the skip path
for thr in (0.3, 0.9):
    print("\n" + "=" * 64)
    print(f"smooth_mesh_interior(skip_threshold={thr}) behaviour")
    print("=" * 64)

    for label, (src, stem) in [
        ("uniform", (SRC_U, STEM_U)),
        ("adapted", (SRC_A, STEM_A)),
    ]:
        print(f"\n[{label}]  skip_threshold={thr}")
        m, T = load_mesh_and_T(src, stem)
        rho = uw.meshing.metric_density_from_gradient(
            m, T, amp=8.0, name=f"probe_{thr}_{label}")
        X_before = np.asarray(m.X.coords).copy()
        uw.meshing.smooth_mesh_interior(
            m, metric=rho, method="anisotropic",
            method_kwargs=dict(resolution_ratio=1.5,
                               relax=0.2, n_outer=12),
            skip_threshold=thr, verbose=True)
        X_after = np.asarray(m.X.coords).copy()
        moved = float(np.linalg.norm(X_after - X_before))
        print(f"  || ΔX || = {moved:.4e}  "
              f"({'moved' if moved > 1e-6 else 'unchanged'})")
