"""Animate SLCN solid-body rotation of a Gaussian on a SphericalManifold.

Runs serial (the 2-rank case has a separate SNES_Projection F1 sizing
issue still to fix). Writes a GIF showing the Gaussian rotating around
the z-axis as the manifold-aware SL trace-back carries it along the
equator.

Output: docs/examples/parallel_point_eval/_advection_rotation.gif

Run:
    python animate_manifold_advection.py
"""

import os

import numpy as np
import sympy

import underworld3 as uw


CELL_SIZE = 0.075
# Exactly one full rotation at omega = 1: t_end = 2π, so step
# count × DT = 2π. 72 steps × 5° each lands the Gaussian back at
# its initial position to compare like-for-like.
N_STEPS = 72
DT = 2 * np.pi / N_STEPS
DIFFUSIVITY = 1.0e-4
GAUSSIAN_SIGMA = 0.3
# Higher-order P_k MeshVariable + RBF trace-back: chord-interior DOFs
# are fine because RBF (Shepard) doesn't need cell-location; more DOFs
# per cell give finer Shepard spacing → less trace-back dissipation.
T_DEGREE = 3
# PETSc convention: qdegree matches the FE element order. UW3 default
# is qdegree=2 (appropriate for P2 default); needs lifting for higher
# orders or the FE assembly is under-integrated.
QDEGREE = T_DEGREE


def main():
    print(f"Building SphericalManifold(cellSize={CELL_SIZE}, qdegree={QDEGREE}) "
          f"for P{T_DEGREE} elements")
    mesh = uw.meshing.SphericalManifold(
        radius=1.0, cellSize=CELL_SIZE, qdegree=QDEGREE,
    )
    print(f"  mesh.dim={mesh.dim}, mesh.cdim={mesh.cdim}")

    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=T_DEGREE)
    coords = np.asarray(T.coords)

    # Initial Gaussian centred at (1, 0, 0) on the equator.
    dx = coords[:, 0] - 1.0
    dy = coords[:, 1]
    dz = coords[:, 2]
    T.data[:, 0] = np.exp(-(dx**2 + dy**2 + dz**2) / (2 * GAUSSIAN_SIGMA**2))

    # Solid-body rotation about z: V = (-y, x, 0). Tangent to the
    # sphere everywhere and div-free.
    x_sym, y_sym, z_sym = mesh.X
    V_sym = sympy.Matrix([[-y_sym, x_sym, sympy.sympify(0)]])

    # monotone_mode="clamp" bounds the trace-back result to the
    # local data range (k = dim+1 nearest psi_star DOFs). Bit-identical
    # to pure FE in smooth regions; prevents runaway with high-order
    # FE on the manifold where chord-interior DOF placement can lead
    # to overshoot.
    adv = uw.systems.AdvDiffusionSLCN(
        mesh, u_Field=T, V_fn=V_sym, order=1, monotone_mode="clamp",
    )
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = DIFFUSIVITY
    adv.f = sympy.Matrix.zeros(1, 1)

    print(f"Stepping {N_STEPS} times at dt={DT:.4f} "
          f"(omega·dt={DT:.3f} rad/step = {np.degrees(DT):.1f}°/step)")
    T_initial = np.asarray(T.data[:, 0]).copy()

    # pyvista output. Open a GIF stream; write one frame per step.
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, T.sym)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    frames_dir = os.path.join(out_dir, "_advection_frames")
    os.makedirs(frames_dir, exist_ok=True)
    gif_path = os.path.join(out_dir, "_advection_rotation.gif")

    # Colormap covers [-0.3, 1.0] so the SLCN BDF-1 undershoot in the
    # wake is visible as a cool-violet tint rather than being clamped
    # to white (which would look like an evaluation failure). The
    # rotating blob still reads as warm against the white background.
    from matplotlib.colors import LinearSegmentedColormap
    field_cmap = LinearSegmentedColormap.from_list(
        "signed_warm",
        [
            (0.00, "#7e2e9d"),   # -0.3 : strong violet (undershoot)
            (0.15, "#bb91dd"),   # -0.105: pale violet
            (0.231, "#ffffff"),  #  0.0  : white (background)
            (0.31, "#cfe5ff"),   #  0.10 : very pale blue
            (0.46, "#4f9aff"),   #  0.30 : mid blue
            (0.65, "#3ec76b"),   #  0.55 : green
            (0.85, "#ffa040"),   #  0.80 : orange
            (1.00, "#c41e1e"),   #  1.00 : intense red
        ],
    )
    clim = (-0.3, 1.0)

    def render_frame(step_idx, t):
        # High-res frames — better than the GIF, and the user can
        # open the directory and step through them or assemble into
        # a higher-quality MP4 with ffmpeg if they want.
        plotter = pv.Plotter(window_size=(1400, 1200), off_screen=True)
        plotter.set_background("white")
        plotter.add_mesh(
            pvmesh,
            scalars="T",
            cmap=field_cmap,
            clim=clim,
            show_edges=False,
            smooth_shading=False,
            scalar_bar_args={"title": "T", "color": "black"},
        )
        # Camera looking down +x with the rotation axis vertical so the
        # Gaussian sweeps left-to-right across the equator.
        plotter.camera_position = [
            (3.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)
        ]
        plotter.add_text(
            f"t = {t:.2f}", position="upper_left",
            color="black", font_size=12,
        )
        path = os.path.join(frames_dir, f"frame_{step_idx:04d}.png")
        plotter.screenshot(path)
        plotter.close()
        return path

    frame_paths = [render_frame(0, 0.0)]

    for step in range(N_STEPS):
        # _evalf=True routes the SL trace-back's `global_evaluate` calls
        # through the RBF (Shepard) path instead of FE/DMInterpolation —
        # which sidesteps the cell-location bug DMInterpolation exhibits
        # on a 2-manifold (16% of trace-back evaluations were producing
        # wildly wrong values via cell-misrouting; validated by manual
        # barycentric probe).
        adv.solve(timestep=DT, _evalf=True)
        t_now = (step + 1) * DT

        # Refresh pyvista field values for this step.
        pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, T.sym)
        frame_paths.append(render_frame(step + 1, t_now))

        if (step + 1) % 8 == 0:
            t_min = float(np.asarray(T.data[:, 0]).min())
            t_max = float(np.asarray(T.data[:, 0]).max())
            t_sum = float(np.asarray(T.data[:, 0]).sum())
            print(f"  step {step + 1:3d}/{N_STEPS}  t={t_now:.2f}  "
                  f"T: min={t_min:+.3f} max={t_max:+.3f} sum={t_sum:.2f}")

    print(f"\nHigh-res PNG frames in: {frames_dir}")
    print(f"  ({len(frame_paths)} frames at 1400×1200 px)")
    print(f"  open the folder with: open {frames_dir}")
    print(f"  assemble to MP4 with: ffmpeg -framerate 12 -i "
          f"{frames_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p "
          f"{out_dir}/rotation.mp4")

    # After one full rotation, T should equal its initial values
    # (modulo numerical dissipation). Report dissipation budget.
    T_final = np.asarray(T.data[:, 0])
    err = T_final - T_initial
    l2_err = float(np.sqrt((err * err).mean()))
    l2_init = float(np.sqrt((T_initial * T_initial).mean()))
    print(f"\nOne-rotation dissipation diagnostic:")
    print(f"  max(T_initial) = {T_initial.max():.4f}")
    print(f"  max(T_final)   = {T_final.max():.4f}   "
          f"(amplitude retained: {T_final.max()/T_initial.max()*100:.1f}%)")
    print(f"  L2 || T_final − T_initial ||  = {l2_err:.4e}")
    print(f"  L2 || T_initial ||             = {l2_init:.4e}")
    print(f"  Relative L2 error per rotation = {l2_err/l2_init*100:.2f}%")

    # Locate the most-negative DOFs and report their position relative
    # to the expected blob centre. If they cluster near the blob,
    # that's classic SLCN dispersion in the wake. If they're scattered,
    # it would point at evaluation failures or trace-back glitches.
    worst_neg = np.argsort(T_final)[:8]
    blob_centre = np.array([1.0, 0.0, 0.0])  # back to start after 2π
    print(f"\n  Most-negative DOFs (8 worst):")
    print(f"    {'idx':>5} {'T':>8} {'coord':>30} "
          f"{'arc to blob':>14}")
    for idx in worst_neg:
        c = coords[idx]
        # Geodesic arc length on the unit sphere = acos(dot(c, centre))
        cos_arc = np.clip(np.dot(c, blob_centre), -1.0, 1.0)
        arc = float(np.degrees(np.arccos(cos_arc)))
        print(f"    {int(idx):>5} {T_final[idx]:>+8.4f} "
              f"[{c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f}]   {arc:>10.1f}°")


if __name__ == "__main__":
    main()
