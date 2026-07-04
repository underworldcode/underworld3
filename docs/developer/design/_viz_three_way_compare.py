"""Three-way comparison at common steps:
  Row 0: P3-FE   (rings catastrophically at step 35)
  Row 1: P3-RBF  (stable but diffusive)
  Row 2: P5-RBF  (stable, recovering accuracy)

Scheme can be 'rk4' or 'rk2'. Steps autodetect from snapshot directories.
"""
import argparse
import os
import numpy as np
import pyvista as pv
import underworld3 as uw
import underworld3.visualisation as vis

pv.OFF_SCREEN = True


CONFIGS = {
    "rk4": [
        ("output/convection_zoo_snapshots_rk4_full",
         "rk4", 3, "RK4 P3-FE  (default, rings @35)"),
        ("output/convection_zoo_snapshots_rk4_rbf_p5",
         "rk4", 5, "RK4 P5-RBF  (clean, more diffusive)"),
        ("output/convection_zoo_snapshots_rk4_monotone_clamp",
         "rk4", 3, "RK4 P3 + B.2 clamp  (preferred)"),
        ("output/convection_zoo_snapshots_rk4_monotone_pick",
         "rk4", 3, "RK4 P3 + B.1 pick  (more smooth)"),
    ],
    "rk2": [
        ("output/convection_zoo_snapshots_rk2_full",
         "rk2", 3, "P3-FE  (default)"),
        ("output/convection_zoo_snapshots_rk2_rbf_traceback",
         "rk2", 3, "P3-RBF  (fix)"),
        ("output/convection_zoo_snapshots_rk2_rbf_p5",
         "rk2", 5, "P5-RBF  (fix + hi-res T)"),
    ],
}


def load_T(snap_dir, scheme, step, degree):
    root = f"uw_{scheme}_step{step:04d}"
    mesh = uw.discretisation.Mesh(
        os.path.join(snap_dir, f"{root}.mesh.00000.h5"))
    pair_tag = f"v2p1"
    T = uw.discretisation.MeshVariable(
        f"T_conv_{pair_tag}", mesh, vtype=uw.VarType.SCALAR,
        degree=degree, continuous=True)
    T.read_timestep(root, f"T_conv_{pair_tag}", 0,
                    outputPath=snap_dir)
    return mesh, T


def available_steps(snap_dir, scheme):
    files = os.listdir(snap_dir) if os.path.isdir(snap_dir) else []
    nums = []
    for f in files:
        if f.startswith(f"uw_{scheme}_step") and f.endswith(
                ".mesh.00000.h5"):
            try:
                nums.append(int(f.split("_step")[1].split(".")[0]))
            except ValueError:
                continue
    return set(nums)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scheme", default="rk4",
                   choices=list(CONFIGS.keys()))
    p.add_argument("--steps", default="5,10,15",
                   help="Comma-separated step numbers.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    configs = CONFIGS[args.scheme]
    wanted = [int(s) for s in args.steps.split(",")]
    # Restrict to steps present in ALL three runs
    common = set(wanted)
    for snap_dir, scheme, _, _ in configs:
        common &= available_steps(snap_dir, scheme)
    steps = sorted(common)
    if not steps:
        print(f"No common steps found among configs for "
              f"requested {wanted}")
        return
    print(f"common steps: {steps}")

    out_png = (args.out
               or f"output/three_way_{args.scheme}_compare.png")

    plotter = pv.Plotter(shape=(len(configs), len(steps)),
                         window_size=(450 * len(steps),
                                      500 * len(configs)),
                         border=False, off_screen=True)
    plotter.set_background("white")

    for row, (snap_dir, scheme, degree, label) in enumerate(configs):
        for col, step in enumerate(steps):
            mesh, T = load_T(snap_dir, scheme, step, degree)
            Tmin = float(T.data[:, 0].min())
            Tmax = float(T.data[:, 0].max())
            print(f"{label} step {step}: T=[{Tmin:+.3f}, "
                  f"{Tmax:+.3f}]")

            pv_T = vis.meshVariable_to_pv_mesh_object(T)
            pv_T.point_data["T"] = np.asarray(T.data[:, 0])
            edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()

            plotter.subplot(row, col)
            plotter.set_background("white")
            show_bar = (row == 0 and col == 0)
            plotter.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
                             clim=(0, 1), show_edges=False,
                             lighting=False,
                             show_scalar_bar=show_bar,
                             scalar_bar_args={
                                 "title": "T",
                                 "vertical": False,
                                 "position_x": 0.15,
                                 "position_y": 0.04,
                                 "width": 0.7,
                                 "height": 0.04,
                                 "title_font_size": 16,
                                 "label_font_size": 14}
                             if show_bar else None)
            plotter.add_mesh(edges, color="black",
                             line_width=0.5, lighting=False)
            plotter.add_text(
                f"{label}\nstep {step}"
                f"   T∈[{Tmin:+.2f}, {Tmax:+.2f}]",
                position="upper_edge", font_size=13,
                color="black")
            plotter.view_xy()

    plotter.screenshot(out_png, transparent_background=False,
                       window_size=(450 * len(steps),
                                    500 * len(configs)))
    plotter.close()
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
