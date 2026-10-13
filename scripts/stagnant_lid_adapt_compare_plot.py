"""Clean comparison: uniform vs adapted mesh.
  Row 1 — T field (no edges)
  Row 2 — |∇T| field + mesh edges overlaid (the diagnostic — is
           the refinement going where the metric demands?)
Loads existing snapshots — no recomputation.
"""
from __future__ import annotations
import os
import numpy as np
import sympy
import underworld3 as uw
import underworld3.visualisation as vis


def required_files(src_dir, stem):
    return [
        os.path.join(src_dir, f"{stem}.mesh.00000.h5"),
        os.path.join(src_dir, f"{stem}.mesh.T_v2p1.00000.h5"),
    ]


def check_required_inputs(cases):
    missing = []

    for label, src_dir, stem in cases:
        for filename in required_files(src_dir, stem):
            if not os.path.exists(filename):
                missing.append((label, filename))

    if missing:
        print("Missing stagnant-lid checkpoint input files:")
        for label, filename in missing:
            print(f"  [{label}] {filename}")

        print(
            "\nThis script only plots existing stagnant-lid snapshots. "
            "Run or copy the required checkpoint outputs first, or update "
            "U_DIR/A_DIR and U_STEM/A_STEM to point to existing files."
        )
        raise SystemExit(1)


def load(src_dir, stem):
    m = uw.discretisation.Mesh(os.path.join(
        src_dir, f"{stem}.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        f"T_view_{id(m)}", m, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep(stem, "T_v2p1", 0, outputPath=src_dir)
    return m, T


U_DIR = os.path.expanduser(
    '~/+Simulations/StagnantLid/uniform_res16_Ra1e7_dEta1e4')
U_STEM = "sl_uniform_res16_Ra1e7_dEta1e4_step00125"
A_DIR = os.path.expanduser(
    '~/+Simulations/StagnantLid/adapted_R15_Ra1e7_dEta1e4')
A_STEM = "adapted"

check_required_inputs([
    ("uniform", U_DIR, U_STEM),
    ("adapted", A_DIR, A_STEM),
])

import pyvista as pv

pv.OFF_SCREEN = True

mu, Tu = load(U_DIR, U_STEM)
ma, Ta = load(A_DIR, A_STEM)


def gradT_mag_sym(mesh, T):
    X = mesh.CoordinateSystem.X
    return sympy.sqrt(T.sym[0].diff(X[0]) ** 2
                      + T.sym[0].diff(X[1]) ** 2)


# Pre-pass: shared color limit for |∇T| so both panels are
# comparable on the same scale.
g_max = 0.0
for (m, T) in [(mu, Tu), (ma, Ta)]:
    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    g = vis.scalar_fn_to_pv_points(pv_T, gradT_mag_sym(m, T))
    g_max = max(g_max, float(np.nanmax(g)))
print(f"|∇T|max (shared clim) = {g_max:.3e}")


pl = pv.Plotter(shape=(2, 2), off_screen=True,
                window_size=(1600, 1600), border=False)
pl.set_background("white")

for col, (label, m, T) in enumerate([
    ("uniform res-16", mu, Tu),
    ("adapted R=1.5  ρ ∝ |∇T|", ma, Ta),
]):
    # Top row: T field (no edges)
    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])
    pl.subplot(0, col)
    pl.add_text(label, font_size=14, color="black")
    pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
                clim=(0.0, 1.0), show_edges=False, lighting=False,
                show_scalar_bar=(col == 1),
                scalar_bar_args=dict(title="T", color="black"))
    pl.view_xy()
    pl.camera.zoom(1.3)

    # Bottom row: |∇T| + mesh edges (the diagnostic)
    pv_g = vis.meshVariable_to_pv_mesh_object(T)
    pv_g.point_data["gradT"] = vis.scalar_fn_to_pv_points(
        pv_g, gradT_mag_sym(m, T))
    edges = vis.mesh_to_pv_mesh(m).extract_all_edges()
    pl.subplot(1, col)
    pl.add_text(f"|∇T| + mesh edges  ({label})",
                font_size=14, color="black")
    pl.add_mesh(pv_g, scalars="gradT", cmap="Greens",
                clim=(0.0, g_max), show_edges=False,
                lighting=False,
                show_scalar_bar=(col == 1),
                scalar_bar_args=dict(title="|∇T|",
                                     color="black"))
    pl.add_mesh(edges, color="black", line_width=0.7,
                lighting=False, opacity=0.55)
    pl.view_xy()
    pl.camera.zoom(1.3)

out = os.path.join(
    A_DIR, "plot_adapt_compare_clean.png")
pl.screenshot(out)
pl.close()
print(f"wrote {out}")
