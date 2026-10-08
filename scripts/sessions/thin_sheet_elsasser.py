"""Elsasser stress diffusion in a viscoelastic thin sheet over a viscous substrate.

An elastic plate (shear modulus mu, thickness L, Maxwell time much longer than
the run) rests on a viscous channel of drag coefficient c = eta_a / h_a. A
step in the velocity of its left edge diffuses into the plate:

    v_t = D v_xx,   D = mu L / c,   v(x, t) = V0 erfc(x / sqrt(4 D t)).

The figure compares the ThinSheet profiles with the erfc solution at several
times. Run from the worktree environment; the figure goes to -uw_output_dir.
"""
import os

import numpy as np
from scipy.special import erfc

import underworld3 as uw

params = uw.Params(
    uw_output_dir=os.path.expanduser("~/+Simulations/thin_sheet_faults/figures"),
    uw_dt=0.0125,
    uw_cell_size=0.1,
    uw_length=4.0,
    uw_times="0.1,0.25,0.5,1.0",
)

mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(params.uw_length, 1.0),
    cellSize=params.uw_cell_size, qdegree=3)
sheet = uw.systems.ThinSheet(mesh)
sheet.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
    sheet.Unknowns, order=1, integrator="etd")
sheet.constitutive_model.Parameters.shear_viscosity_0 = 1.0e6   # t_r = 1e6 >> t
sheet.constitutive_model.Parameters.shear_modulus = 1.0
sheet.basal_drag = 1                                             # D = mu L / c = 1
sheet.tolerance = 1.0e-10
sheet.add_essential_bc([0, 1.0], "Left")
sheet.add_essential_bc([0, 0], "Right")
sheet.add_essential_bc([0, None], "Top")
sheet.add_essential_bc([0, None], "Bottom")

xs = np.linspace(0.0, 3.0, 121)
pts = np.column_stack([xs, np.full_like(xs, 0.5)])
targets = [float(t) for t in params.uw_times.split(",")]
profiles = {}
t, step = 0.0, 0
while targets:
    sheet.solve(timestep=params.uw_dt, zero_init_guess=False)
    t += params.uw_dt
    step += 1
    if abs(t - targets[0]) < 0.5 * params.uw_dt:
        v = uw.function.evaluate(sheet.u.sym, pts).reshape(-1, 2)[:, 1]
        profiles[targets.pop(0)] = v
        uw.pprint(f"t={t:.4f} step {step}: max |v - erfc| = "
                  f"{np.abs(v - erfc(xs / np.sqrt(4.0 * t))).max():.3e}", flush=True)

if uw.mpi.rank == 0:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(params.uw_output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    blues = plt.get_cmap("Blues")
    for k, (tt, v) in enumerate(sorted(profiles.items())):
        colour = blues(0.4 + 0.6 * k / max(len(profiles) - 1, 1))
        ax.plot(xs, erfc(xs / np.sqrt(4.0 * tt)), color="0.25", lw=1.0, ls="--",
                label="erfc solution" if k == 0 else None)
        ax.plot(xs, v, color=colour, lw=2.0, label=f"t = {tt:g}")
        ax.annotate(f"t = {tt:g}", (np.sqrt(4.0 * tt) * 0.9, erfc(0.9) + 0.02),
                    color=colour, fontsize=9)
    ax.set_xlabel("distance from the driven edge  x / sqrt(D)")
    ax.set_ylabel("sheet velocity  v / V0")
    ax.set_xlim(0, 3.0); ax.set_ylim(0, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="0.9", lw=0.6)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Elsasser stress diffusion: viscoelastic thin sheet over a viscous substrate",
                 fontsize=10, loc="left")
    out = os.path.join(params.uw_output_dir, "elsasser-thin-sheet.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    uw.pprint(f"wrote {out}", flush=True)
