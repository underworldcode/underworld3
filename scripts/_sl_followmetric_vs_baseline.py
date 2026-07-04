"""Compare the follow_metric(refinement=3) adapt-loop trajectory
to the validated strategy='med' baseline on the stagnant-lid case.

Both runs: 100 steps, adapt-every-5, anisotropic CFL, surface Nu.
Difference: metric construction (legacy med strategy with manual
dials vs the new single-knob follow_metric API + per-cell rest-
size spring + sliver-floor)."""
import os
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import underworld3 as uw
import underworld3.visualisation as vis

pv.OFF_SCREEN = True

RUNS = [
    ("med (legacy strategy)",
     '~/+Simulations/StagnantLid/adapt_loop_med_every5_aniso',
     'tab:blue'),
    ("follow_metric(refinement=3)",
     '~/+Simulations/StagnantLid/adapt_loop_followmetric_ref3',
     'tab:red'),
]

OUT = os.path.expanduser(
    '~/+Simulations/StagnantLid/adapt_loop_followmetric_ref3')


def load(d):
    return np.load(os.path.expanduser(
        os.path.join(d, 'history.npz')), allow_pickle=True)


# Time-series
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for label, d, c in RUNS:
    h = load(d)
    t = np.asarray(h['t'])
    Nu = np.asarray(h['Nu'])
    vrms = np.asarray(h['vrms'])
    ad = np.asarray(h['adapted']).astype(bool)
    axes[0].plot(t, vrms, color=c, label=label, lw=1.5)
    axes[1].plot(t, Nu, color=c, label=label, lw=1.5)
    if ad.any():
        axes[0].plot(t[ad], vrms[ad], 'o', color=c, ms=6, mfc='none')
        axes[1].plot(t[ad], Nu[ad], 'o', color=c, ms=6, mfc='none')

axes[0].set_ylabel(r'$v_\mathrm{rms}$')
axes[1].set_ylabel('Nu (surface, BdIntegral)')
axes[1].set_xlabel('t')
axes[0].set_title(
    'Stagnant lid Ra=1e7 Δη=1e4, P3-T  ·  100 steps  ·  adapt every 5\n'
    'open circles = adapt step;  follow_metric vs legacy strategy')
for ax in axes:
    ax.grid(alpha=0.3)
axes[0].legend(loc='lower right', fontsize=10)
plt.tight_layout()
out_png = os.path.join(OUT, 'plot_followmetric_vs_baseline.png')
plt.savefig(out_png, dpi=130)
plt.close()
print(f"wrote {out_png}")


# End-state comparison
ncols = 2
pl = pv.Plotter(shape=(1, ncols), off_screen=True,
                window_size=(1500 * ncols, 1500), border=False)
pl.set_background("white")


def last_step(d):
    d_full = os.path.expanduser(d)
    steps = sorted(
        int(f[4:8]) for f in os.listdir(d_full)
        if f.startswith('step') and f.endswith('.mesh.00000.h5'))
    return steps[-1] if steps else None


for col, (label, d, _) in enumerate(RUNS):
    d_full = os.path.expanduser(d)
    s = last_step(d_full)
    stem = f'step{s:04d}'
    m = uw.discretisation.Mesh(
        os.path.join(d_full, f'{stem}.mesh.00000.h5'))
    T = uw.discretisation.MeshVariable(
        f'T_{col}', m, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep(stem, 'T_v2p1', 0, outputPath=d_full)
    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])
    edges = vis.mesh_to_pv_mesh(m).extract_all_edges()
    pl.subplot(0, col)
    pl.add_text(f'{label}\nstep {s}',
                font_size=22, color='black')
    pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
                clim=(0.0, 1.0), show_edges=False,
                lighting=False, show_scalar_bar=False)
    pl.add_mesh(edges, color="black", line_width=1.2,
                lighting=False, opacity=0.85)
    pl.view_xy()
    pl.camera.zoom(1.25)

out_endstate = os.path.join(OUT, 'plot_endstate_vs_baseline.png')
pl.screenshot(out_endstate)
pl.close()
print(f"wrote {out_endstate}")


# Final numeric summary
print()
print("Final-step numbers:")
for label, d, _ in RUNS:
    h = load(d)
    print(
        f'  {label:35s}  '
        f't={float(h["t"][-1]):.4f}  '
        f'Nu={float(h["Nu"][-1]):.3f}  '
        f'vrms={float(h["vrms"][-1]):.1f}  '
        f'wall={float(np.sum(h["wall"])):.0f}s  '
        f'adapts={int(np.sum(np.asarray(h["adapted"]).astype(bool)))}'
    )
