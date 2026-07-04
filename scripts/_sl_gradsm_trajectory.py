"""Compare 100-step trajectories on the stagnant-lid adapt-loop with
three configurations of strategy='med', adapt-every-5:

    1. med                            (no metric smoothing, isotropic CFL)
    2. med + anisotropic CFL          (no metric smoothing)
    3. med + anisotropic CFL + grad_smoothing_length = 2·h0
       (gradient-side screened-Poisson smoothing of the metric source)

Two-panel time-series figure (vrms + Nu vs t) with adapt marks, plus a
two-panel end-state figure (T field + mesh on rows 0, 1) for the two
extreme cases.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

import underworld3 as uw
import underworld3.visualisation as vis


pv.OFF_SCREEN = True
OUT = os.path.expanduser(
    '~/+Simulations/StagnantLid/adapt_loop_med_every5_gradsmL2h0')
RUNS = [
    ('med',
     '~/+Simulations/StagnantLid/adapt_loop_med_every5',
     'tab:gray'),
    ('med + aniso CFL',
     '~/+Simulations/StagnantLid/adapt_loop_med_every5_aniso',
     'tab:blue'),
    ('med + aniso CFL + grad-sm L=2·h0',
     '~/+Simulations/StagnantLid/adapt_loop_med_every5_gradsmL2h0',
     'tab:red'),
]


def load(d):
    p = os.path.expanduser(os.path.join(d, 'history.npz'))
    return np.load(p, allow_pickle=True)


# Time-series panel
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
for label, d, c in RUNS:
    h = load(d)
    t = np.asarray(h['t'])
    Nu = np.asarray(h['Nu'])
    vrms = np.asarray(h['vrms'])
    ad = np.asarray(h['adapted']).astype(bool)
    axes[0].plot(t, vrms, color=c, label=label, lw=1.5)
    axes[1].plot(t, Nu, color=c, label=label, lw=1.5)
    # adapt marks
    if ad.any():
        axes[0].plot(t[ad], vrms[ad], 'o', color=c, ms=6, mfc='none')
        axes[1].plot(t[ad], Nu[ad], 'o', color=c, ms=6, mfc='none')

axes[0].set_ylabel(r'$v_\mathrm{rms}$')
axes[1].set_ylabel('Nu')
axes[1].set_xlabel('t')
axes[0].set_title(
    'Stagnant lid Ra=1e7 Δη=1e4, P3-T  ·  100 steps  ·  '
    'adapt every 5 (strategy=med)\n'
    'open circles = adapt step;  '
    'baseline + aniso CFL + gradient-side metric smoothing'
)
for ax in axes:
    ax.grid(alpha=0.3)
axes[0].legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'plot_timeseries_compare.png'), dpi=130)
plt.close()
print('wrote', os.path.join(OUT, 'plot_timeseries_compare.png'))


# End-state side-by-side (aniso vs aniso+grad-sm) — T + mesh
ENDPAIR = [
    ('aniso, no smoothing',
     '~/+Simulations/StagnantLid/adapt_loop_med_every5_aniso'),
    ('aniso + grad-sm L=2h0',
     '~/+Simulations/StagnantLid/adapt_loop_med_every5_gradsmL2h0'),
]

# pick the highest-numbered step snapshot present in both
def last_step(d):
    d_full = os.path.expanduser(d)
    steps = sorted(
        int(f[4:8]) for f in os.listdir(d_full)
        if f.startswith('step') and f.endswith('.mesh.00000.h5')
    )
    return steps[-1] if steps else None


pl = pv.Plotter(shape=(1, 2), off_screen=True,
                window_size=(1100, 600), border=False)
pl.set_background('white')

for col, (label, d) in enumerate(ENDPAIR):
    d_full = os.path.expanduser(d)
    s = last_step(d_full)
    stem = f'step{s:04d}'
    m = uw.discretisation.Mesh(
        os.path.join(d_full, f'{stem}.mesh.00000.h5'))
    T = uw.discretisation.MeshVariable(
        f'T_{col}', m, vtype=uw.VarType.SCALAR, degree=3, continuous=True)
    T.read_timestep(stem, 'T_v2p1', 0, outputPath=d_full)
    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data['T'] = np.asarray(T.data[:, 0])
    edges = vis.mesh_to_pv_mesh(m).extract_all_edges()
    pl.subplot(0, col)
    pl.add_text(f'{label}\nstep {s}',
                font_size=12, color='black')
    pl.add_mesh(pv_T, scalars='T', cmap='RdBu_r',
                clim=(0.0, 1.0), show_edges=False,
                lighting=False,
                show_scalar_bar=(col == 1),
                scalar_bar_args=dict(title='T', color='black'))
    pl.add_mesh(edges, color='black', line_width=0.7,
                lighting=False, opacity=0.6)
    pl.view_xy()
    pl.camera.zoom(1.25)

out_png = os.path.join(OUT, 'plot_endstate_compare.png')
pl.screenshot(out_png)
pl.close()
print('wrote', out_png)

# Final-number summary
print()
print('Final-step numbers:')
for label, d, _ in RUNS:
    h = load(d)
    print(
        f'  {label:38s}  '
        f't={float(h["t"][-1]):.4f}  '
        f'Nu={float(h["Nu"][-1]):.3f}  '
        f'vrms={float(h["vrms"][-1]):.1f}  '
        f'wall={float(np.sum(h["wall"])):.0f}s  '
        f'adapts={int(np.sum(np.asarray(h["adapted"]).astype(bool)))}'
    )
