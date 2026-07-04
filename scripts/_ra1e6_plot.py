"""Ra=1e6: Nu/vrms timeseries + final-checkpoint T+mesh, two
arms (workable stack: l2 + cold-recover + V,P-remap).

  a16r15_6  R=1.5 equidist adaptive
  u16_6     uniform res-16  (no adaptation)

NB the harness settle detector was tuned for Ra=1e5; at Ra=1e6 it
likely misfires — these are *final-checkpoint* states (where the
detector stopped each run), not certified statistical steady state.
"""
import glob
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

D = "/tmp/metric_mesh/sat"
RUNS = [("a16r15_6", "#1f77b4",
         "R=1.5 equidist (adaptive)"),
        ("u16_6",    "#c0392b",
         "uniform res-16")]

# ---------------- (1) Nu/vrms timeseries -----------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
for tag, col, lab in RUNS:
    z = np.load(f"{D}/sat_{tag}_hist.npz")
    t, Nu, vr = z["t"], z["Nu"], z["vrms"]
    ax[0].plot(t, Nu, "-", color=col, lw=1.7, label=lab)
    ax[1].plot(t, vr, "-", color=col, lw=1.7, label=lab)
    # final-checkpoint annotation (NOT certified steady)
    k = max(1, len(t) // 10)
    nu_f, vr_f = float(Nu[-k:].mean()), float(vr[-k:].mean())
    ax[0].axhline(nu_f, color=col, ls=":", lw=1.0, alpha=0.5)
    print(f"{tag:9s}: {len(t):4d} steps  t_end={t[-1]:.4f}  "
          f"Nu_final≈{nu_f:.3f}  vrms_final≈{vr_f:.1f}")
for a, ttl, yl in ((ax[0], "Nusselt(t)", "Nu"),
                   (ax[1], "vrms(t)", "vrms")):
    a.set_xlabel("dimensionless time")
    a.set_ylabel(yl)
    a.set_title(ttl)
    a.legend(fontsize=10, loc="best")
    a.grid(alpha=0.3)
fig.suptitle("Ra=1e6 — Nu/vrms time histories (final-checkpoint "
             "values; settle detector likely under-tuned for Ra=1e6)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out_ts = f"{D}/ra1e6_timeseries.png"
fig.savefig(out_ts, dpi=140)
print(f"saved {out_ts}")

# ---------------- (2) T field + mesh at final checkpoint -------
pv.OFF_SCREEN = True
pl = pv.Plotter(shape=(1, 2), off_screen=True,
                window_size=(1100, 900), border=True,
                border_color="#888888")
pl.set_background("white")

def _latest(tag):
    fs = glob.glob(f"{D}/sat_{tag}.mesh.T.*.h5")
    ix = [int(re.search(r"\.T\.(\d+)\.h5$", f).group(1))
          for f in fs]
    return max(ix) if ix else None

for c, (tag, _, lab) in enumerate(RUNS):
    idx = _latest(tag)
    m = uw.discretisation.Mesh(f"{D}/sat_{tag}.mesh.{idx:05}.h5")
    Tv = uw.discretisation.MeshVariable(
        "T", m, vtype=uw.VarType.SCALAR, degree=3, continuous=True)
    Tv.read_timestep(f"sat_{tag}", "T", idx, outputPath=D)
    pv_T = vis.meshVariable_to_pv_mesh_object(Tv)
    pv_T.point_data["T"] = np.asarray(Tv.data[:, 0])
    edges = vis.mesh_to_pv_mesh(m).extract_all_edges()
    pl.subplot(0, c)
    pl.add_text(f"{lab}\nfinal ckpt {idx}",
                font_size=14, color="black")
    pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
                clim=(0.0, 1.0), show_edges=False,
                lighting=False,
                show_scalar_bar=(c == len(RUNS) - 1),
                scalar_bar_args=dict(title="T", color="black"))
    pl.add_mesh(edges, color="#202020", line_width=0.7,
                lighting=False)
    pl.view_xy()
    pl.camera.zoom(1.35)
out_fld = f"{D}/ra1e6_fields.png"
pl.screenshot(out_fld)
pl.close()
print(f"saved {out_fld}")
