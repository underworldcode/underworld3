"""Nu(t) and Nu(step) plots comparing RK4 trace-back variants:
  P3-FE (default, rings at step 35)
  P5-RBF (full RBF + hi-res T, more diffusive)
  P3 + B.2 clamp (preferred)
  P3 + B.1 pick (more smooth)
"""
import os
import numpy as np
import matplotlib.pyplot as plt

CSVS = [
    ("/tmp/conv_rk4_full.csv",            "P3-FE (default, rings)",   "C3", "-"),
    ("/tmp/conv_rk4_rbf_p5.csv",          "P5-RBF (full RBF)",        "C0", "-"),
    ("/tmp/conv_rk4_clamp.csv",           "P3 + B.2 clamp",           "C2", "-"),
    ("/tmp/conv_rk4_pick.csv",            "P3 + B.1 pick",            "C1", "-"),
]
OUT_PNG = "output/nu_compare_rk4.png"


def load(csv_path):
    # cols: scheme,step,t,dt,h_pole,vrms,T_avg,Nu,h_max,h_rms,area_uw,
    #       delta_A,wall_step_s
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1,
                         usecols=(1, 2, 5, 7, 8))
    return {"step": data[:, 0].astype(int),
            "t":    data[:, 1],
            "vrms": data[:, 2],
            "Nu":   data[:, 3],
            "h_max":data[:, 4]}


fig, axes = plt.subplots(3, 1, figsize=(11, 11),
                         sharex=True, constrained_layout=True)

for csv, label, color, ls in CSVS:
    if not os.path.exists(csv):
        print(f"missing {csv}")
        continue
    d = load(csv)
    print(f"{label}: {len(d['step'])} steps, "
          f"final Nu={d['Nu'][-1]:.1f}")
    axes[0].plot(d["step"], d["Nu"],
                 color=color, linestyle=ls, marker="o",
                 markersize=3, label=label)
    axes[1].plot(d["step"], d["vrms"],
                 color=color, linestyle=ls, marker="o",
                 markersize=3, label=label)
    axes[2].plot(d["step"], d["h_max"],
                 color=color, linestyle=ls, marker="o",
                 markersize=3, label=label)

# Mark the catastrophe at step 35 for P3-FE
axes[0].axvline(35, color="0.7", linewidth=0.5, linestyle=":")
axes[0].annotate("FE catastrophe →",
                 xy=(35, axes[0].get_ylim()[1] * 0.9),
                 xytext=(28, axes[0].get_ylim()[1] * 0.85),
                 fontsize=9, color="C3",
                 ha="right")

axes[0].set_ylabel("Nu")
axes[0].set_title("Nu vs step — RK4 trace-back variants")
axes[0].legend(loc="upper left", fontsize=10)
axes[0].grid(alpha=0.3)

axes[1].set_ylabel("vrms")
axes[1].grid(alpha=0.3)

axes[2].set_ylabel("h_max  (surface deformation)")
axes[2].set_xlabel("step")
axes[2].grid(alpha=0.3)

fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT_PNG}")
