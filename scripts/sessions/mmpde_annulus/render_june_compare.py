"""Nu and vrms of the H2 rerun of the June benchmark against the June wombat run_log points."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

log, out = sys.argv[1], sys.argv[2]
rows = {}
for line in open(log):
    p = line.split()
    if p and p[0].isdigit():
        rows[int(p[0])] = (float(p[1]), float(p[4]), float(p[5]))
r = np.array([rows[k] for k in sorted(rows)])
# June wombat anim_full_Ra1e7_dEta1e3_res32_R8_mode1 (mid-shell Nu at r = 0.75), from W1
june = np.array([
    (0.00021, 5.1454, 1.0318), (0.00042, 5.7522, 1.0491), (0.00063, 6.2922, 1.0514),
    (0.00509, 83.505, 1.0081), (0.00932, 114.16, 1.8804), (0.01382, 136.10, 3.9492),
    (0.01834, 117.04, 4.3306), (0.02286, 117.08, 4.3179), (0.02735, 126.02, 4.3802),
    (0.03188, 134.54, 4.3650), (0.03646, 127.35, 4.2945), (0.04107, 120.80, 4.2265),
    (0.04569, 118.86, 4.1471)])
fig, (a, b) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
a.plot(r[:, 0], r[:, 2], color="#2a6fdb", lw=1.5, label="H2 rerun (surface Nu)")
a.plot(june[:, 0], june[:, 2], "o", color="#d1495b", label="June wombat (mid-shell Nu, r = 0.75)")
a.axhline(1, color="0.6", lw=0.8)
a.set_ylabel("Nu"); a.legend(frameon=False)
b.plot(r[:, 0], r[:, 1], color="#2a6fdb", lw=1.5, label="H2 rerun")
b.plot(june[:, 0], june[:, 1], "o", color="#d1495b", label="June wombat")
b.set_ylabel("vrms"); b.set_xlabel("t"); b.legend(frameon=False)
for ax in (a, b):
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("June benchmark (Ra 1e7, Δη 1e3, res 32, R 8, adapt every step)")
fig.tight_layout(); fig.savefig(out, dpi=150)
