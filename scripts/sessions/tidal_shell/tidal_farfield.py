"""Net tangential displacement of the shell over one orbit: the far-field
permanent deformation the stripes leave. Reads disp_*.npz (cumulative
displacement at cycle boundaries) of a faulted and an unfaulted run, takes
the last orbit's increment of each, and maps faulted minus unfaulted on the
Mollweide and in the polar view, with the stripes' own slip for scale.
Usage: <sim> <faulted run> <unfaulted run>"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sim, fr, ur = sys.argv[1:4]


def last_orbit(run):
    files = sorted(glob.glob(os.path.join(sim, run, "disp_*.npz")))
    a, b = np.load(files[-2]), np.load(files[-1])
    return b, b["disp_global"] - a["disp_global"], b["disp_polar"] - a["disp_polar"], float(b["t"]) - float(a["t"])


F, dgF, dpF, TF = last_orbit(fr)
U, dgU, dpU, TU = last_orbit(ur)
LON, LAT = F["lon"], F["lat"]; grid = F["grid"]
net_g = dgF - dgU; net_p = dpF - dpU
mag_g = np.linalg.norm(net_g, axis=1).reshape(LAT.shape)
mag_p = np.linalg.norm(net_p, axis=1).reshape(F["rho"].shape)
# the stripes' own net slip for scale
rs = json.load(open(os.path.join(sim, fr, "records.json")))
t = np.array([r["t"] for r in rs]); last = t > t.max() - 1 + 1e-9
names = sorted(rs[0]["slip"].keys())
# the orbit's net slip is the cumulative at its end minus the cumulative at the end
# of the previous orbit (the record BEFORE the first step of the last cycle)
i0 = int(np.argmax(last))
net_slip = np.mean([rs[-1]["slip"][n]["cumulative"] - rs[i0 - 1]["slip"][n]["cumulative"] for n in names])
# the unfaulted shell's own residual (numerical): the noise floor
floor = np.linalg.norm(dgU, axis=1)
print(f"orbit length {TF:.3f}; stripe-mean net slip over the orbit {net_slip:+.5f}")
print(f"far-field net displacement |faulted - unfaulted|: max {mag_g.max():.5f}, median {np.median(mag_g):.5f}; "
      f"unfaulted residual max {floor.max():.6f} (noise floor)")
# where it lives: fraction of the global rms within 0.5 rad of the stripe set's centre
c = F["centre"]; ang = np.degrees(np.arccos(np.clip(grid @ c, -1, 1)))
w = np.cos(LAT).ravel()
for rad in (15, 30, 60, 90):
    m = ang < rad
    print(f"  within {rad:3d} deg of the stripes: rms {np.sqrt((w[m]*np.linalg.norm(net_g[m],axis=1)**2).sum()/w[m].sum()):.5f}; "
          f"beyond: rms {np.sqrt((w[~m]*np.linalg.norm(net_g[~m],axis=1)**2).sum()/max(w[~m].sum(),1e-30)):.5f}")

fig = plt.figure(figsize=(18, 5.2))
ax = fig.add_subplot(1, 2, 1, projection="mollweide")
pc = ax.pcolormesh(LON, LAT, mag_g, cmap="magma", shading="auto")
ax.set_title(f"net tangential displacement over one orbit, faulted minus unfaulted (stripe net slip {net_slip:+.4f})", fontsize=9, loc="left")
ax.grid(alpha=0.3); fig.colorbar(pc, ax=ax, shrink=0.6, orientation="horizontal", pad=0.05)
ax = fig.add_subplot(1, 2, 2)
X, Y = F["rho"] * np.cos(F["psi"]), F["rho"] * np.sin(F["psi"])
pc = ax.pcolormesh(X, Y, mag_p, cmap="magma", shading="auto")
# arrows in the polar projection: the displacement's components along e1, e2
e1, e2 = F["e1"], F["e2"]
pg = F["pgrid"]
rho = F["rho"].ravel(); psi = F["psi"].ravel()
d_ = pg - (pg @ c)[:, None] * c; d_ /= np.maximum(np.linalg.norm(d_, axis=1), 1e-30)[:, None]
# local radial (away from the centre) and azimuthal unit vectors on the sphere
e_r = np.cos(rho)[:, None] * d_ - np.sin(rho)[:, None] * c[None]
e_a = np.cross(pg, e_r)
ur_, ua_ = (net_p * e_r).sum(1), (net_p * e_a).sum(1)
ex = ur_ * np.cos(psi) - ua_ * np.sin(psi); ey = ur_ * np.sin(psi) + ua_ * np.cos(psi)
sk = (slice(None, None, 6), slice(None, None, 4))
ax.quiver(X[sk], Y[sk], ex.reshape(X.shape)[sk], ey.reshape(X.shape)[sk], color="w", scale=mag_p.max() * 25, width=0.003)
for fp in F["faults"]:
    ax.plot(fp[:, 0], fp[:, 1], color="cyan", lw=2)
ax.set_aspect("equal"); ax.set_xlim(-0.9, 0.9); ax.set_ylim(-0.9, 0.9); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("polar view: magnitude and direction of the net displacement", fontsize=9, loc="left")
fig.colorbar(pc, ax=ax, shrink=0.8)
out = os.path.join(sim, f"farfield_{fr}_vs_{ur}.png")
fig.savefig(out, dpi=120, bbox_inches="tight"); print("wrote", out)
