"""Surface strain of the net displacement per orbit (faulted minus unfaulted):
dilatation and maximum shear on the polar azimuthal grid (geodesic polar
coordinates about the stripe set's centre, unit sphere), masked near the centre.
Reports the rms strain by distance from the set against the tidal strain scale A.
Usage: <sim> <faulted run> <unfaulted run> [A]"""
import glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sim, fr, ur = sys.argv[1:4]
A = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01


def last_orbit(run):
    f = sorted(glob.glob(os.path.join(sim, run, "disp_*.npz")))
    a, b = np.load(f[-2]), np.load(f[-1])
    return b, b["disp_polar"] - a["disp_polar"]


F, dF = last_orbit(fr); U, dU = last_orbit(ur)
net = dF - dU
rho, psi = F["rho"], F["psi"]                      # (npsi, nrho) grids
pg = F["pgrid"]; c = F["centre"]
r_ = rho.ravel()
d_ = pg - (pg @ c)[:, None] * c; d_ /= np.maximum(np.linalg.norm(d_, axis=1), 1e-30)[:, None]
e_r = np.cos(r_)[:, None] * d_ - np.sin(r_)[:, None] * c[None]        # geodesic radial direction
e_p = np.cross(pg, e_r)                                                # azimuthal


def strain_of(disp):
    u_r = (disp * e_r).sum(1).reshape(rho.shape); u_p = (disp * e_p).sum(1).reshape(rho.shape)
    r1 = rho[0]; p1 = psi[:, 0]
    du_r_dr = np.gradient(u_r, r1, axis=1); du_p_dr = np.gradient(u_p, r1, axis=1)
    du_r_dp = np.gradient(u_r, p1, axis=0); du_p_dp = np.gradient(u_p, p1, axis=0)
    sr, cr = np.sin(rho), np.cos(rho)
    e_rr = du_r_dr
    e_pp = du_p_dp / sr + u_r * cr / sr
    e_rp = 0.5 * (du_r_dp / sr + du_p_dr - u_p * cr / sr)
    return e_rr + e_pp, np.sqrt((0.5 * (e_rr - e_pp)) ** 2 + e_rp ** 2)


dil, shear = strain_of(net)
# the noise floor of this finite-difference measure: the same strain of the UNFAULTED
# shell's own residual over the orbit (a periodic shell has none; what remains is the
# grid interpolation of a P2 field, differenced)
dil_n, shear_n = strain_of(dU)
mask = rho < 0.03
dil = np.where(mask, np.nan, dil); shear = np.where(mask, np.nan, shear)
# the stripes' own offsets are displacement discontinuities, so the finite-difference
# strain inside the stripe block (corners at ~0.33 rad) is slip over grid spacing, not
# a far-field quantity: the colour scale is set from the field beyond 0.35 rad
far = rho > 0.35
lim_d = np.nanpercentile(np.abs(dil[far]), 99); lim_s = np.nanpercentile(shear[far], 99)
X, Y = rho * np.cos(psi), rho * np.sin(psi)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
for ax, fld, ttl, cmap, sym in ((axes[0], dil, "dilatation of the net displacement per orbit", "RdBu_r", True),
                                (axes[1], shear, "maximum shear strain of the net displacement per orbit", "magma", False)):
    lim = lim_d if sym else lim_s; kw = {"vmin": -lim, "vmax": lim} if sym else {"vmin": 0, "vmax": lim}
    pc = ax.pcolormesh(X, Y, fld, cmap=cmap, shading="auto", **kw)
    for fp in F["faults"]:
        ax.plot(fp[:, 0], fp[:, 1], color="cyan" if not sym else "k", lw=2)
    for rr in (0.25, 0.5, 0.75):
        ax.add_patch(plt.Circle((0, 0), rr, fill=False, color="gray", lw=0.5, alpha=0.6))
    ax.set_aspect("equal"); ax.set_xlim(-0.9, 0.9); ax.set_ylim(-0.9, 0.9); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(ttl, fontsize=9, loc="left"); fig.colorbar(pc, ax=ax, shrink=0.8)
fig.suptitle(f"{fr} minus {ur}: far-field strain per orbit (polar view, tidal strain scale A = {A})", fontsize=10)
out = os.path.join(sim, f"farfield_strain_{fr}_vs_{ur}.png"); fig.savefig(out, dpi=120, bbox_inches="tight"); print("wrote", out)
w = np.sin(rho)
print("  (inside 0.35 rad the strain is the stripes' own offsets over the grid spacing; far-field bands only)")
for lo, hi in ((0.35, 0.5), (0.5, 0.7), (0.7, 0.9)):
    m = (rho >= lo) & (rho < hi)
    rs = lambda f: np.sqrt((w[m] * f[m] ** 2).sum() / w[m].sum())
    print(f"  {np.degrees(lo):3.0f}-{np.degrees(hi):3.0f} deg from the set: rms shear strain {rs(shear):.2e} ({rs(shear)/A:.3f} A), "
          f"rms dilatation {rs(dil):.2e}; NOISE FLOOR (unfaulted residual) shear {rs(shear_n):.2e}, dilatation {rs(dil_n):.2e}")
